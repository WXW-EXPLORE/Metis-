"""Phase 0 合成诊断用的参考线性记忆（无可学习参数）。

状态递推 M_{t+1} = lam * M_t + (1 - lam) * k_t v_t^T，读出 y = M^T q。
对应 docs/design_diagnostics.md §2，与 ideas_Cybernetics.md §5 的基本形式一致。
空步用零键表示，即 M 只按 lam 衰减。
"""
import numpy as np


def simulate(keys, values, lam, M0=None):
    """按序写入 (keys[t], values[t])，返回最终状态 M，形状 (d, dv)。"""
    d, dv = keys.shape[1], values.shape[1]
    M = np.zeros((d, dv)) if M0 is None else M0.copy()
    for k, v in zip(keys, values):
        M = lam * M + (1 - lam) * np.outer(k, v)
    return M


def unit(x):
    return x / np.linalg.norm(x)


def make_target_and_distractors(rng, d, dv, n, rho):
    """生成目标键 k_c、目标值 v_c，以及 n 个与 k_c 余弦严格等于 rho 的干扰键。

    干扰键 k_j = rho * k_c + sqrt(1 - rho^2) * n_j，其中 n_j 为单位向量且与 k_c 正交，
    因此 ||k_j|| = 1 且 k_j · k_c = rho。
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    k_c = unit(rng.standard_normal(d))
    v_c = rng.standard_normal(dv)
    noise = rng.standard_normal((n, d))
    noise -= np.outer(noise @ k_c, k_c)
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    distractor_keys = rho * k_c + np.sqrt(1.0 - rho**2) * noise
    distractor_values = rng.standard_normal((n, dv))
    return k_c, v_c, distractor_keys, distractor_values


def decay_probe(k_c, v_c, lam, age):
    """只写入目标，经过 age 步衰减后读出，返回沿 v_c 方向的振幅。

    理论值为 (1 - lam) * lam**age（对应 ideas §5 的 λ^{T-c}(1-λ)(q·k_c)）。
    """
    L = age + 1
    keys = np.zeros((L, k_c.size))
    values = np.zeros((L, v_c.size))
    keys[0], values[0] = k_c, v_c
    M = simulate(keys, values, lam)
    y = M.T @ k_c
    return float(y @ v_c / (v_c @ v_c))


def crosstalk_probe(rng, d, dv, lam, age, n, rho):
    """固定目标年龄 age，在最后 n 步写入干扰；返回目标项与串扰项的范数及 SNR。

    目标在第 0 步写入，干扰占用第 age-n+1 .. age 步，读出在第 age 步。
    串扰项由线性性直接得到：完整读出减去仅含目标时的读出。
    """
    if n >= age:
        raise ValueError("need n < age so that the target stays at index 0")
    k_c, v_c, dk, dv_ = make_target_and_distractors(rng, d, dv, n, rho)
    L = age + 1
    keys = np.zeros((L, d))
    values = np.zeros((L, dv))
    keys[0], values[0] = k_c, v_c
    keys[L - n:], values[L - n:] = dk, dv_
    M_full = simulate(keys, values, lam)
    keys_t = np.zeros((L, d))
    values_t = np.zeros((L, dv))
    keys_t[0], values_t[0] = k_c, v_c
    M_target = simulate(keys_t, values_t, lam)
    y_target = M_target.T @ k_c
    y_cross = M_full.T @ k_c - y_target
    t_norm = float(np.linalg.norm(y_target))
    c_norm = float(np.linalg.norm(y_cross))
    # 串扰近似为零（数值噪声量级）时 SNR 无意义，记为 NaN
    snr = t_norm / c_norm if c_norm > 1e-12 else float("nan")
    # 新写入目标的振幅为 (1-lam)||v_c||，与衰减无关；串扰相对它的大小不被 lam^age 混淆
    fresh = (1 - lam) * float(np.linalg.norm(v_c))
    return {
        "target_norm": t_norm,
        "crosstalk_norm": c_norm,
        "snr": snr,
        "crosstalk_over_fresh": c_norm / fresh,
    }


def delete_projection(M, k_hat):
    """按键擦除（GDN 式，beta=1）：M <- (I - k̂ k̂^T) M，k̂ 为单位向量。"""
    return M - np.outer(k_hat, k_hat @ M)


def delete_global(M, gamma):
    """全局遗忘式删除：M <- gamma * M，不寻址。"""
    return gamma * M


def deletion_probe(rng, d, dv, lam, age, n, rho, method, gamma=0.0):
    """在写入目标与 n 个干扰之后删除目标，测量残留与误伤。

    返回目标残留比例（删除后目标读出 / 删除前）与其余条目的平均误伤比例。
    误伤定义为 ||ΔM^T k_j|| / ||M_before^T k_j||，对 j 取平均。
    """
    k_c, v_c, dk, dv_ = make_target_and_distractors(rng, d, dv, n, rho)
    L = age + 1
    keys = np.zeros((L, d))
    values = np.zeros((L, dv))
    keys[0], values[0] = k_c, v_c
    keys[L - n:], values[L - n:] = dk, dv_
    M_before = simulate(keys, values, lam)
    if method == "projection":
        M_after = delete_projection(M_before, k_c)
    elif method == "global":
        M_after = delete_global(M_before, gamma)
    else:
        raise ValueError(f"unknown method: {method}")
    y_c_before = np.linalg.norm(M_before.T @ k_c)
    residual = float(np.linalg.norm(M_after.T @ k_c) / y_c_before)
    dM = M_after - M_before
    before_j = np.linalg.norm(dk @ M_before, axis=1)
    delta_j = np.linalg.norm(dk @ dM, axis=1)
    collateral = float(np.mean(delta_j / before_j))
    return {"target_residual": residual, "collateral": collateral}
