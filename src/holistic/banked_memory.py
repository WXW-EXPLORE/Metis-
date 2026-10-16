"""0a 参考记忆：分库（A）、多时间尺度（B）、路由（E）、显式删除（C）、解耦损失（D）、归一化读出（F）。

接口与 Metis 的 local_memory 对齐（见 docs/design_0a.md §1），便于后期接入骨干：
    reset()                      清空状态
    write(k, v, beta=1.0)        按时间顺序写入，k: (B,T,d)，v: (B,T,dv)
    read(q)                      读出，q: (B,d) -> y: (B,dv)；q: (B,n,d) -> (B,n,dv)
    delete(k_hat, beta_d=1.0, value=None)   显式删除
    decoupling_loss()            （v1）库中心正交损失；v2 解耦见 weighted_gram / gram_offdiag_loss
    state / z / P                (B,K,d_b,dv)、(B,K,d_b,1)、(B,K,d_b,d_b)

两种更新规则（update_rule）：
  "gdn"（默认，Metis 的形式，行为与 v1 完全一致）：
      M_b <- alpha_b (M_b - beta g_b kappa kappa^T M_b) + beta g_b kappa v^T
  "rls"（设计 0a §10，增益由协方差解析给出）：
      e = v - M_b^T kappa;  gain = w P kappa / (lambda_b + w kappa^T P kappa)
      M_b <- M_b + gain e^T;  P_b <- (P_b - w P kappa kappa^T P / (lambda_b + w kappa^T P kappa)) / lambda_b
      其中 w = beta * g_b（硬路由时 g_b 为 0/1），lambda_b 为遗忘因子。
      有效极点 lambda_b / (lambda_b + w kappa^T P kappa)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

UPDATE_RULES = ("gdn", "rls")


def weighted_gram(keys, lam, weights=None):
    """可观测性格拉姆矩阵 W = sum_j lam^{2(T-j)} w_j kappa_j kappa_j^T（设计 0a §10.6）。

    keys: (T, d_b) 单位投影键（按时间顺序），lam: 遗忘因子，weights: (T,) 可选写入权重。
    返回 (d_b, d_b)。
    """
    T = keys.shape[0]
    ages = torch.arange(T - 1, -1, -1, dtype=keys.dtype, device=keys.device)  # T-1-j
    coef = lam ** (2 * ages)
    if weights is not None:
        coef = coef * weights
    return (keys * coef[:, None]).T @ keys


def gram_spread_loss(W, eps=1e-12):
    """格拉姆矩阵的特征值对数离散度（与坐标基无关）。

    mu 为 W 的特征值（按迹归一化），损失 = sum_i (log mu_i - mean log mu)^2。
    W 正比于 I（各方向同样可观测）时为 0；离散度越大，可观测性越不均匀。
    注意：非对角项依赖坐标基，不作为损失使用（见 design_0a §10.6 修正）。
    """
    mu = torch.linalg.eigvalsh(W).clamp_min(eps)
    mu = mu / mu.sum()
    logmu = mu.log()
    return ((logmu - logmu.mean()) ** 2).sum()


class BankedLinearMemory(nn.Module):
    def __init__(self, key_dim, value_dim, n_banks=1, bank_key_dim=None, top_k=None,
                 alphas=None, tau=1.0, seed=0, update_rule="gdn", p0=1.0, p_max=None,
                 proj_mode="random", straight_through=False):
        super().__init__()
        self.d = key_dim
        self.dv = value_dim
        self.K = n_banks
        self.d_b = bank_key_dim if bank_key_dim is not None else key_dim
        if not 1 <= self.d_b <= key_dim:
            raise ValueError("bank_key_dim must be in [1, key_dim]")
        if top_k is not None and not 1 <= top_k <= n_banks:
            raise ValueError("top_k must be in [1, n_banks] or None")
        if update_rule not in UPDATE_RULES:
            raise ValueError(f"update_rule must be one of {UPDATE_RULES}")
        self.top_k = top_k
        self.tau = tau
        self.update_rule = update_rule
        self.p0 = float(p0)
        self.p_max = p_max

        if alphas is None:
            alphas = [0.99] * n_banks
        if len(alphas) != n_banks or not all(0.0 < a <= 1.0 for a in alphas):
            raise ValueError("alphas must have one value in (0, 1] per bank")
        # float64 存储：时间常数需要精确表示（例如 0.9 在 float32 中有约 2e-8 的误差）
        self.register_buffer("alpha", torch.tensor(alphas, dtype=torch.float64))

        gen = torch.Generator().manual_seed(seed)
        if proj_mode not in ("random", "block"):
            raise ValueError("proj_mode must be 'random' or 'block'")
        if proj_mode == "block" and n_banks * self.d_b > key_dim:
            raise ValueError("block projection needs n_banks * bank_key_dim <= key_dim")
        self.proj_mode = proj_mode
        self.straight_through = straight_through
        proj = []
        for b in range(n_banks):
            if self.d_b == key_dim:
                proj.append(torch.eye(key_dim))
            elif proj_mode == "block":
                # 分块：库 b 只取坐标 [b*d_b, (b+1)*d_b)，不混合坐标
                proj.append(torch.eye(key_dim)[b * self.d_b:(b + 1) * self.d_b])
            else:
                q, _ = torch.linalg.qr(torch.randn(key_dim, self.d_b, generator=gen))
                proj.append(q.T.contiguous())
        self.register_buffer("proj", torch.stack(proj))  # (K, d_b, d)

        # 可训练的库中心（路由）——经典理论未给出分配规则，因此学习（设计 0a §12）
        self.centers = nn.Parameter(F.normalize(torch.randn(n_banks, key_dim, generator=gen), dim=-1))

        self.record = False
        self.routing_log = []
        self.t = 0                # 全局写入步数（用于计算条目年龄，见 delete）
        self.clamp_count = 0      # RLS：协方差迹超过 p_max 的次数
        self.fallback_count = 0   # RLS 删除：退回投影删除的次数
        self.reset()

    # ── 状态 ──────────────────────────────────────────────────────────────
    def reset(self):
        self._state = None
        self._z = None
        self._P = None

    @property
    def state(self):
        return self._state

    @property
    def z(self):
        return self._z

    @property
    def P(self):
        return self._P

    def _ensure(self, bsz, device, dtype):
        """状态为空时创建零状态（RLS 时同时创建 P = p0 * I）；批量大小不一致时报错。"""
        if self._state is None:
            self._state = torch.zeros(bsz, self.K, self.d_b, self.dv, device=device, dtype=dtype)
            self._z = torch.zeros(bsz, self.K, self.d_b, 1, device=device, dtype=dtype)
            if self.update_rule == "rls":
                eye = torch.eye(self.d_b, device=device, dtype=dtype)
                self._P = (self.p0 * eye).expand(bsz, self.K, self.d_b, self.d_b).clone()
        elif self._state.shape[0] != bsz:
            raise ValueError(
                f"batch size {bsz} does not match memory batch {self._state.shape[0]}; call reset() first"
            )

    # ── 路由与投影 ────────────────────────────────────────────────────────
    def route(self, kn):
        """kn: (B,d) 单位键 -> 路由权重 (B,K)。

        top_k=1 且 straight_through=True 时，前向为硬选择（one-hot），反向经 softmax 传梯度（直通估计）。
        """
        logits = kn @ self.centers.to(kn.dtype).T / self.tau
        if self.top_k is None or self.top_k >= self.K:
            return torch.softmax(logits, dim=-1)
        _, idx = logits.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(logits).scatter(-1, idx, 1.0).bool()
        soft = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
        if self.straight_through and self.top_k == 1:
            # 直通：软分支必须对全部库做 softmax（只对选中项做 softmax 恒为 1，梯度为 0）
            hard = torch.zeros_like(soft).scatter(-1, idx, 1.0)
            soft_all = torch.softmax(logits, dim=-1)
            return hard + (soft_all - soft_all.detach())
        return soft

    def load_balance_loss(self, kn):
        """Switch 式负载均衡：K * sum_b f_b P_b，f_b 为硬选择比例，P_b 为平均软概率。kn: (N,d) 单位键。"""
        logits = kn @ self.centers.to(kn.dtype).T / self.tau
        soft = torch.softmax(logits, dim=-1)
        hard = torch.zeros_like(soft).scatter(-1, logits.argmax(-1, keepdim=True), 1.0)
        f = hard.mean(dim=0)
        P = soft.mean(dim=0)
        return self.K * (f * P).sum()

    def _bank_keys(self, kn):
        """(B,d) -> 各库的单位投影键 (B,K,d_b)。"""
        kb = torch.einsum("kjd,bd->bkj", self.proj.to(kn.dtype), kn)
        return F.normalize(kb, dim=-1)

    def _log_routing(self, g):
        if self.record:
            self.routing_log.append(g.detach().float().cpu())

    # ── 写入 ──────────────────────────────────────────────────────────────
    def write(self, k, v, beta=1.0):
        bsz, T, _ = k.shape
        self._ensure(bsz, k.device, k.dtype)
        kn = F.normalize(k, dim=-1)
        if not torch.is_tensor(beta):
            beta = torch.full((bsz, T), float(beta), device=k.device, dtype=k.dtype)
        for t in range(T):
            self._write_step(kn[:, t], v[:, t], beta[:, t])

    def write_chunk(self, k, v, beta=1.0):
        """RLS 的块更新（信息形式，与逐 token 的 write 数值等价；仅 K=1）。

        信息形式：Lambda_t = lam Lambda_{t-1} + w_t kappa kappa^T，b_t = lam b_{t-1} + w_t kappa v^T，
        M = Lambda^{-1} b。展开 L 步得到闭式，因此一个 chunk 只需一次求逆，结果精确。
        """
        if self.update_rule != "rls" or self.K != 1:
            raise ValueError("write_chunk requires update_rule='rls' and n_banks=1")
        bsz, L, _ = k.shape
        self._ensure(bsz, k.device, k.dtype)
        out_dtype = self._state.dtype
        # 求逆需要至少 fp32（bf16 不支持 linalg.inv）；结果转回状态原 dtype
        cdt = torch.float64 if out_dtype == torch.float64 else torch.float32
        k, v = k.to(cdt), v.to(cdt)
        kn = F.normalize(k, dim=-1)
        kb = torch.einsum("jd,bld->blj", self.proj[0].to(cdt), kn)
        kb = F.normalize(kb, dim=-1)                                      # (B,L,d_b)
        if not torch.is_tensor(beta):
            beta = torch.full((bsz, L), float(beta), device=k.device, dtype=cdt)
        beta = beta.to(cdt)
        lam = self.alpha.to(cdt)[0]
        ages = (L - 1 - torch.arange(L, device=k.device, dtype=kb.dtype))  # 块内每步到块末的衰减次数
        c = beta * lam ** ages                                            # (B,L) 信息权重

        P0 = self._P[:, 0].to(cdt)
        M0 = self._state[:, 0].to(cdt)
        z0 = self._z[:, 0].to(cdt)
        Lam0 = torch.linalg.inv(P0)
        b0 = Lam0 @ M0
        Lam = (lam ** L) * Lam0 + torch.einsum("bl,blj,blk->bjk", c, kb, kb)
        b = (lam ** L) * b0 + torch.einsum("bl,blj,blv->bjv", c, kb, v)
        P = torch.linalg.inv(Lam)
        M = P @ b
        z = (lam ** L) * z0 + torch.einsum("bl,blj->bj", c, kb)[..., None]
        self._P = P[:, None].to(out_dtype)
        self._state = M[:, None].to(out_dtype)
        self._z = z[:, None].to(out_dtype)

    def _write_step(self, kn, vt, beta_t):
        self.t += 1
        g = self.route(kn)                       # (B,K)
        self._log_routing(g)
        kb = self._bank_keys(kn)                 # (B,K,d_b)
        if self.update_rule == "rls":
            self._rls_step(kb, g, vt, beta_t)
        else:
            self._gdn_step(kb, g, vt, beta_t)

    def _gdn_step(self, kb, g, vt, beta_t):
        w = beta_t.view(-1, 1) * g               # (B,K) 有效写入强度
        w4 = w[:, :, None, None]
        a = self.alpha.to(kb.dtype).view(1, self.K, 1, 1)

        M = self._state
        km = (kb.unsqueeze(-1) * M).sum(dim=-2)                       # (B,K,dv) = kappa^T M
        erase = w4 * kb[:, :, :, None] * km[:, :, None, :]
        add = w4 * kb[:, :, :, None] * vt[:, None, None, :]
        self._state = a * (M - erase) + add

        z = self._z
        kz = (kb.unsqueeze(-1) * z).sum(dim=-2)
        erase_z = w4 * kb[:, :, :, None] * kz[:, :, None, :]
        add_z = w4 * kb[:, :, :, None]
        self._z = a * (z - erase_z) + add_z

    def _rls_step(self, kb, g, vt, beta_t):
        """带遗忘因子的 RLS 写入（解析增益，见模块文档）。"""
        w = beta_t.view(-1, 1) * g                               # (B,K)
        lam = self.alpha.to(kb.dtype).view(1, self.K)           # (1,K)
        M = self._state
        P = self._P

        pred = (kb.unsqueeze(-1) * M).sum(dim=-2)                # (B,K,dv) = kappa^T M
        e = vt[:, None, :] - pred                                # 新息
        Pk = torch.einsum("bkij,bkj->bki", P, kb)                # (B,K,d_b) = P kappa
        s = (kb * Pk).sum(dim=-1)                                # (B,K) = kappa^T P kappa
        denom = lam + w * s                                      # (B,K)
        coef = w / denom                                         # (B,K)
        gain = coef[..., None] * Pk                              # (B,K,d_b)
        self._state = M + gain[..., None] * e[:, :, None, :]

        P_new = (P - coef[..., None, None] * (Pk[..., :, None] * Pk[..., None, :])) / lam[..., None, None]
        P_new = self._clamp_P(P_new)
        self._P = P_new

        self._z = lam[..., None, None] * self._z + w[:, :, None, None] * kb[:, :, :, None]

    def _clamp_P(self, P):
        """协方差迹上界（防止未激励方向的 windup），超出时缩放并计数。"""
        if self.p_max is None:
            return P
        tr = P.diagonal(dim1=-2, dim2=-1).sum(dim=-1)            # (B,K)
        over = tr > self.p_max
        self.clamp_count += int(over.sum().item())
        scale = torch.where(over, self.p_max / tr.clamp_min(1e-12), torch.ones_like(tr))
        return P * scale[..., None, None]

    # ── 读出（F）──────────────────────────────────────────────────────────
    def read(self, q):
        """q: (B,d) -> y: (B,dv)；或 q: (B,n,d) -> y: (B,n,dv)（同一记忆对 n 个查询读出）。"""
        if q.dim() == 3:
            return torch.stack([self.read(q[:, i]) for i in range(q.shape[1])], dim=1)
        bsz = q.shape[0]
        self._ensure(bsz, q.device, q.dtype)
        qn = F.normalize(q, dim=-1)
        g = self.route(qn)
        self._log_routing(g)
        qb = self._bank_keys(qn)
        num = (qb.unsqueeze(-1) * self._state).sum(dim=-2)
        den = (qb.unsqueeze(-1) * self._z).sum(dim=-2) + 1.0
        per_bank = num / den
        return (g.unsqueeze(-1) * per_bank).sum(dim=1)

    # ── 显式删除（C）──────────────────────────────────────────────────────
    def delete(self, k_hat, beta_d=1.0, value=None, age=None):
        """删除地址 k_hat 处的内容。

        gdn：沿 kappa 的投影擦除（不需要 value）。
        rls：若给出 value（被删条目的值），用降秩（downdate）撤销其写入；
             条件 w kappa^T P kappa < 1 不满足的库退回投影删除并计数。
             降秩在 lambda=1 时精确，lambda<1 时为近似（写入之后的遗忘未被撤销）。
        """
        if self._state is None:
            return
        if k_hat.shape[0] != self._state.shape[0]:
            raise ValueError("delete: batch size does not match memory batch")
        kn = F.normalize(k_hat, dim=-1)
        g = self.route(kn)
        kb = self._bank_keys(kn)
        if self.update_rule == "rls" and value is not None:
            self._rls_downdate(kb, g, value, beta_d, age)
            return
        if self.update_rule == "rls":
            self.fallback_count += int((g > 0).sum().item())
        w = (beta_d * g)[:, :, None, None]
        M = self._state
        km = (kb.unsqueeze(-1) * M).sum(dim=-2)
        self._state = M - w * kb[:, :, :, None] * km[:, :, None, :]
        z = self._z
        kz = (kb.unsqueeze(-1) * z).sum(dim=-2)
        self._z = z - w * kb[:, :, :, None] * kz[:, :, None, :]

    def _rls_downdate(self, kb, g, value, beta_d, age):
        """降秩删除（设计 0a §10.4 修订）。

        信息形式下，第 j 条写入在 P^{-1} 与 b 中的权重为 c_j = lambda^{age_j} * w_j。
        删除该条即以 c_j 做降秩：M <- M - coef Pk e^T，P <- P + coef Pk Pk^T，
        coef = c/(1 - c s)，s = kappa^T P kappa。对任意 lambda 精确，前提是 age 已知。
        age 未给出且 lambda<1 时，退回投影删除并计数。
        """
        w = beta_d * g                                           # (B,K)
        M = self._state
        P = self._P
        lam = self.alpha.to(kb.dtype).view(1, self.K)
        if age is None:
            c = torch.where(lam == 1.0, w, torch.zeros_like(w))  # 无年龄信息：只对 lambda=1 的库降秩
            known = (lam == 1.0).expand_as(w)
        else:
            c = w * lam ** float(age)
            known = torch.ones_like(w, dtype=torch.bool)
        pred = (kb.unsqueeze(-1) * M).sum(dim=-2)
        e = value[:, None, :] - pred                             # v_del - kappa^T M
        Pk = torch.einsum("bkij,bkj->bki", P, kb)
        s = (kb * Pk).sum(dim=-1)
        cs = c * s
        ok = known & ((cs < 1.0) | (c == 0))
        denom = (1.0 - cs).clamp_min(1e-12)
        coef = torch.where(ok, c / denom, torch.zeros_like(c))
        self._state = M - coef[..., None, None] * (Pk[..., :, None] * e[:, :, None, :])
        self._P = P + coef[..., None, None] * (Pk[..., :, None] * Pk[..., None, :])
        # 无法降秩的活跃库：投影删除（M 与 z）
        bad = (~ok) & (w > 0)
        self.fallback_count += int(bad.sum().item())
        w_bad = torch.where(bad, w, torch.zeros_like(w))[..., None, None]
        km = (kb.unsqueeze(-1) * self._state).sum(dim=-2)
        self._state = self._state - w_bad * kb[:, :, :, None] * km[:, :, None, :]
        z = self._z
        kz = (kb.unsqueeze(-1) * z).sum(dim=-2)
        z_ok = torch.where(ok, c, torch.zeros_like(c))[..., None, None]
        self._z = z - z_ok * kb[:, :, :, None] - w_bad * kb[:, :, :, None] * kz[:, :, None, :]

    # ── 解耦（D）──────────────────────────────────────────────────────────
    def decoupling_loss(self):
        """（v1）库中心的正交损失 ||C C^T - I||_F^2。K=1 时为 0。"""
        c = F.normalize(self.centers, dim=-1)
        gram = c @ c.T
        eye = torch.eye(self.K, device=c.device, dtype=c.dtype)
        return ((gram - eye) ** 2).sum()

    # ── 监控 ──────────────────────────────────────────────────────────────
    def routing_summary(self):
        """路由熵与库利用率（需先设置 record=True 并写入/读出）。"""
        if not self.routing_log:
            return None
        g = torch.cat(self.routing_log, dim=0)
        entropy = -(g.clamp_min(1e-12).log() * g).sum(dim=-1).mean().item()
        max_entropy = float(torch.log(torch.tensor(float(self.K))).item()) if self.K > 1 else 0.0
        return {"entropy": entropy, "max_entropy": max_entropy,
                "utilization": g.mean(dim=0).tolist()}
