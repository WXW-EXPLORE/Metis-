#!/usr/bin/env bash
# E025（G1e）修订版完整训练：读出 abs（1+|q·z|），训练集剔除 >4000 token 的 episode，原型来自 abs 的准备步骤。
# 设计与修订：docs/design_G1e.md §10、docs/design_readout_norm.md。
# 3 个臂 × 3 个种子，顺序执行；已存在 summary.json 的运行跳过（断点续跑）；失败即停止。
# 用法（仓库根目录）：bash experiments/E025_G1e_train/run_all_abs.sh
set -u
cd "$(dirname "$0")/../.."
source /hy-tmp/venv-metis/bin/activate
export PYTHONPATH=.:src
RUNS=experiments/E025_G1e_train/runs_abs
PROTO=experiments/E025_G1e_train/abs/prototypes.pt
mkdir -p "$RUNS"
for seed in 0 1 2; do
  for arm in K1_full K4_CL_cons K4_random; do
    out=$RUNS/${arm}_s${seed}
    if [ -f "$out/summary.json" ]; then
      echo "skip $out (summary exists)"; continue
    fi
    echo "start $out $(date)"
    python experiments/E025_G1e_train/train.py --arm "$arm" --seed "$seed" --out "$out" \
      --prototypes "$PROTO" --readout_den abs --max_train_tokens 3000 \
      > "$out.log" 2>&1 || { echo "FAILED $out (exit $?) $(date)"; exit 1; }
    echo "done $out $(date)"
  done
done
echo "ALL DONE $(date)"
