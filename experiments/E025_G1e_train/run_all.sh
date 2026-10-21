#!/usr/bin/env bash
# E025（G1e）完整训练：3 个臂 × 3 个种子，顺序执行（单张 GPU）。
# 已存在 summary.json 的运行会跳过（断点续跑）。设计：docs/design_G1e.md。
# 用法（仓库根目录）：bash experiments/E025_G1e_train/run_all.sh
set -u
cd "$(dirname "$0")/../.."
mkdir -p experiments/E025_G1e_train/runs
source /hy-tmp/venv-metis/bin/activate
export PYTHONPATH=.:src
for seed in 0 1 2; do
  for arm in K1_full K4_CL_cons K4_random; do
    out=experiments/E025_G1e_train/runs/${arm}_s${seed}
    if [ -f "$out/summary.json" ]; then
      echo "skip $out (summary exists)"; continue
    fi
    echo "start $out $(date)"
    python experiments/E025_G1e_train/train.py --arm "$arm" --seed "$seed" --out "$out" \
      > "$out.log" 2>&1 || { echo "FAILED $out (exit $?) $(date)"; exit 1; }
    echo "done $out $(date)"
  done
done
echo "ALL DONE $(date)"
