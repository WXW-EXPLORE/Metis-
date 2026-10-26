#!/usr/bin/env bash
# E029（B）：RLS 与 GDN 差距的 pilot 诊断。设计：docs/design_E029_gap_diagnosis.md
# 只用 pilot 划分（第 5 个对话），不读测试集。种子 10。顺序执行（单 GPU）。已有 summary.json 的运行跳过。
set -u
cd "$(dirname "$0")/../.."
source /hy-tmp/venv-metis/bin/activate
export PYTHONPATH=.:src
COMMON="--pilot --seed 10 --readout_den abs --max_train_tokens 3000 --n_test 100"
run() {
  local name=$1; shift
  local out=experiments/E029_gap_diagnosis/runs/$name
  if [ -f "$out/summary.json" ]; then echo "skip $name"; return; fi
  echo "start $name $(date)"
  python experiments/E025_G1e_train/train.py $COMMON "$@" --out "$out" > "$out.log" 2>&1 \
    || { echo "FAILED $name (exit $?) $(date)"; exit 1; }
  echo "done $name $(date)"
}
mkdir -p experiments/E029_gap_diagnosis/runs
run P0_K1_base   --arm K1_full --lam 0.9 --lr 1e-4
run P1_K1_lam099 --arm K1_full --lam 0.99 --lr 1e-4
run P2_K1_lr3e4  --arm K1_full --lam 0.9 --lr 3e-4
run P3_GDN_base  --arm GDN --lr 1e-4
echo "ALL DONE $(date)"
