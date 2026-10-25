#!/usr/bin/env bash
# E026（G2c）：先做回归 G2c-0，通过后运行 K1_full（单次，与种子无关）与 K4_CL_cons（3 个种子）。
# 设计：docs/design_G2c.md。用法（仓库根目录）：bash experiments/E026_G2c_composition/run_g2c.sh
set -u
cd "$(dirname "$0")/../.."
source /hy-tmp/venv-metis/bin/activate
export PYTHONPATH=.:src
OUT=experiments/E026_G2c_composition
echo "start regression $(date)"
python $OUT/run.py --regression --out $OUT/regression > $OUT/regression.log 2>&1 || { echo "FAILED regression"; exit 1; }
grep -q "G2c-0 passed = True" $OUT/regression.log || { echo "G2c-0 NOT PASSED"; exit 1; }
echo "regression passed $(date)"
python $OUT/run.py --arm K1_full --seed 0 --out $OUT/runs/K1_full > $OUT/runs/K1_full.log 2>&1 || { echo "FAILED K1_full"; exit 1; }
echo "done K1_full $(date)"
for s in 0 1 2; do
  python $OUT/run.py --arm K4_CL_cons --seed $s --out $OUT/runs/K4_CL_cons_s$s > $OUT/runs/K4_CL_cons_s$s.log 2>&1 || { echo "FAILED K4 s$s"; exit 1; }
  echo "done K4_CL_cons_s$s $(date)"
done
echo "ALL DONE $(date)"
