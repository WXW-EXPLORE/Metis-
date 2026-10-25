#!/usr/bin/env bash
# E026 G2c-b'（第二次登记）：K4 的三个种子与 K1 参照。设计：docs/design_G2c.md §10。
set -u
cd "$(dirname "$0")/../.."
source /hy-tmp/venv-metis/bin/activate
export PYTHONPATH=.:src
OUT=experiments/E026_G2c_composition/g2cb_v2
mkdir -p $OUT
for s in 0 1 2; do
  python experiments/E026_G2c_composition/run_g2cb_v2.py --arm K4_CL_cons --seed $s --out $OUT/K4_CL_cons_s$s > $OUT/K4_CL_cons_s$s.log 2>&1 || { echo "FAILED K4 s$s"; exit 1; }
  echo "done K4 s$s $(date)"
done
python experiments/E026_G2c_composition/run_g2cb_v2.py --arm K1_full --seed 0 --out $OUT/K1_full > $OUT/K1_full.log 2>&1 || { echo "FAILED K1"; exit 1; }
echo "ALL DONE $(date)"
