#!/usr/bin/env bash
# E031：初始不匹配诊断的六个变体（顺序执行）。设计：docs/design_E031_init_mismatch.md
set -u
cd "$(dirname "$0")/../.."
source /hy-tmp/venv-metis/bin/activate
export PYTHONPATH=.:src
O=experiments/E031_init_mismatch/runs
run() { local n=$1; shift; [ -f $O/$n.json ] && { echo "skip $n"; return; }; echo "start $n $(date)"; python experiments/E031_init_mismatch/run_init.py "$@" --out $O/$n.json > $O/$n.log 2>&1 || { echo "FAILED $n"; exit 1; }; echo "done $n $(date)"; }
run G_base      --arm GDN
run R0_base    --arm RLS --readout_den abs
run ZG_nomem   --arm GDN --read_scale 0
run ZR_nomem   --arm RLS --readout_den abs --read_scale 0
run Rorig      --arm RLS --readout_den orig
run SR_0p1     --arm RLS --readout_den abs --read_scale 0.1
echo ALL DONE $(date)
