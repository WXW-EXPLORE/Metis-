#!/usr/bin/env bash
# E031b：mem_norm 有效缩放对初始化的影响（不训练）。设计：docs/design_E031_init_mismatch.md §7
set -u
cd "$(dirname "$0")/../.."
source /hy-tmp/venv-metis/bin/activate
export PYTHONPATH=.:src
O=experiments/E031_init_mismatch/runs
run() { local n=$1; shift; [ -f $O/$n.json ] && { echo "skip $n"; return; }; echo "start $n $(date)"; python -u experiments/E031_init_mismatch/run_init.py "$@" --out $O/$n.json > $O/$n.log 2>&1 || { echo "FAILED $n"; exit 1; }; echo "done $n $(date)"; }
run S1_G   --arm GDN --mn_scale 1.0
run S1_R   --arm RLS --readout_den abs --mn_scale 1.0
run S02_G  --arm GDN --mn_scale 0.2
run S02_R  --arm RLS --readout_den abs --mn_scale 0.2
echo ALL DONE $(date)
