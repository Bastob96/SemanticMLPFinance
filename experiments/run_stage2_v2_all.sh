#!/bin/bash
# Version 2 of Stage 2 in one go, on this computer (needs ANTHROPIC_API_KEY for run C):
#   C: the new loop with Claude (8 rounds of 6 factors, same budget as the 15 Oct run)
#   B: the 48 factors of the 15 Oct run, judged with the new rules (no Claude call)
#   then the comparison of A (15 Oct), B and C, including the one-time Q4 test.
# Usage: bash experiments/run_stage2_v2_all.sh
set -e
cd "$(dirname "$0")/.."
python experiments/run_stage2_loop_v2.py --rounds 8
python experiments/run_stage2_loop_v2.py --proposer replay --replay results/stage2/runs/20261015_1635_claude
python experiments/compare_stage2_runs.py
