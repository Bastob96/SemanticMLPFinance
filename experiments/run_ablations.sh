#!/bin/bash
# Ablation study for the Stage 2 loop (Liu, 22 Oct): switch off one part at a time and see how much it matters.
#
# Full method = run C (20261021_1105_claude): Claude Sonnet, report cards, team trial, pair trials, 8 rounds.
#
#   Claude runs (need ANTHROPIC_API_KEY, about US$1 each):
#     abl_claude_nofeedback   Claude sees only the formulas already tried, no results
#     abl_claude_haiku        a smaller Claude model (Haiku) instead of Sonnet
#     abl_claude_16rounds     run C continued from round 9 to round 16 (more iterations)
#     abl_claude_repeat2/3    run C again twice (is it luck?)
#   No-Claude runs (no API key, no cost):
#     abl_C_check / abl_B_check     C's and B's factors re-judged with the same rules (should match C and B)
#     abl_C_nopairs / abl_B_nopairs the same factors with the pair trials switched off
#     abl_template_seed0/1/2        the same loop with template formulas instead of Claude
# Then the one-time Q4 test for every selected set: results/stage2/ablation/summary.csv
#
# Every step is skipped (or continued) if its folder is already done, so it is safe to run again.
# Usage:  export ANTHROPIC_API_KEY=...   then   bash experiments/run_ablations.sh
#         bash experiments/run_ablations.sh --no-claude     (only the free runs)
set -e
cd "$(dirname "$0")/.."
R=results/stage2/runs_v2
C=$R/20261021_1105_claude
OLD=results/stage2/runs/20261015_1635_claude
RUN="python experiments/run_stage2_loop_v2.py"

if [ "$1" != "--no-claude" ]; then
  if [ -z "$ANTHROPIC_API_KEY" ]; then echo "Set ANTHROPIC_API_KEY first (or use --no-claude)"; exit 1; fi
  echo "== Claude without feedback";  $RUN --rounds 8 --no-feedback --out $R/abl_claude_nofeedback --resume
  echo "== Smaller Claude model";     $RUN --rounds 8 --model claude-haiku-5-5 --out $R/abl_claude_haiku --resume
  echo "== 16 rounds (C continued)"
  [ -d $R/abl_claude_16rounds ] || cp -R $C $R/abl_claude_16rounds
  $RUN --rounds 16 --out $R/abl_claude_16rounds --resume
  echo "== Repeat C (2)";             $RUN --rounds 8 --out $R/abl_claude_repeat2 --resume
  echo "== Repeat C (3)";             $RUN --rounds 8 --out $R/abl_claude_repeat3 --resume
fi

echo "== Re-judge C and B (check) and with pair trials off"
$RUN --proposer replay --replay $C   --out $R/abl_C_check   --resume
$RUN --proposer replay --replay $C   --out $R/abl_C_nopairs --resume --no-pairs
$RUN --proposer replay --replay $OLD --out $R/abl_B_check   --resume
$RUN --proposer replay --replay $OLD --out $R/abl_B_nopairs --resume --no-pairs
for s in 0 1 2; do
  echo "== Template formulas, seed $s"
  $RUN --proposer template --rounds 8 --seed $s --out $R/abl_template_seed$s --resume
done

echo "== One-time Q4 test for every selected set"
python experiments/compare_stage2_runs.py out=results/stage2/ablation \
  full_C=$C full_B=$R/20261021_1110_replay_20261015_1635_claude \
  C_check=$R/abl_C_check B_check=$R/abl_B_check \
  C_no_pairs=$R/abl_C_nopairs B_no_pairs=$R/abl_B_nopairs \
  no_feedback=$R/abl_claude_nofeedback haiku=$R/abl_claude_haiku rounds_16=$R/abl_claude_16rounds \
  C_repeat2=$R/abl_claude_repeat2 C_repeat3=$R/abl_claude_repeat3 \
  template_0=$R/abl_template_seed0 template_1=$R/abl_template_seed1 template_2=$R/abl_template_seed2 \
  one_at_a_time_A=$OLD
echo "Done. Open 'Stage 2_v3_Ablation.ipynb' and Run All."
