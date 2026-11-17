#!/bin/bash
# Batch mode (Liu, 12 Nov): Claude proposes 100 interpretable factors in one call, then the usual search.
#
#   batch100_claude_1/2/3     Claude Sonnet, 3 repeats (randomness was the biggest effect in the ablation)
#   batch100_template_0/1/2   the same search on 100 template formulas (no LLM)
#   batch100_noise_0/1/2      the same search on 100 factors of pure random numbers: how many pass by luck?
# Then the pooled Q4 test (all four Q4 quarters together) for every run with kept factors.
#
# Each step is skipped if its folder is already done, so it is safe to run again.
# Use the same Python as the earlier runs (Anaconda, scikit-learn 1.7.2): leave .venv switched off.
# Usage:  export ANTHROPIC_API_KEY=...   then   bash experiments/run_batch.sh
#         bash experiments/run_batch.sh --no-claude     (only the free runs)
set -e
cd "$(dirname "$0")/.."
R=results/stage2/runs_v2
RUN="python experiments/run_stage2_loop_v2.py --batch 100 --resume"

if [ "$1" != "--no-claude" ]; then
  if [ -z "$ANTHROPIC_API_KEY" ]; then echo "Set ANTHROPIC_API_KEY first (or use --no-claude)"; exit 1; fi
  for i in 1 2 3; do
    echo "== Claude batch of 100, repeat $i"; $RUN --out $R/batch100_claude_$i
  done
fi
for s in 0 1 2; do
  echo "== Template batch of 100, seed $s"; $RUN --proposer template --seed $s --out $R/batch100_template_$s
  echo "== Random-number batch of 100, seed $s"; $RUN --proposer noise --seed $s --out $R/batch100_noise_$s
done

echo "== Pooled Q4 test (all four Q4 quarters together)"
python experiments/run_pooled_test.py
echo "Done. Results: results/stage2/pooled/summary.csv"
