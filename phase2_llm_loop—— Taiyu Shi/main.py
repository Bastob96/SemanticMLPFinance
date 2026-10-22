"""Capstone Phase 2: LLM-driven factor generation with an automated feedback loop.

Examples (run from the project folder):
    python main.py --baseline-only          # Stage 0 only: reproduce the Phase 1 baseline on validation
    python main.py --dry-run                # full pipeline with offline mock candidates (no API calls)
    python main.py                          # live run: 3 rounds x 5 candidates with Claude
    python main.py --rounds 2 --candidates 5 --no-cache
"""

import argparse
import logging
import platform
import sys
import traceback

import config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Capstone Phase 2 LLM factor loop",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Use fixed offline candidates instead of Claude (no API calls, no LLM feedback)")
    mode.add_argument("--baseline-only", action="store_true",
                      help="Run Stage 0 (baseline on validation) only; the test set is not touched")
    parser.add_argument("--rounds", type=int, default=config.DEFAULT_ROUNDS,
                        help=f"Number of feedback rounds (default {config.DEFAULT_ROUNDS})")
    parser.add_argument("--candidates", type=int, default=config.DEFAULT_CANDIDATES,
                        help=f"Candidate factors requested per round (default {config.DEFAULT_CANDIDATES})")
    parser.add_argument("--model", help="Claude model ID (overrides CLAUDE_MODEL in .env)")
    parser.add_argument("--effort", help="Claude effort level: low|medium|high|xhigh|max|none")
    parser.add_argument("--no-cache", action="store_true", help="Do not read or write the LLM response cache")
    parser.add_argument("--no-fallbacks", action="store_true", help="Disable server-side refusal fallback")
    parser.add_argument("--data", help=f"Path to the OHLCV CSV (default {config.DATA_PATH})")
    args = parser.parse_args(argv)
    if args.rounds < 1:
        parser.error("--rounds must be at least 1")
    if not 1 <= args.candidates <= 20:
        parser.error("--candidates must be between 1 and 20")
    return args


def library_versions():
    import anthropic
    import matplotlib
    import numpy
    import pandas
    import sklearn
    return {"python": platform.python_version(), "platform": platform.platform(), "numpy": numpy.__version__,
            "pandas": pandas.__version__, "scikit-learn": sklearn.__version__,
            "matplotlib": matplotlib.__version__, "anthropic": anthropic.__version__}


def make_figures(runner, out, mode_label):
    import visualizer
    importance, corr, candidates = runner.figure_inputs()
    llm_names = {f["name"] for f in runner.selection["selected_llm_factors"]}
    s = runner.splits
    figures = {}
    steps = [
        ("fig1", lambda: visualizer.fig1_final_comparison(runner.test_summary, out.figures_dir, mode_label)),
        ("fig2", lambda: visualizer.fig2_iteration_performance(runner.set_history, out.figures_dir, mode_label)),
        ("fig3", lambda: visualizer.fig3_feature_importance(
            importance, out.figures_dir, f"{s.validation.min().date()} to {s.validation.max().date()}", mode_label)),
        ("fig4", lambda: visualizer.fig4_actual_vs_predicted(runner.test_predictions, out.figures_dir, mode_label)),
        ("fig5", lambda: visualizer.fig5_correlation_heatmap(
            corr, llm_names, out.figures_dir, f"{s.train.min().date()} to {s.train.max().date()}", mode_label)),
        ("fig6", lambda: visualizer.fig6_candidate_outcomes(candidates, out.figures_dir, mode_label)),
    ]
    for key, make in steps:
        try:
            figures[key] = str(make())
        except Exception as exc:   # report, never fabricate
            logging.getLogger("capstone").error("Figure %s failed: %s", key, exc)
            figures[key] = f"FAILED: {exc}"
    return figures


def write_summary(runner, out, settings, figures):
    import pandas as pd
    rows = [{"stage": "baseline" if h["round"] == 0 else f"round_{h['round']}", "round": h["round"],
             "llm_factors": ";".join(h["llm_factors"]), "n_llm_factors": len(h["llm_factors"]),
             "validation_mae": h["metrics"]["mae"], "validation_rmse": h["metrics"]["rmse"],
             "validation_directional_accuracy": h["metrics"]["directional_accuracy"],
             "validation_spearman_ic": h["metrics"]["spearman_ic"],
             "round_evaluated_mae": (h["round_evaluated_metrics"] or {}).get("mae"),
             "round_evaluated_rmse": (h["round_evaluated_metrics"] or {}).get("rmse")}
            for h in runner.set_history]
    out.save_csv("iteration_summary.csv", pd.DataFrame(rows))
    usage = pd.DataFrame(runner.usage_rows)
    totals = {}
    if not usage.empty:
        for column in ("input_tokens", "output_tokens", "estimated_cost_usd"):
            totals[column] = float(pd.to_numeric(usage[column], errors="coerce").fillna(0).sum())
    m = runner.test_summary["metrics"]["refit_train_validation"]
    out.save_json("run_summary.json", {
        "run_id": out.run_id, "mode": "dry_run" if settings.dry_run else "live",
        "live_llm_feedback_demonstrated": not settings.dry_run,
        "rounds": settings.rounds, "candidates_per_round": settings.candidates,
        "selected_round": runner.selection["selected_round"],
        "selected_llm_factors": [f["name"] for f in runner.selection["selected_llm_factors"]],
        "negative_result": runner.selection["negative_result"],
        "validation": {"baseline_mae": runner.baseline_metrics["mae"],
                       "selected_mae": runner.selection["selected_validation_metrics"]["mae"]},
        "test_refit_train_validation": {k: {"mae": v["mae"], "rmse": v["rmse"]} for k, v in m.items()},
        "llm_usage_totals": totals, "model_fits": runner.evaluator.fit_count, "figures": figures,
    })


def main(argv=None):
    args = parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(config.PROJECT_ROOT / ".env", override=False)
    except ImportError:
        pass
    settings = config.RunSettings.from_env(
        rounds=args.rounds, candidates=args.candidates, dry_run=args.dry_run, baseline_only=args.baseline_only,
        model=args.model, effort=args.effort, data_path=args.data)
    if args.no_cache:
        settings.use_cache = False
    if args.no_fallbacks:
        settings.use_fallbacks = False

    from iteration_loop import ExperimentRunner
    from llm_generator import ClaudeFactorGenerator, FatalLLMError, MockFactorGenerator
    from run_logger import RunLogger

    mode = "baseline" if settings.baseline_only else ("dryrun" if settings.dry_run else "live")
    generator = None
    if not settings.baseline_only:
        if settings.dry_run:
            generator = MockFactorGenerator()
        else:
            try:
                generator = ClaudeFactorGenerator(settings)
            except FatalLLMError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 2
    out = RunLogger(mode)
    log = out.logger
    out.save_json("run_config.json", {"run_id": out.run_id, "mode": mode, "settings": settings.to_dict(),
                                      "fixed_config": config.fixed_config_snapshot(),
                                      "library_versions": library_versions()})
    if settings.dry_run:
        log.warning("DRY RUN: candidates are fixed offline examples. This run does NOT demonstrate live "
                    "Claude feedback.")
    runner = ExperimentRunner(settings, generator, out)
    try:
        runner.prepare()
        reproduction = runner.stage0()
        if not reproduction["matches_phase1"]:
            log.warning("Baseline does not match the Phase 1 reference values: %s", reproduction)
        if settings.baseline_only:
            log.info("Baseline-only run finished. Results: %s", out.run_dir)
            return 0
        runner.run_rounds()
        runner.select_final()
        runner.final_test()
        figures = make_figures(runner, out, "DRY RUN, mock candidates" if settings.dry_run else "live Claude run")
        write_summary(runner, out, settings, figures)
        log.info("Run complete. Results: %s | Figures: %s", out.run_dir, out.figures_dir)
        return 0
    except FatalLLMError as exc:
        log.error("Stopped before the final test because of an LLM configuration error: %s", exc)
        log.error("Rounds completed: %d. The test set was not evaluated.", len(runner.round_records))
        return 3
    except Exception:
        log.error("Run failed:\n%s", traceback.format_exc())
        return 1
    finally:
        out.close()


if __name__ == "__main__":
    sys.exit(main())
