"""
Airflow DAG: evaluate-agent
============================
Runs mini-swe-agent on a SWE-bench subset and evaluates the results.

Pipeline:
    prepare_run  →  run_agent  →  run_eval  →  summarize_and_log

Parameters (all configurable from the Airflow UI trigger form):
    split       - SWE-bench split: "test" or "dev"
    subset      - SWE-bench subset: "verified" or "lite"
    workers     - Number of parallel agent workers (1–20)
    model       - LLM model identifier (e.g. "nebius/moonshotai/Kimi-K2.6")
    task_slice  - Python slice of tasks to run, e.g. "0:3" (empty = all)
    run_id      - Custom run identifier (auto-generated timestamp if empty)
    cost_limit  - Per-agent cost ceiling in USD (0 = use model default)

Outputs:
    runs/<run-id>/
        config.json         ← Full run configuration
        run-agent/
            preds.json      ← Predictions in SWE-bench format
            <instance_id>/  ← Per-instance trajectory directories
        run-eval/
            logs/           ← SWE-bench per-instance evaluation logs
            results.json    ← Aggregated SWE-bench results
        metrics.json        ← Parsed metrics (resolve_rate, counts, etc.)
        manifest.json       ← Index to all important files + optional S3 URI
"""

import sys
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param

# ---------------------------------------------------------------------------
# Make src/pipeline.py importable inside Airflow tasks
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

@dag(
    dag_id="evaluate-agent",
    description=(
        "Run mini-swe-agent on a SWE-bench subset, evaluate the patches, "
        "and log results to MLflow."
    ),
    start_date=datetime(2024, 1, 1),
    schedule=None,           # manual trigger only
    catchup=False,
    max_active_runs=3,       # allow up to 3 concurrent experiments
    tags=["swe-bench", "evaluation", "mini-swe-agent"],
    params={
        # Required params
        "split": Param(
            "test",
            type="string",
            enum=["test", "dev"],
            title="SWE-bench split",
            description="Which dataset split to evaluate on.",
        ),
        "subset": Param(
            "verified",
            type="string",
            enum=["verified", "lite"],
            title="SWE-bench subset",
            description="'verified' = SWE-bench Verified (500 tasks). 'lite' = SWE-bench Lite (300 tasks).",
        ),
        "workers": Param(
            5,
            type="integer",
            minimum=1,
            maximum=20,
            title="Parallel workers",
            description="Number of agent workers to run in parallel.",
        ),
        # Optional / tuning params
        "model": Param(
            "nebius/moonshotai/Kimi-K2.6",
            type="string",
            title="Model identifier",
            description="LLM model to use for the agent, in LiteLLM format.",
        ),
        "task_slice": Param(
            "0:3",
            type="string",
            title="Task slice",
            description=(
                "Python slice notation to select a subset of tasks, e.g. '0:3' "
                "for the first 3. Leave empty to run all tasks in the split."
            ),
        ),
        "run_id": Param(
            "",
            type="string",
            title="Run ID (optional)",
            description=(
                "Custom run identifier. If empty, a timestamped ID is "
                "generated automatically, e.g. 'run-20260623-143000'."
            ),
        ),
        "cost_limit": Param(
            0,
            type="number",
            minimum=0,
            title="Cost limit (USD)",
            description="Maximum spend per agent run in USD. 0 = use model default.",
        ),
    },
)
def evaluate_agent_dag():
    """End-to-end evaluation pipeline: agent → eval → MLflow."""

    # ------------------------------------------------------------------
    # Task 1: prepare_run
    # ------------------------------------------------------------------
    @task(task_id="prepare_run")
    def prepare_run(**context) -> dict:
        """Read Airflow params, build run config, create directory structure."""
        from pipeline import build_run_config, prepare_run_dir

        params = context["params"]
        run_config = build_run_config(params)
        run_dir = prepare_run_dir(run_config)

        print(f"Run ID  : {run_config['run_id']}")
        print(f"Run dir : {run_dir}")
        print(f"Config  : {run_config}")

        return {
            "run_config": run_config,
            "run_dir": str(run_dir),
        }

    # ------------------------------------------------------------------
    # Task 2: run_agent
    # ------------------------------------------------------------------
    @task(
        task_id="run_agent",
        retries=1,
        execution_timeout=None,  # agent runs can be long; no hard timeout
    )
    def run_agent(prepare_output: dict) -> dict:
        """Run mini-swe-agent batch and write trajectories + preds.json."""
        from pathlib import Path
        from pipeline import run_agent_batch

        run_config = prepare_output["run_config"]
        run_dir = Path(prepare_output["run_dir"])

        print(f"Starting agent run for run_id={run_config['run_id']}")
        preds_path = run_agent_batch(run_config, run_dir)
        print(f"Agent done. Predictions: {preds_path}")

        return {
            "preds_path": str(preds_path),
            **prepare_output,
        }

    # ------------------------------------------------------------------
    # Task 3: run_eval
    # ------------------------------------------------------------------
    @task(
        task_id="run_eval",
        retries=1,
    )
    def run_eval(agent_output: dict) -> dict:
        """Run the SWE-bench evaluation harness on the produced preds.json."""
        from pathlib import Path
        from pipeline import run_swebench_eval

        run_config = agent_output["run_config"]
        run_dir = Path(agent_output["run_dir"])
        preds_path = Path(agent_output["preds_path"])

        print(f"Starting evaluation for run_id={run_config['run_id']}")
        print(f"Predictions file: {preds_path}")

        eval_dir = run_swebench_eval(run_config, preds_path, run_dir)
        print(f"Evaluation done. Results in: {eval_dir}")

        return {
            "eval_dir": str(eval_dir),
            **agent_output,
        }

    # ------------------------------------------------------------------
    # Task 4: summarize_and_log
    # ------------------------------------------------------------------
    @task(task_id="summarize_and_log")
    def summarize_and_log(eval_output: dict) -> dict:
        """Parse evaluation reports, write metrics.json + manifest.json, log to MLflow."""
        from pathlib import Path
        from pipeline import collect_metrics, build_manifest, log_mlflow_run

        run_config = eval_output["run_config"]
        run_dir = Path(eval_output["run_dir"])
        eval_dir = Path(eval_output["eval_dir"])

        # Collect metrics from evaluation results
        metrics = collect_metrics(eval_dir, run_dir)
        print(f"Metrics: {metrics}")

        # Build the run manifest (no S3 in easy mode — see evaluate_agent_docker.py)
        build_manifest(run_config, run_dir, metrics, s3_uri=None)

        # Log everything to MLflow
        log_mlflow_run(run_config, metrics, run_dir, s3_uri=None)

        print(
            f"\n{'='*60}\n"
            f"Run complete: {run_config['run_id']}\n"
            f"  Resolved : {metrics['resolved_instances']} / {metrics['submitted_instances']}\n"
            f"  Rate     : {metrics['resolve_rate']:.1%}\n"
            f"  Run dir  : {run_dir}\n"
            f"{'='*60}"
        )

        return {
            "run_id": run_config["run_id"],
            "metrics": metrics,
            "run_dir": str(run_dir),
        }

    # ------------------------------------------------------------------
    # Wire up the linear task chain
    # ------------------------------------------------------------------
    prep     = prepare_run()
    agent    = run_agent(prep)
    evaluate = run_eval(agent)
    summary  = summarize_and_log(evaluate)   # noqa: F841


# Instantiate the DAG
evaluate_agent_dag()
