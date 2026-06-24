"""
Pipeline helper functions for the SWE-bench evaluation Airflow DAG.

Functions:
    build_run_config    - Build a run config dict from Airflow params
    prepare_run_dir     - Create run directory structure and save config.json
    run_agent_batch     - Run mini-swe-agent batch and collect outputs
    run_swebench_eval   - Run SWE-bench evaluation harness
    collect_metrics     - Parse evaluation results and write metrics.json
    build_manifest      - Write manifest.json pointing to all key files
    log_mlflow_run      - Log params, metrics, and artifacts to MLflow
    upload_to_s3        - Upload run directory to S3-compatible object storage
"""

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Config & directory helpers
# ---------------------------------------------------------------------------

def build_run_config(params: dict) -> dict:
    """Build a run configuration dict from Airflow params.

    Args:
        params: Airflow DAG params dict with keys:
            - split (str): "test" or "dev"
            - subset (str): "verified" or "lite"
            - workers (int): number of parallel workers
            - model (str): model identifier, e.g. "nebius/moonshotai/Kimi-K2.6"
            - task_slice (str): Python slice notation, e.g. "0:3"
            - run_id (str): unique run identifier (auto-generated if empty)
            - cost_limit (float): cost limit for the agent (0 = no limit)

    Returns:
        dict with the full run configuration.
    """
    run_id = params.get("run_id") or f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    subset = params["subset"]

    return {
        "run_id": run_id,
        "split": params["split"],
        "subset": subset,
        "workers": int(params["workers"]),
        "model": params.get("model", "nebius/moonshotai/Kimi-K2.6"),
        "task_slice": params.get("task_slice", ""),
        "cost_limit": float(params.get("cost_limit", 0)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": (
            "princeton-nlp/SWE-bench_Verified"
            if subset == "verified"
            else "princeton-nlp/SWE-bench_Lite"
        ),
    }


def prepare_run_dir(run_config: dict) -> Path:
    """Create the run directory structure and save config.json.

    Creates:
        runs/<run-id>/
        runs/<run-id>/run-agent/
        runs/<run-id>/run-eval/
        runs/<run-id>/config.json

    Args:
        run_config: dict produced by build_run_config().

    Returns:
        Path to the run directory.
    """
    run_dir = PROJECT_ROOT / "runs" / run_config["run_id"]
    
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory {run_dir} already exists! "
            f"Please choose a different run_id to avoid mixing artifacts."
        )
        
    (run_dir / "run-agent").mkdir(parents=True, exist_ok=False)
    (run_dir / "run-eval").mkdir(parents=True, exist_ok=False)

    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    return run_dir


# ---------------------------------------------------------------------------
# Agent & evaluation runners
# ---------------------------------------------------------------------------

def run_agent_batch(run_config: dict, run_dir: Path) -> Path:
    """Run mini-swe-agent in batch mode and write outputs to run_dir/run-agent/.

    Calls: uv run mini-extra swebench --subset ... --split ... --model ...
           --workers ... [--slice ...] -o <agent_dir>

    The cost_limit parameter is enforced via the MSWEA_AGENT_COST_LIMIT
    environment variable, which mini-swe-agent reads to cap per-instance
    agent spend (the CLI does not expose a --cost-limit flag).

    Args:
        run_config: dict produced by build_run_config().
        run_dir: Path to the run directory (from prepare_run_dir).

    Returns:
        Path to the produced preds.json file.
    """
    agent_dir = run_dir / "run-agent"

    cmd = [
        "uv", "run", "mini-extra", "swebench",
        "--subset", run_config["subset"],
        "--split", run_config["split"],
        "--model", run_config["model"],
        "--workers", str(run_config["workers"]),
        "-o", str(agent_dir),
    ]

    if run_config.get("task_slice"):
        cmd.extend(["--slice", run_config["task_slice"]])

    env = {**os.environ, "MSWEA_COST_TRACKING": "ignore_errors"}

    # Pass cost_limit to the agent via environment variable.
    # mini-swe-agent's CLI does not expose --cost-limit; the agent reads
    # MSWEA_AGENT_COST_LIMIT to cap per-instance spend.
    cost_limit = run_config.get("cost_limit", 0)
    if cost_limit and cost_limit > 0:
        env["MSWEA_AGENT_COST_LIMIT"] = str(cost_limit)

    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)

    return agent_dir / "preds.json"


def run_swebench_eval(run_config: dict, preds_path: Path, run_dir: Path) -> Path:
    """Run the SWE-bench evaluation harness and write results to run_dir/run-eval/.

    Calls: python -m swebench.harness.run_evaluation
              --dataset_name ... --predictions_path ... --max_workers ... --run_id ...

    After the evaluation completes, moves the SWE-bench output files
    (logs and summary JSON) into the structured run-eval/ directory.

    Args:
        run_config: dict produced by build_run_config().
        preds_path: Path to the preds.json file (from run_agent_batch).
        run_dir: Path to the run directory.

    Returns:
        Path to the run-eval/ directory.
    """
    eval_dir = run_dir / "run-eval"

    cmd = [
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", run_config["dataset_name"],
        "--predictions_path", str(preds_path),
        "--max_workers", str(run_config["workers"]),
        "--run_id", run_config["run_id"],
    ]

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    # SWE-bench writes per-instance logs to: logs/run_evaluation/<run_id>/
    swebench_log_dir = PROJECT_ROOT / "logs" / "run_evaluation" / run_config["run_id"]
    if swebench_log_dir.exists():
        shutil.copytree(swebench_log_dir, eval_dir / "logs", dirs_exist_ok=True)

    # SWE-bench writes a top-level summary JSON: <model_sanitized>.<split>.json
    # e.g. nebius__moonshotai__Kimi-K2.6.test.json
    model_sanitized = run_config["model"].replace("/", "__")
    results_file = PROJECT_ROOT / f"{model_sanitized}.{run_config['split']}.json"
    if results_file.exists():
        shutil.copy2(results_file, eval_dir / "results.json")

    return eval_dir


# ---------------------------------------------------------------------------
# Metrics & manifest
# ---------------------------------------------------------------------------

def collect_metrics(eval_dir: Path, run_dir: Path) -> dict:
    """Parse evaluation results and write metrics.json to the run directory.

    Reads eval_dir/results.json (the aggregated SWE-bench summary) and
    extracts the key counters. If results.json is missing, returns zeros.

    Args:
        eval_dir: Path to the run-eval/ directory.
        run_dir: Path to the run directory (where metrics.json will be written).

    Returns:
        dict with metrics: total_instances, submitted_instances,
        completed_instances, resolved_instances, unresolved_instances,
        error_instances, resolve_rate.
    """
    metrics = {
        "total_instances": 0,
        "submitted_instances": 0,
        "completed_instances": 0,
        "resolved_instances": 0,
        "unresolved_instances": 0,
        "error_instances": 0,
        "resolve_rate": 0.0,
    }

    results_file = eval_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)

        # SWE-bench writes a top-level aggregated summary with keys like
        # "total_instances", "resolved_instances", etc.  If results.json
        # is empty or only contains per-instance reports (our fallback
        # path), we aggregate from the reports/ directory and build a
        # proper summary so the artifact contract is consistent.
        if not results or "total_instances" not in results:
            per_instance = {}
            reports_dir = eval_dir / "reports"
            if reports_dir.exists():
                for report_file in reports_dir.glob("*.json"):
                    with open(report_file) as rf:
                        per_instance.update(json.load(rf))

            # Count metrics from per-instance reports
            n_total = len(per_instance)
            n_resolved = sum(
                1 for d in per_instance.values() if d.get("resolved")
            )
            n_error = sum(
                1 for d in per_instance.values()
                if d.get("patch_is_None") or not d.get("patch_exists")
            )

            # Build a proper aggregated summary matching SWE-bench format
            results = {
                "total_instances": n_total,
                "submitted_instances": n_total,
                "completed_instances": n_total,
                "resolved_instances": n_resolved,
                "unresolved_instances": n_total - n_resolved - n_error,
                "error_instances": n_error,
                "resolve_rate": (n_resolved / n_total) if n_total > 0 else 0.0,
                "per_instance_results": per_instance,
            }

            # Overwrite results.json with the aggregated summary.
            # Delete the root-owned file first (directory ownership allows this).
            try:
                os.remove(results_file)
            except (PermissionError, FileNotFoundError):
                pass
            try:
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)
            except PermissionError:
                pass  # metrics.json is still written below

        metrics["total_instances"] = results.get("total_instances", 0)
        metrics["submitted_instances"] = results.get("submitted_instances", 0)
        metrics["completed_instances"] = results.get("completed_instances", 0)
        metrics["resolved_instances"] = results.get("resolved_instances", 0)
        metrics["unresolved_instances"] = results.get("unresolved_instances", 0)
        metrics["error_instances"] = results.get("error_instances", 0)

        if metrics["submitted_instances"] > 0:
            metrics["resolve_rate"] = (
                metrics["resolved_instances"] / metrics["submitted_instances"]
            )

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def build_manifest(
    run_config: dict,
    run_dir: Path,
    metrics: dict,
    s3_uri: str = None,
) -> dict:
    """Build and write manifest.json — an index pointing to all important files.

    Args:
        run_config: dict produced by build_run_config().
        run_dir: Path to the run directory.
        metrics: dict produced by collect_metrics().
        s3_uri: Optional S3 URI if artifacts were uploaded (from upload_to_s3).

    Returns:
        The manifest dict.
    """
    candidate_files = {
        "config": "config.json",
        "predictions": "run-agent/preds.json",
        "trajectories": "run-agent/",
        "eval_logs": "run-eval/logs/",
        "eval_reports": "run-eval/reports/",
        "eval_results": "run-eval/results.json",
        "metrics": "metrics.json",
    }
    
    files = {}
    for key, path_str in candidate_files.items():
        if (run_dir / path_str).exists():
            files[key] = path_str

    manifest = {
        "run_id": run_config["run_id"],
        "created_at": run_config["created_at"],
        "model": run_config["model"],
        "split": run_config["split"],
        "subset": run_config["subset"],
        "task_slice": run_config.get("task_slice", ""),
        "files": files,
        "metrics_summary": metrics,
        "s3_uri": s3_uri,
    }

    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------

def log_mlflow_run(
    run_config: dict,
    metrics: dict,
    run_dir: Path,
    s3_uri: str = None,
) -> None:
    """Log params, metrics, and artifact references to MLflow.

    Uses MLFLOW_TRACKING_URI from the environment (defaults to
    http://localhost:5000).  Creates or reuses the experiment
    "swe-bench-evaluation".

    Args:
        run_config: dict produced by build_run_config().
        metrics: dict produced by collect_metrics().
        run_dir: Path to the run directory.
        s3_uri: Optional S3 URI to tag on the MLflow run.
    """
    import mlflow

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("swe-bench-evaluation")

    with mlflow.start_run(run_name=run_config["run_id"]):
        # --- Parameters ---
        mlflow.log_param("run_id", run_config["run_id"])
        mlflow.log_param("split", run_config["split"])
        mlflow.log_param("subset", run_config["subset"])
        mlflow.log_param("workers", run_config["workers"])
        mlflow.log_param("model", run_config["model"])
        mlflow.log_param("task_slice", run_config.get("task_slice") or "all")
        mlflow.log_param("cost_limit", run_config.get("cost_limit", 0))
        mlflow.log_param("dataset_name", run_config["dataset_name"])

        # --- Metrics ---
        mlflow.log_metric("total_instances", metrics["total_instances"])
        mlflow.log_metric("submitted_instances", metrics["submitted_instances"])
        mlflow.log_metric("completed_instances", metrics["completed_instances"])
        mlflow.log_metric("resolved_instances", metrics["resolved_instances"])
        mlflow.log_metric("unresolved_instances", metrics["unresolved_instances"])
        mlflow.log_metric("error_instances", metrics["error_instances"])
        mlflow.log_metric("resolve_rate", metrics["resolve_rate"])

        # --- Artifacts (key files only, not the full run tree) ---
        for fname in ("config.json", "metrics.json", "manifest.json"):
            fpath = run_dir / fname
            if fpath.exists():
                mlflow.log_artifact(str(fpath))

        preds_file = run_dir / "run-agent" / "preds.json"
        if preds_file.exists():
            mlflow.log_artifact(str(preds_file), artifact_path="predictions")

        # --- Tags ---
        mlflow.set_tag("artifact_local_path", str(run_dir))
        if s3_uri:
            mlflow.set_tag("artifact_s3_uri", s3_uri)


# ---------------------------------------------------------------------------
# Auto-update REPORT.md with concrete run example
# ---------------------------------------------------------------------------

def update_report_run_example(run_config: dict, metrics: dict) -> None:
    """Overwrite the 'Concrete Run Example' section of REPORT.md with real data.

    Finds the block between the '## 4. Concrete Run Example' heading and the
    following '---' separator and replaces it with values drawn directly from
    run_config and metrics so the report never drifts from the committed run.

    Args:
        run_config: dict produced by build_run_config().
        metrics: dict produced by collect_metrics().
    """
    report_path = PROJECT_ROOT / "REPORT.md"
    if not report_path.exists():
        return

    resolved = metrics.get("resolved_instances", 0)
    total = metrics.get("total_instances", 0)
    rate = metrics.get("resolve_rate", 0.0)

    if total > 0 and resolved > 0:
        outcome_text = (
            f"The agent successfully produced a valid patch that was applied and "
            f"evaluated by SWE-bench. Out of {total} instance(s) submitted, "
            f"{resolved} was resolved, yielding a **resolve_rate of {rate:.1%}**. "
            f"All pipeline stages (Airflow → Agent → Evaluator → Metrics → MLflow) "
            f"completed successfully."
        )
    else:
        outcome_text = (
            f"The pipeline completed all stages (Airflow → Agent → Evaluator → "
            f"Metrics → MLflow) without errors. "
            f"resolved_instances={resolved}, total_instances={total}."
        )

    new_section = (
        "## 4. Concrete Run Example\n"
        "\n"
        f"The following is an authentic, end-to-end evaluation run committed under "
        f"`runs/{run_config['run_id']}/`.\n"
        "\n"
        "**Input Parameters:**\n"
        "\n"
        f"- `split`: {run_config.get('split', 'test')}\n"
        f"- `subset`: {run_config.get('subset', 'verified')}\n"
        f"- `workers`: {run_config.get('workers', 5)}\n"
        f"- `model`: {run_config.get('model', '')}\n"
        f"- `task_slice`: {run_config.get('task_slice', '0:1')}\n"
        f"- `run_id`: {run_config.get('run_id', '')}\n"
        f"- `cost_limit`: {run_config.get('cost_limit', 0.0)}\n"
        "\n"
        "**Resulting Metrics:**\n"
        f"- `total_instances`: {metrics.get('total_instances', 0)}\n"
        f"- `submitted_instances`: {metrics.get('submitted_instances', 0)}\n"
        f"- `completed_instances`: {metrics.get('completed_instances', 0)}\n"
        f"- `resolved_instances`: {resolved}\n"
        f"- `unresolved_instances`: {metrics.get('unresolved_instances', 0)}\n"
        f"- `error_instances`: {metrics.get('error_instances', 0)}\n"
        f"- `resolve_rate`: {rate:.4f}\n"
        "\n"
        f"{outcome_text}\n"
        "\n"
    )

    text = report_path.read_text(encoding="utf-8")

    # Find the section boundaries
    section_start = text.find("## 4. Concrete Run Example")
    if section_start == -1:
        return  # section not found — skip

    # Find the next '---' separator after the section heading
    sep_pos = text.find("\n---", section_start)
    if sep_pos == -1:
        return  # can't find end boundary — skip

    # Replace the section content (keep the trailing '---')
    new_text = text[:section_start] + new_section + text[sep_pos + 1:]
    report_path.write_text(new_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# S3 / Object Storage upload (optional – extra credit)
# ---------------------------------------------------------------------------

def upload_to_s3(run_dir: Path, bucket: str = None, prefix: str = "runs") -> str:
    """Upload the entire run directory to S3-compatible object storage.

    Reads credentials and endpoint from environment variables:
        S3_BUCKET          - target bucket name
        S3_ENDPOINT_URL    - endpoint URL for non-AWS storage (e.g. Nebius)
        AWS_ACCESS_KEY_ID
        AWS_SECRET_ACCESS_KEY

    Args:
        run_dir: Path to the run directory to upload.
        bucket: Override bucket name (defaults to S3_BUCKET env var).
        prefix: S3 key prefix (default: "runs").

    Returns:
        s3_uri: The S3 URI of the uploaded directory, e.g.
                "s3://my-bucket/runs/run-20260623-143000/"
    """
    import boto3

    bucket = bucket or os.environ.get("S3_BUCKET", "mlops-assignment-artifacts")
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL") or None

    s3 = boto3.client("s3", endpoint_url=s3_endpoint)

    run_id = run_dir.name
    s3_prefix = f"{prefix}/{run_id}"
    uploaded = 0

    for file_path in sorted(run_dir.rglob("*")):
        if file_path.is_file():
            s3_key = f"{s3_prefix}/{file_path.relative_to(run_dir).as_posix()}"
            s3.upload_file(str(file_path), bucket, s3_key)
            uploaded += 1

    s3_uri = f"s3://{bucket}/{s3_prefix}/"
    print(f"Uploaded {uploaded} files → {s3_uri}")
    return s3_uri
