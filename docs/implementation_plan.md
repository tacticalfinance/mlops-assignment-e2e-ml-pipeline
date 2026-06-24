# Implementation Plan: 100% Score on E2E ML Pipeline Assignment

## Goal

Turn the ad-hoc scripts in `scripts/` into a **production-grade Airflow pipeline** that runs [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) on SWE-bench tasks and evaluates the results — with full artifact structure, MLflow tracking, Docker isolation, and Docker Compose deployment.

---

## Grading Breakdown & Strategy

| Area | Weight | Our Target |
|---|---:|---|
| **Configurable Airflow DAG** | 35% | Full marks — parameterized DAG with all 7 params, 4 tasks, no hard-coded values |
| **Artifact structure & reproducibility** | 20% | Full marks — structured `runs/<run-id>/` tree + S3 upload for extra credit |
| **MLflow tracking** | 15% | Full marks — params, metrics, artifact URI logged; runs comparable in UI |
| **Execution isolation (DockerOperator)** | 10% | Full marks — agent & eval tasks run via `DockerOperator` using the provided `Dockerfile` |
| **Docker Compose deployment** | 10% | Full marks — `docker-compose.yaml` with Airflow + MLflow services |
| **Report & reproducibility** | 10% | Full marks — `REPORT.md` with architecture, trigger instructions, artifact layout, screenshot, rerun guide |

---

## Existing Codebase Summary

### What We Already Have

| File | Purpose |
|---|---|
| [mini-swe-bench-single.py](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/dags/mini-swe-bench-single.py) | Example DAG — runs a single SWE-bench task via subprocess, all values hard-coded |
| [mini-swe-bench-batch.sh](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/scripts/mini-swe-bench-batch.sh) | Shell script — runs `mini-extra swebench` in batch mode with hard-coded params |
| [swe-bench-eval.sh](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/scripts/swe-bench-eval.sh) | Shell script — runs `python -m swebench.harness.run_evaluation` with hard-coded paths |
| [mini-swe-bench-single.sh](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/scripts/mini-swe-bench-single.sh) | Shell script — runs a single task with hard-coded params |
| [Dockerfile](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/Dockerfile) | Ubuntu 24.04 + Docker-in-Docker + uv + project deps |
| [run-airflow-standalone.sh](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/run-airflow-standalone.sh) | Starts Airflow standalone on port 8080 |
| [pyproject.toml](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/pyproject.toml) | Dependencies: `mini-swe-agent>=2.4.1`, `swebench>=4.1.0` |
| [.env.example](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/.env.example) | Contains only `NEBIUS_API_KEY=XXX` |
| `sample/` | Sample outputs: trajectories, preds.json, evaluation logs, report.json, results JSON |

### Key Data Formats (from `sample/`)

**`preds.json`** — dict of `instance_id → {model_name_or_path, instance_id, model_patch}`:
```json
{
  "astropy__astropy-12907": {
    "model_name_or_path": "nebius/moonshotai/Kimi-K2.6",
    "instance_id": "astropy__astropy-12907",
    "model_patch": "diff --git a/..."
  }
}
```

**Evaluation result** (`nebius__moonshotai__Kimi-K2.6.test.json`) — contains `total_instances`, `submitted_instances`, `completed_instances`, `resolved_instances`, `resolved_ids`, `unresolved_ids`, etc.

**Per-instance report** (`report.json`) — contains `resolved: true/false` and detailed test status.

**Trajectory** (`*.traj.json`) — full agent conversation with config, messages, tool calls, and submission diff.

---

## Proposed Changes

### Phase 1: Pipeline Helper Module

#### [NEW] [pipeline.py](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/src/pipeline.py)

Create `src/pipeline.py` (or `src/__init__.py` + `src/pipeline.py`) with these helper functions:

```python
# src/pipeline.py

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_run_config(params: dict) -> dict:
    """Build a run configuration dict from Airflow params.
    
    Params expected:
      - split (str): "test" or "dev"
      - subset (str): "verified" or "lite"  
      - workers (int): number of parallel workers
      - model (str): model identifier, e.g. "nebius/moonshotai/Kimi-K2.6"
      - task_slice (str): Python slice notation, e.g. "0:3"
      - run_id (str): unique run identifier (auto-generated if empty)
      - cost_limit (float): cost limit for the agent
    """
    run_id = params.get("run_id") or f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    
    return {
        "run_id": run_id,
        "split": params["split"],
        "subset": params["subset"],
        "workers": int(params["workers"]),
        "model": params.get("model", "nebius/moonshotai/Kimi-K2.6"),
        "task_slice": params.get("task_slice", ""),
        "cost_limit": float(params.get("cost_limit", 0)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": f"princeton-nlp/SWE-bench_{'Verified' if params['subset'] == 'verified' else 'Lite'}",
    }


def prepare_run_dir(run_config: dict) -> Path:
    """Create the run directory structure and save config.json."""
    run_dir = PROJECT_ROOT / "runs" / run_config["run_id"]
    (run_dir / "run-agent").mkdir(parents=True, exist_ok=True)
    (run_dir / "run-eval").mkdir(parents=True, exist_ok=True)
    
    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)
    
    return run_dir


def run_agent_batch(run_config: dict, run_dir: Path) -> Path:
    """Run mini-swe-agent batch and write outputs to run_dir/run-agent/."""
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
    if run_config.get("cost_limit") is not None:
        cmd.extend(["--cost-limit", str(run_config["cost_limit"])])
    
    env = {**os.environ, "MSWEA_COST_TRACKING": "ignore_errors"}
    
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    
    return agent_dir / "preds.json"


def run_swebench_eval(run_config: dict, preds_path: Path, run_dir: Path) -> Path:
    """Run SWE-bench evaluation and write results to run_dir/run-eval/."""
    eval_dir = run_dir / "run-eval"
    
    cmd = [
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", run_config["dataset_name"],
        "--predictions_path", str(preds_path),
        "--max_workers", str(run_config["workers"]),
        "--run_id", run_config["run_id"],
    ]
    
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    
    # Move SWE-bench outputs into the run-eval directory
    # SWE-bench writes to logs/run_evaluation/<run_id>/<model_name>/
    swebench_log_dir = PROJECT_ROOT / "logs" / "run_evaluation" / run_config["run_id"]
    if swebench_log_dir.exists():
        shutil.copytree(swebench_log_dir, eval_dir / "logs", dirs_exist_ok=True)
    
    # SWE-bench also writes a summary JSON: <model_sanitized>.<split>.json
    model_sanitized = run_config["model"].replace("/", "__")
    results_file = PROJECT_ROOT / f"{model_sanitized}.{run_config['split']}.json"
    if results_file.exists():
        shutil.copy2(results_file, eval_dir / "results.json")
    
    return eval_dir


def collect_metrics(eval_dir: Path, run_dir: Path) -> dict:
    """Parse evaluation reports and write metrics.json."""
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
        metrics["total_instances"] = results.get("total_instances", 0)
        metrics["submitted_instances"] = results.get("submitted_instances", 0)
        metrics["completed_instances"] = results.get("completed_instances", 0)
        metrics["resolved_instances"] = results.get("resolved_instances", 0)
        metrics["unresolved_instances"] = results.get("unresolved_instances", 0)
        metrics["error_instances"] = results.get("error_instances", 0)
        if metrics["submitted_instances"] > 0:
            metrics["resolve_rate"] = metrics["resolved_instances"] / metrics["submitted_instances"]
    
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


def build_manifest(run_config: dict, run_dir: Path, metrics: dict, s3_uri: str = None) -> dict:
    """Build and write manifest.json — points to all important files."""
    manifest = {
        "run_id": run_config["run_id"],
        "created_at": run_config["created_at"],
        "config": "config.json",
        "predictions": "run-agent/preds.json",
        "trajectories": "run-agent/",
        "eval_logs": "run-eval/logs/",
        "eval_results": "run-eval/results.json",
        "metrics": "metrics.json",
        "s3_uri": s3_uri,
    }
    
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    return manifest


def log_mlflow_run(run_config: dict, metrics: dict, run_dir: Path, s3_uri: str = None) -> None:
    """Log params, metrics, and artifact references to MLflow."""
    import mlflow
    
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("swe-bench-evaluation")
    
    with mlflow.start_run(run_name=run_config["run_id"]):
        # Log params
        mlflow.log_param("run_id", run_config["run_id"])
        mlflow.log_param("split", run_config["split"])
        mlflow.log_param("subset", run_config["subset"])
        mlflow.log_param("workers", run_config["workers"])
        mlflow.log_param("model", run_config["model"])
        mlflow.log_param("task_slice", run_config.get("task_slice", "all"))
        mlflow.log_param("cost_limit", run_config.get("cost_limit", 0))
        
        # Log metrics
        mlflow.log_metric("total_instances", metrics["total_instances"])
        mlflow.log_metric("submitted_instances", metrics["submitted_instances"])
        mlflow.log_metric("completed_instances", metrics["completed_instances"])
        mlflow.log_metric("resolved_instances", metrics["resolved_instances"])
        mlflow.log_metric("unresolved_instances", metrics["unresolved_instances"])
        mlflow.log_metric("error_instances", metrics["error_instances"])
        mlflow.log_metric("resolve_rate", metrics["resolve_rate"])
        
        # Log key files as artifacts
        mlflow.log_artifact(str(run_dir / "config.json"))
        mlflow.log_artifact(str(run_dir / "metrics.json"))
        mlflow.log_artifact(str(run_dir / "manifest.json"))
        
        preds_file = run_dir / "run-agent" / "preds.json"
        if preds_file.exists():
            mlflow.log_artifact(str(preds_file), "predictions")
        
        # Log S3 URI as a tag if available
        if s3_uri:
            mlflow.set_tag("artifact_s3_uri", s3_uri)
        
        mlflow.set_tag("artifact_local_path", str(run_dir))
```

> [!IMPORTANT]
> The helper functions are designed to work both in easy-mode (subprocess) and can later be adapted for DockerOperator by adjusting how commands are invoked.

---

### Phase 2: Configurable Airflow DAG (35%)

#### [NEW] [evaluate_agent.py](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/dags/evaluate_agent.py)

```python
import os
from datetime import datetime
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure src/ is importable
import sys
sys.path.insert(0, str(PROJECT_ROOT / "src"))


@dag(
    dag_id="evaluate-agent",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    params={
        "split": Param("test", type="string", enum=["test", "dev"],
                       description="SWE-bench split"),
        "subset": Param("verified", type="string", enum=["verified", "lite"],
                        description="SWE-bench subset"),
        "workers": Param(5, type="integer", minimum=1, maximum=20,
                         description="Number of parallel workers"),
        "model": Param("nebius/moonshotai/Kimi-K2.6", type="string",
                       description="Model identifier for inference"),
        "task_slice": Param("0:3", type="string",
                           description="Python slice notation for tasks, e.g. '0:3'"),
        "run_id": Param("", type="string",
                        description="Custom run ID (auto-generated if empty)"),
        "cost_limit": Param(0, type="number",
                           description="Cost limit for agent (0 = use default)"),
    },
    tags=["swe-bench", "evaluation", "mini-swe-agent"],
)
def evaluate_agent_dag():

    @task
    def prepare_run(**context):
        from pipeline import build_run_config, prepare_run_dir
        
        params = context["params"]
        run_config = build_run_config(params)
        run_dir = prepare_run_dir(run_config)
        
        return {"run_config": run_config, "run_dir": str(run_dir)}

    @task
    def run_agent(prepare_output, **context):
        from pipeline import run_agent_batch
        
        run_config = prepare_output["run_config"]
        run_dir = Path(prepare_output["run_dir"])
        
        preds_path = run_agent_batch(run_config, run_dir)
        
        return {"preds_path": str(preds_path), **prepare_output}

    @task
    def run_eval(agent_output, **context):
        from pipeline import run_swebench_eval
        
        run_config = agent_output["run_config"]
        run_dir = Path(agent_output["run_dir"])
        preds_path = Path(agent_output["preds_path"])
        
        eval_dir = run_swebench_eval(run_config, preds_path, run_dir)
        
        return {"eval_dir": str(eval_dir), **agent_output}

    @task
    def summarize_and_log(eval_output, **context):
        from pipeline import collect_metrics, build_manifest, log_mlflow_run
        
        run_config = eval_output["run_config"]
        run_dir = Path(eval_output["run_dir"])
        eval_dir = Path(eval_output["eval_dir"])
        
        # Collect metrics
        metrics = collect_metrics(eval_dir, run_dir)
        
        # Build manifest (s3_uri=None for now; Phase 3 adds S3)
        build_manifest(run_config, run_dir, metrics)
        
        # Log to MLflow
        log_mlflow_run(run_config, metrics, run_dir)
        
        return {"metrics": metrics, "run_id": run_config["run_id"]}

    # Define task dependencies: linear chain
    prep = prepare_run()
    agent = run_agent(prep)
    evaluation = run_eval(agent)
    summary = summarize_and_log(evaluation)


evaluate_agent_dag()
```

**Key design decisions:**
- All 7 parameters exposed via `Param` with types, defaults, and descriptions
- `Param` enums for `split`/`subset` prevent invalid values
- Tasks pass data via XCom using dicts (Airflow best practice)
- Linear dependency chain: `prepare_run → run_agent → run_eval → summarize_and_log`
- Imports done inside tasks to avoid Airflow DAG parsing overhead

---

### Phase 3: Artifact Structure & Reproducibility (20%)

The `prepare_run_dir`, `collect_metrics`, and `build_manifest` functions in `src/pipeline.py` (Phase 1) produce:

```text
runs/<run-id>/
  config.json            ← Full run configuration (split, subset, model, slice, etc.)
  run-agent/
    preds.json            ← Predictions in SWE-bench format
    <instance_id>/
      <instance_id>.traj.json   ← Full agent trajectory
    exit_statuses_*.yaml  ← Agent exit statuses
    minisweagent.log      ← Agent log
  run-eval/
    logs/                 ← SWE-bench per-instance eval logs
      <model_name>/
        <instance_id>/
          report.json
          run_instance.log
          test_output.txt
          patch.diff
          eval.sh
    results.json          ← Aggregated SWE-bench evaluation results
  metrics.json            ← Parsed metrics (resolved_instances, resolve_rate, etc.)
  manifest.json           ← Index pointing to all important files + S3 URI
```

**`manifest.json` example:**
```json
{
  "run_id": "run-20260623-143000",
  "created_at": "2026-06-23T14:30:00+00:00",
  "config": "config.json",
  "predictions": "run-agent/preds.json",
  "trajectories": "run-agent/",
  "eval_logs": "run-eval/logs/",
  "eval_results": "run-eval/results.json",
  "metrics": "metrics.json",
  "s3_uri": "s3://my-bucket/runs/run-20260623-143000/"
}
```

> [!TIP]
> **Extra credit**: Add S3 upload step. Create `upload_to_s3(run_dir, bucket)` in `src/pipeline.py` that uses `boto3` or the `aws` CLI to upload the entire `runs/<run-id>/` folder to Object Storage.

#### [NEW] S3 Upload Helper (optional for extra credit)

Add to `src/pipeline.py`:

```python
def upload_to_s3(run_dir: Path, bucket: str = None, prefix: str = "runs") -> str:
    """Upload run directory to S3-compatible object storage."""
    import boto3
    
    bucket = bucket or os.environ.get("S3_BUCKET", "mlops-assignment-artifacts")
    s3_endpoint = os.environ.get("S3_ENDPOINT_URL", None)
    
    s3 = boto3.client("s3", endpoint_url=s3_endpoint)
    
    run_id = run_dir.name
    s3_prefix = f"{prefix}/{run_id}"
    
    for file_path in run_dir.rglob("*"):
        if file_path.is_file():
            s3_key = f"{s3_prefix}/{file_path.relative_to(run_dir)}"
            s3.upload_file(str(file_path), bucket, s3_key)
    
    s3_uri = f"s3://{bucket}/{s3_prefix}/"
    return s3_uri
```

---

### Phase 4: MLflow Tracking (15%)

#### MLflow Integration Details

The `log_mlflow_run` function (Phase 1) handles this. Summary of what gets logged:

| MLflow Concept | What We Log |
|---|---|
| **Experiment** | `swe-bench-evaluation` |
| **Run name** | `<run-id>` |
| **Params** | `run_id`, `split`, `subset`, `workers`, `model`, `task_slice`, `cost_limit` |
| **Metrics** | `total_instances`, `submitted_instances`, `completed_instances`, `resolved_instances`, `unresolved_instances`, `error_instances`, `resolve_rate` |
| **Artifacts** | `config.json`, `metrics.json`, `manifest.json`, `preds.json` |
| **Tags** | `artifact_local_path`, `artifact_s3_uri` (if S3 upload enabled) |

#### [MODIFY] [pyproject.toml](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/pyproject.toml)

Add `mlflow` to dependencies:

```diff
 dependencies = [
     "mini-swe-agent>=2.4.1",
     "swebench>=4.1.0",
+    "mlflow>=2.15.0",
+    "boto3>=1.35.0",
 ]
```

---

### Phase 5: Execution Isolation — DockerOperator (10%)

#### [MODIFY] [Dockerfile](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/Dockerfile)

Enhance the existing Dockerfile to also include `src/` and support parameterized invocation:

```diff
 COPY scripts scripts/
+COPY src src/

 # Optional but useful if your script lacks executable bit or shebang issues:
 RUN chmod +x scripts/*.sh
```

#### [NEW] [evaluate_agent_docker.py](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/dags/evaluate_agent_docker.py)

A production-style variant of the DAG that uses `DockerOperator` for agent and eval tasks:

```python
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
```

Key design:
- `prepare_run` and `summarize_and_log` remain as `@task` Python callables (lightweight, need filesystem access)
- `run_agent` and `run_eval` use `DockerOperator` with:
  - The project Docker image (built from `Dockerfile`)
  - Volume mounts for `runs/` directory and Docker socket (`/var/run/docker.sock`)
  - Environment variables passed for `NEBIUS_API_KEY`, `MSWEA_COST_TRACKING`
  - `auto_remove=True`, `network_mode="host"`

```python
run_agent_task = DockerOperator(
    task_id="run_agent",
    image="mlops-pipeline:latest",
    command="bash -c 'uv run mini-extra swebench --subset {{ params.subset }} ...'",
    mounts=[
        Mount(source="/path/to/runs", target="/mlops-assignment/runs", type="bind"),
        Mount(source="/var/run/docker.sock", target="/var/run/docker.sock", type="bind"),
    ],
    environment={
        "NEBIUS_API_KEY": "{{ var.value.NEBIUS_API_KEY }}",
        "MSWEA_COST_TRACKING": "ignore_errors",
    },
    auto_remove=True,
    docker_url="unix://var/run/docker.sock",
    network_mode="host",
    retries=2,
    retry_delay=timedelta(minutes=2),
)
```

> [!NOTE]
> The Docker-in-Docker setup is needed because `mini-swe-agent` itself uses Docker to run tasks inside SWE-bench containers. We mount the host Docker socket so the agent container can spawn sibling containers.

---

### Phase 6: Docker Compose Deployment (10%)

#### [NEW] [docker-compose.yaml](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/docker-compose.yaml)

```yaml
version: '3.8'

x-airflow-common: &airflow-common
  image: apache/airflow:2.10.5-python3.12
  environment: &airflow-env
    AIRFLOW__CORE__EXECUTOR: LocalExecutor
    AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
    AIRFLOW__CORE__FERNET_KEY: ''
    AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'false'
    AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    AIRFLOW__API__AUTH_BACKENDS: 'airflow.api.auth.backend.basic_auth'
    MLFLOW_TRACKING_URI: http://mlflow:5000
    NEBIUS_API_KEY: ${NEBIUS_API_KEY}
  volumes:
    - ./dags:/opt/airflow/dags
    - ./src:/opt/airflow/src
    - ./scripts:/opt/airflow/scripts
    - ./runs:/opt/airflow/runs
    - /var/run/docker.sock:/var/run/docker.sock
  depends_on:
    postgres:
      condition: service_healthy

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "airflow"]
      interval: 10s
      retries: 5

  airflow-init:
    <<: *airflow-common
    entrypoint: /bin/bash
    command:
      - -c
      - |
        airflow db migrate &&
        airflow users create \
          --username admin --password admin \
          --firstname Admin --lastname User \
          --role Admin --email admin@example.com || true
    restart: "no"

  airflow-webserver:
    <<: *airflow-common
    command: airflow webserver --port 8080
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 5
    depends_on:
      airflow-init:
        condition: service_completed_successfully

  airflow-scheduler:
    <<: *airflow-common
    command: airflow scheduler
    depends_on:
      airflow-init:
        condition: service_completed_successfully

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.15.0
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlflow/artifacts
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow

volumes:
  postgres-data:
  mlflow-data:
```

#### [MODIFY] [.env.example](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/.env.example)

```diff
 NEBIUS_API_KEY=XXX
+MLFLOW_TRACKING_URI=http://localhost:5000
+# S3/Object Storage (optional)
+S3_ENDPOINT_URL=
+S3_BUCKET=mlops-assignment-artifacts
+AWS_ACCESS_KEY_ID=
+AWS_SECRET_ACCESS_KEY=
```

---

### Phase 7: REPORT.md (10%)

#### [NEW] [REPORT.md](file:///c:/work/development/Nebius/M3%20-%20assignment%2003/mlops-assignment-e2e-ml-pipeline/REPORT.md)

Sections to include:

1. **Architecture Overview** — Mermaid diagram showing DAG flow and system components
2. **How to Set Up** — VM setup, Docker Compose up, env vars
3. **How to Trigger a Run** — Airflow UI screenshots, parameter descriptions
4. **Artifact Layout** — Tree structure of `runs/<run-id>/`
5. **MLflow Integration** — Screenshot of MLflow UI, how to compare runs
6. **Rerunning by Run ID** — How to re-trigger with the same params
7. **One Completed Evaluation** — Summary of at least one real run with metrics
8. **Production-Style Additions** — DockerOperator explanation, Docker Compose docs

---

### Phase 8: Screenshots & Evidence

#### [NEW] `screenshots/` directory

Capture and commit:

| File | Content |
|---|---|
| `screenshots/airflow_dag.png` | Airflow UI showing the `evaluate-agent` DAG graph view |
| `screenshots/mlflow_runs.png` | MLflow UI showing logged evaluation runs and metrics |
| `screenshots/object_storage_artifacts.png` | S3 console or CLI output showing uploaded run artifacts |
| `screenshots/dag_params.png` | Airflow trigger form showing all configurable parameters |
| `screenshots/run_folder.png` | Terminal `tree` output of a completed `runs/<run-id>/` folder |

---

## File Summary (All Changes)

| Status | File | Purpose |
|---|---|---|
| **[NEW]** | `src/pipeline.py` | Helper functions (build_run_config, prepare_run_dir, run_agent_batch, run_swebench_eval, collect_metrics, build_manifest, log_mlflow_run, upload_to_s3) |
| **[NEW]** | `src/__init__.py` | Make `src` a Python package |
| **[NEW]** | `dags/evaluate_agent.py` | Main configurable Airflow DAG (easy-mode, subprocess-based) |
| **[NEW]** | `dags/evaluate_agent_docker.py` | Production-style DAG using DockerOperator |
| **[NEW]** | `docker-compose.yaml` | Docker Compose for Airflow + MLflow deployment |
| **[NEW]** | `REPORT.md` | Assignment report |
| **[NEW]** | `screenshots/` | Evidence screenshots |
| **[MODIFY]** | `pyproject.toml` | Add `mlflow`, `boto3` dependencies |
| **[MODIFY]** | `.env.example` | Add MLflow, S3 env vars |
| **[MODIFY]** | `Dockerfile` | Copy `src/` into image |
| **[MODIFY]** | `.gitignore` | Add `runs/`, keep sample |

---

## Open Questions

> [!IMPORTANT]
> **Q1: Do you have access to a Nebius VM?** The actual agent runs require a VM with Docker and `NEBIUS_API_KEY`. If you don't have one yet, I can prepare all the code so it's ready to deploy — you just need to `docker compose up` on the VM and trigger the DAG.

> [!IMPORTANT]
> **Q2: Do you want S3/Object Storage upload?** This is extra credit. If you have Nebius Object Storage credentials, I'll implement the full S3 upload flow. Otherwise, I'll document how it would work and log the local path to MLflow.

> [!IMPORTANT]
> **Q3: Do you want me to build both the easy-mode DAG (subprocess) AND the DockerOperator DAG?** The rubric says easy-mode can get "most of the credit" for isolation (10%), but DockerOperator gets full marks. I recommend building both — the subprocess DAG for quick testing and the DockerOperator DAG for the production-style solution.

> [!IMPORTANT]  
> **Q4: Should I implement the plan now or do you want to review/adjust first?** I'm ready to write all the code files immediately upon your approval.

---

## Verification Plan

### Automated Tests
```bash
# 1. Lint the DAG
uv run ruff check dags/ src/

# 2. Validate Airflow DAG parsing
AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags airflow dags list

# 3. Docker Compose validation
docker compose config

# 4. Build the Docker image
docker build -t mlops-pipeline:latest .
```

### Manual Verification
1. Start Airflow + MLflow via `docker compose up -d`
2. Open Airflow UI at http://localhost:8080
3. Trigger `evaluate-agent` DAG with `split=test, subset=verified, workers=3, task_slice=0:3`
4. Wait for completion → verify `runs/<run-id>/` folder structure
5. Check MLflow UI at http://localhost:5000 → verify params, metrics, artifacts
6. Take screenshots for `screenshots/` directory
7. Verify `manifest.json` points to all correct files
8. (If S3 configured) Verify files appear in Object Storage
