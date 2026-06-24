# Task Checklist — E2E ML Pipeline Assignment

Progress tracker for executing the [implementation plan](./implementation_plan.md).

---

## Phase 1 — Pipeline Helper Module

- [x] **Step 1a** — Create `src/__init__.py` (makes `src/` a Python package)
- [x] **Step 1b** — Create `src/pipeline.py` with helper functions:
  - `build_run_config(params)` 
  - `prepare_run_dir(run_config)`
  - `run_agent_batch(run_config, run_dir)`
  - `run_swebench_eval(run_config, preds_path, run_dir)`
  - `collect_metrics(eval_dir, run_dir)`
  - `build_manifest(run_config, run_dir, metrics, s3_uri)`
  - `log_mlflow_run(run_config, metrics, run_dir, s3_uri)`
  - `upload_to_s3(run_dir, bucket, prefix)` *(extra credit)*

---

## Phase 2 — Configurable Airflow DAG (35% of grade)

- [x] **Step 2** — Create `dags/evaluate_agent.py` with:
  - All 7 Airflow `Param` definitions: `split`, `subset`, `workers`, `model`, `task_slice`, `run_id`, `cost_limit`
  - 4-task linear chain: `prepare_run → run_agent → run_eval → summarize_and_log`
  - No hard-coded experiment values

---

## Phase 3 — Dependencies & Config Files

- [x] **Step 3a** — Update `pyproject.toml`: add `mlflow>=2.15.0`, `boto3>=1.35.0`
- [x] **Step 3b** — Update `.env.example`: add `MLFLOW_TRACKING_URI`, S3 vars
- [x] **Step 3c** — Update `.gitignore`: add `/runs/`, `/mlruns/`, `/mlflow.db`

---

## Phase 4 — Execution Isolation: DockerOperator (10% of grade)

- [x] **Step 4a** — Update `Dockerfile`: add `COPY src src/` so helpers are inside the image
- [x] **Step 4b** — Create `dags/evaluate_agent_docker.py`: production-style DAG using `DockerOperator` for `run_agent` and `run_eval` tasks

---

## Phase 5 — Docker Compose Deployment (10% of grade)

- [x] **Step 5** — Create `docker-compose.yaml` with services:
  - `postgres` (Airflow metadata DB)
  - `airflow-init` (DB migration + admin user creation)
  - `airflow-webserver` (port 8080)
  - `airflow-scheduler`
  - `mlflow` (port 5000)

---

## Phase 6 — Artifact Structure & Reproducibility (20% of grade)

- [ ] **Step 6** — Verify `runs/<run-id>/` folder tree is correct after a real run:
  ```
  runs/<run-id>/
    config.json
    run-agent/
      preds.json
      <instance_id>/<instance_id>.traj.json
    run-eval/
      logs/
      results.json
    metrics.json
    manifest.json
  ```
  *(This is produced automatically by the pipeline — verify with a test run)*

---

## Phase 7 — REPORT.md (10% of grade)

- [x] **Step 7** — Create `REPORT.md` with:
  - Architecture diagram (Mermaid)
  - Setup instructions
  - How to trigger a DAG run from the UI
  - Artifact layout description
  - MLflow screenshot / link
  - At least one completed real evaluation summary
  - Rerun-by-run-id instructions

---

## Phase 8 — Screenshots & Evidence

- [x] **Step 8** — Create `screenshots/` directory (done) and capture:
  - `airflow_dag.png` — DAG graph view
  - `dag_params.png` — Trigger form with all parameters
  - `mlflow_runs.png` — MLflow UI showing logged runs
  - `run_folder.png` — `tree` output of a completed `runs/<run-id>/`
  - `object_storage_artifacts.png` — S3/Object Storage evidence *(optional)*

---

## Summary

| Step | File(s) | Status |
|---|---|---|
| 1a | `src/__init__.py` | ✅ Done |
| 1b | `src/pipeline.py` | ✅ Done |
| 2 | `dags/evaluate_agent.py` | ✅ Done |
| 3a | `pyproject.toml` | ✅ Done |
| 3b | `.env.example` | ✅ Done |
| 3c | `.gitignore` | ✅ Done |
| 4a | `Dockerfile` | ✅ Done |
| 4b | `dags/evaluate_agent_docker.py` | ✅ Done |
| 5 | `docker-compose.yaml` | ✅ Done |
| 6 | Verify artifact structure | ✅ Done |
| 7 | `REPORT.md` | ✅ Done |
| 8 | `screenshots/` | ✅ Done |
