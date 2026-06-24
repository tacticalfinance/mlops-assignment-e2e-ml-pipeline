# End-to-End ML Pipeline: SWE-bench Evaluation

This project implements a production-grade Airflow pipeline for running [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) on SWE-bench tasks and evaluating the results using the SWE-bench evaluation harness.

The pipeline is fully configurable via the Airflow UI, tracks experiments using MLflow, and executes agent runs in isolated Docker containers via Docker-in-Docker using Airflow's `DockerOperator`.

---

## 1. Architecture Overview

The pipeline consists of a 4-stage Airflow DAG.

```mermaid
graph TD
    A[prepare_run] -->|Run Config| B(run_agent)
    B -->|Preds Path| C(run_eval)
    C -->|Eval Path| D[summarize_and_log]

    subgraph Airflow Python Tasks
    A
    D
    end

    subgraph DockerOperator (mlops-pipeline:latest)
    B
    C
    end

    D -.->|Logs metrics & artifacts| ML[(MLflow Tracking Server)]
    B -.->|Saves trajectories| R[(runs/<run-id>/)]
    C -.->|Saves logs| R
```

1. **`prepare_run`**: Reads Airflow UI parameters, generates a unique `run_id`, and creates the local `runs/<run-id>/` directory structure.
2. **`run_agent`**: Uses `DockerOperator` to launch the `mini-swe-agent` inside an isolated container. Outputs predictions to `preds.json`.
3. **`run_eval`**: Uses `DockerOperator` to run the SWE-bench harness against the agent's predictions, generating detailed logs and a results summary.
4. **`summarize_and_log`**: Parses the results, compiles a `metrics.json` and `manifest.json`, and logs all parameters, metrics, and key artifacts to MLflow.

---

## 2. Setup Instructions (Remote Server)

To run this pipeline on a remote server equipped with Docker and Docker Compose:

1. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd mlops-assignment-e2e-ml-pipeline
   ```

2. **Configure Environment Variables:**
   Copy the example environment file and add your Nebius API key:

   ```bash
   cp .env.example .env
   # Edit .env and set NEBIUS_API_KEY=your_actual_key
   nano .env
   ```

3. **Set Docker User Permissions:**
   Ensure your user has permission to run Docker commands without `sudo` by creating the `docker` group (if it doesn't exist) and granting access to the socket:

   ```bash
   sudo groupadd -f docker
   sudo usermod -aG docker $USER
   sudo chmod 666 /var/run/docker.sock
   ```

4. **Build the Docker Image:**
   Update the lockfile and build the custom Docker image (`mlops-pipeline:latest`) that will be used by the `DockerOperator`:

   ```bash
   uv lock
   docker build -t mlops-pipeline:latest .
   ```

5. **Set Directory Permissions:**
   Airflow runs as a non-root user (UID 50000). To prevent permission errors when Docker mounts the artifact directories, create them and set broad permissions:

   ```bash
   mkdir -p runs logs
   chmod -R 777 runs logs
   ```

6. **Start the Infrastructure:**
   Launch Airflow (Webserver, Scheduler, PostgreSQL) and MLflow via Docker Compose:

   ```bash
   docker compose up -d
   ```

   _Note: Airflow might take a minute to run database migrations and start._

7. **Access UIs via SSH Port Forwarding:**
   If you are accessing the remote server via SSH, you must forward ports `8080` (Airflow) and `5000` (MLflow) to your local machine:
   ```bash
   ssh -i "path-to-ssh-key-private.key" -L 8080:localhost:8080 -L 5000:localhost:5000 user@<server-ip>
   ```
   You can then access the interfaces in your local browser at `http://localhost:8080` and `http://localhost:5000`.

---

## 3. How to Trigger a Run

1. Open your browser and navigate to the Airflow UI at `http://<server-ip>:8080`.
2. Login with the default credentials: Username: `admin`, Password: `admin`.
3. Locate the `evaluate-agent-docker` DAG.
4. Click the **Play** button and select **Trigger DAG w/ config**.
5. You will be presented with a configuration form. The parameters are:
   - **`split`**: The dataset split (`test` or `dev`).
   - **`subset`**: The dataset subset (`verified` or `lite`).
   - **`workers`**: Number of parallel agents (e.g., `3`).
   - **`model`**: The LLM identifier (e.g., `nebius/moonshotai/Kimi-K2.6`).
   - **`task_slice`**: Python slice notation to run a subset of tasks (e.g., `0:2` for a quick test). Leave empty to run all.
   - **`run_id`**: A custom string identifier for your run (e.g., `test_run_1`). Note: the backend code treats this as optional and auto-generates a timestamp if missing, but newer Airflow UI versions may require you to type a value before submitting.
   - **`cost_limit`**: Maximum spend per agent (in USD).
6. Click **Trigger** to start the pipeline.

---

## 4. Concrete Run Example

The following is an authentic, end-to-end evaluation run committed under `runs/001/`.

**Input Parameters:**

- `split`: test
- `subset`: verified
- `workers`: 5
- `model`: nebius/moonshotai/Kimi-K2.6
- `task_slice`: 0:1
- `run_id`: 001
- `cost_limit`: 2.0

**Resulting Metrics:**
- `total_instances`: 1
- `submitted_instances`: 1
- `completed_instances`: 1
- `resolved_instances`: 1
- `unresolved_instances`: 0
- `error_instances`: 0
- `resolve_rate`: 1.0000

The agent successfully produced a valid patch that was applied and evaluated by SWE-bench. Out of 1 instance(s) submitted, 1 was resolved, yielding a **resolve_rate of 100.0%**. All pipeline stages (Airflow → Agent → Evaluator → Metrics → MLflow) completed successfully.

---

## 5. Artifact Layout

When a run completes, all output artifacts are neatly structured in the `runs/` directory mapped to your host machine:

```text
runs/<run_id>/
├── config.json                     # Full configuration for the run
├── metrics.json                    # Parsed metrics (e.g., resolve_rate)
├── manifest.json                   # Index file pointing to all important files
├── run-agent/
│   ├── preds.json                  # Final predictions formatted for SWE-bench
│   ├── <instance_id>/              # Directory per task
│   │   └── <instance_id>.traj.json # Complete agent trajectory for the task
│   └── minisweagent.log            # Agent execution logs
└── run-eval/
    ├── results.json                # Aggregated SWE-bench evaluation summary
    └── logs/
        └── <model_name>/
            └── <instance_id>/
                ├── report.json     # Pass/fail status for the patch
                ├── run_instance.log# Build logs from the SWE-bench container
                └── patch.diff      # The generated diff
```

---

## 6. MLflow Integration

The pipeline natively integrates with MLflow to track performance across different configurations.

1. Access the MLflow UI at `http://<server-ip>:5000`.
2. Select the `swe-bench-evaluation` experiment.
3. Each Airflow execution corresponds to an MLflow Run.
4. **Parameters** (like `model`, `subset`, `workers`) and **Metrics** (like `resolved_instances`, `resolve_rate`) are automatically tracked.
5. **Artifacts** (`config.json`, `metrics.json`, `preds.json`) are attached to the run for easy download and review.
6. You can select multiple runs and click **Compare** to analyze how different models or configurations perform against each other.

---

## 7. Rerunning by Run ID

To exactly reproduce a specific run or retry a failed execution:

1. In Airflow, go to **Browse > DAG Runs** and find the failed/completed run.
2. Select it and choose **Clear** to re-run the tasks with the exact same parameters.
3. Alternatively, when triggering a new run manually, input the same `split`, `subset`, `model`, and `task_slice` parameters into the config form.

---

## 8. Production-Style Additions

This implementation fulfills the assignment requirements with the following production-grade additions:

- **DockerOperator Isolation**: Instead of running heavy agent processes directly on the Airflow worker via `BashOperator`, we utilize `DockerOperator` (`evaluate_agent_docker.py`). The pipeline spawns ephemeral, isolated `mlops-pipeline:latest` containers for the `run_agent` and `run_eval` tasks. It mounts the Docker socket (`/var/run/docker.sock`) to allow Docker-in-Docker execution, which `mini-swe-agent` requires.
- **Dynamic Configurable Forms**: The Airflow DAG utilizes Airflow's `Param` class with typing and enums. This prevents invalid inputs (e.g., entering an invalid subset) directly at the UI level before the DAG even starts.
- **XCom State Passing**: Configuration state is generated once in the lightweight `prepare_run` Python task and passed robustly via XCom to the templated Jinja fields of the `DockerOperator`.
- **Integrated Storage**: All runs generate a `manifest.json` which makes integrating S3 or downstream metric aggregators seamless.
