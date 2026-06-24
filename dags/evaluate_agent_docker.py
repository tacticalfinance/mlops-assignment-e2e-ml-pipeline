"""
Airflow DAG: evaluate-agent-docker
====================================
Production-style pipeline using DockerOperator.

Runs mini-swe-agent on a SWE-bench subset and evaluates the results,
running the heavy workloads inside isolated Docker containers using the
project's Dockerfile (mlops-pipeline:latest).

Pipeline:
    prepare_run (Python)
    run_agent   (DockerOperator)
    run_eval    (DockerOperator)
    summarize_and_log (Python)
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.decorators import dag, task
from airflow.models.param import Param
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

# Ensure src/ is importable for the Python tasks
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# We need the absolute path of the runs folder on the host machine to mount it into the containers.
# When running in docker-compose, this might be a mounted volume path.
HOST_RUNS_DIR = os.environ.get("HOST_RUNS_DIR", str(_PROJECT_ROOT / "runs"))
HOST_LOGS_DIR = os.environ.get("HOST_LOGS_DIR", str(_PROJECT_ROOT / "logs"))


@dag(
    dag_id="evaluate-agent-docker",
    description="Production-style agent evaluation pipeline using DockerOperator",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=2,
    tags=["swe-bench", "evaluation", "docker", "production"],
    params={
        "split": Param("test", type="string", enum=["test", "dev"]),
        "subset": Param("verified", type="string", enum=["verified", "lite"]),
        "workers": Param(5, type="integer", minimum=1, maximum=20),
        "model": Param("nebius/moonshotai/Kimi-K2.6", type="string"),
        "task_slice": Param("0:3", type="string"),
        "run_id": Param("", type="string"),
        "cost_limit": Param(0, type="number"),
    },
)
def evaluate_agent_docker_dag():

    # 1. Prepare run (Local Python Task - lightweight)
    @task(task_id="prepare_run")
    def prepare_run(**context) -> dict:
        from pipeline import build_run_config, prepare_run_dir
        params = context["params"]
        run_config = build_run_config(params)
        prepare_run_dir(run_config)
        return run_config

    run_config = prepare_run()

    # We use Jinja templating in the Docker commands to inject the run_config values
    # passed from the prepare_run task via XCom.
    
    # We define the mounts. We mount the runs dir so artifacts are persisted.
    # We mount the docker socket because mini-swe-agent uses Docker-in-Docker.
    common_mounts = [
        Mount(source=HOST_RUNS_DIR, target="/mlops-assignment/runs", type="bind"),
        Mount(source=HOST_LOGS_DIR, target="/mlops-assignment/logs", type="bind"),
        Mount(source="/var/run/docker.sock", target="/var/run/docker.sock", type="bind"),
    ]

    # 2. Run Agent (DockerOperator)
    # The command constructs the CLI call based on templated params.
    # NOTE: cost_limit is enforced via the MSWEA_AGENT_COST_LIMIT env var
    # because the mini-extra swebench CLI does not expose a --cost-limit flag.
    agent_cmd = (
        "uv run mini-extra swebench "
        "--subset {{ ti.xcom_pull(task_ids='prepare_run')['subset'] }} "
        "--split {{ ti.xcom_pull(task_ids='prepare_run')['split'] }} "
        "--model '{{ ti.xcom_pull(task_ids='prepare_run')['model'] }}' "
        "--workers {{ ti.xcom_pull(task_ids='prepare_run')['workers'] }} "
        "{% if ti.xcom_pull(task_ids='prepare_run')['task_slice'] %}--slice '{{ ti.xcom_pull(task_ids='prepare_run')['task_slice'] }}' {% endif %}"
        "-o /mlops-assignment/runs/{{ ti.xcom_pull(task_ids='prepare_run')['run_id'] }}/run-agent"
    )

    run_agent = DockerOperator(
        task_id="run_agent",
        image="mlops-pipeline:latest",
        command=f"bash -c \"{agent_cmd}\"",
        mounts=common_mounts,
        environment={
            "NEBIUS_API_KEY": os.environ.get("NEBIUS_API_KEY", ""),
            "MSWEA_COST_TRACKING": "ignore_errors",
            # Pass cost_limit to the agent via environment variable.
            # mini-swe-agent reads MSWEA_AGENT_COST_LIMIT to cap per-instance
            # agent spend; the CLI does not expose a --cost-limit flag.
            "MSWEA_AGENT_COST_LIMIT": "{{ ti.xcom_pull(task_ids='prepare_run')['cost_limit'] }}",
        },
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
        network_mode="host",
        retries=1,
        execution_timeout=None,
    )

    # 3. Run Evaluation (DockerOperator)
    eval_cmd = (
        "python -m swebench.harness.run_evaluation "
        "--dataset_name '{{ ti.xcom_pull(task_ids='prepare_run')['dataset_name'] }}' "
        "--predictions_path /mlops-assignment/runs/{{ ti.xcom_pull(task_ids='prepare_run')['run_id'] }}/run-agent/preds.json "
        "--max_workers {{ ti.xcom_pull(task_ids='prepare_run')['workers'] }} "
        "--run_id {{ ti.xcom_pull(task_ids='prepare_run')['run_id'] }}"
    )

    # A post-eval step inside the container to move results to the right directory
    post_eval_cmd = (
        "RUN_DIR=/mlops-assignment/runs/{{ ti.xcom_pull(task_ids='prepare_run')['run_id'] }} && "
        "LOGS_DIR=/mlops-assignment/logs/run_evaluation/{{ ti.xcom_pull(task_ids='prepare_run')['run_id'] }} && "
        "mkdir -p $RUN_DIR/run-eval/logs $RUN_DIR/run-eval/reports && "
        "cp -r $LOGS_DIR/* $RUN_DIR/run-eval/logs/ || true && "
        "find $LOGS_DIR -name 'report.json' | while read f; do "
        "instance=$(basename $(dirname \"$f\")); "
        "cp \"$f\" \"$RUN_DIR/run-eval/reports/${instance}.json\"; done || true && "
        "MODEL_CLEAN=$(echo '{{ ti.xcom_pull(task_ids='prepare_run')['model'] }}' | tr '/' '__') && "
        "cp /mlops-assignment/${MODEL_CLEAN}.{{ ti.xcom_pull(task_ids='prepare_run')['split'] }}.json "
        "$RUN_DIR/run-eval/results.json || echo '{}' > $RUN_DIR/run-eval/results.json"
    )

    run_eval = DockerOperator(
        task_id="run_eval",
        image="mlops-pipeline:latest",
        command=f"bash -c \"{eval_cmd} && {post_eval_cmd}\"",
        mounts=common_mounts,
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
        network_mode="host",
        retries=1,
    )

    # 4. Summarize and log (Local Python Task - lightweight)
    @task(task_id="summarize_and_log")
    def summarize_and_log(run_config: dict) -> dict:
        from pipeline import collect_metrics, build_manifest, log_mlflow_run, update_report_run_example
        
        run_dir = Path(_PROJECT_ROOT) / "runs" / run_config["run_id"]
        eval_dir = run_dir / "run-eval"

        metrics = collect_metrics(eval_dir, run_dir)
        build_manifest(run_config, run_dir, metrics)
        log_mlflow_run(run_config, metrics, run_dir)
        update_report_run_example(run_config, metrics)

        return metrics

    # Dependencies
    run_config >> run_agent >> run_eval >> summarize_and_log(run_config)

evaluate_agent_docker_dag()
