"""Batch ResearchClawBench evaluation CLI backed by ResearchHarness.

This module intentionally keeps orchestration in ResearchClawBench:
workspace construction, scoring, and summary generation stay here, while the
installed ``researchharness`` package is used only as the agent runtime.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import secrets
import signal
import statistics
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from .config import TASKS_DIR, WORKSPACES_DIR
from .run_task import TaskRunner
from .utils import list_tasks


class EvalConfigError(ValueError):
    """Raised when a batch evaluation config is invalid."""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    api_base: str
    api_key: str
    name_source: str
    api_key_source: str
    api_base_source: str


@dataclass(frozen=True)
class ScorerConfig:
    enabled: bool
    model: str
    api_base: str
    api_key: str
    model_source: str
    api_key_source: str
    api_base_source: str


@dataclass(frozen=True)
class RunSpec:
    task_id: str
    repeat_index: int


@dataclass(frozen=True)
class TaskPlanItem:
    task_id: str
    repeats: int


@dataclass(frozen=True)
class BatchContext:
    batch_id: str
    batch_dir: Path
    config_name: str
    config_path: Path
    model: ModelConfig
    scorer: ScorerConfig
    task_plan: list[TaskPlanItem]
    max_workers: int


@dataclass(frozen=True)
class ActiveRun:
    runner: TaskRunner
    start_time: float
    model_name: str
    config_name: str
    repeat_index: int


_RUN_ALLOCATION_LOCK = threading.Lock()
_ALLOCATED_RUN_IDS: set[str] = set()
_ACTIVE_RUNS_LOCK = threading.Lock()
_ACTIVE_RUNS: dict[str, ActiveRun] = {}
CLI_WORKSPACE_GROUP = "cli_runs"
CLI_RUN_PREFIX = "cli"
EVAL_REPORT_PREFIX = "eval_report"


def _cli_workspaces_dir() -> Path:
    return WORKSPACES_DIR / CLI_WORKSPACE_GROUP


def _eval_report_name(batch_id: str) -> str:
    return f"{EVAL_REPORT_PREFIX}_{batch_id}.md"


def _new_batch_context(
    *,
    config_path: Path,
    config_name: str,
    model: ModelConfig,
    scorer: ScorerConfig,
    task_plan: list[TaskPlanItem],
    max_workers: int,
) -> BatchContext:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for _attempt in range(100):
        batch_id = f"{CLI_RUN_PREFIX}_{timestamp}_{secrets.token_hex(4)}"
        batch_dir = _cli_workspaces_dir() / batch_id
        if not batch_dir.exists():
            batch_dir.mkdir(parents=True)
            return BatchContext(
                batch_id=batch_id,
                batch_dir=batch_dir,
                config_name=config_name,
                config_path=config_path,
                model=model,
                scorer=scorer,
                task_plan=task_plan,
                max_workers=max_workers,
            )
    raise RuntimeError("Failed to allocate a unique CLI batch directory.")


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by user envs
        raise EvalConfigError("PyYAML is required. Install it with: pip install PyYAML") from exc

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvalConfigError(f"Failed to read config file: {path}") from exc
    except Exception as exc:
        raise EvalConfigError(f"Failed to parse YAML config: {exc}") from exc
    if not isinstance(data, dict):
        raise EvalConfigError("Config must be a YAML mapping.")
    return data


def _as_positive_int(value: Any, name: str, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise EvalConfigError(f"{name} is required.")
        value = default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise EvalConfigError(f"{name} must be a positive integer.") from exc
    if parsed <= 0:
        raise EvalConfigError(f"{name} must be a positive integer.")
    return parsed


def _as_optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _as_positive_int(value, name)


def _as_optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise EvalConfigError(f"{name} must be a number.") from exc


def _as_required_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise EvalConfigError(f"{name} must be explicitly set to true or false.")
    return value


def _section(config: dict[str, Any], name: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if name not in config:
        if default is None:
            raise EvalConfigError(f"{name} is required.")
        value = default
    else:
        value = config[name]
    if not isinstance(value, dict):
        raise EvalConfigError(f"{name} must be a mapping.")
    return value


def _resolve_config_value(
    section: dict[str, Any],
    *,
    value_key: str,
    env_key: str,
    label: str,
    required: bool,
) -> tuple[str, str]:
    direct = str(section.get(value_key) or "").strip()
    if direct:
        return direct, value_key
    env_name = str(section.get(env_key) or "").strip()
    if not env_name:
        raise EvalConfigError(f"{label} is required. Set {value_key} or {env_key} explicitly.")
    if env_name:
        value = os.environ.get(env_name, "")
        if value:
            return value, env_name
    if required:
        raise EvalConfigError(f"{label} env var {env_name} is not set.")
    return f"${env_name}", env_name


def _load_model_config(config: dict[str, Any], *, require_secrets: bool) -> ModelConfig:
    if "model" in config:
        raise EvalConfigError("Use agent_model for the evaluated model; the old model section is not supported.")
    model = _section(config, "agent_model")
    name, name_source = _resolve_config_value(
        model,
        value_key="name",
        env_key="name_env",
        label="agent_model.name",
        required=require_secrets,
    )
    api_base, api_base_source = _resolve_config_value(
        model,
        value_key="api_base",
        env_key="api_base_env",
        label="agent_model.api_base",
        required=require_secrets,
    )
    api_key, api_key_source = _resolve_config_value(
        model,
        value_key="api_key",
        env_key="api_key_env",
        label="agent_model.api_key",
        required=require_secrets,
    )
    return ModelConfig(
        name=name,
        api_base=api_base,
        api_key=api_key,
        name_source=name_source,
        api_key_source=api_key_source,
        api_base_source=api_base_source,
    )


def _load_scorer_config(config: dict[str, Any], *, require_secrets: bool, force_disabled: bool) -> ScorerConfig:
    if "scorer" in config:
        raise EvalConfigError("Use judge_model for the scoring model; the old scorer section is not supported.")
    scorer = _section(config, "judge_model")
    yaml_enabled = _as_required_bool(scorer.get("enabled"), "judge_model.enabled")
    enabled = yaml_enabled and not force_disabled
    model = ""
    model_source = ""
    if enabled:
        model, model_source = _resolve_config_value(
            scorer,
            value_key="name",
            env_key="name_env",
            label="judge_model.name",
            required=require_secrets,
        )
        api_base, api_base_source = _resolve_config_value(
            scorer,
            value_key="api_base",
            env_key="api_base_env",
            label="judge_model.api_base",
            required=require_secrets,
        )
        api_key, api_key_source = _resolve_config_value(
            scorer,
            value_key="api_key",
            env_key="api_key_env",
            label="judge_model.api_key",
            required=require_secrets,
        )
    else:
        api_base, api_base_source = "", ""
        api_key, api_key_source = "", ""
    return ScorerConfig(
        enabled=enabled,
        model=model,
        api_base=api_base,
        api_key=api_key,
        model_source=model_source,
        api_key_source=api_key_source,
        api_base_source=api_base_source,
    )


def _resolve_task_plan(config: dict[str, Any], default_repeats: int) -> list[TaskPlanItem]:
    raw_tasks = config.get("tasks")
    if raw_tasks == "all":
        task_plan = [TaskPlanItem(task_id, default_repeats) for task_id in list_tasks()]
    elif isinstance(raw_tasks, str):
        task_plan = [TaskPlanItem(raw_tasks, default_repeats)]
    elif isinstance(raw_tasks, list):
        task_plan = []
        for index, item in enumerate(raw_tasks):
            if isinstance(item, str):
                task_id = item.strip()
                repeats = default_repeats
            elif isinstance(item, dict):
                task_id = str(item.get("id") or item.get("task_id") or "").strip()
                repeats = _as_positive_int(
                    item.get("repeats") or item.get("repeats_per_task"),
                    f"tasks[{index}].repeats",
                    default_repeats,
                )
            else:
                raise EvalConfigError("tasks entries must be task ids or mappings with id/repeats.")
            if task_id:
                task_plan.append(TaskPlanItem(task_id, repeats))
    else:
        raise EvalConfigError("tasks must be a task id, a list of task ids, or 'all'.")
    if not task_plan:
        raise EvalConfigError("No tasks selected.")
    unknown = [item.task_id for item in task_plan if not (TASKS_DIR / item.task_id / "task_info.json").exists()]
    if unknown:
        raise EvalConfigError(f"Unknown task ids: {unknown}")
    return task_plan


def _validate_researchharness_config(config: dict[str, Any]) -> None:
    rh = _section(config, "researchharness", {})
    if "max_llm_calls" in rh:
        raise EvalConfigError("researchharness.max_llm_calls is no longer supported; use researchharness.max_rounds.")


def _load_researchharness():
    try:
        from agent_base.react_agent import default_llm_config
        from benchmarks.ResearchClawBench.adapter import ResearchClawBenchAgent
    except ImportError as exc:
        raise EvalConfigError(
            "researchharness is required. Install it with: pip install researchharness"
        ) from exc
    spec = importlib.util.find_spec("benchmarks.ResearchClawBench.adapter")
    role_prompt_path = Path(spec.origin).with_name("role_prompt.md") if spec and spec.origin else None
    if role_prompt_path is None or not role_prompt_path.exists():
        raise EvalConfigError(
            "Installed researchharness package does not expose benchmarks/ResearchClawBench/role_prompt.md."
        )
    role_prompt = role_prompt_path.read_text(encoding="utf-8")
    return ResearchClawBenchAgent, default_llm_config, role_prompt


def _build_llm_config(default_llm_config, config: dict[str, Any], model: ModelConfig) -> dict[str, Any]:
    rh = _section(config, "researchharness", {})
    llm = default_llm_config(model_name=model.name)
    llm["api_base"] = model.api_base
    llm["api_key"] = model.api_key
    timeout = _as_optional_float(rh.get("timeout_seconds"), "researchharness.timeout_seconds")
    if timeout is not None:
        llm["timeout_seconds"] = timeout
    generate_cfg = dict(llm.get("generate_cfg", {}))
    overrides = {
        "max_input_tokens": _as_optional_int(rh.get("max_input_tokens"), "researchharness.max_input_tokens"),
        "max_output_tokens": _as_optional_int(rh.get("max_output_tokens"), "researchharness.max_output_tokens"),
        "max_retries": _as_optional_int(rh.get("max_retries"), "researchharness.max_retries"),
        "temperature": _as_optional_float(rh.get("temperature"), "researchharness.temperature"),
        "top_p": _as_optional_float(rh.get("top_p"), "researchharness.top_p"),
        "presence_penalty": _as_optional_float(rh.get("presence_penalty"), "researchharness.presence_penalty"),
        "compact_trigger_tokens": rh.get("compact_trigger_tokens"),
    }
    for key, value in overrides.items():
        if value is not None:
            generate_cfg[key] = value
    llm["generate_cfg"] = generate_cfg
    return llm


def _make_unique_runner(task_id: str, repeat_index: int, agent_name: str, batch_dir: Path) -> TaskRunner:
    with _RUN_ALLOCATION_LOCK:
        runner = TaskRunner(task_id, agent_cmd="ResearchHarness Python API", agent_name=agent_name)
        base_timestamp = runner.timestamp
        for _attempt in range(100):
            nonce = secrets.token_hex(4)
            timestamp = base_timestamp
            runner.timestamp = timestamp
            runner.run_id = f"{CLI_RUN_PREFIX}_{task_id}_{timestamp}_{nonce}"
            runner.workspace = batch_dir / runner.run_id
            runner.meta_path = runner.workspace / "_meta.json"
            runner.output_path = runner.workspace / "_agent_output.jsonl"
            runner.instructions_path = runner.workspace / "INSTRUCTIONS.md"
            if runner.run_id not in _ALLOCATED_RUN_IDS and not runner.workspace.exists():
                _ALLOCATED_RUN_IDS.add(runner.run_id)
                return runner
    raise RuntimeError(f"Failed to allocate a unique workspace for {task_id} repeat {repeat_index}.")


def _register_active_run(
    runner: TaskRunner,
    *,
    start_time: float,
    model_name: str,
    config_name: str,
    repeat_index: int,
) -> None:
    with _ACTIVE_RUNS_LOCK:
        _ACTIVE_RUNS[runner.run_id] = ActiveRun(
            runner=runner,
            start_time=start_time,
            model_name=model_name,
            config_name=config_name,
            repeat_index=repeat_index,
        )


def _unregister_active_run(run_id: str) -> None:
    with _ACTIVE_RUNS_LOCK:
        _ACTIVE_RUNS.pop(run_id, None)


def _append_interruption_event(runner: TaskRunner, reason: str) -> None:
    try:
        runner.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(runner.output_path, "a", encoding="utf-8") as output_f:
            output_f.write(json.dumps({"type": "error", "error": reason}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _mark_active_runs_interrupted(reason: str, exit_code: int) -> None:
    with _ACTIVE_RUNS_LOCK:
        active_runs = list(_ACTIVE_RUNS.values())
        _ACTIVE_RUNS.clear()
    for active in active_runs:
        runner = active.runner
        if not runner.workspace.exists():
            continue
        duration = round(time.time() - active.start_time)
        _append_interruption_event(runner, reason)
        try:
            _write_meta(
                runner,
                "failed",
                {
                    "exit_code": exit_code,
                    "model": active.model_name,
                    "duration_seconds": duration,
                    "termination": "interrupted",
                    "error": reason,
                    "evaluation_config": active.config_name,
                    "repeat_index": active.repeat_index,
                },
            )
        except Exception:
            pass


def _install_interrupt_handlers() -> dict[int, Any]:
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        reason = f"CLI evaluation interrupted by {signal_name}."
        exit_code = 128 + signum
        print(f"\n{reason} Marking active runs as failed and exiting.", file=sys.stderr, flush=True)
        _mark_active_runs_interrupted(reason, exit_code)
        os._exit(exit_code)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)
    return previous_handlers


def _restore_interrupt_handlers(previous_handlers: dict[int, Any]) -> None:
    for signum, previous in previous_handlers.items():
        signal.signal(signum, previous)


def _write_meta(runner: TaskRunner, status: str, extra: dict[str, Any]) -> None:
    runner._write_meta(status, extra)


def _score_completed_run(run_id: str, scorer: ScorerConfig) -> tuple[float | None, dict[str, Any] | None, str]:
    if not scorer.enabled:
        return None, None, ""
    os.environ["JUDGE_API_KEY"] = scorer.api_key
    os.environ["JUDGE_API_BASE"] = scorer.api_base
    os.environ["JUDGE_MODEL_NAME"] = scorer.model
    from . import score as score_module

    score_module.JUDGE_MODEL_NAME = scorer.model
    score_data = score_module.score_run(run_id)
    if not isinstance(score_data, dict):
        return None, None, "Scorer returned a non-dict result."
    if score_data.get("error"):
        return None, score_data, str(score_data["error"])
    total = score_data.get("total_score")
    return (float(total) if total is not None else None), score_data, ""


def _run_one(
    spec: RunSpec,
    *,
    config: dict[str, Any],
    config_name: str,
    model: ModelConfig,
    scorer: ScorerConfig,
    batch_dir: Path,
    role_prompt: str,
    ResearchClawBenchAgent,
    default_llm_config,
) -> dict[str, Any]:
    start = time.time()
    rh = _section(config, "researchharness", {})
    agent_name = str(config.get("agent_name") or f"ResearchHarness ({model.name})")
    runner = _make_unique_runner(spec.task_id, spec.repeat_index, agent_name, batch_dir)
    session: dict[str, Any] = {}
    score_error = ""
    score_value: float | None = None
    registered_active = False
    try:
        runner.setup_workspace()
        _register_active_run(
            runner,
            start_time=start,
            model_name=model.name,
            config_name=config_name,
            repeat_index=spec.repeat_index,
        )
        registered_active = True
        trace_dir = batch_dir / f"{runner.run_id}_trace"
        llm = _build_llm_config(default_llm_config, config, model)
        agent = ResearchClawBenchAgent(
            llm=llm,
            role_prompt=role_prompt,
            trace_dir=str(trace_dir),
            max_rounds=_as_optional_int(rh.get("max_rounds"), "researchharness.max_rounds"),
            max_runtime_seconds=_as_optional_int(rh.get("max_runtime_seconds"), "researchharness.max_runtime_seconds"),
        )
        prompt = runner.instructions_path.read_text(encoding="utf-8")
        with open(runner.output_path, "a", encoding="utf-8") as output_f:
            def event_callback(event: dict[str, Any]) -> None:
                output_f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
                output_f.flush()

            session = agent._run_session(
                prompt,
                workspace_root=str(runner.workspace.resolve()),
                event_callback=event_callback,
            )

        duration = round(time.time() - start)
        report_exists = (runner.workspace / "report" / "report.md").exists()
        termination = str(session.get("termination") or "")
        completed = termination == "result" and report_exists
        status = "completed" if completed else "failed"
        if completed:
            try:
                score_value, _score_data, score_error = _score_completed_run(runner.run_id, scorer)
            except Exception as exc:
                score_error = f"{type(exc).__name__}: {exc}"
        _write_meta(
            runner,
            status,
            {
                "exit_code": 0 if status == "completed" else 1,
                "model": model.name,
                "duration_seconds": duration,
                "termination": termination,
                "trace_path": session.get("trace_path", ""),
                "session_state_path": session.get("session_state_path", ""),
                "evaluation_config": config_name,
                "repeat_index": spec.repeat_index,
                "score_error": score_error,
            },
        )
        return {
            "task_id": spec.task_id,
            "repeat": spec.repeat_index,
            "run_id": runner.run_id,
            "status": status,
            "score": score_value,
            "duration_seconds": duration,
            "model": model.name,
            "workspace": str(runner.workspace),
            "report_exists": report_exists,
            "termination": termination,
            "trace_path": session.get("trace_path", ""),
            "score_error": score_error,
        }
    except Exception as exc:
        duration = round(time.time() - start)
        error = f"{type(exc).__name__}: {exc}"
        if runner.workspace.exists():
            try:
                (runner.workspace / "_agent_output.jsonl").parent.mkdir(parents=True, exist_ok=True)
                with open(runner.output_path, "a", encoding="utf-8") as output_f:
                    output_f.write(
                        json.dumps(
                            {"type": "error", "error": error, "traceback": traceback.format_exc()},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                _write_meta(
                    runner,
                    "failed",
                    {
                        "exit_code": 1,
                        "model": model.name,
                        "duration_seconds": duration,
                        "error": error,
                        "evaluation_config": config_name,
                        "repeat_index": spec.repeat_index,
                    },
                )
            except Exception:
                pass
        return {
            "task_id": spec.task_id,
            "repeat": spec.repeat_index,
            "run_id": runner.run_id,
            "status": "failed",
            "score": None,
            "duration_seconds": duration,
            "model": model.name,
            "workspace": str(runner.workspace),
            "report_exists": (runner.workspace / "report" / "report.md").exists(),
            "termination": "exception",
            "trace_path": session.get("trace_path", "") if session else "",
            "score_error": error,
        }
    finally:
        if registered_active:
            _unregister_active_run(runner.run_id)


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _variance(values: list[float]) -> float | None:
    return round(statistics.pvariance(values), 4) if len(values) > 1 else 0.0 if values else None


def _std(values: list[float]) -> float | None:
    return round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0 if values else None


def _summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_ids = sorted({row["task_id"] for row in rows})
    task_summary: list[dict[str, Any]] = []
    for task_id in task_ids:
        subset = [row for row in rows if row["task_id"] == task_id]
        scores = [float(row["score"]) for row in subset if row.get("score") is not None]
        task_summary.append(
            {
                "task_id": task_id,
                "runs": len(subset),
                "scored_runs": len(scores),
                "mean": _mean(scores),
                "std": _std(scores),
                "variance": _variance(scores),
                "min": round(min(scores), 4) if scores else None,
                "max": round(max(scores), 4) if scores else None,
                "success_rate": round(sum(1 for row in subset if row["status"] == "completed") / len(subset), 4),
            }
        )
    all_scores = [float(row["score"]) for row in rows if row.get("score") is not None]
    overall = {
        "tasks": len(task_ids),
        "runs": len(rows),
        "scored_runs": len(all_scores),
        "mean": _mean(all_scores),
        "std": _std(all_scores),
        "variance": _variance(all_scores),
        "min": round(min(all_scores), 4) if all_scores else None,
        "max": round(max(all_scores), 4) if all_scores else None,
        "success_rate": round(sum(1 for row in rows if row["status"] == "completed") / len(rows), 4) if rows else 0.0,
    }
    return task_summary, overall


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _render_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "(no rows)"
    widths = {
        col: max(len(col), *(len(_format_value(row.get(col))) for row in rows))
        for col in columns
    }
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    sep = "  ".join("-" * widths[col] for col in columns)
    body = ["  ".join(_format_value(row.get(col)).ljust(widths[col]) for col in columns) for row in rows]
    return "\n".join([header, sep, *body])


def _escape_markdown_cell(value: Any) -> str:
    text = _format_value(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _render_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_escape_markdown_cell(row.get(col)) for col in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _run_columns() -> list[str]:
    run_columns = [
        "task_id",
        "repeat",
        "run_id",
        "status",
        "score",
        "duration_seconds",
        "model",
        "report_exists",
        "termination",
        "trace_path",
        "score_error",
    ]
    return run_columns


def _task_columns() -> list[str]:
    return ["task_id", "runs", "scored_runs", "mean", "std", "variance", "min", "max", "success_rate"]


def _safe_source(value: str) -> str:
    if not value:
        return ""
    if value in {"api_key", "agent_model.api_key", "judge_model.api_key", "name"}:
        return "inline"
    return value


def _format_task_plan(task_plan: list[TaskPlanItem]) -> str:
    return ", ".join(f"{item.task_id} x{item.repeats}" for item in task_plan)


def _planned_runs(task_plan: list[TaskPlanItem]) -> int:
    return sum(item.repeats for item in task_plan)


def _write_eval_report(
    batch: BatchContext,
    *,
    rows: list[dict[str, Any]],
    task_summary: list[dict[str, Any]],
    overall: dict[str, Any],
) -> Path:
    run_columns = _run_columns()
    task_columns = _task_columns()
    run_links = []
    for row in rows:
        workspace_rel = Path(row["workspace"]).relative_to(batch.batch_dir)
        trace_path = row.get("trace_path") or ""
        trace_rel = ""
        if trace_path:
            try:
                trace_rel = str(Path(trace_path).relative_to(batch.batch_dir))
            except ValueError:
                trace_rel = trace_path
        trace_note = f"; trace: `{trace_rel}`" if trace_rel else ""
        run_links.append(f"- `{row['run_id']}`: `{workspace_rel}`{trace_note}")
    lines = [
        "# ResearchClawBench CLI Evaluation Report",
        "",
        "## Batch Metadata",
        "",
        f"- Batch ID: `{batch.batch_id}`",
        f"- Batch directory: `{batch.batch_dir}`",
        f"- Config file: `{batch.config_path}`",
        f"- Config name: `{batch.config_name}`",
        f"- Agent model: `{batch.model.name}`",
        f"- Agent model name source: `{_safe_source(batch.model.name_source)}`",
        f"- Agent model API base source: `{_safe_source(batch.model.api_base_source)}`",
        f"- Agent model API key source: `{_safe_source(batch.model.api_key_source)}`",
        f"- Scoring enabled: `{batch.scorer.enabled}`",
        f"- Judge model: `{batch.scorer.model}`",
        f"- Judge model name source: `{_safe_source(batch.scorer.model_source)}`",
        f"- Judge model API base source: `{_safe_source(batch.scorer.api_base_source)}`",
        f"- Judge model API key source: `{_safe_source(batch.scorer.api_key_source)}`",
        f"- Task plan: `{_format_task_plan(batch.task_plan)}`",
        f"- Max concurrent runs: `{batch.max_workers}`",
        f"- Total planned runs: `{_planned_runs(batch.task_plan)}`",
        "",
        "## Run Directories",
        "",
        *run_links,
        "",
        "## Per-Run Results",
        "",
        _render_markdown_table(rows, run_columns),
        "",
        "## Per-Task Summary",
        "",
        _render_markdown_table(task_summary, task_columns),
        "",
        "## Overall Summary",
        "",
        "```json",
        json.dumps(overall, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    report_path = batch.batch_dir / _eval_report_name(batch.batch_id)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _print_dry_run(task_plan: list[TaskPlanItem], max_workers: int, model: ModelConfig, scorer: ScorerConfig) -> None:
    print("Dry run OK.")
    print(f"Agent model: {model.name}")
    print(f"Agent model name source: {model.name_source}")
    print(f"Agent model API base source: {model.api_base_source}")
    print(f"Agent model API key source: {model.api_key_source}")
    print(f"Scoring: {'enabled' if scorer.enabled else 'disabled'}")
    if scorer.enabled:
        print(f"Judge model: {scorer.model}")
        print(f"Judge model name source: {scorer.model_source}")
        print(f"Judge model API base source: {scorer.api_base_source}")
        print(f"Judge model API key source: {scorer.api_key_source}")
    print(f"Tasks: {len(task_plan)}")
    print(f"Task plan: {_format_task_plan(task_plan)}")
    print(f"Max concurrent runs: {max_workers}")
    print(f"Planned runs: {_planned_runs(task_plan)}")
    for item in task_plan:
        print(f"  - {item.task_id} x{item.repeats}")


def run_eval(config_path: Path, *, dry_run: bool, no_score: bool, skip_secret_check: bool) -> int:
    config = _load_yaml(config_path)
    _validate_researchharness_config(config)
    repeats = _as_positive_int(config.get("repeats_per_task"), "repeats_per_task", 1)
    task_plan = _resolve_task_plan(config, repeats)
    max_workers = _as_positive_int(config.get("max_concurrent_runs"), "max_concurrent_runs", 1)
    model = _load_model_config(config, require_secrets=not skip_secret_check)
    scorer = _load_scorer_config(config, require_secrets=not skip_secret_check, force_disabled=no_score)
    ResearchClawBenchAgent, default_llm_config, role_prompt = _load_researchharness()

    if dry_run:
        _print_dry_run(task_plan, max_workers, model, scorer)
        return 0

    config_name = str(config.get("name") or config_path.stem)
    batch = _new_batch_context(
        config_path=config_path,
        config_name=config_name,
        model=model,
        scorer=scorer,
        task_plan=task_plan,
        max_workers=max_workers,
    )
    specs = [
        RunSpec(item.task_id, repeat)
        for item in task_plan
        for repeat in range(1, item.repeats + 1)
    ]
    rows: list[dict[str, Any]] = []
    print(f"Starting {len(specs)} runs with max_concurrent_runs={max_workers}")
    print(f"Batch: {batch.batch_dir}")

    previous_handlers = _install_interrupt_handlers()
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_one,
                    spec,
                    config=config,
                    config_name=config_name,
                    model=model,
                    scorer=scorer,
                    batch_dir=batch.batch_dir,
                    role_prompt=role_prompt,
                    ResearchClawBenchAgent=ResearchClawBenchAgent,
                    default_llm_config=default_llm_config,
                )
                for spec in specs
            ]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(
                    f"[{len(rows)}/{len(specs)}] {row['task_id']} repeat={row['repeat']} "
                    f"status={row['status']} score={_format_value(row.get('score'))} run_id={row['run_id']}"
                )
    finally:
        _restore_interrupt_handlers(previous_handlers)

    rows.sort(key=lambda row: (row["task_id"], int(row["repeat"])))
    task_summary, overall = _summarize(rows)
    print("\nPer-run results:")
    print(_render_table(rows, _run_columns()))
    print("\nPer-task summary:")
    print(_render_table(task_summary, _task_columns()))
    print("\nOverall summary:")
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    report_path = _write_eval_report(batch, rows=rows, task_summary=task_summary, overall=overall)
    print(f"\nEvaluation report: {report_path}")
    return 0 if all(row["status"] == "completed" for row in rows) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run batch ResearchClawBench evaluations with ResearchHarness.")
    parser.add_argument("config", help="Path to an eval YAML config.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dependencies without creating workspaces.",
    )
    parser.add_argument("--no-score", action="store_true", help="Run agents but skip LLM scoring.")
    parser.add_argument(
        "--skip-secret-check",
        action="store_true",
        help="Do not require API key/base env vars during validation. Intended for config linting only.",
    )
    args = parser.parse_args(argv)
    if args.skip_secret_check and not args.dry_run:
        print("--skip-secret-check is only allowed with --dry-run.", file=sys.stderr)
        return 2
    try:
        return run_eval(
            Path(args.config).resolve(),
            dry_run=args.dry_run,
            no_score=args.no_score,
            skip_secret_check=args.skip_secret_check,
        )
    except EvalConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
