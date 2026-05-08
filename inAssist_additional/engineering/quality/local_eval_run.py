"""
 Модуль: local_eval_run.py
 Назначение: Локальная оценка GGUF-модели через llama.cpp на общем eval-наборе
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

import contextlib
import io
import json
import hashlib
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from llama_cpp import Llama
try:
    from llama_cpp import LlamaDiskCache
except ImportError:
    LlamaDiskCache = None

from common import (
    SUPPORTED_METRICS,
    append_jsonl,
    auto_time_paths,
    build_prompt,
    build_report_payload,
    build_runtime_context,
    build_user_payload,
    compare_paths,
    compute_episode_success,
    deterministic_field_paths,
    extract_json_object,
    is_critical_step,
    load_config,
    load_test_payload,
    manual_macro_f1,
    normalize_cases,
    normalize_step_payload,
    resolve_path,
    save_json,
    select_system_prompt,
    step_action_label,
    analyze_action_label,
    validate_analyze_schema,
    validate_step_schema,
)

import app.prompts as service_prompts


if LlamaDiskCache is not None:
    class QuietLlamaDiskCache(LlamaDiskCache):
        """Disk cache without llama-cpp-python cache hit/save stderr chatter."""

        def __getitem__(self, key):
            with contextlib.redirect_stderr(io.StringIO()):
                return super().__getitem__(key)

        def __setitem__(self, key, value):
            with contextlib.redirect_stderr(io.StringIO()):
                super().__setitem__(key, value)
else:
    QuietLlamaDiskCache = None


def _selected_cases(raw_samples: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    normalized = normalize_cases(raw_samples)
    step_cases = [sample for sample in normalized if str(sample.get("stage", "")).lower() == "step"]
    if step_cases:
        return "step", step_cases
    analyze_cases = [sample for sample in normalized if str(sample.get("stage", "")).lower() == "analyze"]
    return "analyze", analyze_cases


def _prompt_cache_enabled(config: Dict[str, Any]) -> bool:
    value = config.get("use_llama_prompt_cache", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _prompt_cache_capacity_bytes(config: Dict[str, Any]) -> int:
    raw_value = config.get("llama_cache_size_bytes", 1024 ** 3)
    try:
        return max(int(raw_value), 64 * 1024 * 1024)
    except (TypeError, ValueError):
        return 1024 ** 3


def _model_cache_key(model_path: Path) -> str:
    resolved_path = model_path.resolve()
    try:
        stat = resolved_path.stat()
        signature = (
            f"{resolved_path}|{stat.st_size}|{int(stat.st_mtime)}|"
            f"{service_prompts.PROMPT_CACHE_VERSION}"
        )
    except OSError:
        signature = f"{resolved_path}|{service_prompts.PROMPT_CACHE_VERSION}"
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]


def _configure_prompt_cache(llm: Llama, model_path: Path, config: Dict[str, Any]) -> bool:
    if not _prompt_cache_enabled(config):
        return False
    if QuietLlamaDiskCache is None:
        return False

    try:
        cache_root = resolve_path(config.get("llama_cache_dir", "reports/llama_cache"))
        cache_dir = cache_root / _model_cache_key(model_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        llm.set_cache(
            QuietLlamaDiskCache(
                cache_dir=str(cache_dir),
                capacity_bytes=_prompt_cache_capacity_bytes(config),
            )
        )
        return True
    except Exception as exc:
        print(f"llama prompt cache: setup failed: {exc}")
        return False


def _warm_prompt_cache(llm: Llama, sample: Dict[str, Any]) -> None:
    try:
        prompt = build_prompt(select_system_prompt(sample), "{}", build_runtime_context(sample))
        llm(
            prompt,
            max_tokens=1,
            temperature=0.0,
            stop=["<|eot_id|>", "<|end_of_text|>"],
            echo=False,
        )
    except Exception as exc:
        print(f"llama prompt cache: warmup failed: {exc}")


def _evaluate_case(sample: Dict[str, Any], generated: str) -> Dict[str, Any]:
    stage = str(sample.get("stage", "step")).lower()
    expected_raw = sample.get("expected_json", {})
    expected = normalize_step_payload(expected_raw) if stage == "step" else expected_raw

    is_json, parsed = extract_json_object(generated)
    parsed_normalized = normalize_step_payload(parsed) if (is_json and stage == "step") else (
        parsed if is_json else None)

    if stage == "step":
        schema_valid = is_json and validate_step_schema(parsed_normalized or {})
        expected_action = step_action_label(expected)
        predicted_action = step_action_label(parsed_normalized or {})
    else:
        schema_valid = is_json and validate_analyze_schema(parsed or {})
        expected_action = analyze_action_label(expected)
        predicted_action = analyze_action_label(parsed or {})

    action_exact = expected_action == predicted_action

    field_paths = deterministic_field_paths(sample, expected)
    fields_scored = len(field_paths) > 0
    fields_exact = None
    field_mismatches: List[str] = []
    if fields_scored:
        fields_exact, field_mismatches = compare_paths(expected, parsed_normalized or parsed or {}, field_paths)

    time_paths = auto_time_paths(sample, expected)
    time_scored = len(time_paths) > 0
    time_exact = None
    time_mismatches: List[str] = []
    if time_scored:
        time_exact, time_mismatches = compare_paths(expected, parsed_normalized or parsed or {}, time_paths)

    step_success = bool(
        is_json
        and schema_valid
        and action_exact
        and (fields_exact if fields_scored else True)
        and (time_exact if time_scored else True)
    )

    return {
        "case_id": sample.get("case_id"),
        "episode_id": sample.get("episode_id", sample.get("case_id")),
        "episode_type": sample.get("episode_type", "default"),
        "step_number": sample.get("step_number", 1),
        "stage": stage,
        "user_text": sample.get("user_text", ""),
        "critical": is_critical_step(sample),
        "expected_action_class": expected_action,
        "predicted_action_class": predicted_action,
        "action_exact": action_exact,
        "field_paths": field_paths,
        "fields_scored": fields_scored,
        "fields_exact": fields_exact,
        "field_mismatches": field_mismatches,
        "time_paths": time_paths,
        "time_scored": time_scored,
        "time_exact": time_exact,
        "time_mismatches": time_mismatches,
        "step_success": step_success,
        "expected": expected,
        "llm_response_text": generated,
        "generated_text": generated,
        "parsed": parsed if is_json else None,
        "parsed_normalized": parsed_normalized if is_json else None,
        "json_valid": is_json,
        "schema_valid": schema_valid,
    }


def _compute_metrics(details: List[Dict[str, Any]], episode_total_steps: Dict[str, int]) -> Tuple[
    Dict[str, float], int]:
    total_count = len(details)
    expected_labels = [detail["expected_action_class"] for detail in details]
    predicted_labels = [detail["predicted_action_class"] for detail in details]

    field_details = [detail for detail in details if detail.get("fields_scored")]
    time_details = [detail for detail in details if detail.get("time_scored")]
    json_valid_count = sum(1 for detail in details if detail.get("json_valid"))
    schema_valid_count = sum(1 for detail in details if detail.get("schema_valid"))

    episode_success_rate, completed_episodes = compute_episode_success(
        details,
        episode_total_steps,
        only_completed_episodes=True,
    )

    metrics = {
        "step_action_macro_f1": manual_macro_f1(expected_labels, predicted_labels),
        "step_action_accuracy": (
            sum(1 for expected, predicted in zip(expected_labels, predicted_labels) if
                expected == predicted) / total_count
            if total_count
            else 0.0
        ),
        "deterministic_fields_exact_rate": (
            sum(1 for detail in field_details if detail.get("fields_exact")) / len(field_details)
            if field_details
            else 0.0
        ),
        "time_exact_rate_conditional": (
            sum(1 for detail in time_details if detail.get("time_exact")) / len(time_details)
            if time_details
            else 0.0
        ),
        "episode_success_rate": episode_success_rate,
        "json_valid_rate": (json_valid_count / total_count) if total_count else 0.0,
        "schema_valid_rate": (schema_valid_count / total_count) if total_count else 0.0,
    }
    return metrics, completed_episodes


def main() -> None:
    """Запускает локальный GGUF eval и пишет отчёт с метриками."""
    config = load_config()
    model_path = resolve_path(config.get("gguf_model_path", "../../models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"))
    test_path = resolve_path(config["test_data_path"])
    report_path = resolve_path(config["metrics_path"])
    details_jsonl_path = resolve_path(config.get("details_jsonl_path", "reports/step_details.jsonl"))

    if details_jsonl_path.exists():
        details_jsonl_path.unlink()

    if not model_path.exists():
        raise FileNotFoundError(f"GGUF model not found: {model_path}")

    metric_name, raw_samples = load_test_payload(test_path)
    stage_mode, samples = _selected_cases(raw_samples)
    if not samples:
        raise ValueError("No evaluation samples found")

    episode_total_steps: Dict[str, int] = {}
    for sample in samples:
        episode_id = str(sample.get("episode_id", sample.get("case_id")))
        episode_total_steps[episode_id] = episode_total_steps.get(episode_id, 0) + 1

    llm = Llama(model_path=str(model_path), n_ctx=4096, n_threads=18, n_gpu_layers=0, verbose=False)
    if _configure_prompt_cache(llm, model_path, config):
        _warm_prompt_cache(llm, samples[0])
    selected_metric_name = metric_name if metric_name in SUPPORTED_METRICS else (
        "episode_success_rate" if stage_mode == "step" else "step_action_macro_f1"
    )

    details: List[Dict[str, Any]] = []
    last_reported_percent = 0
    sample_count = len(samples)

    for idx, sample in enumerate(samples):
        prompt = build_prompt(
            select_system_prompt(sample),
            build_user_payload(sample),
            build_runtime_context(sample),
        )
        out = llm(
            prompt,
            max_tokens=int(config.get("max_new_tokens", 512)),
            temperature=float(config.get("temperature", 0.0)),
            stop=["<|eot_id|>", "<|end_of_text|>"],
            echo=False,
        )
        generated = out["choices"][0]["text"].strip()

        detail = _evaluate_case(sample, generated)
        details.append(detail)
        append_jsonl(details_jsonl_path, detail)

        metrics, completed_episodes = _compute_metrics(details, episode_total_steps)
        field_scored_count = sum(1 for row in details if row.get("fields_scored"))
        time_scored_count = sum(1 for row in details if row.get("time_scored"))

        save_json(
            report_path,
            build_report_payload(
                config=config,
                mode="local_gguf",
                selected_metric_name=selected_metric_name,
                sample_count=sample_count,
                completed_samples=idx + 1,
                completed_episodes=completed_episodes,
                time_scored_count=time_scored_count,
                field_scored_count=field_scored_count,
                supported_metrics=metrics,
                details=details,
                extra={
                    "tested_at_utc": datetime.now(timezone.utc).isoformat(),
                    "model_path": str(model_path),
                    "stage_mode": stage_mode,
                    "details_jsonl_path": str(details_jsonl_path),
                },
            ),
        )

        current_percent = int(((idx + 1) * 100) / sample_count) if sample_count else 100
        if current_percent > last_reported_percent:
            print(
                f"Progress: {current_percent}% "
                f"({idx + 1}/{sample_count}) "
                f"selected_metric={selected_metric_name}:{metrics[selected_metric_name]:.4f}"
            )
            last_reported_percent = current_percent

    final_metrics, completed_episodes = _compute_metrics(details, episode_total_steps)
    field_scored_count = sum(1 for row in details if row.get("fields_scored"))
    time_scored_count = sum(1 for row in details if row.get("time_scored"))

    save_json(
        report_path,
        build_report_payload(
            config=config,
            mode="local_gguf",
            selected_metric_name=selected_metric_name,
            sample_count=sample_count,
            completed_samples=sample_count,
            completed_episodes=completed_episodes,
            time_scored_count=time_scored_count,
            field_scored_count=field_scored_count,
            supported_metrics=final_metrics,
            details=details,
            extra={
                "tested_at_utc": datetime.now(timezone.utc).isoformat(),
                "model_path": str(model_path),
                "stage_mode": stage_mode,
                "details_jsonl_path": str(details_jsonl_path),
            },
        ),
    )

    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))
    print(f"report: {report_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
