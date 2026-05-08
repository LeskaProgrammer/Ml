"""
 Модуль: evaluate.py
 Назначение: Оценка LoRA/adapter-модели на step-agent или legacy eval-датасете
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common import (
    SUPPORTED_METRICS,
    analyze_action_label,
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
    validate_analyze_schema,
    validate_step_schema,
)


def _lazy_imports() -> Dict[str, Any]:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install dependencies: transformers peft torch") from exc

    return {
        "PeftModel": PeftModel,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
    }


def _generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated.strip()


def _selected_cases(raw_samples: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    normalized = normalize_cases(raw_samples)
    step_cases = [sample for sample in normalized if str(sample.get("stage", "")).lower() == "step"]
    if step_cases:
        return "step", step_cases
    analyze_cases = [sample for sample in normalized if str(sample.get("stage", "")).lower() == "analyze"]
    return "analyze", analyze_cases


def _evaluate_case(
        sample: Dict[str, Any],
        generated: str,
) -> Dict[str, Any]:
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

    action_accuracy = (
        sum(1 for expected, predicted in zip(expected_labels, predicted_labels) if expected == predicted) / total_count
        if total_count
        else 0.0
    )

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
        "step_action_accuracy": action_accuracy,
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
    """Запускает eval, сохраняет пошаговые детали и итоговые метрики."""
    config = load_config()
    libs = _lazy_imports()

    use_adapter = bool(config.get("use_adapter", True))
    adapter_path = resolve_path(config["adapter_path"])
    test_path = resolve_path(config["test_data_path"])
    metrics_path = resolve_path(config["metrics_path"])
    details_jsonl_path = resolve_path(config.get("details_jsonl_path", "reports/step_details.jsonl"))

    if details_jsonl_path.exists():
        details_jsonl_path.unlink()

    if use_adapter and not adapter_path.exists():
        raise FileNotFoundError(f"Adapter/model path not found: {adapter_path}")

    metric_name, raw_samples = load_test_payload(test_path)
    stage_mode, samples = _selected_cases(raw_samples)
    sample_count = len(samples)
    if sample_count == 0:
        raise ValueError("No evaluation samples found")

    episode_total_steps: Dict[str, int] = {}
    for sample in samples:
        episode_id = str(sample.get("episode_id", sample.get("case_id")))
        episode_total_steps[episode_id] = episode_total_steps.get(episode_id, 0) + 1

    tokenizer = libs["AutoTokenizer"].from_pretrained(config["base_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = libs["AutoModelForCausalLM"].from_pretrained(
        config["base_model"],
        torch_dtype="auto",
        device_map="auto",
    )
    model = libs["PeftModel"].from_pretrained(base_model, str(adapter_path)) if use_adapter else base_model
    model.eval()

    if metric_name in SUPPORTED_METRICS:
        selected_metric_name = metric_name
    else:
        selected_metric_name = "episode_success_rate" if stage_mode == "step" else "step_action_macro_f1"

    details: List[Dict[str, Any]] = []
    last_reported_percent = 0

    for idx, sample in enumerate(samples):
        prompt = build_prompt(
            select_system_prompt(sample),
            build_user_payload(sample),
            build_runtime_context(sample),
        )
        generated = _generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=int(config["max_new_tokens"]),
            temperature=float(config["temperature"]),
        )

        detail = _evaluate_case(sample, generated)
        details.append(detail)
        append_jsonl(details_jsonl_path, detail)

        metrics, completed_episodes = _compute_metrics(details, episode_total_steps)
        field_scored_count = sum(1 for row in details if row.get("fields_scored"))
        time_scored_count = sum(1 for row in details if row.get("time_scored"))

        save_json(
            metrics_path,
            build_report_payload(
                config=config,
                mode="adapter" if use_adapter else "baseline",
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
        metrics_path,
        build_report_payload(
            config=config,
            mode="adapter" if use_adapter else "baseline",
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
                "stage_mode": stage_mode,
                "details_jsonl_path": str(details_jsonl_path),
            },
        ),
    )

    print(f"Selected metric: {selected_metric_name}={final_metrics[selected_metric_name]:.4f}")
    print(f"Step action macro F1: {final_metrics['step_action_macro_f1']:.4f}")
    print(f"Step action accuracy: {final_metrics['step_action_accuracy']:.4f}")
    print(f"Deterministic fields exact rate: {final_metrics['deterministic_fields_exact_rate']:.4f}")
    print(f"Conditional time exact rate: {final_metrics['time_exact_rate_conditional']:.4f}")
    print(f"Episode success rate: {final_metrics['episode_success_rate']:.4f}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
