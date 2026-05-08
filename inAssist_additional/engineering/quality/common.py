"""
 Модуль: common.py
 Назначение: Общие функции загрузки датасетов, построения промптов и расчёта метрик качества
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

VALID_TOOL_NAMES = {
    "create_event",
    "update_event",
    "find_free_slot",
    "split_task",
    "summarize_week",
    "general_chat",
    "clarification_needed",
}
VALID_REQUIREMENTS = {"none", "slots", "events"}
VALID_STEP_RESPONSE_TYPES = {"tool_call", "clarify", "finish"}
VALID_GATEWAY_TOOLS = {
    "find_event",
    "list_events",
    "get_free_slots",
    "create_event",
    "update_event",
    "delete_event",
}
SUPPORTED_METRICS = {
    "step_action_macro_f1",
    "step_action_accuracy",
    "deterministic_fields_exact_rate",
    "time_exact_rate_conditional",
    "episode_success_rate",
    "json_valid_rate",
    "schema_valid_rate",
}

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import app.prompts as service_prompts


def module_root() -> Path:
    return Path(__file__).resolve().parent


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    path = module_root() / config_path
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def resolve_path(relative_path: str) -> Path:
    return (module_root() / relative_path).resolve()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_test_payload(path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    """Загружает eval-датасет и возвращает выбранную метрику вместе со списком samples."""
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found: {path}")

    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"Test dataset is empty: {path}")

    raw = json.loads(text)

    if isinstance(raw, list):
        return "episode_success_rate", raw

    if isinstance(raw, dict):
        metric = raw.get("metric", "episode_success_rate")
        samples = raw.get("samples", [])
        if not isinstance(metric, str):
            raise ValueError("Field 'metric' must be a string")
        if not isinstance(samples, list):
            raise ValueError("Field 'samples' must be a list")
        return metric, samples

    raise ValueError("Test dataset must be a JSON array or object with fields 'metric' and 'samples'")


def _normalize_step_case(
        step: Dict[str, Any],
        *,
        sample_index: int,
        episode_id: str,
        episode_type: Optional[str],
        step_number: int,
) -> Optional[Dict[str, Any]]:
    request = step.get("step_request") or step.get("request")
    expected = step.get("expected_step_response") or step.get("expected_json")
    if not isinstance(request, dict) or not isinstance(expected, dict):
        return None

    return {
        "case_id": f"{episode_id}:step_{step_number}",
        "episode_id": episode_id,
        "episode_type": episode_type or "default",
        "step_number": step_number,
        "stage": "step",
        "user_text": request.get("text", ""),
        "context": request.get("context", {}),
        "state": request.get("state"),
        "conversation": request.get("conversation"),
        "all_context": request.get("all_context"),
        "expected_json": expected,
        "evaluation": step.get("evaluation", {}),
        "source_index": sample_index,
    }


def normalize_cases(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Приводит legacy и step samples к единому списку eval-case словарей."""
    cases: List[Dict[str, Any]] = []

    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            continue

        sample_id = str(sample.get("sample_id", sample.get("id", f"sample_{index}")))

        steps = sample.get("steps")
        if isinstance(steps, list):
            episode_id = str(sample.get("episode_id", sample_id))
            episode_type = sample.get("episode_type")
            for step_number, step in enumerate(steps, start=1):
                if not isinstance(step, dict):
                    continue
                normalized = _normalize_step_case(
                    step,
                    sample_index=index,
                    episode_id=episode_id,
                    episode_type=episode_type,
                    step_number=step_number,
                )
                if normalized:
                    cases.append(normalized)
            continue

        step_request = sample.get("step_request") or sample.get("request")
        expected_step = sample.get("expected_step_response")
        if isinstance(step_request, dict) and isinstance(expected_step, dict):
            normalized = _normalize_step_case(
                sample,
                sample_index=index,
                episode_id=str(sample.get("episode_id", sample_id)),
                episode_type=sample.get("episode_type"),
                step_number=int(sample.get("step_number", 1)),
            )
            if normalized:
                cases.append(normalized)
            continue

        analyze_request = sample.get("analyze_request")
        expected_analyze = sample.get("expected_analyze_response")
        if isinstance(analyze_request, dict) and isinstance(expected_analyze, dict):
            cases.append(
                {
                    "case_id": f"{sample_id}:analyze",
                    "episode_id": str(sample.get("episode_id", sample_id)),
                    "episode_type": sample.get("episode_type", "legacy"),
                    "step_number": int(sample.get("step_number", 1)),
                    "stage": "analyze",
                    "user_text": analyze_request.get("text", ""),
                    "context": analyze_request.get("context", {}),
                    "conversation": analyze_request.get("conversation"),
                    "all_context": analyze_request.get("all_context"),
                    "expected_json": expected_analyze,
                    "evaluation": sample.get("evaluation", {}),
                }
            )

        execute_request = sample.get("execute_request")
        expected_execute = sample.get("expected_execute_response")
        if isinstance(execute_request, dict) and isinstance(expected_execute, dict):
            cases.append(
                {
                    "case_id": f"{sample_id}:execute",
                    "episode_id": str(sample.get("episode_id", sample_id)),
                    "episode_type": sample.get("episode_type", "legacy"),
                    "step_number": int(sample.get("step_number", 1)),
                    "stage": "execute",
                    "user_text": execute_request.get("text", ""),
                    "context": execute_request.get("context", {}),
                    "conversation": execute_request.get("conversation"),
                    "all_context": execute_request.get("all_context"),
                    "expected_json": expected_execute,
                    "intent": execute_request.get("tool_name") or expected_execute.get("tool_name"),
                    "fetched_events": execute_request.get("fetched_events", []),
                    "fetched_slots": execute_request.get("fetched_slots", []),
                    "evaluation": sample.get("evaluation", {}),
                }
            )

        if "expected_json" in sample:
            case = dict(sample)
            case.setdefault("case_id", sample_id)
            case.setdefault("episode_id", str(sample.get("episode_id", sample_id)))
            case.setdefault("episode_type", sample.get("episode_type", "legacy"))
            case.setdefault("step_number", int(sample.get("step_number", 1)))
            case.setdefault("evaluation", {})
            cases.append(case)

    cases.sort(key=lambda case: (str(case.get("episode_id", "")), int(case.get("step_number", 1)),
                                 str(case.get("case_id", ""))))
    return cases


def build_user_payload(sample: Dict[str, Any]) -> str:
    stage = str(sample.get("stage", "step")).lower()

    if stage == "step":
        payload = {
            "text": sample.get("user_text", ""),
            "context": sample.get("context", {}),
        }
        if isinstance(sample.get("state"), dict):
            payload["state"] = sample.get("state")
        if sample.get("conversation") is not None:
            payload["conversation"] = sample.get("conversation")
        if sample.get("all_context"):
            payload["all_context"] = sample.get("all_context")
        return json.dumps(payload, ensure_ascii=False)

    payload = {
        "text": sample.get("user_text", ""),
        "context": sample.get("context", {}),
    }
    if sample.get("conversation") is not None:
        payload["conversation"] = sample.get("conversation")
    if sample.get("all_context"):
        payload["all_context"] = sample.get("all_context")
    return json.dumps(payload, ensure_ascii=False)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _derive_slot_hints(sample: Dict[str, Any]) -> Dict[str, str]:
    slot_hints = sample.get("slot_hints")
    if isinstance(slot_hints, dict):
        return {
            "best_slot_time": str(slot_hints.get("best_slot_time", "10:00")),
            "alt_slot_time": str(slot_hints.get("alt_slot_time", "11:00")),
        }

    fetched_slots = sample.get("fetched_slots")
    if isinstance(fetched_slots, list) and fetched_slots:
        best_slot = fetched_slots[0]
        alt_slot = fetched_slots[1] if len(fetched_slots) > 1 else fetched_slots[0]
        return {
            "best_slot_time": str(best_slot.get("start", "10:00")).split("T")[-1][:5],
            "alt_slot_time": str(alt_slot.get("start", "11:00")).split("T")[-1][:5],
        }

    return {"best_slot_time": "10:00", "alt_slot_time": "11:00"}


def build_runtime_context(sample: Dict[str, Any]) -> str:
    stage = str(sample.get("stage", "step")).lower()
    expected = sample.get("expected_json") if isinstance(sample.get("expected_json"), dict) else {}
    context = sample.get("context") if isinstance(sample.get("context"), dict) else {}

    parts = [
        service_prompts.RUNTIME_CONTEXT_PROMPT.format(
            current_time=context.get("current_time") or "",
            timezone=context.get("timezone") or "UTC",
        ).strip()
    ]

    if stage == "execute":
        tool_name = str(sample.get("intent") or expected.get("tool_name") or "").lower()
        dynamic_input: Dict[str, Any] = {"user_request": sample.get("user_text", "")}

        if tool_name in {"summarize_week", "update_event"}:
            events = sample.get("fetched_events", [])
            dynamic_input["events"] = events if isinstance(events, list) else []
        elif tool_name == "find_free_slot":
            dynamic_input.update(_derive_slot_hints(sample))

        parts.append(
            service_prompts.DYNAMIC_INPUT_PROMPT.format(
                dynamic_input_json=json.dumps(
                    {"dynamic_input": dynamic_input},
                    ensure_ascii=False,
                )
            ).strip()
        )

    return "\n\n".join(part for part in parts if part)


def select_system_prompt(sample: Dict[str, Any]) -> str:
    """Выбирает системный промпт сервиса под stage и intent конкретного eval-case."""
    stage = str(sample.get("stage", "step")).lower()
    expected = sample.get("expected_json") if isinstance(sample.get("expected_json"), dict) else {}

    if stage == "step":
        return service_prompts.STEP_SYSTEM_PROMPT.format(
            persona=service_prompts.CHAT_PERSONA,
        )

    tool_name = str(sample.get("intent") or expected.get("tool_name") or "").lower()
    if stage == "execute":
        if tool_name == "summarize_week":
            return service_prompts.SUMMARIZE_PROMPT.format(
                persona=service_prompts.CHAT_PERSONA,
            )

        if tool_name == "find_free_slot":
            return service_prompts.SLOT_FOUND_PROMPT.format(
                persona=service_prompts.CHAT_PERSONA,
            )

        if tool_name == "split_task":
            return service_prompts.SPLIT_TASK_PROMPT.format(
                persona=service_prompts.CHAT_PERSONA,
            )

        if tool_name == "update_event":
            return service_prompts.UPDATE_EVENT_PROMPT.format(
                persona=service_prompts.CHAT_PERSONA,
            )

    return service_prompts.ROUTER_SYSTEM_PROMPT.format(
        persona=service_prompts.CHAT_PERSONA,
    )


def build_prompt(system_prompt: str, user_payload: str, runtime_context: Optional[str] = None) -> str:
    model_user_text = user_payload
    if runtime_context and runtime_context.strip():
        model_user_text = f"{runtime_context.strip()}\n\nUSER PAYLOAD:\n{user_payload}"

    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{model_user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def extract_json_object(text: str) -> Tuple[bool, Dict[str, Any]]:
    if not text:
        return False, {}

    left = text.find("{")
    right = text.rfind("}")
    if left == -1 or right == -1 or right <= left:
        return False, {}

    try:
        return True, json.loads(text[left: right + 1])
    except json.JSONDecodeError:
        return False, {}


def normalize_step_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(payload or {})

    if not isinstance(data.get("response_payload"), dict):
        data["response_payload"] = {}

    if not isinstance(data.get("state_patch"), dict):
        data["state_patch"] = {}

    if "next_gateway_action" not in data:
        root_tool_call = data.get("tool_call")
        if isinstance(root_tool_call, dict):
            data["next_gateway_action"] = {"type": "tool_call", "tool_call": root_tool_call}
        else:
            data["next_gateway_action"] = {"type": "none", "tool_call": None}

    gateway_action = data.get("next_gateway_action")
    if isinstance(gateway_action, dict):
        if "type" not in gateway_action:
            gateway_action["type"] = "tool_call" if gateway_action.get("tool_call") else "none"
        if "tool_call" not in gateway_action:
            gateway_action["tool_call"] = None
        data["next_gateway_action"] = gateway_action
    else:
        data["next_gateway_action"] = {"type": "none", "tool_call": None}

    if "response_type" not in data:
        if data["next_gateway_action"].get("type") == "tool_call":
            data["response_type"] = "tool_call"
        elif data.get("assistant_message"):
            data["response_type"] = "finish"
        else:
            data["response_type"] = "clarify"

    if "assistant_message" not in data:
        data["assistant_message"] = None

    return data


def _validate_slot(slot: Any) -> bool:
    return isinstance(slot, dict) and isinstance(slot.get("start"), str) and isinstance(slot.get("end"), str)


def _validate_update_params(parameters: Any) -> bool:
    if not isinstance(parameters, dict):
        return False
    if not isinstance(parameters.get("event_id"), str):
        return False
    updates = parameters.get("updates")
    if not isinstance(updates, dict):
        return False
    allowed = {"title", "start_time", "duration_minutes"}
    if not any(key in updates for key in allowed):
        return False
    if "title" in updates and updates["title"] is not None and not isinstance(updates["title"], str):
        return False
    if "start_time" in updates and updates["start_time"] is not None and not isinstance(updates["start_time"], str):
        return False
    if "duration_minutes" in updates and updates["duration_minutes"] is not None and not isinstance(
            updates["duration_minutes"], (int, float)
    ):
        return False
    return True


def validate_execute_schema(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    tool_name = payload.get("tool_name")
    if tool_name not in VALID_TOOL_NAMES:
        return False
    if not isinstance(payload.get("reply_text"), str):
        return False

    parameters = payload.get("parameters", {})
    if not isinstance(parameters, dict):
        return False

    if tool_name == "create_event":
        return (
                isinstance(parameters.get("title"), str)
                and isinstance(parameters.get("start_time"), str)
                and isinstance(parameters.get("duration_minutes"), (int, float))
        )

    if tool_name == "update_event":
        return _validate_update_params(parameters)

    if tool_name == "find_free_slot":
        ranked_slots = parameters.get("ranked_slots")
        reasoning = parameters.get("reasoning")
        return (
                isinstance(ranked_slots, list)
                and all(_validate_slot(slot) for slot in ranked_slots)
                and isinstance(reasoning, str)
        )

    if tool_name == "split_task":
        subtasks = parameters.get("subtasks")
        if not isinstance(parameters.get("main_task"), str) or not isinstance(subtasks, list):
            return False
        return all(
            isinstance(task, dict)
            and isinstance(task.get("title"), str)
            and isinstance(task.get("duration_minutes"), (int, float))
            for task in subtasks
        )

    return True


def validate_analyze_schema(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    tool_name = payload.get("tool_name")
    requirement = payload.get("requirement")
    if tool_name not in VALID_TOOL_NAMES:
        return False
    if requirement not in VALID_REQUIREMENTS:
        return False

    if requirement == "none":
        final_response = payload.get("final_response")
        return final_response is None or validate_execute_schema(final_response)

    data_params = payload.get("data_params")
    if not isinstance(data_params, dict):
        return False
    return "start" in data_params and "end" in data_params


def validate_step_schema(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False

    normalized = normalize_step_payload(payload)
    response_type = normalized.get("response_type")
    if response_type not in VALID_STEP_RESPONSE_TYPES:
        return False

    if response_type in {"clarify", "finish"}:
        return normalized.get("assistant_message") is None or isinstance(normalized.get("assistant_message"), str)

    next_gateway_action = normalized.get("next_gateway_action")
    if not isinstance(next_gateway_action, dict):
        return False
    if next_gateway_action.get("type") != "tool_call":
        return False

    tool_call = next_gateway_action.get("tool_call")
    if not isinstance(tool_call, dict):
        return False
    tool_name = tool_call.get("tool_name")
    arguments = tool_call.get("arguments")
    if tool_name not in VALID_GATEWAY_TOOLS or not isinstance(arguments, dict):
        return False

    if tool_name == "find_event":
        return isinstance(arguments.get("query"), str)
    if tool_name == "list_events":
        return isinstance(arguments.get("start"), str) and isinstance(arguments.get("end"), str)
    if tool_name == "get_free_slots":
        return isinstance(arguments.get("start"), str) and isinstance(arguments.get("end"), str)
    if tool_name == "create_event":
        return (
                isinstance(arguments.get("title"), str)
                and isinstance(arguments.get("start_time"), str)
                and isinstance(arguments.get("duration_minutes"), (int, float))
        )
    if tool_name == "update_event":
        return _validate_update_params(arguments)
    if tool_name == "delete_event":
        return isinstance(arguments.get("event_id"), str)

    return False


def step_action_label(payload: Dict[str, Any], invalid_label: str = "__invalid__") -> str:
    if not isinstance(payload, dict):
        return invalid_label

    normalized = normalize_step_payload(payload)
    response_type = normalized.get("response_type")
    if response_type == "tool_call":
        next_gateway_action = normalized.get("next_gateway_action") or {}
        tool_call = next_gateway_action.get("tool_call") if isinstance(next_gateway_action, dict) else None
        tool_name = tool_call.get("tool_name") if isinstance(tool_call, dict) else None
        if isinstance(tool_name, str):
            return f"tool_call/{tool_name}"
        return invalid_label

    if response_type in {"clarify", "finish"}:
        return response_type

    return invalid_label


def analyze_action_label(payload: Dict[str, Any], invalid_label: str = "__invalid__|__invalid__") -> str:
    if not isinstance(payload, dict):
        return invalid_label
    tool_name = payload.get("tool_name")
    requirement = payload.get("requirement")
    if not isinstance(tool_name, str) or not isinstance(requirement, str):
        return invalid_label
    return f"{tool_name}|{requirement}"


def get_by_path(obj: Any, path: str) -> Any:
    current = obj
    token = ""
    i = 0
    while i < len(path):
        char = path[i]
        if char == ".":
            if token:
                if not isinstance(current, dict) or token not in current:
                    return None
                current = current[token]
                token = ""
            i += 1
            continue

        if char == "[":
            if token:
                if not isinstance(current, dict) or token not in current:
                    return None
                current = current[token]
                token = ""
            end = path.find("]", i)
            if end == -1:
                return None
            index_text = path[i + 1: end]
            try:
                idx = int(index_text)
            except ValueError:
                return None
            if not isinstance(current, list) or idx >= len(current):
                return None
            current = current[idx]
            i = end + 1
            continue

        token += char
        i += 1

    if token:
        if not isinstance(current, dict) or token not in current:
            return None
        current = current[token]

    return current


def _value_equal(expected: Any, predicted: Any) -> bool:
    if isinstance(expected, (dict, list)) or isinstance(predicted, (dict, list)):
        return json.dumps(expected, ensure_ascii=False, sort_keys=True) == json.dumps(
            predicted, ensure_ascii=False, sort_keys=True
        )
    return str(expected) == str(predicted)


def compare_paths(expected: Dict[str, Any], predicted: Dict[str, Any], paths: Iterable[str]) -> Tuple[bool, List[str]]:
    mismatches: List[str] = []
    for path in paths:
        expected_value = get_by_path(expected, path)
        predicted_value = get_by_path(predicted, path)
        if expected_value is None and predicted_value is None:
            continue
        if expected_value is None and predicted_value is not None:
            mismatches.append(f"unexpected:{path}={predicted_value}")
            continue
        if expected_value is not None and predicted_value is None:
            mismatches.append(f"missing:{path}")
            continue
        if not _value_equal(expected_value, predicted_value):
            mismatches.append(f"diff:{path}: expected={expected_value} predicted={predicted_value}")
    return len(mismatches) == 0, mismatches


def auto_time_paths(sample: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    evaluation = sample.get("evaluation", {})
    if isinstance(evaluation, dict):
        explicit = evaluation.get("time_paths")
        if isinstance(explicit, list):
            return [str(item) for item in explicit if isinstance(item, str)]

    stage = str(sample.get("stage", "step")).lower()
    if stage == "step":
        label = step_action_label(expected)
        if label == "tool_call/create_event":
            if get_by_path(expected, "next_gateway_action.tool_call.arguments.start_time") is not None:
                return ["next_gateway_action.tool_call.arguments.start_time"]
        if label == "tool_call/update_event":
            if get_by_path(expected, "next_gateway_action.tool_call.arguments.updates.start_time") is not None:
                return ["next_gateway_action.tool_call.arguments.updates.start_time"]
        return []

    if stage == "analyze":
        tool_name = expected.get("tool_name")
        requirement = expected.get("requirement")
        if requirement != "none":
            return []
        if tool_name == "create_event" and get_by_path(expected, "final_response.parameters.start_time") is not None:
            return ["final_response.parameters.start_time"]
        if tool_name == "update_event" and get_by_path(expected,
                                                       "final_response.parameters.updates.start_time") is not None:
            return ["final_response.parameters.updates.start_time"]
    return []


def deterministic_field_paths(sample: Dict[str, Any], expected: Dict[str, Any]) -> List[str]:
    evaluation = sample.get("evaluation", {})
    if isinstance(evaluation, dict):
        explicit = evaluation.get("field_paths")
        if isinstance(explicit, list):
            return [str(item) for item in explicit if isinstance(item, str)]

    stage = str(sample.get("stage", "step")).lower()
    if stage == "step":
        label = step_action_label(expected)
        if label == "tool_call/find_event":
            return ["next_gateway_action.tool_call.arguments.query"]
        if label == "tool_call/list_events":
            return ["next_gateway_action.tool_call.arguments.start", "next_gateway_action.tool_call.arguments.end"]
        if label == "tool_call/get_free_slots":
            return [
                "next_gateway_action.tool_call.arguments.start",
                "next_gateway_action.tool_call.arguments.end",
                "next_gateway_action.tool_call.arguments.min_duration_minutes",
            ]
        if label == "tool_call/create_event":
            return [
                "next_gateway_action.tool_call.arguments.title",
                "next_gateway_action.tool_call.arguments.duration_minutes",
            ]
        if label == "tool_call/update_event":
            return ["next_gateway_action.tool_call.arguments.event_id"]
        if label == "tool_call/delete_event":
            return ["next_gateway_action.tool_call.arguments.event_id"]
    return []


def is_critical_step(sample: Dict[str, Any]) -> bool:
    evaluation = sample.get("evaluation", {})
    if isinstance(evaluation, dict) and "critical" in evaluation:
        return bool(evaluation.get("critical"))
    return True


def manual_macro_f1(expected_labels: List[str], predicted_labels: List[str]) -> float:
    if not expected_labels:
        return 0.0

    labels = sorted(set(expected_labels) | set(predicted_labels))
    if not labels:
        return 0.0

    scores = []
    for label in labels:
        tp = sum(
            1
            for expected, predicted in zip(expected_labels, predicted_labels)
            if expected == label and predicted == label
        )
        fp = sum(
            1
            for expected, predicted in zip(expected_labels, predicted_labels)
            if expected != label and predicted == label
        )
        fn = sum(
            1
            for expected, predicted in zip(expected_labels, predicted_labels)
            if expected == label and predicted != label
        )

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))

    return sum(scores) / len(scores)


def compute_episode_success(
        details: List[Dict[str, Any]],
        episode_total_steps: Dict[str, int],
        *,
        only_completed_episodes: bool,
) -> Tuple[float, int]:
    """Считает долю эпизодов, где все критичные шаги прошли успешно."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for detail in details:
        grouped[str(detail.get("episode_id", detail.get("case_id", "unknown")))].append(detail)

    eligible_episode_ids: List[str] = []
    success_count = 0

    for episode_id, episode_details in grouped.items():
        total_steps = episode_total_steps.get(episode_id, len(episode_details))
        if only_completed_episodes and len(episode_details) < total_steps:
            continue
        eligible_episode_ids.append(episode_id)

        critical_steps = [detail for detail in episode_details if detail.get("critical", True)]
        if not critical_steps:
            success_count += 1
            continue

        if all(bool(detail.get("step_success")) for detail in critical_steps):
            success_count += 1

    if not eligible_episode_ids:
        return 0.0, 0
    return success_count / len(eligible_episode_ids), len(eligible_episode_ids)


def build_report_payload(
        *,
        config: Dict[str, Any],
        mode: str,
        selected_metric_name: str,
        sample_count: int,
        completed_samples: int,
        completed_episodes: int,
        time_scored_count: int,
        field_scored_count: int,
        supported_metrics: Dict[str, float],
        details: List[Dict[str, Any]],
        extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "run_name": config.get("run_name", "llama_step_eval"),
        "mode": mode,
        "sample_count": sample_count,
        "completed_samples": completed_samples,
        "completed_episodes": completed_episodes,
        "progress_percent": (completed_samples / sample_count * 100.0) if sample_count else 0.0,
        "time_scored_count": time_scored_count,
        "field_scored_count": field_scored_count,
        "selected_metric": selected_metric_name,
        "selected_metric_value": supported_metrics[selected_metric_name],
        "metrics": supported_metrics,
        "details": details,
    }
    if extra:
        payload.update(extra)
    return payload
