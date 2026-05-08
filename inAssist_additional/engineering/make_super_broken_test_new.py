"""
Create a deterministic noisy robustness dataset from quality/data/test_new.json.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from random import Random
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "engineering" / "quality" / "data" / "test_new.json"
DEFAULT_OUTPUT = ROOT / "engineering" / "quality" / "data" / "super_broken_test_new.json"
DEFAULT_SEED = 20260507

KEYBOARD_ROWS = [
    "йцукенгшщзхъ",
    "фывапролджэ",
    "ячсмитьбю",
]
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+", re.UNICODE)
DIGIT_RE = re.compile(r"\d+(?::\d+)?")
PUNCT_RE = re.compile(r"[,.!?\"«»]")

MONTH_WORDS = {
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
}

PROTECTED_TIME_WORDS = {
    *MONTH_WORDS,
    "январь",
    "февраль",
    "март",
    "апрель",
    "май",
    "июнь",
    "июль",
    "август",
    "сентябрь",
    "октябрь",
    "ноябрь",
    "декабрь",
    "сегодня",
    "завтра",
    "послезавтра",
    "вчера",
    "понедельник",
    "понедельника",
    "понедельнику",
    "вторник",
    "вторника",
    "вторнику",
    "среда",
    "среду",
    "среды",
    "четверг",
    "четверга",
    "четвергу",
    "пятница",
    "пятницу",
    "пятницы",
    "суббота",
    "субботу",
    "субботы",
    "воскресенье",
    "воскресенья",
    "воскресенью",
    "утром",
    "утро",
    "днем",
    "днём",
    "день",
    "вечером",
    "вечеру",
    "вечер",
    "обед",
    "обеда",
    "после",
    "до",
    "час",
    "часа",
    "часов",
    "минут",
    "минуты",
    "минуту",
    "полчаса",
    "полтора",
    "полторы",
}

SLANG_REPLACEMENTS = [
    (re.compile(r"\bпожалуйста\b", re.IGNORECASE), ["пж", "плиз"]),
    (re.compile(r"\bсейчас\b", re.IGNORECASE), ["щас"]),
    (re.compile(r"\bсобытие\b", re.IGNORECASE), ["событ"]),
    (re.compile(r"\bсобытия\b", re.IGNORECASE), ["событ"]),
]

ASR_LIKE_REPLACEMENTS = [
    (re.compile(r"\bпожалуйста\b", re.IGNORECASE), ["пожалста"]),
    (re.compile(r"\bсобытие\b", re.IGNORECASE), ["событье"]),
    (re.compile(r"\bкалендарь\b", re.IGNORECASE), ["календар"]),
    (re.compile(r"\bсвободное\b", re.IGNORECASE), ["свободно"]),
    (re.compile(r"\bокно\b", re.IGNORECASE), ["акно"]),
]


def keyboard_neighbors() -> dict[str, list[str]]:
    coords: dict[str, tuple[float, float]] = {}
    for row_index, row in enumerate(KEYBOARD_ROWS):
        offset = 0.5 * row_index
        for column_index, char in enumerate(row):
            coords[char] = (column_index + offset, row_index)

    neighbors: dict[str, list[str]] = {}
    for char, (x, y) in coords.items():
        items: list[tuple[float, str]] = []
        for other, (other_x, other_y) in coords.items():
            if other == char:
                continue
            distance = math.hypot(x - other_x, y - other_y)
            if distance <= 1.25:
                items.append((distance, other))
        neighbors[char] = [other for _, other in sorted(items)]
    return neighbors


NEIGHBORS = keyboard_neighbors()


def stable_rng(seed: int, text: str) -> Random:
    digest = hashlib.sha256(f"{seed}|{text}".encode("utf-8")).hexdigest()
    return Random(int(digest[:16], 16))


def word_spans(text: str) -> list[tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group(0).lower()) for match in WORD_RE.finditer(text)]


def digit_parts(text: str) -> list[str]:
    return DIGIT_RE.findall(text or "")


def month_parts(text: str) -> list[str]:
    return [word for _, _, word in word_spans(text or "") if word in MONTH_WORDS]


def protected_mask(text: str) -> list[bool]:
    mask = [False] * len(text)
    digit_ranges: list[tuple[int, int]] = []

    for match in DIGIT_RE.finditer(text):
        digit_ranges.append((match.start(), match.end()))
        start = max(0, match.start() - 1)
        end = min(len(text), match.end() + 1)
        for index in range(start, end):
            mask[index] = True

    for start, end, word in word_spans(text):
        protect = word in PROTECTED_TIME_WORDS
        if not protect:
            for digit_start, digit_end in digit_ranges:
                if 0 <= start - digit_end <= 2 or 0 <= digit_start - end <= 2:
                    protect = True
                    break
        if protect:
            for index in range(start, end):
                mask[index] = True

    return mask


def letter_candidates(text: str, mask: list[bool], *, min_word_len: int = 1) -> list[int]:
    word_ranges = [
        (start, end)
        for start, end, word in word_spans(text)
        if len(word) >= min_word_len
    ]
    result: list[int] = []
    for index, char in enumerate(text):
        if mask[index] or not char.isalpha():
            continue
        if not any(start <= index < end for start, end in word_ranges):
            continue
        result.append(index)
    return result


def keyboard_candidates(text: str, mask: list[bool]) -> list[int]:
    return [index for index, char in enumerate(text) if not mask[index] and NEIGHBORS.get(char.lower())]


def apply_keyboard_neighbor(text: str, rng: Random) -> tuple[str, bool]:
    candidates = keyboard_candidates(text, protected_mask(text))
    if not candidates:
        return text, False
    index = rng.choice(candidates)
    old = text[index]
    new = rng.choice(NEIGHBORS[old.lower()])
    if old.isupper():
        new = new.upper()
    return text[:index] + new + text[index + 1:], True


def apply_delete_letter(text: str, rng: Random) -> tuple[str, bool]:
    candidates = letter_candidates(text, protected_mask(text), min_word_len=5)
    if not candidates:
        return text, False
    index = rng.choice(candidates)
    return text[:index] + text[index + 1:], True


def apply_repeat_letter(text: str, rng: Random) -> tuple[str, bool]:
    candidates = letter_candidates(text, protected_mask(text), min_word_len=4)
    if not candidates:
        return text, False
    index = rng.choice(candidates)
    return text[:index] + text[index] + text[index:], True


def apply_e_yo_mix(text: str, rng: Random) -> tuple[str, bool]:
    mask = protected_mask(text)
    candidates = [
        index
        for index, char in enumerate(text)
        if not mask[index] and char.lower() in {"е", "ё"}
    ]
    if not candidates:
        return text, False
    index = rng.choice(candidates)
    old = text[index]
    new = "ё" if old.lower() == "е" else "е"
    if old.isupper():
        new = new.upper()
    return text[:index] + new + text[index + 1:], True


def surrounding_words(text: str, space_index: int) -> tuple[str, str]:
    left = ""
    right = ""
    for start, end, word in word_spans(text):
        if end == space_index:
            left = word
        if start == space_index + 1:
            right = word
    return left, right


def apply_remove_space(text: str, rng: Random) -> tuple[str, bool]:
    mask = protected_mask(text)
    candidates: list[int] = []
    for index, char in enumerate(text):
        if char != " " or index == 0 or index >= len(text) - 1:
            continue
        if mask[index - 1] or mask[index + 1]:
            continue
        if not (text[index - 1].isalpha() and text[index + 1].isalpha()):
            continue
        left, right = surrounding_words(text, index)
        if len(left) < 4 or len(right) < 4:
            continue
        if left in PROTECTED_TIME_WORDS or right in PROTECTED_TIME_WORDS:
            continue
        candidates.append(index)
    if not candidates:
        return text, False
    index = rng.choice(candidates)
    return text[:index] + text[index + 1:], True


def apply_remove_punctuation(text: str, rng: Random) -> tuple[str, bool]:
    candidates = [match.start() for match in PUNCT_RE.finditer(text)]
    if not candidates:
        return text, False
    index = rng.choice(candidates)
    if text[index] in {"\"", "«", "»"}:
        updated = text.replace("\"", "").replace("«", "").replace("»", "")
        return updated, updated != text
    return text[:index] + text[index + 1:], True


def apply_case_noise(text: str, rng: Random) -> tuple[str, bool]:
    if rng.random() < 0.55 and text != text.lower():
        return text.lower(), True

    candidates = [
        (start, end)
        for start, end, word in word_spans(text)
        if len(word) >= 4 and word not in PROTECTED_TIME_WORDS
    ]
    if not candidates:
        return text, False
    start, end = rng.choice(candidates)
    word = text[start:end]
    if word.isupper():
        return text, False
    return text[:start] + word.upper() + text[end:], True


def apply_regex_replacement(
        text: str,
        rng: Random,
        replacements: list[tuple[re.Pattern[str], list[str]]],
) -> tuple[str, bool]:
    possible = [
        (pattern, variants)
        for pattern, variants in replacements
        if pattern.search(text)
    ]
    if not possible:
        return text, False
    pattern, variants = rng.choice(possible)
    replacement = rng.choice(variants)
    return pattern.sub(replacement, text, count=1), True


def apply_slang(text: str, rng: Random) -> tuple[str, bool]:
    return apply_regex_replacement(text, rng, SLANG_REPLACEMENTS)


def apply_asr_like(text: str, rng: Random) -> tuple[str, bool]:
    return apply_regex_replacement(text, rng, ASR_LIKE_REPLACEMENTS)


Operation = Callable[[str, Random], tuple[str, bool]]


def noisy_text(text: str, seed: int) -> str:
    if not text or not text.strip():
        return text

    rng = stable_rng(seed, text)
    target = rng.randint(2, 4) if len(text) >= 35 else rng.randint(1, 3)
    character_operations: list[Operation] = [
        apply_keyboard_neighbor,
        apply_keyboard_neighbor,
        apply_delete_letter,
        apply_repeat_letter,
        apply_e_yo_mix,
    ]
    surface_operations: list[Operation] = [
        apply_remove_space,
        apply_remove_punctuation,
        apply_case_noise,
        apply_slang,
        apply_asr_like,
    ]

    updated = text
    used_surface_noise = False
    applied = 0
    attempts = 0

    while applied < target and attempts < 48:
        attempts += 1
        if not used_surface_noise and rng.random() < 0.34:
            operation = rng.choice(surface_operations)
        else:
            operation = rng.choice(character_operations)

        candidate, ok = operation(updated, rng)
        if not ok or candidate == updated:
            continue

        if operation in surface_operations:
            used_surface_noise = True
        updated = candidate
        applied += 1

    if updated == text:
        updated, _ = apply_keyboard_neighbor(text, rng)
    return updated


def mutate_user_text_fields(data: dict[str, Any], seed: int) -> tuple[int, int, int, dict[str, str]]:
    mapping: dict[str, str] = {}
    step_text_changed = 0
    state_message_changed = 0
    conversation_changed = 0

    def broken(text: str) -> str:
        if text not in mapping:
            mapping[text] = noisy_text(text, seed)
        return mapping[text]

    for episode in data.get("samples", []):
        for step in episode.get("steps", []):
            request = step.get("step_request")
            if not isinstance(request, dict):
                continue

            text = request.get("text")
            if isinstance(text, str) and text:
                new_text = broken(text)
                request["text"] = new_text
                if new_text != text:
                    step_text_changed += 1

            state = request.get("state") if isinstance(request.get("state"), dict) else {}
            messages = state.get("messages") if isinstance(state.get("messages"), list) else []
            for message in messages:
                if not isinstance(message, dict) or message.get("role") != "user":
                    continue
                message_text = message.get("text")
                if isinstance(message_text, str) and message_text:
                    new_text = broken(message_text)
                    message["text"] = new_text
                    if new_text != message_text:
                        state_message_changed += 1

            conversation = request.get("conversation")
            if isinstance(conversation, dict):
                for key in ("last_user_message",):
                    value = conversation.get(key)
                    if isinstance(value, str) and value:
                        new_value = broken(value)
                        conversation[key] = new_value
                        if new_value != value:
                            conversation_changed += 1
                recent = conversation.get("recent_messages")
                if isinstance(recent, list):
                    for message in recent:
                        if not isinstance(message, dict) or message.get("role") != "user":
                            continue
                        value = message.get("text")
                        if isinstance(value, str) and value:
                            new_value = broken(value)
                            message["text"] = new_value
                            if new_value != value:
                                conversation_changed += 1

    return step_text_changed, state_message_changed, conversation_changed, mapping


def validate_unchanged_structure(source: dict[str, Any], result: dict[str, Any]) -> None:
    source_samples = source.get("samples", [])
    result_samples = result.get("samples", [])
    if len(source_samples) != len(result_samples):
        raise ValueError("Sample count changed")

    for source_episode, result_episode in zip(source_samples, result_samples):
        if source_episode.get("episode_id") != result_episode.get("episode_id"):
            raise ValueError("Episode id changed")
        source_steps = source_episode.get("steps", [])
        result_steps = result_episode.get("steps", [])
        if len(source_steps) != len(result_steps):
            raise ValueError(f"{source_episode.get('episode_id')}: step count changed")

        for source_step, result_step in zip(source_steps, result_steps):
            episode_id = source_episode.get("episode_id")
            step_id = source_step.get("step_id")
            if source_step.get("expected_step_response") != result_step.get("expected_step_response"):
                raise ValueError(f"{episode_id} step {step_id}: expected response changed")
            if source_step.get("evaluation") != result_step.get("evaluation"):
                raise ValueError(f"{episode_id} step {step_id}: evaluation changed")

            source_request = source_step.get("step_request", {})
            result_request = result_step.get("step_request", {})
            for key in ("context", "all_context"):
                if source_request.get(key) != result_request.get(key):
                    raise ValueError(f"{episode_id} step {step_id}: request.{key} changed")

            source_state = source_request.get("state", {})
            result_state = result_request.get("state", {})
            for key in ("completed_actions", "tool_observations", "working_state", "memory_summary"):
                if source_state.get(key) != result_state.get(key):
                    raise ValueError(f"{episode_id} step {step_id}: state.{key} changed")

            validate_user_text_surface(
                source_request.get("text", ""),
                result_request.get("text", ""),
                f"{episode_id} step {step_id}: step_request.text",
            )

            validate_messages(
                source_state.get("messages", []),
                result_state.get("messages", []),
                f"{episode_id} step {step_id}: state.messages",
            )

            validate_conversation(
                source_request.get("conversation"),
                result_request.get("conversation"),
                f"{episode_id} step {step_id}: conversation",
            )


def validate_user_text_surface(source: Any, result: Any, label: str) -> None:
    if not isinstance(source, str) or not isinstance(result, str):
        if source != result:
            raise ValueError(f"{label}: non-string value changed")
        return
    if digit_parts(source) != digit_parts(result):
        raise ValueError(f"{label}: numeric/time parts changed")
    if month_parts(source) != month_parts(result):
        raise ValueError(f"{label}: month words changed")


def validate_messages(source_messages: Any, result_messages: Any, label: str) -> None:
    if not isinstance(source_messages, list) or not isinstance(result_messages, list):
        if source_messages != result_messages:
            raise ValueError(f"{label}: message container changed")
        return
    if len(source_messages) != len(result_messages):
        raise ValueError(f"{label}: message count changed")
    for index, (source_message, result_message) in enumerate(zip(source_messages, result_messages)):
        if not isinstance(source_message, dict) or not isinstance(result_message, dict):
            if source_message != result_message:
                raise ValueError(f"{label}[{index}]: message changed")
            continue
        if source_message.get("role") != result_message.get("role"):
            raise ValueError(f"{label}[{index}]: role changed")
        if source_message.get("role") == "user":
            validate_user_text_surface(
                source_message.get("text", ""),
                result_message.get("text", ""),
                f"{label}[{index}].text",
            )
            source_rest = {key: value for key, value in source_message.items() if key != "text"}
            result_rest = {key: value for key, value in result_message.items() if key != "text"}
            if source_rest != result_rest:
                raise ValueError(f"{label}[{index}]: non-text fields changed")
        elif source_message != result_message:
            raise ValueError(f"{label}[{index}]: non-user message changed")


def validate_conversation(source: Any, result: Any, label: str) -> None:
    if source is None and result is None:
        return
    if not isinstance(source, dict) or not isinstance(result, dict):
        if source != result:
            raise ValueError(f"{label}: conversation changed")
        return

    source_copy = copy.deepcopy(source)
    result_copy = copy.deepcopy(result)
    for key in ("last_user_message",):
        if key in source_copy or key in result_copy:
            validate_user_text_surface(source_copy.get(key, ""), result_copy.get(key, ""), f"{label}.{key}")
            source_copy.pop(key, None)
            result_copy.pop(key, None)

    if "recent_messages" in source_copy or "recent_messages" in result_copy:
        validate_messages(
            source_copy.get("recent_messages", []),
            result_copy.get("recent_messages", []),
            f"{label}.recent_messages",
        )
        source_copy.pop("recent_messages", None)
        result_copy.pop("recent_messages", None)

    if source_copy != result_copy:
        raise ValueError(f"{label}: non-user conversation fields changed")


def build_dataset(source: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    result = copy.deepcopy(source)
    result["dataset_name"] = "calendar_step_agent_super_broken_test_new_mixed_noise"
    result["split"] = "super_broken_test_new"
    result["status"] = "surface_filled_ready_mixed_user_noise"
    result["noise_profile"] = {
        "source_file": "test_new.json",
        "type": "mixed_moderate_user_surface_noise",
        "seed": seed,
        "changed_fields": [
            "step_request.text",
            "step_request.state.messages[role=user].text",
            "step_request.conversation user text fields when present",
        ],
        "noise_types": [
            "keyboard-neighbor substitution",
            "rare missing space",
            "letter deletion",
            "letter repetition",
            "e/yo mixing",
            "punctuation removal",
            "lowercase/uppercase noise",
            "colloquial abbreviations",
            "small ASR-like reductions",
        ],
        "protected_content": [
            "digits",
            "clock times",
            "month names",
            "relative time words",
            "expected JSON",
            "tool observations",
            "completed actions",
        ],
    }
    step_changed, state_changed, conversation_changed, mapping = mutate_user_text_fields(result, seed)
    stats = {
        "episodes": len(result.get("samples", [])),
        "steps": sum(len(episode.get("steps", [])) for episode in result.get("samples", [])),
        "unique_user_requests": len(mapping),
        "step_request_text_changed": step_changed,
        "state_user_messages_changed": state_changed,
        "conversation_user_text_changed": conversation_changed,
    }
    validate_unchanged_structure(source, result)
    return result, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Create super_broken_test_new.json with deterministic user-text noise.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8-sig"))
    result, stats = build_dataset(source, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(args.output), **stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
