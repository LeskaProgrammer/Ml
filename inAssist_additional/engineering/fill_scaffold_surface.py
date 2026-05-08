"""
 Модуль: fill_scaffold_surface.py
 Назначение: Заполнение [[FILL::...]] плейсхолдеров в scaffold-датасетах живыми русскими формулировками
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_PATH = ROOT / "engineering" / "quality" / "data" / "test_new.json"
PLACEHOLDER_RE = re.compile(r"\[\[FILL::[^\]]+\]\]")
MONTHS_GEN = [
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
]
ROLE_UNDER_REPLACEMENTS = {
    "у редактором": "у редактора",
    "у куратором": "у куратора",
    "у дизайнером": "у дизайнера",
    "у адвокатом": "у адвоката",
    "у репетитором": "у репетитора",
    "у бухгалтером": "у бухгалтера",
    "у подрядчиком": "у подрядчика",
    "у иллюстратором": "у иллюстратора",
    "у маркетологом": "у маркетолога",
    "у арендодателем": "у арендодателя",
    "у архитектором": "у архитектора",
    "у стоматологом": "у стоматолога",
    "у поставщиком": "у поставщика",
    "у юристом": "у юриста",
    "у аналитиком": "у аналитика",
    "у логистом": "у логиста",
}


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def pick(rng: Random, variants: list[str]) -> str:
    return variants[rng.randrange(len(variants))]


def format_day(dt: datetime) -> str:
    return f"{dt.day} {MONTHS_GEN[dt.month - 1]}"


def format_day_time(dt: datetime) -> str:
    return f"{format_day(dt)} в {dt:%H:%M}"


def format_time_range(start: datetime, end: datetime) -> str:
    return f"{start:%H:%M}-{end:%H:%M}"


def duration_phrase(minutes: int, rng: Random) -> str:
    variants = {
        30: ["на полчаса", "на 30 минут"],
        45: ["на 45 минут"],
        60: ["на час", "на 60 минут"],
        90: ["на полтора часа", "на 90 минут"],
        120: ["на два часа", "на 120 минут"],
    }
    return pick(rng, variants.get(minutes, [f"на {minutes} минут"]))


def vague_hint_phrase(raw: str) -> str:
    normalized = (raw or "").strip().lower()
    if "lunch" in normalized:
        return "после обеда"
    return raw.strip()


def sort_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(items, key=lambda item: item.get("start", ""))


def replace_live_fields(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [replace_live_fields(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: replace_live_fields(item, mapping) for key, item in value.items()}
    return value


def normalize_string_artifacts(value: Any) -> Any:
    if isinstance(value, str):
        updated = value
        for old, new in ROLE_UNDER_REPLACEMENTS.items():
            updated = updated.replace(old, new)
        return updated
    if isinstance(value, list):
        return [normalize_string_artifacts(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_string_artifacts(item) for key, item in value.items()}
    return value


@dataclass
class EpisodeFiller:
    """Заполняет плейсхолдеры одного эпизода детерминированными вариантами текста."""

    episode: dict[str, Any]

    def __post_init__(self) -> None:
        self.rng = Random(stable_seed(self.episode["episode_id"]))
        self.mapping: dict[str, str] = {}

    def fill(self) -> None:
        for step in self.episode["steps"]:
            fill_instructions = step.get("fill_instructions", {})
            for key in ("user_text", "assistant_message", "assistant_gateway_message"):
                payload = fill_instructions.get(key)
                if not isinstance(payload, dict):
                    continue
                slot = payload.get("slot")
                if not slot or slot in self.mapping:
                    continue
                if key == "user_text":
                    self.mapping[slot] = self.render_user_text(step, payload)
                else:
                    self.mapping[slot] = self.render_assistant_text(step, payload)

        for step in self.episode["steps"]:
            step["step_request"]["text"] = replace_live_fields(step["step_request"]["text"], self.mapping)
            step["step_request"]["state"]["messages"] = replace_live_fields(
                step["step_request"]["state"].get("messages", []),
                self.mapping,
            )
            step["expected_step_response"]["assistant_message"] = replace_live_fields(
                step["expected_step_response"].get("assistant_message"),
                self.mapping,
            )

    def render_user_text(self, step: dict[str, Any], payload: dict[str, Any]) -> str:
        mode = payload["mode"]
        if mode == "direct_create":
            return self.render_direct_create(payload)
        if mode == "find_then_update":
            return self.render_find_then_update(payload)
        if mode == "internal_repeat":
            return self.fallback_previous_user_text()
        if mode == "list_events":
            return self.render_list_events(step)
        if mode == "get_slots":
            return self.render_get_slots(step, payload)
        if mode == "ambiguous_find_for_update":
            return self.render_ambiguous_update(payload)
        if mode == "choose_second_candidate":
            return self.render_choose_second(payload)
        if mode == "clarify_vague_time":
            return self.render_vague_create(payload)
        if mode == "find_then_delete":
            return self.render_find_then_delete(payload)
        if mode == "slot_then_create":
            return self.render_slot_then_create(step, payload)
        if mode == "switch_to_second_slot":
            return pick(
                self.rng,
                [
                    "Не первый слот, а второй, пожалуйста.",
                    "Давай лучше второй вариант по времени.",
                    "Перенеси тогда во второе окно.",
                    "Выбери, пожалуйста, не первый, а второй слот.",
                ],
            )
        raise ValueError(f"Unsupported user_text mode: {mode}")

    def render_assistant_text(self, step: dict[str, Any], payload: dict[str, Any]) -> str:
        mode = payload["mode"]
        if mode == "finish_after_update":
            return self.render_finish_after_update(step, mention_second=False)
        if mode == "finish_after_ambiguous_update":
            return self.render_finish_after_update(step, mention_second=True)
        if mode == "finish_after_list":
            return self.render_finish_after_list(step, hard=bool(payload.get("hard")))
        if mode == "finish_after_slots":
            return self.render_finish_after_slots(step, hard=bool(payload.get("hard")))
        if mode == "clarify_missing_time":
            return self.render_clarify_missing_time(step)
        if mode == "clarify_two_candidates":
            return self.render_clarify_two_candidates(step)
        if mode == "finish_after_delete":
            return self.render_finish_after_delete(step)
        if mode == "gateway_offers_two_slots":
            return self.render_gateway_offers_two_slots(step)
        if mode == "finish_after_slot_switch":
            return self.render_finish_after_slot_switch(step)
        raise ValueError(f"Unsupported assistant mode: {mode}")

    def render_direct_create(self, payload: dict[str, Any]) -> str:
        dt = parse_dt(payload["start_time"])
        title = payload["title"]
        duration = duration_phrase(int(payload["duration_minutes"]), self.rng)
        common = [
            f"Добавь в календарь событие \"{title}\" на {format_day_time(dt)}, {duration}.",
            f"Запланируй событие \"{title}\" на {format_day_time(dt)}, {duration}.",
            f"Поставь в календарь \"{title}\" на {format_day_time(dt)}, {duration}.",
        ]
        if payload.get("hard"):
            common.append(f"Зафиксируй, пожалуйста, событие \"{title}\" на {format_day_time(dt)}, {duration}.")
        return pick(self.rng, common)

    def render_find_then_update(self, payload: dict[str, Any]) -> str:
        dt = parse_dt(payload["new_start_time"])
        query = payload["query"]
        return pick(
            self.rng,
            [
                f"Перенеси {query} на {format_day_time(dt)}.",
                f"Сдвинь {query} на {format_day_time(dt)}.",
                f"Поставь {query} на {format_day_time(dt)}.",
            ],
        )

    def render_ambiguous_update(self, payload: dict[str, Any]) -> str:
        dt = parse_dt(payload["new_start_time"])
        query = payload["query"]
        return pick(
            self.rng,
            [
                f"Перенеси {query} на {format_day_time(dt)}.",
                f"Нужно передвинуть {query} на {format_day_time(dt)}.",
                f"Смести {query} на {format_day_time(dt)}.",
            ],
        )

    def render_choose_second(self, payload: dict[str, Any]) -> str:
        dt = parse_dt(payload["new_start_time"])
        return pick(
            self.rng,
            [
                f"Не первое, а второе. Перенеси на {format_day_time(dt)}.",
                f"Нужно второе событие, поставь его на {format_day_time(dt)}.",
                f"Второй вариант, и время пусть будет {format_day_time(dt)}.",
            ],
        )

    def render_vague_create(self, payload: dict[str, Any]) -> str:
        title = payload["title"]
        hint = vague_hint_phrase(payload["vague_time_hint"])
        return pick(
            self.rng,
            [
                f"Запланируй {title} {hint}.",
                f"Нужно поставить {title} {hint}.",
                f"Добавь {title}, как будет окно {hint}.",
            ],
        )

    def render_find_then_delete(self, payload: dict[str, Any]) -> str:
        query = payload["query"]
        return pick(
            self.rng,
            [
                f"Удали {query}.",
                f"Убери из календаря {query}.",
                f"Снеси, пожалуйста, {query} из расписания.",
            ],
        )

    def render_list_events(self, step: dict[str, Any]) -> str:
        args = step["expected_step_response"]["next_gateway_action"]["tool_call"]["arguments"]
        start = parse_dt(args["start"])
        day = format_day(start)
        return pick(
            self.rng,
            [
                f"Что у меня стоит на {day}?",
                f"Покажи расписание на {day}.",
                f"Какие события у меня на {day}?",
            ],
        )

    def render_get_slots(self, step: dict[str, Any], payload: dict[str, Any]) -> str:
        args = step["expected_step_response"]["next_gateway_action"]["tool_call"]["arguments"]
        start = parse_dt(args["start"])
        day = format_day(start)
        minutes = int(payload["min_duration_minutes"])
        duration = duration_phrase(minutes, self.rng)
        if payload.get("hard"):
            return pick(
                self.rng,
                [
                    f"Посмотри, куда на {day} можно вставить окно {duration}.",
                    f"Найди на {day} свободный слот хотя бы {duration}.",
                    f"Проверь, есть ли на {day} окно {duration}.",
                ],
            )
        return pick(
            self.rng,
            [
                f"Найди на {day} свободное окно {duration}.",
                f"Подбери на {day} слот {duration}.",
                f"Есть ли на {day} окно {duration}?",
            ],
        )

    def render_slot_then_create(self, step: dict[str, Any], payload: dict[str, Any]) -> str:
        args = step["expected_step_response"]["next_gateway_action"]["tool_call"]["arguments"]
        start = parse_dt(args["start"])
        day = format_day(start)
        duration = duration_phrase(int(payload["min_duration_minutes"]), self.rng)
        title = payload["title"]
        return pick(
            self.rng,
            [
                f"Найди на {day} окно {duration} и сразу поставь туда {title}.",
                f"Подбери на {day} свободный слот {duration} и запланируй туда {title}.",
                f"Если на {day} найдется окно {duration}, поставь туда {title}.",
            ],
        )

    def render_finish_after_update(self, step: dict[str, Any], mention_second: bool) -> str:
        update_event = self.last_event_observation(step, "update_event")
        summary = update_event["summary"]
        start = parse_dt(update_event["start"])
        end = parse_dt(update_event["end"])
        prefix = "Готово, второе событие перенес" if mention_second else pick(
            self.rng,
            ["Готово, перенес", "Сделано, перенес", "Готово, сдвинул"],
        )
        return f"{prefix} {summary} на {format_day(start)}, {format_time_range(start, end)}."

    def render_finish_after_list(self, step: dict[str, Any], hard: bool) -> str:
        items = sort_items(self.last_items_observation(step, "list_events"))
        day = format_day(parse_dt(items[0]["start"]))
        preview = "; ".join(f"{parse_dt(item['start']):%H:%M} — {item['summary']}" for item in items[:3])
        if len(items) > 3:
            preview += f"; и еще {len(items) - 3}"
        if hard:
            return f"На {day} у тебя {len(items)} события: {preview}."
        return pick(
            self.rng,
            [
                f"На {day} у тебя {len(items)} события: {preview}.",
                f"На {day} в календаре вот что стоит: {preview}.",
            ],
        )

    def render_finish_after_slots(self, step: dict[str, Any], hard: bool) -> str:
        slots = self.last_slots_observation(step)
        parts = [format_time_range(parse_dt(slot["start"]), parse_dt(slot["end"])) for slot in slots]
        joined = ", ".join(parts)
        if hard:
            return f"Нашел свободные окна: {joined}."
        return pick(
            self.rng,
            [
                f"Есть свободные окна: {joined}.",
                f"Свободные слоты такие: {joined}.",
            ],
        )

    def render_clarify_missing_time(self, step: dict[str, Any]) -> str:
        title = None
        user_payload = step.get("fill_instructions", {}).get("user_text")
        if isinstance(user_payload, dict):
            title = user_payload.get("title")
        if title:
            return pick(
                self.rng,
                [
                    f"Уточни, пожалуйста, во сколько поставить {title}.",
                    f"Нужно точнее время для события {title}. Во сколько его поставить?",
                    f"Подскажи точное время для {title}, чтобы я его запланировал.",
                ],
            )
        return "Уточни, пожалуйста, точное время."

    def render_clarify_two_candidates(self, step: dict[str, Any]) -> str:
        items = sort_items(self.last_items_observation(step, "find_event"))
        left = items[0]
        right = items[1]
        return (
            f"Нашел два похожих события: {parse_dt(left['start']):%H:%M} — {left['summary']}; "
            f"{parse_dt(right['start']):%H:%M} — {right['summary']}. Какое из них нужно изменить?"
        )

    def render_finish_after_delete(self, step: dict[str, Any]) -> str:
        items = self.last_items_observation(step, "find_event")
        summary = items[0]["summary"] if items else "событие"
        return pick(
            self.rng,
            [
                f"Готово, удалил {summary}.",
                f"Сделано, {summary} больше нет в календаре.",
                f"Удалил {summary}.",
            ],
        )

    def render_gateway_offers_two_slots(self, step: dict[str, Any]) -> str:
        slots = self.last_slots_observation(step)
        first = format_time_range(parse_dt(slots[0]["start"]), parse_dt(slots[0]["end"]))
        second = format_time_range(parse_dt(slots[1]["start"]), parse_dt(slots[1]["end"]))
        return (
            f"Нашел два окна: {first} и {second}. "
            f"Я уже поставил встречу в первый слот, но могу перенести ее во второй."
        )

    def render_finish_after_slot_switch(self, step: dict[str, Any]) -> str:
        update_event = self.last_event_observation(step, "update_event")
        start = parse_dt(update_event["start"])
        end = parse_dt(update_event["end"])
        return pick(
            self.rng,
            [
                f"Готово, перенес встречу во второй слот: {format_day(start)}, {format_time_range(start, end)}.",
                f"Сделано, теперь событие стоит во втором окне: {format_day(start)}, {format_time_range(start, end)}.",
            ],
        )

    def fallback_previous_user_text(self) -> str:
        for value in self.mapping.values():
            return value
        raise ValueError(f"Could not resolve internal_repeat in {self.episode['episode_id']}")

    def last_items_observation(self, step: dict[str, Any], tool_name: str) -> list[dict[str, Any]]:
        for observation in reversed(step["step_request"]["state"].get("tool_observations", [])):
            if observation.get("tool_name") == tool_name:
                return observation["result"]["items"]
        raise ValueError(f"Missing {tool_name} items observation in {self.episode['episode_id']}")

    def last_slots_observation(self, step: dict[str, Any]) -> list[dict[str, Any]]:
        for observation in reversed(step["step_request"]["state"].get("tool_observations", [])):
            if observation.get("tool_name") == "get_free_slots":
                return observation["result"]["slots"]
        raise ValueError(f"Missing slots observation in {self.episode['episode_id']}")

    def last_event_observation(self, step: dict[str, Any], tool_name: str) -> dict[str, Any]:
        for observation in reversed(step["step_request"]["state"].get("tool_observations", [])):
            if observation.get("tool_name") == tool_name:
                return observation["result"]["event"]
        raise ValueError(f"Missing {tool_name} event observation in {self.episode['episode_id']}")


def validate_filled_dataset(raw: dict[str, Any]) -> None:
    """Проверяет, что после заполнения не осталось плейсхолдеров и пустых реплик."""
    errors: list[str] = []
    for episode in raw.get("samples", []):
        for step in episode.get("steps", []):
            live_fields = [
                ("step_request.text", step["step_request"].get("text")),
                ("expected_step_response.assistant_message", step["expected_step_response"].get("assistant_message")),
            ]
            for message in step["step_request"]["state"].get("messages", []):
                live_fields.append(("state.messages.text", message.get("text")))
            for label, value in live_fields:
                if isinstance(value, str) and PLACEHOLDER_RE.search(value):
                    errors.append(f"{episode['episode_id']} step {step['step_id']} still has placeholder in {label}")
                if label != "expected_step_response.assistant_message" and (value is None or value == ""):
                    errors.append(f"{episode['episode_id']} step {step['step_id']} has empty {label}")

            response = step["expected_step_response"]
            response_type = response["response_type"]
            gateway = response["next_gateway_action"]
            if response_type == "tool_call" and gateway["type"] != "tool_call":
                errors.append(f"{episode['episode_id']} step {step['step_id']} has mismatched tool_call gateway action")
            if response_type in {"finish", "clarify"}:
                if gateway["type"] != "none" or gateway["tool_call"] is not None:
                    errors.append(
                        f"{episode['episode_id']} step {step['step_id']} terminal action still has gateway tool call")
                if not isinstance(response.get("assistant_message"), str) or not response["assistant_message"].strip():
                    errors.append(
                        f"{episode['episode_id']} step {step['step_id']} terminal action has empty assistant message")

    if errors:
        preview = "\n".join(errors[:20])
        raise ValueError(f"Filled dataset validation failed with {len(errors)} issue(s):\n{preview}")


def main() -> None:
    """Заполняет выбранный датасет и пишет результат рядом или поверх исходного файла."""
    parser = argparse.ArgumentParser(description="Fill scaffold placeholder text in a generated dataset.")
    parser.add_argument("--path", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite the input file instead of writing a sibling *_filled file.")
    args = parser.parse_args()

    path = args.path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or "samples" not in raw:
        raise ValueError("Expected top-level object with 'samples'.")

    for episode in raw["samples"]:
        EpisodeFiller(episode).fill()
        for step in episode.get("steps", []):
            step.pop("fill_instructions", None)

    raw = normalize_string_artifacts(raw)
    raw["scaffold_only"] = False
    raw["status"] = "surface_filled_ready"

    validate_filled_dataset(raw)

    output_path = path if args.in_place else path.with_name(f"{path.stem}_filled{path.suffix}")
    output_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(output_path))
    print(f"episodes={len(raw['samples'])}")
    print(f"steps={sum(len(ep.get('steps', [])) for ep in raw['samples'])}")


if __name__ == "__main__":
    main()
