"""
 Модуль: retouch_ready_surface.py
 Назначение: Финальная ретушь уже заполненных scaffold-датасетов и нормализация русских формулировок
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
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


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def pick(rng: Random, variants: list[str]) -> str:
    return variants[rng.randrange(len(variants))]


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def format_day(dt: datetime) -> str:
    return f"{dt.day} {MONTHS_GEN[dt.month - 1]}"


def format_day_time(value: str) -> str:
    dt = parse_dt(value)
    return f"{format_day(dt)} в {dt:%H:%M}"


def format_range(start: str, end: str) -> str:
    return f"{parse_dt(start):%H:%M}-{parse_dt(end):%H:%M}"


def duration_phrase(minutes: int, rng: Random) -> str:
    variants = {
        30: ["на полчаса", "на 30 минут"],
        45: ["на 45 минут"],
        60: ["на час", "на 60 минут"],
        90: ["на полтора часа", "на 90 минут"],
        120: ["на два часа", "на 120 минут"],
    }
    return pick(rng, variants.get(minutes, [f"на {minutes} минут"]))


def normalize_strings(value: Any) -> Any:
    if isinstance(value, str):
        updated = value
        for old, new in ROLE_UNDER_REPLACEMENTS.items():
            updated = updated.replace(old, new)
        return updated
    if isinstance(value, list):
        return [normalize_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_strings(item) for key, item in value.items()}
    return value


def set_repeated_user_text(step: dict[str, Any], text: str) -> None:
    step["step_request"]["text"] = text


def set_single_user_message(step: dict[str, Any], text: str) -> None:
    step["step_request"]["state"]["messages"] = [{"role": "user", "text": text}]


def set_messages(step: dict[str, Any], messages: list[dict[str, str]]) -> None:
    step["step_request"]["state"]["messages"] = messages


def last_items(step: dict[str, Any], tool_name: str) -> list[dict[str, Any]]:
    for observation in reversed(step["step_request"]["state"].get("tool_observations", [])):
        if observation.get("tool_name") == tool_name:
            return observation["result"]["items"]
    raise ValueError(f"Missing {tool_name} observation")


def last_slots(step: dict[str, Any]) -> list[dict[str, Any]]:
    for observation in reversed(step["step_request"]["state"].get("tool_observations", [])):
        if observation.get("tool_name") == "get_free_slots":
            return observation["result"]["slots"]
    raise ValueError("Missing get_free_slots observation")


def last_event(step: dict[str, Any], tool_name: str) -> dict[str, Any]:
    for observation in reversed(step["step_request"]["state"].get("tool_observations", [])):
        if observation.get("tool_name") == tool_name:
            return observation["result"]["event"]
    raise ValueError(f"Missing {tool_name} observation")


def render_direct_create(step: dict[str, Any], rng: Random) -> str:
    args = step["expected_step_response"]["next_gateway_action"]["tool_call"]["arguments"]
    return pick(
        rng,
        [
            f'Добавь в календарь событие "{args["title"]}" на {format_day_time(args["start_time"])}, {duration_phrase(int(args["duration_minutes"]), rng)}.',
            f'Запланируй событие "{args["title"]}" на {format_day_time(args["start_time"])}, {duration_phrase(int(args["duration_minutes"]), rng)}.',
            f'Поставь в календарь "{args["title"]}" на {format_day_time(args["start_time"])}, {duration_phrase(int(args["duration_minutes"]), rng)}.',
        ],
    )


def render_find_update_request(steps: list[dict[str, Any]], rng: Random) -> str:
    query = steps[0]["expected_step_response"]["next_gateway_action"]["tool_call"]["arguments"]["query"]
    new_start = steps[1]["expected_step_response"]["next_gateway_action"]["tool_call"]["arguments"]["updates"][
        "start_time"]
    return pick(
        rng,
        [
            f'Перенеси событие "{query}" на {format_day_time(new_start)}.',
            f'Сдвинь событие "{query}" на {format_day_time(new_start)}.',
            f'Поставь событие "{query}" на {format_day_time(new_start)}.',
        ],
    )


def render_find_delete_request(steps: list[dict[str, Any]], rng: Random) -> str:
    query = steps[0]["expected_step_response"]["next_gateway_action"]["tool_call"]["arguments"]["query"]
    return pick(
        rng,
        [
            f'Удали событие "{query}".',
            f'Убери из календаря событие "{query}".',
            f'Снеси событие "{query}" из расписания.',
        ],
    )


def render_list_request(step: dict[str, Any], rng: Random) -> str:
    args = step["expected_step_response"]["next_gateway_action"]["tool_call"]["arguments"]
    day = format_day(parse_dt(args["start"]))
    return pick(
        rng,
        [
            f"Что у меня стоит на {day}?",
            f"Покажи расписание на {day}.",
            f"Какие события у меня на {day}?",
        ],
    )


def render_slots_request(step: dict[str, Any], rng: Random) -> str:
    args = step["expected_step_response"]["next_gateway_action"]["tool_call"]["arguments"]
    day = format_day(parse_dt(args["start"]))
    duration = duration_phrase(int(args["min_duration_minutes"]), rng)
    return pick(
        rng,
        [
            f"Найди на {day} свободное окно {duration}.",
            f"Подбери на {day} слот {duration}.",
            f"Есть ли на {day} окно {duration}?",
        ],
    )


def render_slots_then_create_request(steps: list[dict[str, Any]], rng: Random) -> str:
    args = steps[0]["expected_step_response"]["next_gateway_action"]["tool_call"]["arguments"]
    create_args = steps[1]["expected_step_response"]["next_gateway_action"]["tool_call"]["arguments"]
    day = format_day(parse_dt(args["start"]))
    duration = duration_phrase(int(args["min_duration_minutes"]), rng)
    return pick(
        rng,
        [
            f'Найди на {day} окно {duration} и сразу поставь туда событие "{create_args["title"]}".',
            f'Подбери на {day} свободный слот {duration} и запланируй туда событие "{create_args["title"]}".',
            f'Если на {day} найдется окно {duration}, поставь туда событие "{create_args["title"]}".',
        ],
    )


def retouch_episode(episode: dict[str, Any]) -> None:
    """Обновляет пользовательские и ассистентские реплики одного эпизода по его scaffold-family."""
    family = episode.get("scaffold_meta", {}).get("family")
    rng = Random(stable_seed(episode["episode_id"]))
    steps = episode["steps"]

    if family in {"ordinary_direct_create", "hard_direct_create_searchy"}:
        steps[0]["step_request"]["text"] = render_direct_create(steps[0], rng)
        return

    if family == "ordinary_find_update_finish":
        request_text = render_find_update_request(steps, rng)
        for step in steps:
            set_repeated_user_text(step, request_text)
        set_single_user_message(steps[1], request_text)
        set_single_user_message(steps[2], request_text)
        event = last_event(steps[2], "update_event")
        steps[2]["expected_step_response"]["assistant_message"] = pick(
            rng,
            [
                f'Готово, перенес событие "{event["summary"]}" на {format_day(parse_dt(event["start"]))}, {format_range(event["start"], event["end"])}.',
                f'Сделано, событие "{event["summary"]}" теперь стоит на {format_day(parse_dt(event["start"]))}, {format_range(event["start"], event["end"])}.',
            ],
        )
        return

    if family == "ordinary_find_delete_finish":
        request_text = render_find_delete_request(steps, rng)
        for step in steps:
            set_repeated_user_text(step, request_text)
        set_single_user_message(steps[1], request_text)
        set_single_user_message(steps[2], request_text)
        item = last_items(steps[1], "find_event")[0]
        steps[2]["expected_step_response"]["assistant_message"] = pick(
            rng,
            [
                f'Готово, удалил событие "{item["summary"]}".',
                f'Сделано, события "{item["summary"]}" больше нет в календаре.',
            ],
        )
        return

    if family in {"ordinary_list_then_finish", "hard_list_then_finish"}:
        request_text = render_list_request(steps[0], rng)
        for step in steps:
            set_repeated_user_text(step, request_text)
        set_single_user_message(steps[1], request_text)
        items = sorted(last_items(steps[1], "list_events"), key=lambda item: item["start"])
        preview = "; ".join(f'{parse_dt(item["start"]):%H:%M} — {item["summary"]}' for item in items[:3])
        if len(items) > 3:
            preview += f"; и еще {len(items) - 3}"
        steps[1]["expected_step_response"][
            "assistant_message"] = f'На {format_day(parse_dt(items[0]["start"]))} у тебя {len(items)} события: {preview}.'
        return

    if family in {"ordinary_slots_then_finish", "hard_slots_then_finish"}:
        request_text = render_slots_request(steps[0], rng)
        for step in steps:
            set_repeated_user_text(step, request_text)
        set_single_user_message(steps[1], request_text)
        slots = last_slots(steps[1])
        joined = ", ".join(format_range(slot["start"], slot["end"]) for slot in slots)
        steps[1]["expected_step_response"]["assistant_message"] = pick(
            rng,
            [
                f"Есть свободные окна: {joined}.",
                f"Свободные слоты такие: {joined}.",
                f"Нашел свободные окна: {joined}.",
            ],
        )
        return

    if family == "hard_ambiguous_find_clarify_update_finish":
        request_text = render_find_update_request([steps[0], steps[2]], rng)
        followup_time = steps[2]["expected_step_response"]["next_gateway_action"]["tool_call"]["arguments"]["updates"][
            "start_time"]
        followup_text = pick(
            rng,
            [
                f"Не первое, а второе. Перенеси на {format_day_time(followup_time)}.",
                f"Нужно второе событие, поставь его на {format_day_time(followup_time)}.",
                f"Второй вариант, и время пусть будет {format_day_time(followup_time)}.",
            ],
        )
        steps[0]["step_request"]["text"] = request_text
        steps[1]["step_request"]["text"] = request_text
        set_single_user_message(steps[1], request_text)
        items = sorted(last_items(steps[1], "find_event"), key=lambda item: item["start"])
        clarify_text = (
            f'Нашел два похожих события: {parse_dt(items[0]["start"]):%H:%M} — {items[0]["summary"]}; '
            f'{parse_dt(items[1]["start"]):%H:%M} — {items[1]["summary"]}. Какое из них нужно изменить?'
        )
        steps[1]["expected_step_response"]["assistant_message"] = clarify_text
        steps[2]["step_request"]["text"] = followup_text
        set_messages(
            steps[2],
            [
                {"role": "user", "text": request_text},
                {"role": "assistant", "text": clarify_text},
            ],
        )
        steps[3]["step_request"]["text"] = followup_text
        set_messages(
            steps[3],
            [
                {"role": "user", "text": request_text},
                {"role": "assistant", "text": clarify_text},
                {"role": "user", "text": followup_text},
            ],
        )
        event = last_event(steps[3], "update_event")
        steps[3]["expected_step_response"]["assistant_message"] = (
            f'Готово, второе событие перенес на {format_day(parse_dt(event["start"]))}, '
            f'{format_range(event["start"], event["end"])}: "{event["summary"]}".'
        )
        return

    if family == "hard_mixed_second_slot_update":
        request_text = render_slots_then_create_request(steps, rng)
        followup_text = pick(
            rng,
            [
                "Не первый слот, а второй, пожалуйста.",
                "Давай лучше второй вариант по времени.",
                "Перенеси тогда во второе окно.",
                "Выбери, пожалуйста, не первый, а второй слот.",
            ],
        )
        steps[0]["step_request"]["text"] = request_text
        steps[1]["step_request"]["text"] = request_text
        set_single_user_message(steps[1], request_text)
        slots = last_slots(steps[1])
        gateway_text = (
            f"Нашел два окна: {format_range(slots[0]['start'], slots[0]['end'])} и {format_range(slots[1]['start'], slots[1]['end'])}. "
            "Я уже поставил встречу в первый слот, но могу перенести ее во второй."
        )
        steps[2]["step_request"]["text"] = followup_text
        set_messages(
            steps[2],
            [
                {"role": "user", "text": request_text},
                {"role": "assistant", "text": gateway_text},
            ],
        )
        steps[3]["step_request"]["text"] = followup_text
        set_messages(
            steps[3],
            [
                {"role": "user", "text": request_text},
                {"role": "assistant", "text": gateway_text},
                {"role": "user", "text": followup_text},
            ],
        )
        event = last_event(steps[3], "update_event")
        steps[3]["expected_step_response"]["assistant_message"] = pick(
            rng,
            [
                f"Готово, перенес встречу во второй слот: {format_day(parse_dt(event['start']))}, {format_range(event['start'], event['end'])}.",
                f"Сделано, теперь событие стоит во втором окне: {format_day(parse_dt(event['start']))}, {format_range(event['start'], event['end'])}.",
            ],
        )
        return


def main() -> None:
    """Применяет ретушь ко всем переданным JSON-датасетам на месте."""
    parser = argparse.ArgumentParser(
        description="Retouch already filled scaffold datasets with UTF-8-safe surface text.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.paths:
        target = path.resolve()
        raw = json.loads(target.read_text(encoding="utf-8-sig"))
        raw = normalize_strings(raw)
        for episode in raw.get("samples", []):
            retouch_episode(episode)
        target.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        print(target)


if __name__ == "__main__":
    main()
