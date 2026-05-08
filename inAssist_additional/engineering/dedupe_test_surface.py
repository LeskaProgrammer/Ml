"""
 Модуль: dedupe_test_surface.py
 Назначение: Ручная дедупликация поверхностных текстов test_new.json относительно train_new.json
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "engineering" / "training" / "data" / "train_new.json"
TEST_PATH = ROOT / "engineering" / "quality" / "data" / "test_new.json"


def set_user_text(ep: dict, step_ids: list[int], text: str) -> None:
    for step in ep["steps"]:
        if step["step_id"] not in step_ids:
            continue
        step["step_request"]["text"] = text
        for message in step["step_request"]["state"].get("messages", []):
            if message.get("role") == "user":
                message["text"] = text


def set_assistant_text(ep: dict, step_id: int, text: str) -> None:
    target = next(step for step in ep["steps"] if step["step_id"] == step_id)
    target["expected_step_response"]["assistant_message"] = text
    for later in ep["steps"]:
        if later["step_id"] <= step_id:
            continue
        for message in later["step_request"]["state"].get("messages", []):
            if message.get("role") == "assistant":
                message["text"] = text


def canon(value: str) -> str:
    return " ".join(value.strip().lower().split())


def main() -> None:
    """Правит известные пересечения train/test и проверяет, что дубликатов не осталось."""
    train = json.loads(TRAIN_PATH.read_text(encoding="utf-8-sig"))
    test = json.loads(TEST_PATH.read_text(encoding="utf-8-sig"))
    by_id = {ep["episode_id"]: ep for ep in test["samples"]}

    set_user_text(by_id["test_new_ep_0042"], [1, 2], "Посмотри, что у меня в календаре на 25 декабря.")

    for episode_id in ["test_new_ep_0061", "test_new_ep_0066", "test_new_ep_0087", "test_new_ep_0089"]:
        set_user_text(by_id[episode_id], [3, 4], "Возьми, пожалуйста, второй слот.")

    set_user_text(by_id["test_new_ep_0075"], [3, 4], "Тогда выбери второе окно по времени.")
    set_user_text(by_id["test_new_ep_0104"], [3, 4], "Бери второе событие и поставь его на 4 января в 16:30.")

    set_assistant_text(by_id["test_new_ep_0023"], 2, "Вижу такие свободные интервалы: 10:15-11:00 и 12:15-13:00.")
    set_assistant_text(by_id["test_new_ep_0037"], 2, "Могу предложить два слота: 09:30-10:00 и 11:30-12:00.")
    set_assistant_text(by_id["test_new_ep_0041"], 2, "Подходящие свободные окна: 10:00-10:45 и 12:00-12:45.")
    set_assistant_text(by_id["test_new_ep_0047"], 2, "На этот день свободны окна 11:00-12:30 и 13:00-14:30.")
    set_assistant_text(by_id["test_new_ep_0050"], 2, "Нашёл два удобных интервала: 11:30-12:15 и 13:30-14:15.")
    set_assistant_text(by_id["test_new_ep_0052"], 2, "Свободные промежутки такие: 10:00-10:45 и 12:00-12:45.")
    set_assistant_text(by_id["test_new_ep_0066"], 4, "Готово, переставил встречу во второе окно: 20 июня, 11:00-12:00.")
    set_assistant_text(by_id["test_new_ep_0080"], 2, "Есть два подходящих слота: 09:15-10:45 и 11:15-12:45.")
    set_assistant_text(by_id["test_new_ep_0101"], 2, "Свободные окна на этот день: 09:00-10:30 и 11:00-12:30.")

    train_texts = {
        canon(step["step_request"]["text"])
        for episode in train["samples"]
        for step in episode["steps"]
    }
    test_texts = {
        canon(step["step_request"]["text"])
        for episode in test["samples"]
        for step in episode["steps"]
    }
    train_msgs = {
        canon(step["expected_step_response"]["assistant_message"])
        for episode in train["samples"]
        for step in episode["steps"]
        if isinstance(step["expected_step_response"].get("assistant_message"), str)
    }
    test_msgs = {
        canon(step["expected_step_response"]["assistant_message"])
        for episode in test["samples"]
        for step in episode["steps"]
        if isinstance(step["expected_step_response"].get("assistant_message"), str)
    }

    overlap_texts = sorted(train_texts & test_texts)
    overlap_msgs = sorted(train_msgs & test_msgs)
    if overlap_texts or overlap_msgs:
        raise ValueError(
            f"Still have overlaps after dedupe: texts={len(overlap_texts)} msgs={len(overlap_msgs)}"
        )

    TEST_PATH.write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")
    print("deduped")


if __name__ == "__main__":
    main()
