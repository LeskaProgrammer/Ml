"""
 Модуль: generate_scaffold_expansion.py
 Назначение: Генерация расширенных train/test scaffold-датасетов для step-agent сценариев календаря
 Разработчик: Симонов Алексей Дмитриевич
 Дата: 2026-01-31
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

MSK = timezone(timedelta(hours=3))
WORK_START_HOUR = 9
WORK_END_HOUR = 18
FIND_EVENT_MAX_RESULTS = 10
LIST_EVENTS_MAX_RESULTS = 20
VALID_RESPONSE_TYPES = {"tool_call", "clarify", "finish"}
VALID_TOOLS = {"find_event", "list_events", "get_free_slots", "create_event", "update_event", "delete_event"}

ROOT = Path(__file__).resolve().parents[1]
TRAIN_SOURCE = ROOT / "engineering" / "training" / "data" / "train.json"
TEST_SOURCE = ROOT / "engineering" / "quality" / "data" / "test.json"
TRAIN_TARGET = ROOT / "engineering" / "training" / "data" / "train_new.json"
TEST_TARGET = ROOT / "engineering" / "quality" / "data" / "test_new.json"

ROLE_POOLS = {
    "train": ["редактором", "куратором", "дизайнером", "адвокатом", "репетитором", "бухгалтером", "подрядчиком",
              "иллюстратором"],
    "test": ["маркетологом", "арендодателем", "архитектором", "стоматологом", "поставщиком", "юристом", "аналитиком",
             "логистом"],
}
TOPIC_POOLS = {
    "train": ["подкасту", "смете", "документам", "презентации", "каталогам", "обложке", "практике", "математике"],
    "test": ["тендеру", "сценарию эфира", "страховке", "образцам", "сделке", "мебели", "кампании", "графику поставок"],
}
TITLE_PATTERNS = [
    "планёрка с {role} по {topic}",
    "созвон с {role} по {topic}",
    "бриф с {role} по {topic}",
    "сверка с {role} по {topic}",
    "проверка документов у {role} по {topic}",
    "урок с {role} по {topic}",
]
VAGUE_TIMES = ["после обеда", "ближе к вечеру", "во второй половине дня", "как будет окно после lunch"]


def load_samples(path: Path) -> List[Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    return raw["samples"] if isinstance(raw, dict) else raw


def count_steps(samples: Sequence[Dict[str, Any]]) -> int:
    return sum(len(sample.get("steps", [])) for sample in samples)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def ctx(now: datetime) -> Dict[str, Any]:
    return {
        "current_time": iso(now),
        "timezone": "Europe/Moscow",
        "work_start_hour": WORK_START_HOUR,
        "work_end_hour": WORK_END_HOUR,
    }


def ph(ep: str, slot: str) -> str:
    return f"[[FILL::{ep}::{slot}]]"


def empty_state() -> Dict[str, Any]:
    return {"messages": [], "completed_actions": [], "tool_observations": [], "working_state": {}, "memory_summary": ""}


def act(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {"tool_name": tool_name, "arguments": arguments}


def msg(role: str, text: str) -> Dict[str, str]:
    return {"role": role, "text": text}


def mk_request(text: str, now: datetime, state: Dict[str, Any]) -> Dict[str, Any]:
    return {"text": text, "context": ctx(now), "state": state, "conversation": None, "all_context": None}


def mk_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "response_type": "tool_call",
        "assistant_message": None,
        "response_payload": {},
        "next_gateway_action": {"type": "tool_call", "tool_call": {"tool_name": tool_name, "arguments": arguments}},
        "state_patch": {},
    }


def mk_finish(text: str) -> Dict[str, Any]:
    return {
        "response_type": "finish",
        "assistant_message": text,
        "response_payload": {},
        "next_gateway_action": {"type": "none", "tool_call": None},
        "state_patch": {},
    }


def mk_clarify(text: str) -> Dict[str, Any]:
    return {
        "response_type": "clarify",
        "assistant_message": text,
        "response_payload": {},
        "next_gateway_action": {"type": "none", "tool_call": None},
        "state_patch": {},
    }


def eval_for(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    field_paths = ["response_type", "next_gateway_action.type", "next_gateway_action.tool_call.tool_name"]
    time_paths: List[str] = []
    if tool_name == "create_event":
        field_paths += ["next_gateway_action.tool_call.arguments.title",
                        "next_gateway_action.tool_call.arguments.duration_minutes"]
        time_paths += ["next_gateway_action.tool_call.arguments.start_time"]
    elif tool_name == "update_event":
        field_paths += ["next_gateway_action.tool_call.arguments.event_id"]
        updates = arguments["updates"]
        if "duration_minutes" in updates:
            field_paths += ["next_gateway_action.tool_call.arguments.updates.duration_minutes"]
        if "title" in updates:
            field_paths += ["next_gateway_action.tool_call.arguments.updates.title"]
        if "start_time" in updates:
            time_paths += ["next_gateway_action.tool_call.arguments.updates.start_time"]
    elif tool_name == "find_event":
        field_paths += ["next_gateway_action.tool_call.arguments.query",
                        "next_gateway_action.tool_call.arguments.max_results"]
        time_paths += ["next_gateway_action.tool_call.arguments.time_min",
                       "next_gateway_action.tool_call.arguments.time_max"]
    elif tool_name == "list_events":
        field_paths += ["next_gateway_action.tool_call.arguments.max_results"]
        time_paths += ["next_gateway_action.tool_call.arguments.start", "next_gateway_action.tool_call.arguments.end"]
    elif tool_name == "get_free_slots":
        field_paths += ["next_gateway_action.tool_call.arguments.min_duration_minutes"]
        time_paths += ["next_gateway_action.tool_call.arguments.start", "next_gateway_action.tool_call.arguments.end"]
    elif tool_name == "delete_event":
        field_paths += ["next_gateway_action.tool_call.arguments.event_id"]
    return {"critical": True, "field_paths": field_paths, "time_paths": time_paths}


def eval_terminal() -> Dict[str, Any]:
    return {"critical": True, "field_paths": ["response_type", "next_gateway_action.type"], "time_paths": []}


def event_item(event_id: str, title: str, start: datetime, duration: int, status: str = "confirmed") -> Dict[str, Any]:
    return {"id": event_id, "summary": title, "start": iso(start), "end": iso(start + timedelta(minutes=duration)),
            "status": status}


def obs_events(tool_name: str, items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {"tool_name": tool_name, "result": {"items": list(items)}}


def obs_slots(slots: Sequence[Tuple[datetime, datetime]]) -> Dict[str, Any]:
    return {"tool_name": "get_free_slots", "result": {"slots": [{"start": iso(s), "end": iso(e)} for s, e in slots]}}


def obs_event_result(tool_name: str, event_id: str, title: str, start: datetime, duration: int) -> Dict[str, Any]:
    return {"tool_name": tool_name, "result": {"event": event_item(event_id, title, start, duration)}}


def obs_delete(event_id: str) -> Dict[str, Any]:
    return {"tool_name": "delete_event", "result": {"deleted_event_id": event_id}}


def copy_state(state: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(state))


class Gen:
    """Детерминированный генератор синтетических календарных эпизодов."""

    def __init__(self, prefix: str, split: str, seed: int) -> None:
        self.prefix = prefix
        self.split = split
        self.rng = random.Random(seed)
        self.ep_i = 1
        self.ev_i = 1

    def next_episode_id(self) -> str:
        value = f"{self.prefix}_ep_{self.ep_i:04d}"
        self.ep_i += 1
        return value

    def next_event_id(self) -> str:
        value = f"{self.prefix}_ev_{self.ev_i:05d}"
        self.ev_i += 1
        return value

    def now(self) -> datetime:
        return datetime(2026, self.rng.randint(4, 12), self.rng.randint(1, 25),
                        self.rng.choice([8, 9, 10, 11, 13, 14, 15, 16]), self.rng.choice([0, 15, 30, 45]), tzinfo=MSK)

    def day_after(self, current: datetime, lo: int, hi: int) -> date:
        return (current + timedelta(days=self.rng.randint(lo, hi))).date()

    def duration(self, long: bool = False) -> int:
        return self.rng.choice([60, 90, 120] if long else [30, 45, 60, 90])

    def work_start(self, duration: int) -> Tuple[int, int]:
        latest = max(WORK_START_HOUR, WORK_END_HOUR - max(1, math.ceil(duration / 60)))
        return self.rng.randint(WORK_START_HOUR, latest), self.rng.choice([0, 15, 30, 45])

    def title(self, verbose: bool = False) -> str:
        title = self.rng.choice(TITLE_PATTERNS).format(role=self.rng.choice(ROLE_POOLS[self.split]),
                                                       topic=self.rng.choice(TOPIC_POOLS[self.split]))
        return f"{title} по финальной версии" if verbose and self.rng.random() < 0.35 else title


def short_query(title: str) -> str:
    return title.split(" по ", 1)[0] if " по " in title else title


def make_step(step_id: int, request: Dict[str, Any], expected: Dict[str, Any], evaluation: Dict[str, Any],
              fill: Dict[str, Any]) -> Dict[str, Any]:
    return {"step_id": step_id, "step_request": request, "expected_step_response": expected, "evaluation": evaluation,
            "fill_instructions": fill}


def family_direct_create(gen: Gen, hard: bool = False) -> Dict[str, Any]:
    ep = gen.next_episode_id()
    now = gen.now()
    day = gen.day_after(now, 1, 25)
    title = gen.title(verbose=hard)
    duration = gen.duration()
    hour, minute = gen.work_start(duration)
    start = datetime(day.year, day.month, day.day, hour, minute, tzinfo=MSK)
    args = {"title": title, "start_time": iso(start), "duration_minutes": duration}
    step = make_step(1, mk_request(ph(ep, "user_request_1"), now, empty_state()), mk_tool("create_event", args),
                     eval_for("create_event", args), {
                         "user_text": {"slot": ph(ep, "user_request_1"), "mode": "direct_create", "hard": hard,
                                       "title": title, "start_time": iso(start), "duration_minutes": duration}})
    return {"episode_id": ep, "episode_type": "direct",
            "scaffold_meta": {"difficulty_bucket": "hard" if hard else "ordinary",
                              "family": "hard_direct_create_searchy" if hard else "ordinary_direct_create"},
            "steps": [step]}


def family_list_finish(gen: Gen, hard: bool = False) -> Dict[str, Any]:
    ep = gen.next_episode_id()
    now = gen.now()
    day = gen.day_after(now, 1, 14)
    start = datetime(day.year, day.month, day.day, 0, 0, tzinfo=MSK)
    end = start + timedelta(days=1)
    user = ph(ep, "user_request_1")
    finish = ph(ep, "assistant_finish_2")
    args = {"start": iso(start), "end": iso(end), "max_results": LIST_EVENTS_MAX_RESULTS}
    items = [event_item(gen.next_event_id(), gen.title(),
                        datetime(day.year, day.month, day.day, gen.rng.choice([9, 11, 14, 16]),
                                 gen.rng.choice([0, 15, 30]), tzinfo=MSK), gen.duration()) for _ in
             range(gen.rng.randint(2, 4))]
    s2 = empty_state()
    s2["messages"] = [msg("user", user)]
    s2["completed_actions"] = [act("list_events", args)]
    s2["tool_observations"] = [obs_events("list_events", items)]
    steps = [
        make_step(1, mk_request(user, now, empty_state()), mk_tool("list_events", args), eval_for("list_events", args),
                  {"user_text": {"slot": user, "mode": "list_events"}}),
        make_step(2, mk_request(user, now, s2), mk_finish(finish), eval_terminal(),
                  {"assistant_message": {"slot": finish, "mode": "finish_after_list", "hard": hard}}),
    ]
    return {"episode_id": ep, "episode_type": "agent_only",
            "scaffold_meta": {"difficulty_bucket": "hard" if hard else "ordinary",
                              "family": "hard_list_then_finish" if hard else "ordinary_list_then_finish"},
            "steps": steps}


def family_slots_finish(gen: Gen, hard: bool = False) -> Dict[str, Any]:
    ep = gen.next_episode_id()
    now = gen.now()
    day = gen.day_after(now, 1, 12)
    user = ph(ep, "user_request_1")
    finish = ph(ep, "assistant_finish_2")
    duration = gen.duration(long=hard)
    args = {"start": iso(datetime(day.year, day.month, day.day, 9, 0, tzinfo=MSK)),
            "end": iso(datetime(day.year, day.month, day.day, 18, 0, tzinfo=MSK)), "min_duration_minutes": duration}
    slot_a = datetime(day.year, day.month, day.day, gen.rng.choice([9, 10, 11]), gen.rng.choice([0, 15, 30]),
                      tzinfo=MSK)
    slot_b = slot_a + timedelta(hours=2)
    s2 = empty_state()
    s2["messages"] = [msg("user", user)]
    s2["completed_actions"] = [act("get_free_slots", args)]
    s2["tool_observations"] = [
        obs_slots([(slot_a, slot_a + timedelta(minutes=duration)), (slot_b, slot_b + timedelta(minutes=duration))])]
    steps = [
        make_step(1, mk_request(user, now, empty_state()), mk_tool("get_free_slots", args),
                  eval_for("get_free_slots", args),
                  {"user_text": {"slot": user, "mode": "get_slots", "hard": hard, "min_duration_minutes": duration}}),
        make_step(2, mk_request(user, now, s2), mk_finish(finish), eval_terminal(),
                  {"assistant_message": {"slot": finish, "mode": "finish_after_slots", "hard": hard}}),
    ]
    return {"episode_id": ep, "episode_type": "agent_only",
            "scaffold_meta": {"difficulty_bucket": "hard" if hard else "ordinary",
                              "family": "hard_slots_then_finish" if hard else "ordinary_slots_then_finish"},
            "steps": steps}


def family_find_update(gen: Gen) -> Dict[str, Any]:
    ep = gen.next_episode_id()
    now = gen.now()
    day = gen.day_after(now, 1, 20)
    title = gen.title()
    query = short_query(title)
    duration = gen.duration()
    old = datetime(day.year, day.month, day.day, 11, 0, tzinfo=MSK)
    new = old + timedelta(hours=1)
    event_id = gen.next_event_id()
    user = ph(ep, "user_request_1")
    finish = ph(ep, "assistant_finish_3")
    find_args = {"query": query, "time_min": iso(datetime(day.year, day.month, day.day, 0, 0, tzinfo=MSK)),
                 "time_max": iso(datetime(day.year, day.month, day.day, 23, 59, tzinfo=MSK)),
                 "max_results": FIND_EVENT_MAX_RESULTS}
    update_args = {"event_id": event_id, "updates": {"start_time": iso(new), "duration_minutes": duration}}
    s2 = empty_state();
    s2["messages"] = [msg("user", user)];
    s2["completed_actions"] = [act("find_event", find_args)];
    s2["tool_observations"] = [obs_events("find_event", [event_item(event_id, title, old, duration)])]
    s3 = copy_state(s2);
    s3["completed_actions"].append(act("update_event", update_args));
    s3["tool_observations"].append(obs_event_result("update_event", event_id, title, new, duration))
    steps = [
        make_step(1, mk_request(user, now, empty_state()), mk_tool("find_event", find_args),
                  eval_for("find_event", find_args), {
                      "user_text": {"slot": user, "mode": "find_then_update", "query": query,
                                    "new_start_time": iso(new)}}),
        make_step(2, mk_request(user, now, s2), mk_tool("update_event", update_args),
                  eval_for("update_event", update_args), {"user_text": {"slot": user, "mode": "internal_repeat"}}),
        make_step(3, mk_request(user, now, s3), mk_finish(finish), eval_terminal(),
                  {"assistant_message": {"slot": finish, "mode": "finish_after_update"}}),
    ]
    return {"episode_id": ep, "episode_type": "agent_only",
            "scaffold_meta": {"difficulty_bucket": "ordinary", "family": "ordinary_find_update_finish"}, "steps": steps}


def family_find_delete(gen: Gen) -> Dict[str, Any]:
    ep = gen.next_episode_id()
    now = gen.now()
    day = gen.day_after(now, 1, 20)
    title = gen.title()
    query = short_query(title)
    duration = gen.duration()
    start = datetime(day.year, day.month, day.day, 13, 0, tzinfo=MSK)
    event_id = gen.next_event_id()
    user = ph(ep, "user_request_1")
    finish = ph(ep, "assistant_finish_3")
    find_args = {"query": query, "time_min": iso(datetime(day.year, day.month, day.day, 0, 0, tzinfo=MSK)),
                 "time_max": iso(datetime(day.year, day.month, day.day, 23, 59, tzinfo=MSK)),
                 "max_results": FIND_EVENT_MAX_RESULTS}
    del_args = {"event_id": event_id}
    s2 = empty_state();
    s2["messages"] = [msg("user", user)];
    s2["completed_actions"] = [act("find_event", find_args)];
    s2["tool_observations"] = [obs_events("find_event", [event_item(event_id, title, start, duration)])]
    s3 = copy_state(s2);
    s3["completed_actions"].append(act("delete_event", del_args));
    s3["tool_observations"].append(obs_delete(event_id))
    steps = [
        make_step(1, mk_request(user, now, empty_state()), mk_tool("find_event", find_args),
                  eval_for("find_event", find_args),
                  {"user_text": {"slot": user, "mode": "find_then_delete", "query": query}}),
        make_step(2, mk_request(user, now, s2), mk_tool("delete_event", del_args), eval_for("delete_event", del_args),
                  {"user_text": {"slot": user, "mode": "internal_repeat"}}),
        make_step(3, mk_request(user, now, s3), mk_finish(finish), eval_terminal(),
                  {"assistant_message": {"slot": finish, "mode": "finish_after_delete"}}),
    ]
    return {"episode_id": ep, "episode_type": "agent_only",
            "scaffold_meta": {"difficulty_bucket": "ordinary", "family": "ordinary_find_delete_finish"}, "steps": steps}


def family_clarify(gen: Gen) -> Dict[str, Any]:
    ep = gen.next_episode_id()
    now = gen.now()
    user = ph(ep, "user_request_1")
    ask = ph(ep, "assistant_clarify_1")
    step = make_step(1, mk_request(user, now, empty_state()), mk_clarify(ask), eval_terminal(), {
        "user_text": {"slot": user, "mode": "clarify_vague_time", "title": gen.title(),
                      "vague_time_hint": gen.rng.choice(VAGUE_TIMES)},
        "assistant_message": {"slot": ask, "mode": "clarify_missing_time"}})
    return {"episode_id": ep, "episode_type": "direct",
            "scaffold_meta": {"difficulty_bucket": "hard", "family": "hard_clarify_missing_time"}, "steps": [step]}


def family_ambiguous_update(gen: Gen) -> Dict[str, Any]:
    ep = gen.next_episode_id()
    now = gen.now()
    day = gen.day_after(now, 1, 15)
    title = gen.title()
    query = short_query(title)
    duration = gen.duration()
    first_id, second_id = gen.next_event_id(), gen.next_event_id()
    first_start = datetime(day.year, day.month, day.day, 11, 0, tzinfo=MSK)
    second_start = datetime(day.year, day.month, day.day, 15, 0, tzinfo=MSK)
    new_start = datetime(day.year, day.month, day.day, 16, 30, tzinfo=MSK)
    user = ph(ep, "user_request_1")
    ask = ph(ep, "assistant_clarify_2")
    follow = ph(ep, "user_followup_3")
    finish = ph(ep, "assistant_finish_4")
    find_args = {"query": query, "time_min": iso(datetime(day.year, day.month, day.day, 0, 0, tzinfo=MSK)),
                 "time_max": iso(datetime(day.year, day.month, day.day, 23, 59, tzinfo=MSK)),
                 "max_results": FIND_EVENT_MAX_RESULTS}
    upd_args = {"event_id": second_id, "updates": {"start_time": iso(new_start), "duration_minutes": duration}}
    s2 = empty_state();
    s2["messages"] = [msg("user", user)];
    s2["completed_actions"] = [act("find_event", find_args)];
    s2["tool_observations"] = [obs_events("find_event",
                                          [event_item(first_id, f"{query} по первой версии", first_start, duration),
                                           event_item(second_id, f"{query} по второй версии", second_start, duration)])]
    s3 = copy_state(s2);
    s3["messages"] = [msg("user", user), msg("assistant", ask)]
    s4 = copy_state(s3);
    s4["messages"].append(msg("user", follow));
    s4["completed_actions"].append(act("update_event", upd_args));
    s4["tool_observations"].append(
        obs_event_result("update_event", second_id, f"{query} по второй версии", new_start, duration))
    steps = [
        make_step(1, mk_request(user, now, empty_state()), mk_tool("find_event", find_args),
                  eval_for("find_event", find_args), {
                      "user_text": {"slot": user, "mode": "ambiguous_find_for_update", "query": query,
                                    "new_start_time": iso(new_start)}}),
        make_step(2, mk_request(user, now, s2), mk_clarify(ask), eval_terminal(),
                  {"assistant_message": {"slot": ask, "mode": "clarify_two_candidates"}}),
        make_step(3, mk_request(follow, now, s3), mk_tool("update_event", upd_args), eval_for("update_event", upd_args),
                  {"user_text": {"slot": follow, "mode": "choose_second_candidate", "new_start_time": iso(new_start)}}),
        make_step(4, mk_request(follow, now, s4), mk_finish(finish), eval_terminal(),
                  {"assistant_message": {"slot": finish, "mode": "finish_after_ambiguous_update"}}),
    ]
    return {"episode_id": ep, "episode_type": "user_correction",
            "scaffold_meta": {"difficulty_bucket": "hard", "family": "hard_ambiguous_find_clarify_update_finish"},
            "steps": steps}


def family_mixed_slot_switch(gen: Gen) -> Dict[str, Any]:
    ep = gen.next_episode_id()
    now = gen.now()
    day = gen.day_after(now, 1, 12)
    duration = gen.duration(long=True)
    title = gen.title()
    event_id = gen.next_event_id()
    slot_a = datetime(day.year, day.month, day.day, 9, 0, tzinfo=MSK)
    slot_b = datetime(day.year, day.month, day.day, 11, 0, tzinfo=MSK)
    user = ph(ep, "user_request_1")
    gateway = ph(ep, "assistant_gateway_2")
    follow = ph(ep, "user_followup_3")
    finish = ph(ep, "assistant_finish_4")
    slots_args = {"start": iso(datetime(day.year, day.month, day.day, 9, 0, tzinfo=MSK)),
                  "end": iso(datetime(day.year, day.month, day.day, 18, 0, tzinfo=MSK)),
                  "min_duration_minutes": duration}
    create_args = {"title": title, "start_time": iso(slot_a), "duration_minutes": duration}
    update_args = {"event_id": event_id, "updates": {"start_time": iso(slot_b), "duration_minutes": duration}}
    s2 = empty_state();
    s2["messages"] = [msg("user", user)];
    s2["completed_actions"] = [act("get_free_slots", slots_args)];
    s2["tool_observations"] = [
        obs_slots([(slot_a, slot_a + timedelta(minutes=duration)), (slot_b, slot_b + timedelta(minutes=duration))])]
    s3 = copy_state(s2);
    s3["messages"] = [msg("user", user), msg("assistant", gateway)];
    s3["completed_actions"].append(act("create_event", create_args));
    s3["tool_observations"].append(obs_event_result("create_event", event_id, title, slot_a, duration))
    s4 = copy_state(s3);
    s4["messages"].append(msg("user", follow));
    s4["completed_actions"].append(act("update_event", update_args));
    s4["tool_observations"].append(obs_event_result("update_event", event_id, title, slot_b, duration))
    steps = [
        make_step(1, mk_request(user, now, empty_state()), mk_tool("get_free_slots", slots_args),
                  eval_for("get_free_slots", slots_args), {
                      "user_text": {"slot": user, "mode": "slot_then_create", "title": title,
                                    "min_duration_minutes": duration}}),
        make_step(2, mk_request(user, now, s2), mk_tool("create_event", create_args),
                  eval_for("create_event", create_args),
                  {"assistant_gateway_message": {"slot": gateway, "mode": "gateway_offers_two_slots"}}),
        make_step(3, mk_request(follow, now, s3), mk_tool("update_event", update_args),
                  eval_for("update_event", update_args),
                  {"user_text": {"slot": follow, "mode": "switch_to_second_slot"}}),
        make_step(4, mk_request(follow, now, s4), mk_finish(finish), eval_terminal(),
                  {"assistant_message": {"slot": finish, "mode": "finish_after_slot_switch"}}),
    ]
    return {"episode_id": ep, "episode_type": "mixed",
            "scaffold_meta": {"difficulty_bucket": "hard", "family": "hard_mixed_second_slot_update"}, "steps": steps}


ORDINARY = [(family_direct_create, 0.34), (family_list_finish, 0.18), (family_slots_finish, 0.18),
            (family_find_update, 0.15), (family_find_delete, 0.15)]
HARD = [(lambda g: family_direct_create(g, hard=True), 0.22), (family_clarify, 0.18), (family_ambiguous_update, 0.22),
        (lambda g: family_list_finish(g, hard=True), 0.14), (lambda g: family_slots_finish(g, hard=True), 0.12),
        (family_mixed_slot_switch, 0.12)]


def choose(rng: random.Random, weighted: Sequence[Tuple[Any, float]]) -> Any:
    x = rng.random()
    total = 0.0
    for fn, weight in weighted:
        total += weight
        if x <= total:
            return fn
    return weighted[-1][0]


def validate(samples: Sequence[Dict[str, Any]]) -> None:
    """Проверяет структурную связность сгенерированных многошаговых эпизодов."""
    errors: List[str] = []
    for sample in samples:
        steps = sample.get("steps", [])
        for i, step in enumerate(steps, start=1):
            if step.get("step_id") != i:
                errors.append(f"{sample['episode_id']}: bad step_id {i}")
            expected = step["expected_step_response"]
            rt = expected["response_type"]
            if rt not in VALID_RESPONSE_TYPES:
                errors.append(f"{sample['episode_id']}: bad response_type")
            nga = expected["next_gateway_action"]
            if rt == "tool_call":
                tc = nga["tool_call"]
                tool = tc["tool_name"]
                if tool not in VALID_TOOLS:
                    errors.append(f"{sample['episode_id']}: bad tool {tool}")
            else:
                if nga["type"] != "none":
                    errors.append(f"{sample['episode_id']}: terminal step has tool")
            if i > 1 and steps[i - 2]["expected_step_response"]["response_type"] == "tool_call":
                prev_tool = steps[i - 2]["expected_step_response"]["next_gateway_action"]["tool_call"]["tool_name"]
                state = step["step_request"]["state"]
                if not state["completed_actions"] or state["completed_actions"][-1]["tool_name"] != prev_tool:
                    errors.append(f"{sample['episode_id']}: missing completed action for step {i}")
                if not state["tool_observations"] or state["tool_observations"][-1]["tool_name"] != prev_tool:
                    errors.append(f"{sample['episode_id']}: missing observation for step {i}")
    if errors:
        raise ValueError("\n".join(errors[:20]))


def build_split(source_path: Path, prefix: str, split: str, seed: int, ratio: float, hard_share: float) -> Dict[
    str, Any]:
    source = load_samples(source_path)
    base_steps = count_steps(source)
    target_steps = max(1, round(base_steps * ratio))
    ordinary_target = target_steps - max(1, round(target_steps * hard_share))
    hard_target = target_steps - ordinary_target
    gen = Gen(prefix, split, seed)
    ordinary_steps = 0
    hard_steps = 0
    samples: List[Dict[str, Any]] = []
    signatures = set()
    while ordinary_steps < ordinary_target:
        sample = choose(gen.rng, ORDINARY)(gen)
        sig = (sample["scaffold_meta"]["family"],
               json.dumps(sample["steps"][0]["expected_step_response"], ensure_ascii=False, sort_keys=True))
        if sig in signatures:
            continue
        signatures.add(sig)
        samples.append(sample)
        ordinary_steps += len(sample["steps"])
    while hard_steps < hard_target:
        sample = choose(gen.rng, HARD)(gen)
        sig = (sample["scaffold_meta"]["family"],
               json.dumps(sample["steps"][0]["expected_step_response"], ensure_ascii=False, sort_keys=True))
        if sig in signatures:
            continue
        signatures.add(sig)
        samples.append(sample)
        hard_steps += len(sample["steps"])
    samples.sort(key=lambda x: x["episode_id"])
    validate(samples)
    action_counts = Counter()
    for sample in samples:
        for step in sample["steps"]:
            resp = step["expected_step_response"]
            action_counts[(
                resp["next_gateway_action"]["tool_call"]["tool_name"] if resp["response_type"] == "tool_call" else resp[
                    "response_type"])] += 1
    return {
        "dataset_name": f"calendar_step_agent_{split}_new_scaffold",
        "schema_version": "1.1-scaffold",
        "split": f"{split}_new",
        "scaffold_only": True,
        "status": "skeleton_needs_llm_surface_fill",
        "source_snapshot": {"episodes": len(source), "steps": base_steps},
        "target_size": {"step_multiplier_vs_source": ratio, "steps_target_total": target_steps,
                        "ordinary_steps_target": ordinary_target, "hard_steps_target": hard_target},
        "generated_summary": {"episodes": len(samples), "steps": sum(len(s["steps"]) for s in samples),
                              "ordinary_steps": ordinary_steps, "hard_steps": hard_steps,
                              "action_counts": dict(action_counts),
                              "episode_type_counts": dict(Counter(s["episode_type"] for s in samples))},
        "fill_contract": {
            "note": "Заполни все строки [[FILL::...]] естественным русским текстом. Структурные действия, даты, длительности и observations уже зафиксированы программно."},
        "samples": samples,
    }


def main() -> None:
    """Создаёт train_new.json и test_new.json из базовых датасетов."""
    parser = argparse.ArgumentParser(description="Generate train_new/test_new scaffold datasets.")
    parser.add_argument("--seed", type=int, default=20260330)
    parser.add_argument("--ratio", type=float, default=1.5)
    parser.add_argument("--hard-share", type=float, default=0.5)
    args = parser.parse_args()

    train = build_split(TRAIN_SOURCE, "train_new", "train", args.seed, args.ratio, args.hard_share)
    test = build_split(TEST_SOURCE, "test_new", "test", args.seed + 1, args.ratio, args.hard_share)
    TRAIN_TARGET.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    TEST_TARGET.write_text(json.dumps(test, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps({"train_new": train["generated_summary"], "test_new": test["generated_summary"]}, ensure_ascii=False,
                   indent=2))


if __name__ == "__main__":
    main()
