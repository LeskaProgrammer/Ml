# Engineering: Step Quality

Этот пайплайн измеряет качество новой `/api/v1/step` архитектуры.

Главная единица оценки теперь не отдельный router intent, а следующий шаг в контексте:

- `text + state -> next step`

То есть в тесте можно хранить эпизоды из 2-5 шагов:

- пользовательский запрос
- continuation шаги самой LLM
- пользовательские коррекции по ходу

## Что считается

Основные метрики:

- `step_action_macro_f1`
- `step_action_accuracy`
- `deterministic_fields_exact_rate`
- `time_exact_rate_conditional`
- `episode_success_rate`
- `json_valid_rate`
- `schema_valid_rate`

### Как трактуются метрики

- `step_action_*`
  Проверяют класс следующего шага:
  - `tool_call/find_event`
  - `tool_call/list_events`
  - `tool_call/get_free_slots`
  - `tool_call/create_event`
  - `tool_call/update_event`
  - `tool_call/delete_event`
  - `clarify`
  - `finish`

- `deterministic_fields_exact_rate`
  Считается только на шагах, где в sample явно или автоматически определены поля, которые должны совпасть точно.
  Например:
  - `find_event.query`
  - `update_event.event_id`
  - `delete_event.event_id`

- `time_exact_rate_conditional`
  Считается только там, где время действительно детерминировано.
  По умолчанию это:
  - `tool_call/create_event.arguments.start_time`
  - `tool_call/update_event.arguments.updates.start_time`
  Если sample не требует точного времени, он не штрафуется.

- `episode_success_rate`
  Эпизод считается успешным, если все его critical steps успешны.

Это специально сделано так, чтобы не штрафовать модель за допустимые варианты ответа там, где единственного правильного текста нет.

## Формат test.json

Рекомендуемый формат — эпизоды:

```json
{
  "metric": "episode_success_rate",
  "samples": [
    {
      "episode_id": "swap_001",
      "episode_type": "mixed",
      "steps": [
        {
          "step_id": 1,
          "step_request": {
            "text": "поменяй местами события стоматолога и футбол",
            "context": {
              "current_time": "2026-03-25T12:00:00+03:00",
              "timezone": "Europe/Moscow",
              "work_start_hour": 9,
              "work_end_hour": 18
            },
            "state": {
              "messages": [
                {
                  "role": "user",
                  "text": "поменяй местами события стоматолога и футбол"
                }
              ],
              "completed_actions": [],
              "tool_observations": [],
              "working_state": {
                "status": "in_progress"
              }
            }
          },
          "expected_step_response": {
            "response_type": "tool_call",
            "assistant_message": null,
            "response_payload": {},
            "next_gateway_action": {
              "type": "tool_call",
              "tool_call": {
                "tool_name": "find_event",
                "arguments": {
                  "query": "стоматолог"
                }
              }
            },
            "state_patch": {}
          },
          "evaluation": {
            "critical": true,
            "field_paths": [
              "next_gateway_action.tool_call.arguments.query"
            ]
          }
        }
      ]
    }
  ]
}
```

### evaluation

Поле `evaluation` опционально.

Поддерживаемые флаги:

- `critical: true/false`
  Учитывать ли этот шаг в `episode_success_rate`

- `field_paths: [...]`
  Какие детерминированные поля должны совпасть точно

- `time_paths: [...]`
  Какие time-поля сравнивать точно

Если `field_paths`/`time_paths` не указаны, пайплайн пытается безопасно вывести их автоматически только для однозначных сценариев.

## Что сохраняется

После каждого sample:

- полный `llm_response_text`
- parsed JSON
- normalized parsed JSON
- текущие метрики

Также результаты пишутся в:

- `reports/step_metrics.json`
- `reports/step_details.jsonl`

`step_details.jsonl` удобен для ручного разбора провалов по шагам.

## Запуск

Из корня репозитория:

```powershell
python engineering/quality/evaluate.py
```

Локальная GGUF-проверка:

```powershell
python engineering/quality/local_eval_run.py
```
