# Session 4 — Claude integration + story adapter (design)

**Date:** 2026-04-25
**Status:** Design approved; ready for implementation plan.
**Spec:** plan §8 Session 4. **Predecessor:** Session 3 (`story_curator` CLI), commit `4a27281`.

## Goal

Take a curator-approved Story (Session 3 deliverable) and produce, via three Claude
calls, a polished narration script + scene list + per-scene visual prompts. Persist
everything in `story.json` and into the SQLite projection. By end of session,
`platinum adapt --story <id>` walks an approved story all the way through to
`visual_prompts` COMPLETE.

## Decisions (six brainstorming forks)

| # | Question | Decision |
|---|---|---|
| 1 | Default model | **Opus 4.7** (`claude-opus-4-7`) for all three calls. Configurable per-call so we can dial back later. |
| 2 | Test fixture strategy | **Recorded fixtures** (home-rolled VCR). Tests run offline; `PLATINUM_RECORD_FIXTURES=1` re-captures. |
| 3 | Structured output | **Tool-use mode** (forced). Schema enforced server-side; SDK returns parsed dict; no JSON parsing of free text. |
| 4 | Stage granularity | **Three separate `Stage` subclasses**. Resume-from-failure preserves successful upstream Opus calls. |
| 5 | Length tolerance | **Regen once with deviation feedback; accept second pass**. Flag `in_tolerance: false` if still off — pipeline does not halt for a 6% miss. |
| 6 | CLI surface / pricing | **One verb `platinum adapt`** walks eligible stories. **Hardcoded pricing table** in `utils/claude.py`. |

## Architecture

```
                        utils/claude.py (Anthropic wrapper)
                          - call(model, system, messages, tool, ...)
                          - prompt cache (system blocks + tool def)
                          - retry decorator (429 / 5xx)
                          - cost tracker -> ApiUsageRow
                          - injectable recorder (test fixtures)

                                       |
                +----------------------+----------------------+
                |                      |                      |
                v                      v                      v
   pipeline/story_adapter.py  pipeline/scene_breakdown.py  pipeline/visual_prompts.py
       adapt(story, cfg)         breakdown(story, cfg)        visual_prompts(story, cfg)
       -> Adapted                -> list[Scene],              -> list[(visual, neg)]
                                    BreakdownReport
       |                          |                            |
       v                          v                            v
   StoryAdapterStage         SceneBreakdownStage         VisualPromptsStage
       |                          |                            |
       +-------- Story.save -> SQLite project (existing) ------+

   config/prompts/atmospheric_horror/
       system.j2  adapt.j2  breakdown.j2  visual_prompts.j2
```

### Architectural rules

1. **Pure core / impure shell.** Each pipeline module exports a function that
   takes `(story, track_cfg, *, claude_call=...)`. Never reads `os.environ`
   or hits the network directly. Stage subclasses do the I/O wiring.
2. **Stage = thin wrapper.** Builds call params, invokes the pure function,
   mutates the Story, returns artifacts dict. Knows nothing about `anthropic`.
3. **`utils/claude.py` is the only file that imports `anthropic`.** Single
   integration point; one place to update SDK; one owner of `ApiUsageRow`.

## Components

### `src/platinum/utils/claude.py` (~200 LOC)

```python
@dataclass(frozen=True)
class ClaudeUsage:
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    cost_usd: float

@dataclass(frozen=True)
class ClaudeResult:
    tool_input: dict[str, Any]
    text: str
    usage: ClaudeUsage
    raw: dict[str, Any]

def call(
    *,
    model: str,
    system: list[dict],          # last block carries cache_control
    messages: list[dict],
    tool: dict,                  # exactly one tool, forced
    story_id: str | None,
    stage: str,
    db_path: Path,
    client_factory: Callable[[], Anthropic] | None = None,
    recorder: Recorder | None = None,
) -> ClaudeResult: ...
```

- **Pricing table:** `_PRICING_USD_PER_MTOK = {"claude-opus-4-7": (15.0, 75.0)}`
  (input, output USD per million tokens). Cache-read priced at 10% of input rate.
- **Retry:** wraps `anthropic.RateLimitError` and `anthropic.APIStatusError(5xx)`
  using existing `utils/retry.py`. Hard-fail on 4xx other than 429.
- **Cost tracking:** writes one `ApiUsageRow` per successful call; failures
  bubble; `provider="anthropic"`. DB write is best-effort (logged warning,
  result still returned).
- **Recorder:** default `None` (live calls). Tests inject a `FixtureRecorder`
  keyed by `(stage, test_name, attempt_index)`. `PLATINUM_RECORD_FIXTURES=1`
  switches recorder to record-and-save mode against the live API.
- **Prompt cache:** last `system` block carries
  `cache_control={"type":"ephemeral"}`. Tool definition is also cacheable.
- **API key:** read from `os.environ["ANTHROPIC_API_KEY"]` at call time;
  missing key raises a clear `RuntimeError` (not Anthropic's generic 401).

### `tests/_fixtures.py` recorder + `tests/fixtures/anthropic/...` (~80 LOC + JSON)

```json
{
  "request":  {"model": "...", "system": [...], "messages": [...], "tools": [...], "tool_choice": {...}},
  "response": {"id": "...", "content": [{"type":"tool_use","input":{...}}], "usage": {...}}
}
```

Path: `tests/fixtures/anthropic/<stage>/<test_name>__<attempt>.json`.
Replay-mode missing fixture raises `FixtureMissingError` with clear hint to
re-run with `PLATINUM_RECORD_FIXTURES=1`.

### `config/prompts/atmospheric_horror/` (4 templates)

- `system.j2` — voice direction, visual aesthetic, palette, influences,
  emotion-tag legend. Track-config-driven; cached.
- `adapt.j2` — wraps source `raw_text`, `title`, `author`, `target_seconds`,
  `pace_wpm`. Asks for arc-structured polished narration.
- `breakdown.j2` — wraps adapted `narration_script`, `target_seconds`,
  `pace_wpm`, deviation feedback (empty on first attempt).
- `visual_prompts.j2` — wraps each scene's `narration_text` plus track
  visual aesthetic; asks for `(visual_prompt, negative_prompt)` per scene.

Folder-per-track means Sessions later can drop in
`config/prompts/<track>/` without code changes.

### `src/platinum/pipeline/story_adapter.py` (~120 LOC)

```python
def adapt(
    story: Story,
    track_cfg: dict,
    *,
    claude_call: ClaudeCallFn,
    db_path: Path,
) -> Adapted:
```

Builds system blocks + user message via Jinja, calls `claude_call`, parses
`tool_input` -> `Adapted(title, synopsis, narration_script,
estimated_duration_seconds, tone_notes, arc={setup, rising, climax,
resolution})`. `estimated_duration_seconds = word_count / pace_wpm * 60`.

`StoryAdapterStage.run`: calls `adapt`, sets `story.adapted = result`, returns
`{"model","input_tokens","output_tokens","cost_usd"}`.

#### Tool schema

```python
ADAPT_TOOL = {
    "name": "submit_adapted_story",
    "input_schema": {
        "type":"object",
        "required":["title","synopsis","narration_script","tone_notes","arc"],
        "properties":{
            "title":{"type":"string"},
            "synopsis":{"type":"string","maxLength":400},
            "narration_script":{"type":"string"},
            "tone_notes":{"type":"string"},
            "arc":{
                "type":"object",
                "required":["setup","rising","climax","resolution"],
                "properties":{
                    "setup":{"type":"string"},
                    "rising":{"type":"string"},
                    "climax":{"type":"string"},
                    "resolution":{"type":"string"},
                },
            },
        },
    },
}
```

### `src/platinum/pipeline/scene_breakdown.py` (~150 LOC)

```python
def breakdown(
    story: Story,
    track_cfg: dict,
    *,
    claude_call: ClaudeCallFn,
    db_path: Path,
) -> tuple[list[Scene], BreakdownReport]:
```

Pure regen-once flow:

```python
deviation_feedback = ""
for attempt in (1, 2):
    result = claude_call(...)
    scenes = _scenes_from_tool(result.tool_input)
    total = _estimate_total_seconds(scenes, pace_wpm)
    if target_min <= total <= target_max:
        return scenes, BreakdownReport(attempts=attempt, final_seconds=total, in_tolerance=True)
    deviation_feedback = f"Previous breakdown totalled {total:.0f}s; target is {target}s plus/minus {tol}s. {'Lengthen' if total < target_min else 'Shorten'} scenes to land in range."
return scenes, BreakdownReport(attempts=2, final_seconds=total, in_tolerance=False)
```

#### Tool schema

```python
BREAKDOWN_TOOL = {
    "name":"submit_scene_breakdown",
    "input_schema":{
        "type":"object","required":["scenes"],
        "properties":{"scenes":{"type":"array","minItems":4,"maxItems":20,"items":{
            "type":"object",
            "required":["index","narration_text","mood","sfx_cues"],
            "properties":{
                "index":{"type":"integer","minimum":1},
                "narration_text":{"type":"string"},
                "mood":{"type":"string"},
                "sfx_cues":{"type":"array","items":{"type":"string"}},
            }}}},
    },
}
```

### `src/platinum/pipeline/visual_prompts.py` (~110 LOC)

```python
def visual_prompts(
    story: Story,
    track_cfg: dict,
    *,
    claude_call: ClaudeCallFn,
    db_path: Path,
) -> list[tuple[str, str]]:
```

One Claude call. Tool returns `{scenes: [{index, visual_prompt,
negative_prompt}, ...]}`. We zip back into existing `story.scenes` by index.

#### Tool schema

```python
VISUAL_PROMPTS_TOOL = {
    "name":"submit_visual_prompts",
    "input_schema":{
        "type":"object","required":["scenes"],
        "properties":{"scenes":{"type":"array","items":{
            "type":"object",
            "required":["index","visual_prompt","negative_prompt"],
            "properties":{
                "index":{"type":"integer","minimum":1},
                "visual_prompt":{"type":"string"},
                "negative_prompt":{"type":"string"},
            }}}},
    },
}
```

### CLI: `src/platinum/cli.py` `adapt` command (~40 LOC of new code)

```
platinum adapt [--story <id>] [--track <name>]
```

Walks stories where `latest_stage_run("story_curator") is COMPLETE` and
`latest_stage_run("visual_prompts") is not COMPLETE`. Per story, runs the
three Stages via the orchestrator's existing `run_stages` (which already
handles skip-if-complete, log StageRun, persist Story+SQLite). Exit 0
including "no eligible stories"; non-zero only on unhandled exception.

## Data flow (one story end-to-end)

Cask of Amontillado (~2300 words) example.

### Pre-state (after Session 3 curate)

```jsonc
{"id":"story_2026_04_24_001","track":"atmospheric_horror",
 "source":{"type":"gutenberg","title":"The Cask of Amontillado",
           "author":"Edgar Allan Poe","raw_text":"...","license":"PD-US"},
 "adapted":null,"scenes":[],
 "stages":[{"stage":"source_fetcher","status":"complete"},
           {"stage":"story_curator","status":"complete",
            "artifacts":{"decision":"approved"}}]}
```

### After `story_adapter`

```jsonc
"adapted":{
  "title":"The Cask of Amontillado",
  "synopsis":"A man lures his rival into the catacombs to seal him alive...",
  "narration_script":"The thousand injuries I had borne as best I could...",
  "estimated_duration_seconds":612.3,
  "tone_notes":"Restrained, first-person, slow build...",
  "arc":{"setup":"...","rising":"...","climax":"...","resolution":"..."}
}
"stages":[..., {"stage":"story_adapter","status":"complete",
                "artifacts":{"model":"claude-opus-4-7",
                             "input_tokens":3250,"output_tokens":1840,
                             "cost_usd":0.187}}]
```

### After `scene_breakdown`

`story.scenes = [Scene(id="scene_001", index=1, narration_text="...",
music_cue="ambient_drone", sfx_cues=["clock_ticking_distant"]), ...]`.
Stage artifacts include `BreakdownReport(attempts, final_seconds,
in_tolerance)`.

### After `visual_prompts`

Each Scene now has `visual_prompt` and `negative_prompt` populated. Story
is ready for Session 5 (aesthetics validation) and Session 6 (keyframe gen).

### Cost profile

| Stage | Input | Output | Cache reads | Cost |
|---|---|---|---|---|
| story_adapter | ~3.5k | ~1.8k | 0 (seeds cache) | ~$0.19 |
| scene_breakdown (1 attempt) | ~2.5k | ~1.2k | ~2k | ~$0.13 |
| scene_breakdown (regen, ~10% of cases) | +2.5k | +1.2k | +2k | +$0.13 |
| visual_prompts | ~3k | ~1.5k | ~2k | ~$0.16 |
| **Total per story** | | | | **~$0.48** typical, **~$0.61** w/ regen |

## Error handling

1. **Transient (429, 5xx, network)** — exponential backoff 1/2/4/8/16s; orchestrator catches eventual failure -> `StageRun(status=FAILED)`.
2. **Hard 4xx (401 auth, 400 malformed)** — no retry; raise immediately. Missing `ANTHROPIC_API_KEY` surfaces a clear `RuntimeError` ahead of any HTTP call.
3. **Tool-schema mismatch** — defensive `ClaudeProtocolError` with offending payload. No regen (this is a prompt or schema bug).
4. **Length tolerance miss after regen** — accept; `in_tolerance: false` flag in artifacts. Not an error.
5. **Source > 80k chars** — truncate with `[...]` marker; warning into `StageRun.artifacts.warnings`.
6. **Replay-mode fixture miss** — `FixtureMissingError` with `PLATINUM_RECORD_FIXTURES=1` hint.
7. **`ApiUsageRow` write failure** — logged warning; call still returns. Cost tracking is observability, not correctness.
8. **Mid-stage process death** — atomic `Story.save` + atomic StageRun append handles resume cleanly.
9. **Stage failure** — orchestrator catches, writes `StageRun(status=FAILED, error=...)`, halts. User re-runs `platinum adapt --story <id>` to resume.

Explicitly out of scope:
- Cost runaway (bounded by N=2 attempts).
- Anthropic schema drift (recorded fixture replay catches it on re-record; live smoke catches it interactively).
- Concurrent `platinum adapt` on the same story (sequential CLI use only).

## Testing

Tests are offline + deterministic via injected `claude_call` (unit) and the
fixture recorder (integration). Live API only during fixture (re)recording
or final smoke.

| Layer | New tests |
|---|---|
| Unit — `claude_util` | 10 |
| Unit — `story_adapter` | 6 |
| Unit — `scene_breakdown` | 8 |
| Unit — `visual_prompts` | 5 |
| Unit — `recorder` | 5 |
| Integration — Stages | 6 |
| Integration — CLI | 5 |
| **New** | **45** |
| Existing | 104 |
| **Final** | **149** |

### Quality gates

- `pytest -q` — 149 pass, 0 fail, 0 skip. ~5s offline.
- `ruff check src tests` — clean.
- `mypy` on the four new files — clean.
- Live smoke: `python -m platinum fetch ... && curate && PLATINUM_RECORD_FIXTURES=1 adapt && status` — verifies live API end-to-end (~$0.50).

## Files

### New

- `src/platinum/utils/claude.py`
- `src/platinum/pipeline/story_adapter.py`
- `src/platinum/pipeline/scene_breakdown.py`
- `src/platinum/pipeline/visual_prompts.py`
- `config/prompts/atmospheric_horror/system.j2`
- `config/prompts/atmospheric_horror/adapt.j2`
- `config/prompts/atmospheric_horror/breakdown.j2`
- `config/prompts/atmospheric_horror/visual_prompts.j2`
- `tests/_fixtures.py` (recorder helper)
- `tests/fixtures/anthropic/<stage>/<test_name>__<attempt>.json` (recorded fixtures)
- `tests/unit/test_claude_util.py`
- `tests/unit/test_story_adapter.py`
- `tests/unit/test_scene_breakdown.py`
- `tests/unit/test_visual_prompts.py`
- `tests/unit/test_recorder.py`
- `tests/integration/test_adapt_stages.py`
- `tests/integration/test_adapt_command.py`

### Modified

- `src/platinum/cli.py` — replace `adapt` stub with real implementation.
- `src/platinum/pipeline/orchestrator.py` — register the three new Stages.
- `tasks/todo.md` — Session 4 plan + checklist.

## Open follow-ups (deferred)

- Multi-track prompts: only `atmospheric_horror/` authored this session.
  Other four tracks gain prompt folders when their first story runs.
- Claude prompt iteration on real horror text — likely needs 1–2 rounds of
  prompt tuning post-smoke. Not blocking; doesn't change architecture.
- `tracks/` Python module: a per-track config loader is on the deferred list.
  For Session 4 we read the track YAML directly with PyYAML.
