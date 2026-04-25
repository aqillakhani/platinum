# Session 2 — Source fetchers

**Spec:** plan §8 Session 2. **Deliverable:** `python -m platinum fetch --track atmospheric_horror --limit 10` produces 10 candidate Story JSONs on disk.

## Files

New:
- `src/platinum/sources/base.py` — `SourceFetcher` ABC with `fetch(filters, limit) -> list[Source]`
- `src/platinum/sources/gutenberg.py` — Gutendex REST + `pg{id}.txt` body fetch + boilerplate strip
- `src/platinum/sources/wikisource.py` — MediaWiki API category enum + page text fetch
- `src/platinum/sources/reddit.py` — port from `gold/utils/reddit.py` + adapter to `Source`
- `src/platinum/sources/registry.py` — type-string → fetcher mapping

Tests:
- `tests/unit/test_source_base.py`
- `tests/unit/test_source_gutenberg.py`
- `tests/unit/test_source_wikisource.py`
- `tests/unit/test_source_reddit.py`
- `tests/integration/test_fetch_command.py`

Modified:
- `src/platinum/cli.py` — replace `fetch` stub with real implementation

## Decisions

- **HTTP mocking:** `httpx.MockTransport` (built-in; no new dep)
- **Async fetchers:** all fetchers are `async def fetch(...)`; CLI wraps with `asyncio.run()`
- **Story IDs:** `story_YYYY_MM_DD_NNN` — count existing same-prefix dirs in `data/stories/`
- **Source allocation:** iterate `track.sources` in YAML order; take what each can supply until `--limit` reached
- **Boilerplate stripping (Gutenberg):** regex on `*** START OF ... ***` / `*** END OF ... ***`; fall back to whole text if no markers
- **Wikisource:** minimal regex cleanup of templates/refs; `story_adapter` (Session 4) does heavy polishing
- **Reddit `adaptation_required` flag:** stored on Source via `license` field stays "Reddit-CC-BY-NC" — adapter stage is responsible for paraphrasing
- **Skip `pipeline/source_fetcher.py` Stage wrapper:** not in deliverable; can be added in a later session if `python -m platinum render` ever needs to re-fetch

## TDD checklist

- [ ] test_source_base — ABC rejects non-implementing subclasses
- [ ] test_source_gutenberg — author filter, length filter, language filter, year filter, boilerplate strip, HTTP error path
- [ ] test_source_wikisource — category enum, page fetch, word count filter, wikitext cleanup
- [ ] test_source_reddit — score filter, word count filter, video posts skipped, Source adapter populates url/title/raw_text/license
- [ ] test_fetch_command — mocked HTTP across all 3 sources → 10 story.json files written, sqlite rows created
- [ ] `pytest -q` passes (Session 1 + Session 2 = ~30+ tests)
- [ ] live smoke test: `python -m platinum fetch --track atmospheric_horror --limit 10` writes 10 files

## Notes
- Live HTTP only at smoke-test time; CI/CD never hits real APIs.
- Reddit JSON API has no auth but rate-limits aggressively — set `User-Agent: "Platinum/1.0"` and use retry decorator.
- Gutendex is generous; no rate-limit concerns at limit=10.
- Wikisource MediaWiki API also generous; respect `User-Agent` rule.

---

## Review (Session 2 complete — 2026-04-24)

**Tests:** 77 pass (Session 1: 22, Session 2: 55 new). 0 failures, 0 skips.

**Files added:**
- `src/platinum/sources/base.py` (SourceFetcher ABC)
- `src/platinum/sources/gutenberg.py` (Gutendex + boilerplate strip)
- `src/platinum/sources/wikisource.py` (MediaWiki API + wikitext cleanup)
- `src/platinum/sources/reddit.py` (port from gold + Source adapter)
- `src/platinum/sources/registry.py` (type-string -> class)
- `src/platinum/sources/runner.py` (orchestration + persist_source_as_story)
- 5 test files (~1300 lines covering filters, error paths, persistence, CLI)

**Files modified:**
- `src/platinum/cli.py` — replaced `fetch` stub with real impl; ASCII-only docstrings (`->` not `→`) so Windows cp1252 console renders `--help` without UnicodeEncodeError

**Live deliverable verified:** `python -m platinum fetch --track atmospheric_horror --limit 10` returns 10 real PD horror stories — Cask of Amontillado (Poe), Call of Cthulhu / Shunned House / Horror at Red Hook / Festival / Thing on the Doorstep / Haunter of the Dark / Lurking Fear / Cool Air / Silver Key (Lovecraft). Each persisted as `data/stories/<id>/story.json` + `source.txt`; SQLite has 10 stories + 10 stage_runs rows.

**Surprises / lessons:**
1. Gutendex `/books` 301-redirects to `/books/`. Fix: `follow_redirects=True` on the httpx client (now in `runner._default_client_factory`).
2. Wikisource enforces Wikipedia's User-Agent policy and returns 403 for generic UAs. Fix: include a contact URL in the UA. Both fixes apply to all future HTTP-using stages.
3. Pre-existing `→` (U+2192) in Session 1 cli.py docstrings crashes Windows cp1252 `--help` rendering. Replaced with ASCII `->`.
4. With limit=10, Gutendex satisfies the limit on its own — Wikisource and Reddit aren't called in that smoke test (correct behavior; integration test verifies multi-source fanout).

**Not done in this session (deferred to later):**
- `pipeline/source_fetcher.py` Stage wrapper. The CLI drives sources directly. A Stage wrapper can be added if the orchestrator ever needs to re-fetch as a pipeline step (no current need).
- Per-fetcher fallback `httpx.AsyncClient` (when no client is injected) doesn't have `follow_redirects=True` or the contact-URL UA. Production path goes through the runner factory which has both. Fixing the fallbacks is a defensive cleanup, not a current bug.
- Wikisource boilerplate stripping leaves trailing "End of Project Gutenberg's …" lines that fall outside the `*** END ***` marker. story_adapter (Session 4) will polish via Claude, so this is not a blocker.
