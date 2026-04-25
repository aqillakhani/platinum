# Platinum — Cinematic AI Short-Film Pipeline

Pipeline that adapts public-domain literature into 3–15 minute cinematic short films across five genre tracks. Track 1 (Atmospheric Horror) ships first.

> **Quality bar:** 10/10 cinematic — indistinguishable from a real animated short film studio's work, not "good for AI."

## Architecture (one-liner)

Public domain text → Claude API adapts to script → split into scenes → Flux.1 Dev keyframes (with IP-Adapter FaceID + ControlNet Depth for continuity) → Wan 2.2 I2V clips → RealESRGAN upscale to 1080p → Chatterbox-Turbo narration cloned from LibriVox reference → curated music + Freesound SFX → Whisper subtitles → FFmpeg assemble + per-clip normalize + LUT grade → human review → YouTube publish.

GPU stages run on a long-lived vast.ai RTX 4090; orchestration, FFmpeg, and review UIs run locally. Three mandatory human review gates: story curation, keyframe approval, final review.

See `short_film_pipeline_prd.md` for the full PRD.

## Project layout

```
config/      track configs, ComfyUI workflows, prompt templates, LUTs
library/     voices/, music/, sfx/ — curated assets with tag metadata
data/        runtime: SQLite DB, per-story directories with intermediates
src/platinum/
  cli.py         Typer CLI entry point
  config.py      YAML + .env loader
  models/        Story dataclass, SQLAlchemy schema
  utils/         shared utilities (Claude client, ComfyUI remote, FFmpeg, validators)
  sources/       Gutenberg, Wikisource, Reddit fetchers
  tracks/        per-track engine implementations
  pipeline/      one module per stage; orchestrator wires them
  publish/       YouTube adapter
  review_ui/     Flask app for keyframe + final review gates
scripts/     one-shot scripts (vast.ai setup, asset acquisition, library curation)
tests/       unit / integration / e2e
```

## Quick start (when a session is built)

```bash
# Install
pip install -e ".[dev]"

# Provision the cloud instance once
bash scripts/vast_setup.sh

# Acquire voice + LUT + music library (one-time)
python scripts/acquire_librivox_voice.py --track atmospheric_horror
python scripts/seed_lut_library.py
python scripts/curate_music_library.py --interactive

# Run the pipeline (Track 1 example)
platinum fetch --track atmospheric_horror --limit 10
platinum curate
platinum render <story_id>      # runs all stages with checkpointing; pauses at each human gate
platinum review keyframes <story_id>   # opens Flask gallery
platinum review final <story_id>       # opens final-review UI
platinum publish <story_id>
```

## Status

Build is incremental, ~16 sessions per the implementation plan. See the plan file at `~/.claude/plans/i-added-a-prd-concurrent-book.md` for the session-by-session breakdown.
