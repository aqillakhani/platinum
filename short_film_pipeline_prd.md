# Short Film Pipeline — Product Requirements Document

**Owner:** Aqil
**Status:** Draft v1
**Last updated:** April 2026
**Quality target:** 10/10 cinematic output. Quality is non-negotiable. Cost minimization is secondary.

---

## 1. Vision

Build an automated pipeline that produces 10/10 cinematic short films (3–15 minutes) by adapting public-domain literature, folktales, and curated story sources into narrated, visually rich AI-generated films across five thematic genre tracks. The pipeline is self-hosted-first, designed to drive marginal cost per film toward zero while never compromising on output quality.

The output should be indistinguishable in production value from a high-end animated short film studio's work — not "good for AI," but actually good.

## 2. Success criteria

A film is considered acceptable for publish only if it meets all of the following:

- Visual coherence across all scenes (no jarring style breaks, no obvious AI artifacts in finals)
- Narration sounds like a professional voice actor — no robotic cadence, no flat affect
- Audio mix is balanced: narration intelligible, music supportive not competing, SFX present and grounding
- Color grade is unified across all clips — film looks like one piece, not a stitched assembly
- Story has clear arc: setup, rising action, climax, resolution — even in 3-minute shorts
- No watermarks, no AI-generation tells visible, no continuity errors that pull the viewer out

If a film fails any of these, it does not publish. The pipeline includes mandatory human review gates for v1.

## 3. Out of scope for v1

Explicitly **not** building in v1:

- Character LoRA training for cross-scene character consistency (use reference image chaining instead)
- On-screen lip-synced dialogue (narrator-driven only — no characters speaking on camera)
- Live-action style with recognizable real people
- Multi-language dubbing (English narration only, v1)
- Real-time generation or streaming
- Mobile or web UI for the pipeline (CLI-driven only)
- Automated publishing without human review (v2 feature once quality is consistent)
- Multi-channel cross-promotion automation
- A/B testing infrastructure for thumbnails and titles
- Comments management or community tools

## 4. Genre tracks

The pipeline supports five genre tracks. Each shares the same core architecture but differs in visual style configuration, story sources, voice direction, music selection, and pacing rules.

### Track 1: Atmospheric Horror & Dark Tales **(SHIPS FIRST — v1.0)**

**Why first:** Visual style is the most forgiving of current AI video weaknesses. Shadows, fog, candlelight, and dim interiors hide the artifacts that still plague brightly-lit AI footage. Strong narrator-driven format with massive proven YouTube audience. Public domain corpus is enormous (Poe, Lovecraft, Bierce, Hodgson, M.R. James, Le Fanu, Algernon Blackwood).

**Visual aesthetic:** Cinematic dark, low-key lighting, heavy atmosphere, muted desaturated palette with occasional crimson or amber accent. Inspired by Guillermo del Toro, Mike Flanagan, classic Hammer horror. Film grain heavy.

**Voice direction:** Slow, measured, low-pitched male narrator. Deliberate pauses. No theatrics — the restraint sells the dread.

**Music direction:** Sparse drones, slow strings, single piano notes, ambient unease. No melodic motifs that compete with narration.

**Length target:** 8–12 minutes per film.

**Source priority:**
1. Project Gutenberg (Poe, Lovecraft, Bierce, M.R. James — all public domain)
2. Wikisource translations of foreign weird fiction
3. r/nosleep top-of-all-time (paraphrased and adapted, never verbatim)
4. r/shortscarystories for micro-shorts

### Track 2: Folktales & World Myths

**Visual aesthetic:** Stylized painterly, inspired by Cartoon Saloon (Song of the Sea, The Secret of Kells), traditional Japanese ink-wash for Asian tales, woodcut illustration aesthetic for Northern European tales. Region-specific palette per origin.

**Voice direction:** Warm, storyteller cadence. Older narrator voice. Gentle authority.

**Music direction:** Region-appropriate traditional instrumentation (shakuhachi for Japan, kora for West Africa, fiddle for Celtic, etc.).

**Length target:** 6–10 minutes.

**Source priority:**
1. Public domain folklore collections (Yei Theodora Ozaki, Andrew Lang's Fairy Books, Joseph Jacobs collections)
2. Wikisource translations of folk tale anthologies
3. Cultural archives in the public domain

### Track 3: Children's Fables

**Visual aesthetic:** Warm, soft animation style. Studio Ghibli daytime palette, Pixar lighting principles, gentle character shapes. Bright but never saturated.

**Voice direction:** Female narrator, warm, slightly playful. Clear enunciation. Gentle pacing.

**Music direction:** Acoustic instrumentation — piano, ukulele, gentle strings. Light, hopeful melodies.

**Length target:** 4–8 minutes.

**Source priority:**
1. Aesop's Fables (public domain)
2. Hans Christian Andersen (public domain)
3. Brothers Grimm — softer tales only (public domain)
4. Beatrix Potter (public domain in most jurisdictions)
5. Cleaned-up versions of folktales suitable for children

### Track 4: Slice-of-Life Meditative Shorts

**Visual aesthetic:** Ghibli-inspired, soft-focus, painterly. Heavy emphasis on environment over character. Rain on windows, steam from tea, light through trees, snow falling. Long, slow shots.

**Voice direction:** Soft female or gentle male narrator. Contemplative pacing. Room for silence.

**Music direction:** Lo-fi, ambient piano, ASMR-adjacent. Joe Hisaishi-influenced.

**Length target:** 5–15 minutes (built for sleep / study audience).

**Source priority:**
1. Original short adaptations of Japanese slice-of-life literature in the public domain
2. Translated short fiction from Wikisource (Kawabata-era and earlier)
3. Curated "calm moment" vignettes adapted from longer public domain works

### Track 5: Sci-Fi Concept Pieces

**Visual aesthetic:** Cinematic sci-fi, Denis Villeneuve / Blade Runner 2049 / Arrival influence. Dramatic lighting, scale, atmospheric haze. Alien environments, far-future technology, abstract megastructures.

**Voice direction:** Clear, measured, slightly cold. Could be male or female. Documentary-like authority.

**Music direction:** Synth-driven, ambient, drones with melodic motifs. Ben Frost / Jóhann Jóhannsson influence.

**Length target:** 6–12 minutes.

**Source priority:**
1. Public domain pulp sci-fi (early Asimov, classic short stories from pre-1929 sources)
2. r/HFY top stories (paraphrased and adapted)
3. Original concept-piece adaptations from public domain hard SF

## 5. System architecture

The pipeline is a directed graph of modules. Each module reads from and writes to a structured `Story` object that flows through the system.

```
[1] source_fetcher
        ↓
[2] story_curator (human review checkpoint)
        ↓
[3] story_adapter (Claude API)
        ↓
[4] scene_breakdown (Claude API)
        ↓
[5] visual_prompt_generator (Claude API)
        ↓
[6] keyframe_generator (Flux Dev)
        ↓
[7] keyframe_review (human review checkpoint)
        ↓
[8] video_generator (Wan 2.2 I2V)
        ↓
[9] voice_generator (Chatterbox-Turbo)
        ↓
[10] music_selector (royalty-free library)
        ↓
[11] sfx_layer (Freesound library)
        ↓
[12] subtitle_generator (Whisper)
        ↓
[13] assembly_engine (FFmpeg)
        ↓
[14] color_grade (FFmpeg + LUT)
        ↓
[15] final_review (human review checkpoint)
        ↓
[16] thumbnail_generator (Flux Dev + PIL)
        ↓
[17] publisher (YouTube API, manual trigger v1)
```

Each module operates on the `Story` object, enriches it, and passes it forward. Failures at any stage halt the pipeline for that story without affecting others.

## 6. Data model

The single source of truth is the `Story` object, persisted as JSON on disk per story.

```json
{
  "id": "story_2026_04_24_001",
  "track": "atmospheric_horror",
  "source": {
    "type": "project_gutenberg",
    "url": "https://www.gutenberg.org/...",
    "title": "The Tell-Tale Heart",
    "author": "Edgar Allan Poe",
    "raw_text": "...",
    "fetched_at": "2026-04-24T14:00:00Z"
  },
  "adapted": {
    "title": "The Tell-Tale Heart",
    "synopsis": "...",
    "narration_script": "...",
    "estimated_duration_seconds": 540,
    "tone_notes": "Increasing paranoia, narrator's mental state degrading"
  },
  "scenes": [
    {
      "id": "scene_001",
      "narration_text": "True! nervous, very, very dreadfully nervous I had been...",
      "narration_audio_path": "/data/story_001/audio/scene_001.wav",
      "narration_duration_seconds": 8.4,
      "visual_prompt": "Close-up of pale hands trembling on a wooden table, single candle flame, dark Victorian interior, oil painting style, chiaroscuro lighting, film grain",
      "negative_prompt": "...",
      "keyframe_path": "/data/story_001/keyframes/scene_001.png",
      "video_path": "/data/story_001/clips/scene_001.mp4",
      "video_duration_seconds": 8,
      "music_cue": "ambient_drone_low",
      "sfx_cues": ["clock_ticking_distant", "wind_through_window"],
      "review_status": "approved"
    }
  ],
  "audio": {
    "narrator_voice_id": "horror_male_01",
    "music_track_path": "/library/music/horror/drone_unease_01.mp3",
    "final_mix_path": "/data/story_001/audio/full_mix.wav"
  },
  "video": {
    "assembled_path": "/data/story_001/output/assembled.mp4",
    "graded_path": "/data/story_001/output/graded.mp4",
    "final_path": "/data/story_001/output/final.mp4",
    "duration_seconds": 547,
    "resolution": "1920x1080",
    "fps": 24
  },
  "publish": {
    "title": "The Tell-Tale Heart | A Tale of Madness by Edgar Allan Poe",
    "description": "...",
    "tags": ["horror", "classic literature", "edgar allan poe", "narrated story"],
    "thumbnail_path": "/data/story_001/output/thumb.jpg",
    "published_url": null,
    "published_at": null
  },
  "review_gates": {
    "story_curated": true,
    "keyframes_approved": true,
    "final_approved": false
  }
}
```

## 7. Module specifications

### Module 1: source_fetcher
**Purpose:** Pull raw stories from configured sources.
**Inputs:** Track configuration (which sources, filters).
**Outputs:** Raw `Story` objects with `source` field populated.
**Implementation:** Python scripts per source type. Project Gutenberg via official feeds, Wikisource via API, Reddit via PRAW, simple HTTP scraping for public archives.
**Success criteria:** Returns at least 10 candidate stories per run, each meeting length filters.

### Module 2: story_curator (HUMAN REVIEW)
**Purpose:** Human picks which stories advance.
**Implementation:** CLI tool that lists candidate stories with metadata, shows preview, accepts y/n/skip per story.
**Success criteria:** Curator can review 20 candidates in under 10 minutes.

### Module 3: story_adapter
**Purpose:** Take raw source text, produce a polished narration script.
**Inputs:** `Story.source.raw_text`, track configuration (tone, length target, voice direction).
**Outputs:** `Story.adapted` populated.
**Implementation:** Single Claude API call with carefully designed prompt per genre track. Prompt includes target length, tone, pacing rules, narrative structure requirements.
**Success criteria:** Output reads as a coherent narration script with clear arc.

### Module 4: scene_breakdown
**Purpose:** Split narration script into scenes with timing.
**Inputs:** `Story.adapted.narration_script`.
**Outputs:** `Story.scenes` populated with narration_text, estimated duration, music/sfx cues.
**Implementation:** Claude API call. Each scene should be 5–10 seconds of narration aligning to one visual.
**Success criteria:** Total scene duration within 5% of target film length.

### Module 5: visual_prompt_generator
**Purpose:** Generate Flux-optimized image prompts per scene.
**Inputs:** `Story.scenes[].narration_text`, track visual aesthetic config.
**Outputs:** `Story.scenes[].visual_prompt` and `negative_prompt` populated.
**Implementation:** Claude API call with track-specific style guide injected. Prompts use natural language (Flux convention, not SDXL tag-style).
**Success criteria:** Prompts include subject, action, environment, lighting, style, mood, technical qualifiers.

### Module 6: keyframe_generator
**Purpose:** Generate one high-quality keyframe per scene.
**Inputs:** `Story.scenes[].visual_prompt`.
**Outputs:** `Story.scenes[].keyframe_path` populated.
**Implementation:** ComfyUI workflow with Flux.1 Dev. Generate 3 candidates per scene, auto-select best via aesthetic scoring model, fall back to first if scoring fails.
**Success criteria:** Each keyframe is 1920x1080, no obvious anatomical errors, matches genre aesthetic.

### Module 7: keyframe_review (HUMAN REVIEW)
**Purpose:** Human approves keyframes before expensive video generation step.
**Implementation:** Web-based gallery viewer (simple Flask app) showing all scenes side-by-side with prompts. Approve / reject / regenerate per scene.
**Success criteria:** Reviewer can approve a 60-scene film in under 15 minutes.

### Module 8: video_generator
**Purpose:** Animate each approved keyframe into a video clip.
**Inputs:** `Story.scenes[].keyframe_path`, narration duration per scene.
**Outputs:** `Story.scenes[].video_path` populated.
**Implementation:** ComfyUI workflow with Wan 2.2 I2V. Clip duration matches narration duration (5–10s typical). Last frame of clip N saved as reference for clip N+1's first frame to maintain continuity where scenes connect.
**Success criteria:** Smooth motion, no flickering, no temporal artifacts, matches keyframe style.

### Module 9: voice_generator
**Purpose:** Generate narration audio.
**Inputs:** `Story.scenes[].narration_text`, track voice configuration.
**Outputs:** `Story.scenes[].narration_audio_path` populated.
**Implementation:** Chatterbox-Turbo locally with cloned narrator voice from a 10-second high-quality reference. Per-track voice profile.
**Success criteria:** Natural cadence, correct emotion per track, no glitches or mispronunciations on a sample of 10 random sentences.

### Module 10: music_selector
**Purpose:** Select appropriate background music from curated library.
**Inputs:** Track configuration, scene mood cues, total film duration.
**Outputs:** `Story.audio.music_track_path` populated.
**Implementation:** Tagged music library on disk. Selector matches by genre track + mood. May concatenate multiple tracks for longer films with mood transitions.
**Success criteria:** Music fits scene mood, supports rather than competes with narration.

### Module 11: sfx_layer
**Purpose:** Layer ambient and specific sound effects per scene.
**Inputs:** `Story.scenes[].sfx_cues`, scene timing.
**Outputs:** SFX track aligned to film timeline.
**Implementation:** Tagged Freesound library. Cues like "clock_ticking_distant" match a tag in the library, pick first random match.
**Success criteria:** SFX present but never overpowering narration.

### Module 12: subtitle_generator
**Purpose:** Generate SRT subtitle file from final narration.
**Inputs:** Final narration audio.
**Outputs:** SRT file aligned to video.
**Implementation:** Whisper large-v3 locally.
**Success criteria:** 99%+ accuracy on narration audio, correctly timed within 200ms.

### Module 13: assembly_engine
**Purpose:** Stitch all video clips, mix all audio, burn subtitles.
**Inputs:** All clips, narration, music, SFX, subtitles.
**Outputs:** `Story.video.assembled_path` populated.
**Implementation:** FFmpeg complex filtergraph, scripted in Python. Audio levels balanced per track config (narration -6dB, music -18dB, sfx -12dB as defaults).
**Success criteria:** Clean assembly, correct timing, balanced audio.

### Module 14: color_grade
**Purpose:** Apply unified color grade across full film.
**Inputs:** Assembled video, track-specific LUT.
**Outputs:** `Story.video.graded_path` populated.
**Implementation:** FFmpeg LUT application. Per-track LUT in `/library/luts/`.
**Success criteria:** Film feels cohesive, color tells story, no clip stands out as unprocessed.

### Module 15: final_review (HUMAN REVIEW)
**Purpose:** Final pass before publish. Catches anything the automation missed.
**Implementation:** Plays graded video, checklist UI: visual cohesion, audio mix, narration quality, story arc, no artifacts, ready to publish.
**Success criteria:** Reviewer can complete in under one viewing of the film.

### Module 16: thumbnail_generator
**Purpose:** Generate compelling thumbnail.
**Inputs:** Best keyframe (manually selected or top-scored), title text.
**Outputs:** `Story.publish.thumbnail_path` populated.
**Implementation:** Flux Dev for new framing if needed, PIL for text overlay with track-specific typography.
**Success criteria:** Thumbnail is visually striking at small sizes, readable text, intrigue without clickbait.

### Module 17: publisher
**Purpose:** Upload to YouTube.
**Implementation:** YouTube Data API. v1 is manual trigger only after final review.
**Success criteria:** Successful upload with all metadata, scheduled or immediate per config.

## 8. Infrastructure

### Hosting strategy
**Primary:** Self-hosted on local hardware where capable, vast.ai overflow for heavy generation.
**Fallback:** fal.ai or Replicate API for any module on demand if local infrastructure is unavailable.

### Hardware requirements
- **Minimum local:** RTX 4090 24GB or RTX 3090 24GB for Flux Dev + Wan 2.2 I2V
- **Acceptable local:** RTX 4070 Ti 12GB+ with quantized models (Flux GGUF Q8, Wan 2.2 GGUF)
- **Vast.ai overflow:** RTX 4090 instances for batch processing
- **Storage:** ~5GB per finished film including all intermediates

### Software stack
- Python 3.11+
- ComfyUI (latest) for image/video generation
- Chatterbox-Turbo for voice
- Whisper large-v3 for subtitles
- FFmpeg for assembly
- Claude API for adaptation/scripting
- Project structure: standard Python package with config files per genre track

### Configuration
All track-specific config (visual style, voice profile, music library subset, LUT path, length target, tone notes) lives in YAML files under `/config/tracks/`. Adding a new genre track means adding one YAML file plus matching library content.

## 9. Quality gates

Three mandatory human review checkpoints in v1:

1. **Story curation gate** — before script adaptation
2. **Keyframe approval gate** — before video generation (most expensive step)
3. **Final review gate** — before publish

These gates exist because (a) AI pipelines fail silently and bad output looks superficially like good output, (b) the cost of catching a problem early is far less than the cost of regenerating a finished film, (c) consistent quality is the entire competitive moat.

v2 may add automated quality scoring to reduce manual review burden, but v1 ships with full human-in-the-loop.

## 10. Cost model

Target cost per film, fully self-hosted on owned hardware:
- Claude API (script + scenes + visual prompts): $1–3
- Electricity for local GPU: negligible
- Music & SFX library: amortized across all films (one-time setup cost)
- **Per-film marginal cost: under $3**

Target cost per film on vast.ai (when local hardware unavailable):
- GPU rental for full generation cycle: $4–8
- Claude API: $1–3
- **Per-film marginal cost: under $11**

## 11. Build plan — Claude Code sessions

Each session is a discrete Claude Code task with a clear deliverable. Git commit between each. Validate working before moving to next.

### Session 1: Project scaffolding
- Project structure, dependencies, config system, logging, base Story dataclass with serialization
- Genre track YAML schema, atmospheric_horror.yaml as first config
- **Deliverable:** Empty pipeline with track config loading, `Story` model with save/load to JSON

### Session 2: Source fetchers
- Project Gutenberg fetcher (highest priority, cleanest source)
- Wikisource fetcher
- Reddit fetcher (PRAW-based)
- **Deliverable:** Can run `fetch --track atmospheric_horror --limit 10` and get 10 candidate stories saved to disk

### Session 3: Story curator CLI
- CLI tool to review fetched candidates, approve/reject
- Persists curation decisions
- **Deliverable:** Interactive curation flow, marks stories as approved for next stage

### Session 4: Story adapter + scene breakdown
- Claude API integration
- Track-specific adaptation prompts
- Scene breakdown with timing estimation
- **Deliverable:** Approved story → polished narration script → scene list with prompts

### Session 5: Visual prompt generator
- Per-scene Flux-optimized prompt generation
- Track aesthetic config injected into prompts
- **Deliverable:** Each scene has a strong visual prompt ready for image generation

### Session 6: Keyframe generation via ComfyUI
- ComfyUI integration (API-based or subprocess)
- Flux.1 Dev workflow
- Multi-candidate generation with selection logic
- **Deliverable:** Each scene has a generated keyframe at 1920x1080

### Session 7: Keyframe review web UI
- Simple Flask app showing scene gallery
- Approve / reject / regenerate per scene
- **Deliverable:** Browser-based gate for keyframe approval

### Session 8: Video generation via Wan 2.2 I2V
- ComfyUI Wan 2.2 workflow integration
- Per-scene clip generation matching narration duration
- Frame-chaining for continuity between adjacent scenes
- **Deliverable:** Each approved keyframe becomes a video clip

### Session 9: Voice generation via Chatterbox-Turbo
- Chatterbox local installation and integration
- Per-track narrator voice profile
- Per-scene narration audio generation
- **Deliverable:** Each scene has matching narration audio

### Session 10: Music + SFX layering
- Library structure on disk with tag metadata
- Selection logic per track and mood
- Audio track preparation for assembly
- **Deliverable:** Full film has music bed and SFX layer ready

### Session 11: Assembly engine
- FFmpeg orchestration in Python
- Audio mixing with proper levels
- Subtitle generation via Whisper
- Subtitle burn-in
- **Deliverable:** All elements combined into a single MP4

### Session 12: Color grade + final review
- LUT application via FFmpeg
- Final review CLI/web checklist
- **Deliverable:** Graded final film ready for publish review

### Session 13: First end-to-end run
- Pick one Poe story, run entire pipeline manually through every gate
- Document failure points, jank, quality issues
- **Deliverable:** One complete published-quality film

### Session 14: Thumbnail + publisher
- Thumbnail generation with text overlay
- YouTube Data API integration
- Manual-trigger publish
- **Deliverable:** Can publish a finished film with one command

### Session 15: Polish, iteration, second film
- Whatever needs fixing from Session 13 learnings
- Run a second film fully through to confirm repeatability
- **Deliverable:** Two finished films, repeatable process

## 12. v2 considerations (not for now)

Listed for context only. Do not build until v1 is shipping films consistently.

- Automated quality scoring to reduce manual review
- Multi-channel orchestration across all five genre tracks
- Character LoRA training for recurring narrators or characters
- Multi-language adaptation
- Long-form content (30+ minute compilations for sleep audience)
- Live narrator voice replacement via Chatterbox cloning
- Performance optimization for batch overnight processing
- A/B testing for thumbnails and titles
- Analytics ingestion to inform future story selection

## 13. Open questions to resolve before Session 1

- Confirm narrator voice reference source for Track 1 (need a 10-second clean recording in target style)
- Confirm music library acquisition strategy (curate manually from YouTube Audio Library + Pixabay, or invest in Artlist/Epidemic subscription)
- Confirm review UI preference: CLI vs lightweight web app for the three gates
- Confirm local hardware capability or default to vast.ai from session 6 onward

## 14. Definition of done for v1.0

- Pipeline can take an Edgar Allan Poe story from Project Gutenberg and produce a published-quality 8–12 minute atmospheric horror film
- Two complete films published to YouTube as proof
- All 17 modules functional
- Three review gates operational
- Per-film marginal cost confirmed under $11 (vast.ai) or under $3 (local)
- Process documented well enough that running another film is mechanical
