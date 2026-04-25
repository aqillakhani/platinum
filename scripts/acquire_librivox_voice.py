"""Acquire a 10-second clean reference clip from a LibriVox public-domain reading.

Usage:
    python scripts/acquire_librivox_voice.py --track atmospheric_horror \
        --url https://archive.org/download/<librivox-zip-or-mp3>/<file>.mp3 \
        --start 00:01:30 --duration 10 \
        --narrator "Phil Chenevert" --title "The Tell-Tale Heart" --year 1843

Behaviour:
    1. Downloads the source audio (mp3/wav).
    2. Extracts a clean 10-second segment with ffmpeg (mono 22kHz, no music/noise).
    3. Saves to library/voices/<track>/reference.wav
    4. Writes library/voices/<track>/source.md with attribution and the URL.

Why not auto-pick: voice selection is a human judgement call (the narrator's tone
must match the track's voice direction). This script automates the *acquisition*,
not the choice. Run after auditioning a few LibriVox readings.

Catalog references for Track 1 (atmospheric horror — slow, measured, low-pitched male):
  - Phil Chenevert's Poe readings: https://librivox.org/author/154
  - Bob Neufeld's gothic readings: https://librivox.org/reader/2275
  - Dale Grothmann's weird fiction: search librivox.org by reader.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
log = logging.getLogger("acquire_librivox_voice")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--track", required=True, help="Track id, e.g. atmospheric_horror")
    p.add_argument("--url", required=True, help="Direct URL to the LibriVox source file (.mp3 or .wav)")
    p.add_argument("--start", default="00:00:30", help="Start timestamp HH:MM:SS (skip intro)")
    p.add_argument("--duration", type=int, default=10, help="Seconds of reference to extract (default 10)")
    p.add_argument("--narrator", required=True, help="Narrator name for attribution")
    p.add_argument("--title", required=True, help="Title of the recording for attribution")
    p.add_argument("--year", type=int, default=0, help="Year of the source text (PD-status check)")
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root (default: parent of scripts/)",
    )
    return p.parse_args()


def download(url: str, dest: Path) -> None:
    log.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def extract_clip(src: Path, dest: Path, start: str, duration: int) -> None:
    log.info("Extracting %ss segment from %s starting at %s", duration, src.name, start)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", start,
        "-i", str(src),
        "-t", str(duration),
        "-ac", "1",
        "-ar", "22050",
        "-acodec", "pcm_s16le",
        str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("ffmpeg failed: %s", result.stderr)
        sys.exit(1)


def write_attribution(dest: Path, *, url: str, narrator: str, title: str, year: int, start: str, duration: int) -> None:
    body = f"""# Voice reference attribution

- **Track:** {dest.parent.name}
- **Source URL:** {url}
- **Narrator:** {narrator}
- **Recording title:** {title}
- **Source year:** {year if year else "unknown"}
- **Segment:** {duration}s starting at {start}
- **Format:** mono 22050 Hz PCM 16-bit
- **License:** Public Domain (LibriVox readings are dedicated to the public domain;
  underlying texts must also be PD)

Used as the reference clip for Chatterbox-Turbo voice cloning. The cloned output is
not the narrator's voice — it is a synthetic voice modeled on this reference.
"""
    dest.write_text(body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    voice_dir = root / "library" / "voices" / args.track
    voice_dir.mkdir(parents=True, exist_ok=True)

    raw = voice_dir / "_source_raw.mp3"
    reference = voice_dir / "reference.wav"
    attribution = voice_dir / "source.md"

    download(args.url, raw)
    extract_clip(raw, reference, args.start, args.duration)
    write_attribution(
        attribution,
        url=args.url,
        narrator=args.narrator,
        title=args.title,
        year=args.year,
        start=args.start,
        duration=args.duration,
    )
    raw.unlink(missing_ok=True)

    log.info("Reference saved: %s", reference)
    log.info("Attribution: %s", attribution)
    log.info("Listen and confirm match with the track's voice direction before using.")


if __name__ == "__main__":
    main()
