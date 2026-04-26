"""Tag downloaded music or SFX files into the platinum library tag index.

Workflow (manual download, scripted tagging):
  1. Manually download free tracks from YouTube Audio Library, Pixabay Music, or Freesound.
  2. Drop files into library/music/<library_subset>/ or library/sfx/<library_subset>/
     (e.g. library/music/horror/ for the atmospheric_horror track's music subset).
  3. Run this script:  python scripts/curate_music_library.py --kind music --subset horror
  4. For each new untagged file it prompts for mood tags, BPM (music only), and license.
  5. Writes/updates library/music/tags.json (or library/sfx/tags.json).

The selector at runtime queries that JSON to find tracks matching a track config's
`music.moods` or a scene's `sfx_cues`.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
log = logging.getLogger("curate")

ALLOWED_KINDS = {"music", "sfx"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--kind", required=True, choices=sorted(ALLOWED_KINDS), help="music or sfx")
    p.add_argument("--subset", required=True, help="library subset name (e.g. horror, folktales)")
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root (default: parent of scripts/)",
    )
    p.add_argument(
        "--non-interactive", action="store_true",
        help="Skip prompts; only re-scan durations",
    )
    return p.parse_args()


def get_duration_seconds(path: Path) -> float:
    """Use ffprobe to extract duration; returns 0.0 on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError):
        log.warning("Could not probe duration for %s", path.name)
        return 0.0


def prompt_for_metadata(path: Path, kind: str) -> dict:
    """Interactive prompt for one file. Press Enter to accept defaults shown in [].

    Returns a metadata dict ready to write into tags.json.
    """
    print(f"\n--- {path.name} ---")
    moods_raw = input("Moods (comma-separated, e.g. ambient_drone,unease) [skip]: ").strip()
    moods = [m.strip() for m in moods_raw.split(",") if m.strip()] if moods_raw else []

    bpm: int | None = None
    if kind == "music":
        bpm_raw = input("BPM [optional]: ").strip()
        if bpm_raw.isdigit():
            bpm = int(bpm_raw)

    license_str = input("License [CC0]: ").strip() or "CC0"
    source = input("Source URL [optional]: ").strip()
    notes = input("Notes [optional]: ").strip()

    return {
        "moods": moods,
        "bpm": bpm,
        "license": license_str,
        "source": source,
        "notes": notes,
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    kind_dir = root / "library" / args.kind / args.subset
    if not kind_dir.exists():
        log.error("Directory does not exist: %s", kind_dir)
        log.error("Create it and drop audio files in, then re-run.")
        return

    tags_path = root / "library" / args.kind / "tags.json"
    tags: dict[str, dict] = {}
    if tags_path.exists():
        tags = json.loads(tags_path.read_text(encoding="utf-8"))

    audio_files = sorted([p for p in kind_dir.iterdir() if p.suffix.lower() in AUDIO_EXTS])
    log.info("Found %d audio file(s) in %s", len(audio_files), kind_dir)

    for path in audio_files:
        rel_key = f"{args.kind}/{args.subset}/{path.name}"
        existing = tags.get(rel_key)

        if existing and not args.non_interactive:
            print(f"Already tagged: {rel_key}  (moods={existing.get('moods')})")
            redo = input("  Re-tag? [y/N]: ").strip().lower()
            if redo != "y":
                # Refresh duration in case file was replaced
                existing["duration_seconds"] = get_duration_seconds(path)
                continue

        if args.non_interactive:
            tags.setdefault(rel_key, {})
            tags[rel_key]["duration_seconds"] = get_duration_seconds(path)
            continue

        meta = prompt_for_metadata(path, args.kind)
        meta["duration_seconds"] = get_duration_seconds(path)
        meta["filename"] = path.name
        meta["subset"] = args.subset
        tags[rel_key] = meta

    tags_path.write_text(json.dumps(tags, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Wrote %d entries to %s", len(tags), tags_path)


if __name__ == "__main__":
    main()
