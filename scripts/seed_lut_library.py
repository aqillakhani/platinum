"""Seed the LUT library with free film-look LUTs and a README of attributions.

This script does NOT auto-download. LUT licensing varies — some require account
sign-up, some require attribution, some are CC0. Instead, it prints a manual
acquisition checklist and creates the README skeleton; you drop the .cube files
into config/luts/ yourself.

Recommended free sources:
  - Freepik film LUT packs (free tier, account required)
  - GitHub: search "free-film-luts" for CC0 / MIT-licensed packs
  - DaVinci Resolve built-in (export to .cube via Color tab → right-click LUT)

Per-track recommendations:
  atmospheric_horror   warm-shadow + cool-highlight, low saturation, raised black
  folktales_world_myths   region-dependent — Celtic warm earth, Japanese cool ink, African ochre
  childrens_fables     soft pastel, slight warm shift, no hard contrast
  slice_of_life        Ghibli-leaning natural, soft contrast, slightly cool
  scifi_concept        teal-orange Villeneuve-style, raised contrast, sodium-vapor
"""

from __future__ import annotations

import argparse
from pathlib import Path

CHECKLIST_TEXT = """\
# LUT acquisition checklist

For each track in config/tracks/, drop a `.cube` file at the path referenced by
its `color_grade.lut` setting:

| Track | Expected path | Style direction |
|---|---|---|
| atmospheric_horror | config/luts/atmospheric_horror.cube | warm shadow, cool highlight, low saturation, raised black |
| folktales_world_myths | config/luts/folktales_world_myths.cube | regional — keep neutral until per-tale grading exists |
| childrens_fables | config/luts/childrens_fables.cube | soft pastel, warm shift, low contrast |
| slice_of_life | config/luts/slice_of_life.cube | Ghibli natural, soft contrast, slight cool tint |
| scifi_concept | config/luts/scifi_concept.cube | teal-orange, raised contrast, sodium-vapor amber |

## Verifying a LUT works

```
ffmpeg -i sample_clip.mp4 -vf "lut3d=config/luts/atmospheric_horror.cube" out.mp4
```

If `out.mp4` plays and looks right, the LUT is valid.

## Recording attribution

When you add a LUT, append a row to `config/luts/README.md` with: filename,
source URL, license, attribution required (yes/no), notes.

The platinum pipeline will not apply a LUT it cannot find — `color_grade` stage
fails fast. So unbacked tracks (folktales / fables / slice / scifi) will not run
to publish until their LUT is in place.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root (default: parent of scripts/)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    luts_dir = Path(args.root) / "config" / "luts"
    luts_dir.mkdir(parents=True, exist_ok=True)

    readme = luts_dir / "README.md"
    if not readme.exists():
        readme.write_text(
            "# LUT library\n\n"
            "Cube files referenced by track configs (`color_grade.lut`).\n"
            "Add attribution rows below as you acquire each LUT.\n\n"
            "| Filename | Source URL | License | Attribution required | Notes |\n"
            "|---|---|---|---|---|\n",
            encoding="utf-8",
        )

    checklist_path = luts_dir / "ACQUISITION_CHECKLIST.md"
    checklist_path.write_text(CHECKLIST_TEXT, encoding="utf-8")

    print(f"Wrote checklist to {checklist_path}")
    print(f"Attribution log skeleton at {readme}")
    print()
    print("Drop your .cube files in:", luts_dir)
    print("Track 1 (atmospheric_horror) is the only LUT required to ship v1.0.")


if __name__ == "__main__":
    main()
