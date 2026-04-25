"""Jinja2 prompt-template rendering used by the three Session-4 stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jinja2


def render_template(
    *,
    prompts_dir: Path,
    track: str,
    name: str,
    context: dict[str, Any],
) -> str:
    """Render `<prompts_dir>/<track>/<name>` with `context`.

    Raises FileNotFoundError with the path on miss.
    """
    template_path = Path(prompts_dir) / track / name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_path.parent)),
        autoescape=False,
        keep_trailing_newline=True,
        undefined=jinja2.StrictUndefined,
    )
    return env.get_template(name).render(**context)
