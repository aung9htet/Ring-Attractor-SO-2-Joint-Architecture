"""Static files of the graph editor, served by the dashboard server (plan phase 6).

Plain HTML, CSS and JavaScript with SVG; no build step, no dependency.
"""

import os
from typing import Dict, Optional

UI_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_FILES: Dict[str, str] = {
    "editor.html": "text/html; charset=utf-8",
    "editor.js": "application/javascript; charset=utf-8",
    "editor.css": "text/css; charset=utf-8",
}


def static_path(name: str) -> Optional[str]:
    if name not in STATIC_FILES:
        return None
    return os.path.join(UI_DIR, name)


def read_static(name: str) -> Optional[bytes]:
    path = static_path(name)
    if path is None:
        return None
    with open(path, "rb") as stream:
        return stream.read()


__all__ = ["STATIC_FILES", "UI_DIR", "read_static", "static_path"]
