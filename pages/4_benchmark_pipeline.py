"""Create and launch a managed SPEED-Bench task."""

import sys
from pathlib import Path

# Streamlit executes pages as scripts, without a package context.
if not __package__:
    _app_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_app_root.parent))
    __package__ = f"{_app_root.name}.pages"

from ..ui.pipeline.launch import render_launch_page


render_launch_page()
