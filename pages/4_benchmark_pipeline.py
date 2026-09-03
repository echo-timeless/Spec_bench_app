"""Create and launch a managed SPEED-Bench task."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bench_app.ui.pipeline.launch import render_launch_page


render_launch_page()
