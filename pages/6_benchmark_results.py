"""Saved SPEED-Bench result lookup and comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bench_app.ui.pipeline.results import render_results_page


render_results_page()
