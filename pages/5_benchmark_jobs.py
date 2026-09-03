"""SPEED-Bench managed jobs and live logs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from bench_app.ui.pipeline.jobs import render_jobs_page


render_jobs_page()
