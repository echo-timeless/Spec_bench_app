"""Compatibility facade for the focused Pipeline UI modules."""

from .pipeline import (
    render_jobs_page,
    render_launch_page,
    render_results_page,
)

__all__ = ["render_launch_page", "render_jobs_page", "render_results_page"]
