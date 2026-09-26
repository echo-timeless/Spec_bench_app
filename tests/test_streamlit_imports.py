"""Run Streamlit pages while forbidding imports from the sibling application."""
import builtins
import importlib
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


APP_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("page", [
    "1_speedup_calculator.py",
    "4_benchmark_pipeline.py",
    "5_benchmark_jobs.py",
    "6_benchmark_results.py",
])
def test_pages_run_without_sibling_bench_app(page, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "bench_app" or name.startswith("bench_app."):
            raise AssertionError(f"Imported sibling repository: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    app = AppTest.from_file(str(APP_ROOT / "pages" / page)).run(timeout=30)
    assert not app.exception, [exc.message for exc in app.exception]
    package = __package__.rsplit(".", 1)[0]
    for suffix in ("ui.benchmark_pipeline", "ui.pipeline.launch", "ui.pipeline.jobs",
                   "ui.pipeline.results", "core.pipeline_runtime", "core.pipeline.config"):
        module = importlib.import_module(f"{package}.{suffix}")
        assert Path(module.__file__).resolve().is_relative_to(APP_ROOT)
    config = importlib.import_module(f"{package}.core.pipeline.config")
    assert config.CONFIG_PATH == APP_ROOT / "benchmark_pipeline.toml"
