"""Load Pipeline environment settings and form defaults from TOML."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from bench_app.core.pipeline.models import (
    BenchmarkParameters,
    PipelineError,
    PipelineSettings,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "benchmark_pipeline.toml"


def _resolve_config_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_settings(config_path: str | Path = CONFIG_PATH) -> PipelineSettings:
    """Load the static pipeline environment and form defaults from TOML."""

    path = Path(config_path).resolve()
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    try:
        environment = raw["environment"]
        defaults = raw["defaults"]
        base_dir = path.parent
        workspace_root = _resolve_config_path(
            str(environment["workspace_root"]), base_dir
        )
        saved_results_value = environment.get("saved_results_root")
        saved_results_root = (
            _resolve_config_path(str(saved_results_value), base_dir)
            if saved_results_value
            else workspace_root / "saved_results"
        )
        comparison = raw.get("comparison") or {}
        raw_default_keys = comparison.get("default_keys") or ()
        if isinstance(raw_default_keys, str) or not isinstance(
            raw_default_keys, (list, tuple)
        ):
            raise PipelineError("comparison.default_keys must be a list of keys")
        default_comparison_keys = tuple(
            dict.fromkeys(
                str(key).strip() for key in raw_default_keys if str(key).strip()
            )
        )
        return PipelineSettings(
            python_executable=_resolve_config_path(
                str(environment["python_executable"]), base_dir
            ),
            sglang_repo=_resolve_config_path(str(environment["sglang_repo"]), base_dir),
            dataset_root=_resolve_config_path(
                str(environment["dataset_root"]), base_dir
            ),
            workspace_root=workspace_root,
            saved_results_root=saved_results_root,
            host=str(defaults["host"]),
            port=int(defaults["port"]),
            ready_timeout=int(defaults["ready_timeout"]),
            dataset_size=str(defaults["dataset_size"]),
            category=str(defaults.get("category", "")),
            num_prompts=int(defaults["num_prompts"]),
            output_len=int(defaults["output_len"]),
            max_concurrency=int(defaults["max_concurrency"]),
            request_rate=str(defaults["request_rate"]),
            seed=int(defaults["seed"]),
            warmup_requests=int(defaults["warmup_requests"]),
            extra_request_body=str(defaults.get("extra_request_body", "")),
            output_details=bool(defaults.get("output_details", False)),
            default_comparison_keys=default_comparison_keys,
            builtin_dataset_root=_resolve_config_path(
                str(
                    environment.get(
                        "builtin_dataset_root",
                        base_dir / "datasets",
                    )
                ),
                base_dir,
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"Invalid pipeline config {path}: {exc}") from exc


def default_parameters(settings: PipelineSettings) -> BenchmarkParameters:
    """Create page parameters from the configured defaults."""

    return BenchmarkParameters(
        host=settings.host,
        port=settings.port,
        dataset_size=settings.dataset_size,
        category=settings.category,
        num_prompts=settings.num_prompts,
        output_len=settings.output_len,
        max_concurrency=settings.max_concurrency,
        request_rate=settings.request_rate,
        seed=settings.seed,
        warmup_requests=settings.warmup_requests,
        extra_request_body=settings.extra_request_body,
        output_details=settings.output_details,
        benchmark_family="performance",
        dataset_id="speed-bench",
        dataset_variant=settings.dataset_size,
    )
