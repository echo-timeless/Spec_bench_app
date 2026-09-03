"""Dataset registry and adapters used by the benchmark pipeline.

The serving benchmark only needs prompts.  Task answers and test cases remain
in the source files for a future evaluator and are deliberately not sent to
the model.  Single-turn datasets are materialized into the OpenAI JSONL format
already supported by SGLang's serving benchmark.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from bench_app.core.pipeline.models import PipelineError, PipelineSettings

PERFORMANCE_FAMILY = "performance"
TASK_EVALUATION_FAMILY = "task_evaluation"

SPEED_BENCH_SIZES = ("1k", "2k", "8k", "16k", "32k")
TASK_GROUPS = ("综合", "数学", "代码", "对话与指令")


@dataclass(frozen=True)
class DatasetSpec:
    """A selectable dataset or a dataset variant."""

    dataset_id: str
    label: str
    family: str
    task_group: str | None = None
    source_kind: str = "builtin"
    source_relpath: str = ""
    loader_name: str = "openai"
    variants: tuple[str, ...] = ()
    supported: bool = True
    unsupported_reason: str = ""

    def source_path(self, settings: PipelineSettings, variant: str = "") -> Path:
        if self.source_kind == "speed_bench":
            if variant not in SPEED_BENCH_SIZES:
                raise PipelineError(
                    f"SPEED-Bench variant must be one of {SPEED_BENCH_SIZES}"
                )
            return settings.dataset_root / f"throughput_{variant}" / "test.jsonl"
        if self.source_kind == "speed_bench_qualitative":
            return settings.dataset_root / "qualitative" / "test.jsonl"
        if self.dataset_id == "mbpp" and variant == "sanitized":
            return settings.builtin_dataset_root / "mbpp" / "sanitized-mbpp.json"
        return settings.builtin_dataset_root / self.source_relpath


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_id="speed-bench",
        label="SPEED-Bench",
        family=PERFORMANCE_FAMILY,
        source_kind="speed_bench",
        loader_name="speed-bench",
        variants=SPEED_BENCH_SIZES,
    ),
    DatasetSpec(
        dataset_id="speed-bench-qualitative",
        label="SPEED-Bench Qualitative",
        family=TASK_EVALUATION_FAMILY,
        task_group="综合",
        source_kind="speed_bench_qualitative",
        loader_name="speed-bench",
    ),
    DatasetSpec(
        dataset_id="spec-bench",
        label="Spec-Bench",
        family=TASK_EVALUATION_FAMILY,
        task_group="综合",
        source_relpath="spec_bench/question.jsonl",
        supported=False,
        unsupported_reason="包含多轮样本，当前 BenchAPP 仅支持单轮发压",
    ),
    DatasetSpec(
        dataset_id="gsm8k",
        label="GSM8K",
        family=TASK_EVALUATION_FAMILY,
        task_group="数学",
        source_relpath="gsm8k/test.jsonl",
    ),
    DatasetSpec(
        dataset_id="math500",
        label="MATH-500",
        family=TASK_EVALUATION_FAMILY,
        task_group="数学",
        source_relpath="math500/test.jsonl",
    ),
    DatasetSpec(
        dataset_id="humaneval",
        label="HumanEval",
        family=TASK_EVALUATION_FAMILY,
        task_group="代码",
        source_relpath="humaneval/HumanEval.jsonl",
    ),
    DatasetSpec(
        dataset_id="mbpp",
        label="MBPP",
        family=TASK_EVALUATION_FAMILY,
        task_group="代码",
        source_relpath="mbpp/mbpp.jsonl",
        variants=("full", "sanitized"),
    ),
    DatasetSpec(
        dataset_id="alpaca",
        label="Alpaca",
        family=TASK_EVALUATION_FAMILY,
        task_group="对话与指令",
        source_relpath="alpaca/alpaca_data.json",
    ),
    DatasetSpec(
        dataset_id="mt-bench",
        label="MT-Bench",
        family=TASK_EVALUATION_FAMILY,
        task_group="对话与指令",
        source_relpath="mt_bench/question.jsonl",
        supported=False,
        unsupported_reason="全部样本为多轮对话，当前 BenchAPP 仅支持单轮发压",
    ),
)


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    for spec in DATASET_SPECS:
        if spec.dataset_id == dataset_id:
            return spec
    raise PipelineError(f"Unknown dataset: {dataset_id!r}")


def selectable_dataset_specs(*, task_group: str | None = None) -> tuple[DatasetSpec, ...]:
    return tuple(
        spec
        for spec in DATASET_SPECS
        if spec.supported
        and spec.family == TASK_EVALUATION_FAMILY
        and (task_group is None or spec.task_group == task_group)
    )


def pending_dataset_specs(*, task_group: str | None = None) -> tuple[DatasetSpec, ...]:
    return tuple(
        spec
        for spec in DATASET_SPECS
        if not spec.supported
        and spec.family == TASK_EVALUATION_FAMILY
        and (task_group is None or spec.task_group == task_group)
    )


def resolve_dataset_spec(
    settings: PipelineSettings,
    *,
    dataset_id: str,
    variant: str = "",
) -> tuple[DatasetSpec, Path]:
    spec = get_dataset_spec(dataset_id)
    if not spec.supported:
        raise PipelineError(f"{spec.label} 暂不支持：{spec.unsupported_reason}")
    if spec.variants:
        if variant not in spec.variants:
            raise PipelineError(
                f"{spec.label} variant must be one of {spec.variants}; got {variant!r}"
            )
    elif variant:
        raise PipelineError(f"{spec.label} does not support variant {variant!r}")

    source_path = spec.source_path(settings, variant)
    if not source_path.is_file():
        raise PipelineError(f"Dataset not found: {source_path}")
    if spec.loader_name == "openai":
        return spec, _materialize_openai_dataset(spec, source_path, settings, variant)
    return spec, source_path


def _iter_source_rows(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, list):
            raise PipelineError(f"Expected a JSON list in {path}")
        for row in payload:
            if isinstance(row, dict):
                yield row
        return

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(
                    f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
                ) from exc
            if isinstance(row, dict):
                yield row


def _prompt_for_row(spec: DatasetSpec, row: dict[str, Any], variant: str) -> str:
    if spec.dataset_id == "alpaca":
        instruction = str(row.get("instruction", "")).strip()
        extra_input = str(row.get("input", "")).strip()
        return f"{instruction}\n\n{extra_input}" if extra_input else instruction
    if spec.dataset_id == "gsm8k":
        return str(row.get("question", "")).strip()
    if spec.dataset_id == "math500":
        return str(row.get("problem", "")).strip()
    if spec.dataset_id == "humaneval":
        return str(row.get("prompt", "")).strip()
    if spec.dataset_id == "mbpp":
        key = "text" if variant == "full" else "prompt"
        return str(row.get(key, "")).strip()
    raise PipelineError(f"No prompt adapter for {spec.dataset_id!r}")


def _cache_path(settings: PipelineSettings, spec: DatasetSpec, variant: str) -> Path:
    suffix = f"_{variant}" if variant else ""
    return settings.workspace_root / "dataset_cache" / f"{spec.dataset_id}{suffix}.jsonl"


def _materialize_openai_dataset(
    spec: DatasetSpec,
    source_path: Path,
    settings: PipelineSettings,
    variant: str,
) -> Path:
    cache_path = _cache_path(settings, spec, variant)
    source_mtime = source_path.stat().st_mtime_ns
    if cache_path.is_file() and cache_path.stat().st_mtime_ns >= source_mtime:
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.tmp"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            count = 0
            for row in _iter_source_rows(source_path):
                prompt = _prompt_for_row(spec, row, variant)
                if not prompt:
                    continue
                json.dump(
                    {"messages": [{"role": "user", "content": prompt}]},
                    output,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                output.write("\n")
                count += 1
        if count == 0:
            raise PipelineError(f"No usable prompts found in {source_path}")
        os.replace(temporary_path, cache_path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return cache_path


def dataset_context(dataset_name: str, dataset_path: str | Path) -> dict[str, str]:
    """Infer stable dataset labels from the generated command's path."""

    path = Path(dataset_path)
    context = {"dataset_name": dataset_name}
    if dataset_name == "speed-bench":
        context["dataset_id"] = "speed-bench"
        if path.parent.name == "qualitative":
            context["dataset_id"] = "speed-bench-qualitative"
            context["dataset_variant"] = "qualitative"
            context["benchmark_family"] = TASK_EVALUATION_FAMILY
        elif path.parent.name.startswith("throughput_"):
            context["dataset_variant"] = path.parent.name.removeprefix("throughput_")
            context["benchmark_family"] = PERFORMANCE_FAMILY
    elif path.parent.name == "dataset_cache":
        context["benchmark_family"] = TASK_EVALUATION_FAMILY
        stem = path.stem
        if stem.endswith("_full") or stem.endswith("_sanitized"):
            context["dataset_id"] = "mbpp"
            context["dataset_variant"] = stem.rsplit("_", 1)[-1]
        else:
            context["dataset_id"] = stem
    return context
