"""Structural regression tests for the three independent Pipeline pages."""

import ast
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent.parent
PAGE_RENDERERS = {
    "4_benchmark_pipeline.py": (
        "bench_app.ui.pipeline.launch",
        "render_launch_page",
    ),
    "5_benchmark_jobs.py": ("bench_app.ui.pipeline.jobs", "render_jobs_page"),
    "6_benchmark_results.py": (
        "bench_app.ui.pipeline.results",
        "render_results_page",
    ),
}


def test_pipeline_pages_use_explicit_render_entrypoints() -> None:
    for page_name, (expected_module, expected_renderer) in PAGE_RENDERERS.items():
        source = (APP_ROOT / "pages" / page_name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        renderer_calls = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("render_")
        }
        renderer_imports = {
            (node.module, alias.name)
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name.startswith("render_")
        }

        assert renderer_calls == {expected_renderer}
        assert renderer_imports == {(expected_module, expected_renderer)}
        assert "runpy" not in source
        assert "_PIPELINE_PAGE_MODE" not in source


def test_shared_ui_exports_each_page_renderer_without_mode_switching() -> None:
    module_by_renderer = {
        renderer: f"{module.rsplit('.', 1)[-1]}.py"
        for module, renderer in PAGE_RENDERERS.values()
    }
    for renderer, module_name in module_by_renderer.items():
        source = (APP_ROOT / "ui" / "pipeline" / module_name).read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        exported_renderers = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("render_")
        }

        assert exported_renderers == {renderer}
        assert "_PIPELINE_PAGE_MODE" not in source
        assert "_PAGE_MODE" not in source


def test_results_page_has_one_column_selection_entrypoint() -> None:
    source = (APP_ROOT / "ui" / "pipeline" / "results.py").read_text(
        encoding="utf-8"
    )

    assert 'f"对比列（可选 {len(available_keys)} 项）"' in source
    assert "历史记录默认展示列" not in source
    assert "更新标签与展示列" not in source
    assert "_render_key_picker" not in source


def test_launch_page_exposes_required_result_metadata_inline() -> None:
    source = (APP_ROOT / "ui" / "pipeline" / "launch.py").read_text(
        encoding="utf-8"
    )

    assert 'st.markdown("##### 结果名称与配置")' in source
    assert "st.popover(" not in source
    assert "开始发压前请设置结果名称；模型名称为必填项。" not in source
