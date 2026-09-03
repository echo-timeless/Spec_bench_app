"""Speculative decoding speedup calculation — pure functions, zero framework dependencies."""

from __future__ import annotations


def compute_avg_accept_length(accept_rates: list[float]) -> float:
    """Compute average accepted token length from per-head accept rates.

    Formula: avg_accept_len = 1 + r1 + r1*r2 + r1*r2*r3 + ...

    Args:
        accept_rates: Per-head acceptance rates, e.g. [0.8, 0.74, 0.67]

    Returns:
        Average accepted token length (always >= 1.0)
    """
    if not accept_rates:
        return 1.0

    cumulative = 1.0  # baseline: always get 1 token from target model
    product = 1.0
    for rate in accept_rates:
        product *= rate
        cumulative += product
    return cumulative


def compute_speedup(
    t_target_ms: float,
    t_target_verify_ms: float,
    t_draft_ms: float,
    accept_rates: list[float],
) -> float:
    """Compute theoretical speedup ratio for speculative decoding.

    Formula: speedup = (t_target * avg_accept_len) / (t_target_verify + t_draft)

    Args:
        t_target_ms: Baseline target model single-step latency (ms)
        t_target_verify_ms: Target model verify step latency (ms), typically 1.0~1.5x of t_target
        t_draft_ms: Total draft model (proposer) latency (ms) for all heads
        accept_rates: Per-head acceptance rates

    Returns:
        Speedup ratio (>1.0 means faster than baseline)
    """
    if t_target_verify_ms + t_draft_ms <= 0:
        return 0.0

    avg_accept_len = compute_avg_accept_length(accept_rates)
    return (t_target_ms * avg_accept_len) / (t_target_verify_ms + t_draft_ms)


def compute_incremental_table(
    t_target_ms: float,
    t_verify_per_step: list[float],
    t_draft_per_step: list[float],
    accept_rates: list[float],
) -> list[dict]:
    """Compute per-step incremental speedup table.

    This mirrors the table in compute_ratio.md showing how adding each
    additional MTP head affects speedup.

    Args:
        t_target_ms: Baseline target model single-step latency (ms)
        t_verify_per_step: Target verify latency for [1-head, 2-head, 3-head, ...]
        t_draft_per_step: Draft latency for [1-head, 2-head, 3-head, ...]
        accept_rates: Per-head acceptance rates [r1, r2, r3, ...]

    Returns:
        List of dicts, one per step:
        {
            "num_draft_tokens": int,
            "t_verify_ms": float,
            "t_draft_ms": float,
            "t_total_ms": float,
            "cumulative_accept_len": float,
            "speedup_vs_baseline": float,
            "speedup_vs_previous": float,
        }
    """
    if not accept_rates:
        return []

    n_steps = min(len(t_verify_per_step), len(t_draft_per_step), len(accept_rates))
    rows: list[dict] = []

    prev_accept_len = 1.0
    prev_total_ms = t_target_ms

    for i in range(n_steps):
        # Cumulative accept length up to step i+1
        current_accept_len = compute_avg_accept_length(accept_rates[: i + 1])

        t_verify = t_verify_per_step[i]
        t_draft = t_draft_per_step[i]
        t_total = t_verify + t_draft

        # Speedup vs baseline: (accept_len / 1) / (t_total / t_target)
        speedup_vs_baseline = (current_accept_len / 1.0) / (t_total / t_target_ms) if t_total > 0 else 0.0

        # Speedup vs previous step
        if i == 0:
            speedup_vs_previous = speedup_vs_baseline
        else:
            prev_row = rows[i - 1]
            ratio_accept = current_accept_len / prev_accept_len if prev_accept_len > 0 else 0.0
            ratio_time = t_total / prev_total_ms if prev_total_ms > 0 else 0.0
            speedup_vs_previous = ratio_accept / ratio_time if ratio_time > 0 else 0.0

        rows.append(
            {
                "num_draft_tokens": i + 1,
                "t_verify_ms": t_verify,
                "t_draft_ms": t_draft,
                "t_total_ms": t_total,
                "cumulative_accept_len": current_accept_len,
                "speedup_vs_baseline": speedup_vs_baseline,
                "speedup_vs_previous": speedup_vs_previous,
            }
        )

        prev_accept_len = current_accept_len
        prev_total_ms = t_total

    return rows
