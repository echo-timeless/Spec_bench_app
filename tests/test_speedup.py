"""Tests for core/speedup.py — verified against compute_ratio.md known values."""

import pytest

from ..core.speedup import (
    compute_avg_accept_length,
    compute_incremental_table,
    compute_speedup,
)


class TestComputeAvgAcceptLength:
    def test_empty_rates(self):
        assert compute_avg_accept_length([]) == 1.0

    def test_single_head_80pct(self):
        # 1 + 0.8 = 1.8
        result = compute_avg_accept_length([0.8])
        assert abs(result - 1.8) < 1e-9

    def test_three_heads_uniform_80pct(self):
        # From compute_ratio.md: uniform 80% → 1 + 0.8 + 0.64 + 0.512 = 2.952
        result = compute_avg_accept_length([0.8, 0.8, 0.8])
        assert abs(result - 2.952) < 1e-9

    def test_three_heads_eb5_rates(self):
        # From compute_ratio.md: [0.8, 0.74, 0.67]
        # 1 + 0.8 + 0.8*0.74 + 0.8*0.74*0.67
        # = 1 + 0.8 + 0.592 + 0.39664 = 2.78864
        # Doc says ≈2.79 (rounded)
        result = compute_avg_accept_length([0.8, 0.74, 0.67])
        assert abs(result - 2.78864) < 1e-4

    def test_two_heads_eb5_rates(self):
        # 1 + 0.8 + 0.8*0.74 = 1 + 0.8 + 0.592 = 2.392
        # Doc says 2.39 (rounded to 2dp in table)
        result = compute_avg_accept_length([0.8, 0.74])
        assert abs(result - 2.392) < 1e-9

    def test_perfect_acceptance(self):
        # All heads accept 100%: 1 + 1 + 1 + 1 = 4
        result = compute_avg_accept_length([1.0, 1.0, 1.0])
        assert abs(result - 4.0) < 1e-9

    def test_zero_acceptance(self):
        # First head 0%: 1 + 0 + 0 + 0 = 1
        result = compute_avg_accept_length([0.0, 0.8, 0.8])
        assert abs(result - 1.0) < 1e-9


class TestComputeSpeedup:
    def test_eb5_three_step(self):
        # From compute_ratio.md: t_target=76, t_verify=117, t_draft=22, rates=[0.8,0.74,0.67]
        # speedup = 76 * 2.78864 / (117 + 22) = 212.03 / 139 ≈ 1.525
        # Doc says ≈1.53
        result = compute_speedup(76.0, 117.0, 22.0, [0.8, 0.74, 0.67])
        assert abs(result - 1.525) < 0.01

    def test_baseline_no_spec(self):
        # No speculation: just target alone
        # speedup = t_target * 1 / (t_target + 0) = 1.0
        result = compute_speedup(76.0, 76.0, 0.0, [])
        assert abs(result - 1.0) < 1e-9

    def test_zero_draft_time(self):
        # Zero draft overhead, single head 80%
        # speedup = 76 * 1.8 / (90 + 0) = 136.8 / 90 = 1.52
        result = compute_speedup(76.0, 90.0, 0.0, [0.8])
        assert abs(result - (76 * 1.8 / 90)) < 1e-9

    def test_zero_total_time(self):
        result = compute_speedup(76.0, 0.0, 0.0, [0.8])
        assert result == 0.0

    def test_eb5_single_step(self):
        # From doc: t_verify=90, t_draft=8, accept_len=1.8
        # speedup = 76 * 1.8 / (90 + 8) = 136.8 / 98 ≈ 1.396
        # Doc says 1.39
        result = compute_speedup(76.0, 90.0, 8.0, [0.8])
        assert abs(result - (76 * 1.8 / 98)) < 1e-9


class TestComputeIncrementalTable:
    def test_eb5_example(self):
        # From compute_ratio.md table
        t_target = 76.0
        t_verify_per_step = [90.0, 104.0, 117.0]
        t_draft_per_step = [8.0, 14.0, 22.0]
        accept_rates = [0.8, 0.74, 0.67]

        table = compute_incremental_table(t_target, t_verify_per_step, t_draft_per_step, accept_rates)

        assert len(table) == 3

        # Step 1: accept_len=1.8, total=98, speedup_vs_baseline = 1.8 / (98/76) ≈ 1.396
        assert abs(table[0]["cumulative_accept_len"] - 1.8) < 1e-9
        assert abs(table[0]["t_total_ms"] - 98.0) < 1e-9
        assert abs(table[0]["speedup_vs_baseline"] - (1.8 / (98 / 76))) < 0.01

        # Step 2: accept_len=2.392, total=118, speedup_vs_baseline ≈ 1.54
        assert abs(table[1]["cumulative_accept_len"] - 2.392) < 1e-9
        assert abs(table[1]["t_total_ms"] - 118.0) < 1e-9
        expected_speedup_2 = 2.392 / (118 / 76)
        assert abs(table[1]["speedup_vs_baseline"] - expected_speedup_2) < 0.01

        # Step 3: accept_len≈2.789, total=139, speedup_vs_baseline ≈ 1.525
        assert abs(table[2]["t_total_ms"] - 139.0) < 1e-9
        expected_speedup_3 = table[2]["cumulative_accept_len"] / (139 / 76)
        assert abs(table[2]["speedup_vs_baseline"] - expected_speedup_3) < 0.01

    def test_empty_input(self):
        assert compute_incremental_table(76.0, [], [], []) == []

    def test_single_step(self):
        table = compute_incremental_table(100.0, [120.0], [10.0], [0.9])
        assert len(table) == 1
        assert table[0]["num_draft_tokens"] == 1
        assert abs(table[0]["cumulative_accept_len"] - 1.9) < 1e-9
