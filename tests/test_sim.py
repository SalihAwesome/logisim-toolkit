"""Simulation through the real Logisim: combinational verify and clocked sequences."""

import pytest

from logisim import sim
from logisim.builder import Project

from test_builder import counter2, full_adder

pytestmark = pytest.mark.jar


def test_verify_full_adder(tmp_path):
    f = tmp_path / "fa.circ"
    Project(full_adder()).save(f)
    r = sim.verify(f, lambda A, B, Cin: {"S": A ^ B ^ Cin, "Cout": (A + B + Cin) >> 1})
    assert r.ok and r.passed == 8, r.report()


def test_verify_reports_failures(tmp_path):
    f = tmp_path / "fa.circ"
    Project(full_adder()).save(f)
    r = sim.verify(f, lambda A, B, Cin: {"S": A | B | Cin})
    assert not r.ok and r.failed == 3  # OR and XOR differ only where exactly two inputs are 1
    assert all(f.expected["S"] != f.actual["S"] for f in r.failures)


def test_run_sequence_counts(tmp_path):
    f = tmp_path / "cnt2.circ"
    Project(counter2()).save(f)
    trace = sim.run_sequence(f, [{}] * 6, "cnt2")
    assert [(r["Q1"], r["Q0"]) for r in trace.rows] == [(0, 1), (1, 0), (1, 1), (0, 0), (0, 1), (1, 0)]


def test_verify_sequence_counter(tmp_path):
    f = tmp_path / "cnt2.circ"
    Project(counter2()).save(f)
    expected = [{"Q0": k & 1, "Q1": (k >> 1) & 1} for k in range(1, 9)]
    r = sim.verify_sequence(f, [{}] * 8, expected, "cnt2")
    assert r.ok and r.passed == 8, r.report()
    wrong = [{"Q0": 0, "Q1": 0}] + expected[1:]
    assert sim.verify_sequence(f, [{}] * 8, wrong, "cnt2").failed == 1
