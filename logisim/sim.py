"""Run circuits in the real Logisim (headless) and compare against Python.

    from logisim import sim
    t = sim.truth_table("full_adder.circ")          # rows of {"A": 0, "B": 1, ...}
    r = sim.verify("full_adder.circ", lambda A, B, Cin: {"S": A ^ B ^ Cin, "Cout": (A + B + Cin) >> 1})
    assert r.ok, r.report()

Combinational checks use Logisim's test-vector mode (-w). Sequential checks
(`verify_sequence`) wrap the circuit in a generated harness: a counter steps
through a ROM of input values on each clock edge, and `-tty table` with a
`halt` pin records the outputs after every edge.
"""

from __future__ import annotations

import inspect
import itertools
import os
import random
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import io, jar
from .model import Project

TAB, NL = chr(9), chr(10)
Value = int | str  # str when some bits are unknown ("U"), error ("E") or don't care ("x")


@dataclass
class PinInfo:
    name: str
    width: int


@dataclass
class Table:
    inputs: list[str]
    outputs: list[str]
    rows: list[dict[str, Value]]

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def to_text(self) -> str:
        cols = self.inputs + self.outputs
        lines = [" ".join(cols)]
        for r in self.rows:
            lines.append(" ".join(str(r[c]) for c in cols))
        return "\n".join(lines)


@dataclass
class Failure:
    vector: int
    inputs: dict[str, Value]
    expected: dict[str, Value]
    actual: dict[str, Value]

    def __str__(self) -> str:
        ins = " ".join(f"{k}={v}" for k, v in self.inputs.items())
        diffs = ", ".join(f"{k}: got {self.actual.get(k)} expected {v}"
                          for k, v in self.expected.items() if k in self.actual)
        return f"vector {self.vector} ({ins}): {diffs}"


@dataclass
class VerifyResult:
    passed: int
    failed: int
    failures: list[Failure] = field(default_factory=list)
    exhaustive: bool = True
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.passed > 0

    def report(self, limit: int = 10) -> str:
        head = (f"{'PASS' if self.ok else 'FAIL'}: {self.passed} passed, {self.failed} failed"
                f" ({'exhaustive' if self.exhaustive else 'sampled'})")
        return "\n".join([head] + [f"  {f}" for f in self.failures[:limit]])


class SimError(RuntimeError):
    pass


# ------------------------------------------------------------------ pins

def pin_info(project: Project, circuit: str | None = None) -> tuple[list[PinInfo], list[PinInfo]]:
    circ = project.circuit(circuit) if circuit else project.main_circuit
    ins = [PinInfo(c.label, int(c.get("width", "1"))) for c in circ.inputs() if c.label]
    outs = [PinInfo(c.label, int(c.get("width", "1"))) for c in circ.outputs() if c.label]
    unlabeled = [c for c in circ.pins() if not c.label]
    if unlabeled:
        raise SimError(f"circuit {circ.name!r} has {len(unlabeled)} unlabeled pin(s); label every pin")
    return ins, outs


def parse_value(token: str) -> Value:
    if re.fullmatch(r"[01]+", token):
        return int(token, 2)
    return token


# ------------------------------------------------------------------ truth table

def truth_table(circ_file: str | os.PathLike, circuit: str | None = None, *, timeout: float = 300) -> Table:
    """Logisim's exhaustive truth table (`-tty table`) for a combinational circuit."""
    args = ["-tty", "table"]
    if circuit:
        args += ["--toplevel-circuit", circuit]
    r = jar.run_jar(args + [str(circ_file)], timeout=timeout)
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if not lines:
        raise SimError(f"Logisim produced no table:\n{r.output}")
    proj = io.load(circ_file)
    ins, outs = pin_info(proj, circuit)
    header = lines[0].split()
    in_names = [h for h in header if h in {p.name for p in ins}]
    out_names = [h for h in header if h not in in_names]
    rows = []
    for ln in lines[1:]:
        toks = ln.split()
        if len(toks) != len(header):
            raise SimError(f"unexpected table line {ln!r}")
        rows.append({h: parse_value(t) for h, t in zip(header, toks)})
    return Table(in_names, out_names, rows)


# ------------------------------------------------------------------ test vectors

def _fmt(v: Value, width: int) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return "x" * width
    return format(v & ((1 << width) - 1), f"0{width}b")


def write_vectors(path: str | os.PathLike, inputs: list[PinInfo], outputs: list[PinInfo],
                  rows: list[tuple[dict[str, Value], dict[str, Value]]]) -> Path:
    """Write a Logisim test-vector file. An expected value of None means don't care."""
    def col(p: PinInfo) -> str:
        return p.name if p.width == 1 else f"{p.name}[{p.width}]"
    for p in inputs + outputs:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", p.name):
            raise SimError(f"pin label {p.name!r} cannot be a test-vector column")
    lines = ["# generated by logisim-toolkit", " ".join(col(p) for p in inputs + outputs)]
    for ins, outs in rows:
        cells = [_fmt(ins[p.name], p.width) for p in inputs]
        cells += [_fmt(outs.get(p.name), p.width) for p in outputs]
        lines.append(" ".join(cells))
    path = Path(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def input_space(inputs: list[PinInfo], *, max_exhaustive_bits: int = 14, samples: int = 1024,
                seed: int = 1) -> tuple[list[dict[str, int]], bool]:
    """All input combinations, or a reproducible sample (with corner cases) when there are too many."""
    bits = sum(p.width for p in inputs)
    if bits <= max_exhaustive_bits:
        ranges = [range(1 << p.width) for p in inputs]
        return [dict(zip([p.name for p in inputs], combo)) for combo in itertools.product(*ranges)], True
    rng = random.Random(seed)
    rows = []
    corners = [0, -1, 1]
    for c in corners:
        rows.append({p.name: c & ((1 << p.width) - 1) for p in inputs})
    while len(rows) < samples:
        rows.append({p.name: rng.getrandbits(p.width) for p in inputs})
    return rows, False


def call_reference(fn: Callable, ins: dict[str, int]) -> dict[str, Value]:
    """Call fn with pin values as keyword args (or one dict argument)."""
    params = inspect.signature(fn).parameters
    if len(params) == 1 and next(iter(params.values())).kind == inspect.Parameter.POSITIONAL_ONLY:
        out = fn(ins)
    elif any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()) or \
            set(ins) <= set(params):
        out = fn(**{k: v for k, v in ins.items() if k in params or
                    any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())})
    else:
        out = fn(ins)
    if not isinstance(out, dict):
        raise SimError("reference function must return a dict of output pin values")
    return out


def run_vectors(circ_file: str | os.PathLike, vector_file: str | os.PathLike,
                circuit: str | None = None, *, timeout: float = 300) -> VerifyResult:
    proj = io.load(circ_file)
    name = circuit or proj.main_circuit.name
    r = jar.run_jar(["-w", name, str(vector_file), str(circ_file)], timeout=timeout)
    text = r.stdout + r.stderr
    m = re.search(r"Passed:\s*(\d+),\s*Failed:\s*(\d+)", text)
    if not m:
        raise SimError(f"Logisim did not run the vectors:\n{text.strip()}")
    res = VerifyResult(int(m.group(1)), int(m.group(2)), raw=text)
    vec_lines = [ln for ln in Path(vector_file).read_text().splitlines()
                 if ln.split("#")[0].strip()]
    header = [h.split("[")[0] for h in vec_lines[0].split()]
    # stderr lists "Error on test vector N:"; the "  name = got (expected want)" details
    # go to stdout after the progress line "N"
    failed = [int(k) for k in re.findall(r"Error on test vector (\d+)", text)]
    details: dict[int, dict[str, tuple[str, str]]] = {}
    k = 0
    for ln in r.stdout.splitlines():
        if re.fullmatch(r"\d+", ln.strip()):
            k = int(ln.strip())
            continue
        am = re.match(r"\s+(\S+) = (\S+) \(expected (\S+)\)", ln)
        if am:
            details.setdefault(k, {})[am.group(1)] = (am.group(2), am.group(3))
    in_names = {p.name for p in pin_info(proj, name)[0]}
    for k in failed:
        toks = vec_lines[k].split() if k < len(vec_lines) else []
        got = details.get(k, {})
        res.failures.append(Failure(k, {h: t for h, t in zip(header, toks) if h in in_names},
                                    {n: e for n, (_, e) in got.items()}, {n: a for n, (a, _) in got.items()}))
    return res


def verify(circ_file: str | os.PathLike, expected_fn: Callable, circuit: str | None = None, *,
           max_exhaustive_bits: int = 14, samples: int = 1024) -> VerifyResult:
    """Run the real Logisim on every input combination (or a sample) and compare with expected_fn.

    expected_fn receives the input pin values (as keyword arguments named after
    the pin labels, or a single dict) and returns {output label: value}; omit an
    output or use None for "don't care".
    """
    proj = io.load(circ_file)
    ins, outs = pin_info(proj, circuit)
    space, exhaustive = input_space(ins, max_exhaustive_bits=max_exhaustive_bits, samples=samples)
    rows = [(row, call_reference(expected_fn, row)) for row in space]
    with tempfile.TemporaryDirectory() as td:
        vf = write_vectors(Path(td) / "vectors.txt", ins, outs, rows)
        res = run_vectors(circ_file, vf, circuit)
    res.exhaustive = exhaustive
    return res


# ------------------------------------------------------------------ sequential

@dataclass
class Trace:
    outputs: list[str]
    rows: list[dict[str, Value]]  # one per clock cycle, after the rising edge


def run_sequence(circ_file: str | os.PathLike, steps: list[dict[str, int]], circuit: str | None = None,
                 *, clock: str = "clk", timeout: float = 300) -> Trace:
    """Apply `steps` (input values per cycle) and record outputs after each rising edge.

    Row k is the state after edge k+1: inputs from steps[k] were sampled on that
    edge. Outputs are read while steps[k+1] (or the last step) is applied, which
    matters only for Mealy outputs.
    """
    from .builder import Circuit as BCircuit
    from .builder import Project as BProject

    base = io.load(circ_file)
    name = circuit or base.main_circuit.name
    ins, outs = pin_info(base, name)
    data_ins = [p for p in ins if p.name != clock]
    if clock not in {p.name for p in ins}:
        raise SimError(f"circuit {name!r} has no input pin {clock!r}")
    n = len(steps)
    word = sum(p.width for p in data_ins) or 1
    contents = []
    for st in steps + [steps[-1]]:
        v, shift = 0, 0
        for p in data_ins:
            v |= (st.get(p.name, 0) & ((1 << p.width) - 1)) << shift
            shift += p.width
        contents.append(v)
    aw = max(1, (len(contents) - 1).bit_length())
    hname = next(f"seq_harness{k or ''}" for k in itertools.count() if not base.has_circuit(f"seq_harness{k or ''}"))
    harness = BCircuit(hname, spacing="compact")
    bp = BProject(harness, base=base)
    clk = harness.clock("clk_h")
    cnt = harness.counter(clk, width=aw, on_goal="stay", max_value=n)
    rom = harness.rom(cnt, contents, word)
    pieces = harness.split(rom, [p.width for p in data_ins]) if len(data_ins) > 1 else [rom]
    conns = {p.name: s for p, s in zip(data_ins, pieces)} if data_ins else {}
    conns[clock] = clk
    dut = harness.instance(name, **conns)
    harness.output("step", cnt, width=aw)
    for p in outs:
        harness.output("o_" + p.name, dut[p.name])
    goal = harness.const(n, aw)
    harness.output("halt", harness.comparator(cnt, goal).eq)
    model = bp.to_model()
    model.main = hname
    hc = model.circuit(hname)
    # -tty prints output pins (minus halt) top-down then left-right, tab separated, no header
    out_pins = sorted((c for c in hc.outputs() if c.label != "halt"), key=lambda c: (c.loc[1], c.loc[0]))
    out_cols = [c.label for c in out_pins]
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "harness.circ"
        io.save(model, f)
        r = jar.run_jar(["-tty", "table", "--toplevel-circuit", hname, str(f)], timeout=timeout)
    rows: list[dict[str, Value]] = []
    for ln in r.stdout.splitlines():
        cells = [c.replace(" ", "") for c in ln.split(TAB)]
        if len(cells) != len(out_cols):
            continue
        rows.append({c: parse_value(t) for c, t in zip(out_cols, cells)})
    if not rows or any(not isinstance(row["step"], int) for row in rows):
        raise SimError("harness produced no usable output:" + NL + r.output)
    by_step: dict[int, dict[str, Value]] = {}
    for row in rows:
        by_step[int(row["step"])] = {k[2:]: v for k, v in row.items() if k != "step"}
    trace = [by_step[k] for k in range(1, n + 1) if k in by_step]
    return Trace([p.name for p in outs], trace)


def verify_sequence(circ_file: str | os.PathLike, steps: list[dict[str, int]],
                    expected: list[dict[str, Value]], circuit: str | None = None,
                    *, clock: str = "clk") -> VerifyResult:
    trace = run_sequence(circ_file, steps, circuit, clock=clock)
    res = VerifyResult(0, 0, exhaustive=False)
    for k, exp in enumerate(expected):
        got = trace.rows[k] if k < len(trace.rows) else {}
        bad = {o: v for o, v in exp.items() if v is not None and got.get(o) != v}
        if bad:
            res.failed += 1
            res.failures.append(Failure(k + 1, steps[k], {o: exp[o] for o in bad},
                                        {o: got.get(o) for o in bad}))
        else:
            res.passed += 1
    return res
