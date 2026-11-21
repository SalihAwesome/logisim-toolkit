"""Static checks for .circ files: wiring mistakes and layout problems.

    from logisim import lint
    for issue in lint.lint_file("cpu.circ"):
        print(issue)

Each check is a function in CHECKS (per circuit) or PROJECT_CHECKS (whole file)
that yields Issues. Errors make Logisim misbehave (two drivers on a net, width
conflicts, components sharing a location, which hangs Logisim on load); warnings
are likely mistakes or untidy drawings.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterator

from . import geometry, io, netlist
from .geometry import Bounds
from .model import Circuit, Component, Point, Project, Wire

G = 10

# inputs Logisim gives a sensible default when left unconnected
OPTIONAL_INPUTS = {"reset", "preset", "en", "clr", "cin", "bin", "upper"}
# Counter ports come from the JAR oracle as p0..p7 (out, in, clk, clr, ld, ud, en, carry)
OPTIONAL_POSITIONAL = {"Counter": {"p1", "p3", "p4", "p5", "p6"}}
# components whose outputs may share a net (tri-state)
TRISTATE = {"Controlled Buffer", "Controlled Inverter"}


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str  # "error" | "warning" | "info"
    circuit: str
    message: str
    at: Point | None = None

    def __str__(self) -> str:
        where = f" at ({self.at[0]},{self.at[1]})" if self.at else ""
        return f"{self.circuit}: {self.severity} [{self.code}]{where}: {self.message}"


@dataclass
class LintConfig:
    min_wire_gap: int = 20  # parallel wires of different nets closer than this
    min_component_gap: int = 10  # bodies closer than this
    max_crossings: int = 20  # per circuit


@dataclass
class Ctx:
    project: Project
    circ: Circuit
    config: LintConfig
    nl: netlist.Netlist
    bounds: dict[int, Bounds] = field(default_factory=dict)  # component index -> absolute bounds

    def describe(self, ci: int) -> str:
        c = self.circ.components[ci]
        return f"{c.name} {c.label!r}" if c.label else f"{c.name} at ({c.loc[0]},{c.loc[1]})"

    def issue(self, code: str, severity: str, message: str, at: Point | None = None) -> Issue:
        return Issue(code, severity, self.circ.name, message, at)


def _driven(net: netlist.Net, circ: Circuit) -> bool:
    """A net counts as driven by an output end, or by an inout end other than a tunnel
    (a splitter or pull resistor end may carry a value in)."""
    return any(p.port.dir == "out" or p.port.dir == "inout" and circ.components[p.comp].name != "Tunnel"
               for p in net.ports)


def _consumed(net: netlist.Net, circ: Circuit) -> bool:
    return any(p.port.dir == "in" or p.port.dir == "inout" and circ.components[p.comp].name != "Tunnel"
               for p in net.ports)


def _optional(comp: Component, port: str) -> bool:
    return port in OPTIONAL_INPUTS or port in OPTIONAL_POSITIONAL.get(comp.name, ())


def _overlap(a: Bounds, b: Bounds, gap: int = 0) -> bool:
    """True if the boxes, grown by `gap` on every side, overlap with positive area."""
    return (a[0] - gap < b[0] + b[2] and b[0] < a[0] + a[2] + gap and
            a[1] - gap < b[1] + b[3] and b[1] < a[1] + a[3] + gap)


def _wire_box(w: Wire) -> Bounds:
    (x0, y0), (x1, y1) = w.a, w.b
    return (min(x0, x1), min(y0, y1), abs(x1 - x0) or 0.001, abs(y1 - y0) or 0.001)


# ------------------------------------------------------------------ connectivity

def floating_inputs(ctx: Ctx) -> Iterator[Issue]:
    for net in ctx.nl.nets:
        if _driven(net, ctx.circ):
            continue
        for p in net.sinks():
            comp = ctx.circ.components[p.comp]
            if _optional(comp, p.port.name):
                continue
            what = "output pin" if comp.name == "Pin" else f"input {p.port.name} of {ctx.describe(p.comp)}"
            if comp.name == "Pin":
                what += f" {comp.label!r}"
            yield ctx.issue("floating-input", "warning", f"{what} is not driven", p.port.pos)


def unused_outputs(ctx: Ctx) -> Iterator[Issue]:
    by_comp: dict[int, list[netlist.PortRef]] = defaultdict(list)
    for net in ctx.nl.nets:
        for p in net.drivers():
            by_comp[p.comp].append(p)
    for ci, outs in by_comp.items():
        if any(_consumed(ctx.nl.net_of(ci, p.port.name), ctx.circ) for p in outs):
            continue
        yield ctx.issue("unused-output", "warning", f"nothing reads the output of {ctx.describe(ci)}",
                        outs[0].port.pos)


def multiple_drivers(ctx: Ctx) -> Iterator[Issue]:
    for net in ctx.nl.nets:
        drivers = [p for p in net.drivers() if ctx.circ.components[p.comp].name not in TRISTATE]
        if len(drivers) > 1:
            names = ", ".join(p.describe(ctx.circ) for p in drivers)
            yield ctx.issue("multiple-drivers", "error", f"{len(drivers)} outputs drive one net: {names}",
                            drivers[0].port.pos)


def width_mismatch(ctx: Ctx) -> Iterator[Issue]:
    for net in ctx.nl.nets:
        if len(net.widths()) > 1:
            ends = ", ".join(f"{p.describe(ctx.circ)}={p.port.width}" for p in net.ports if p.port.width)
            yield ctx.issue("width-mismatch", "error", f"bit widths differ on one net: {ends}",
                            net.ports[0].port.pos)


def dangling_wires(ctx: Ctx) -> Iterator[Issue]:
    circ = ctx.circ
    horiz, vert = netlist._index_wires(circ)
    port_points = {p.port.pos for n in ctx.nl.nets for p in n.ports}
    for i, w in enumerate(circ.wires):
        for end in (w.a, w.b):
            others = [j for j in netlist.wires_through(circ, end, horiz, vert) if j != i]
            if not others and end not in port_points:
                yield ctx.issue("dangling-wire", "warning", "wire end connects to nothing", end)


def merged_junctions(ctx: Ctx) -> Iterator[Issue]:
    """Two straight wires meeting on another wire: Logisim merges them and they only cross it."""
    circ = ctx.circ
    horiz, vert = netlist._index_wires(circ)
    ends: dict[Point, list[int]] = defaultdict(list)
    for i, w in enumerate(circ.wires):
        ends[w.a].append(i)
        ends[w.b].append(i)
    port_points = {p.port.pos for n in ctx.nl.nets for p in n.ports}
    for pt, at in ends.items():
        if pt in port_points:
            continue
        through = netlist.wires_through(circ, pt, horiz, vert)
        merged = netlist.merged_orientation(circ, pt, at, through)
        if merged is not None and any(circ.wires[i].horizontal != merged for i in through):
            yield ctx.issue("merged-junction", "warning",
                            "wires meeting here in a straight line are merged by Logisim, so they do not "
                            "connect to the wire they cross; split that wire here or move the junction", pt)


def tunnels(ctx: Ctx) -> Iterator[Issue]:
    counts = Counter(c.label for c in ctx.circ.components if c.name == "Tunnel")
    for c in ctx.circ.components:
        if c.name == "Tunnel" and not c.label:
            yield ctx.issue("unlabeled-tunnel", "warning", "tunnel has no label", c.loc)
        elif c.name == "Tunnel" and counts[c.label] == 1:
            yield ctx.issue("lonely-tunnel", "warning", f"tunnel {c.label!r} has no partner", c.loc)


def pins(ctx: Ctx) -> Iterator[Issue]:
    for c in ctx.circ.pins():
        if not c.label:
            yield ctx.issue("unlabeled-pin", "warning", "pin has no label (tests and subcircuit "
                            "instances need one)", c.loc)
    # Logisim clears one of two non-tunnel labels that are equal ignoring case
    seen: dict[str, Component] = {}
    for c in ctx.circ.components:
        if not c.label or c.name == "Tunnel":
            continue
        other = seen.setdefault(c.label.lower(), c)
        if other is not c:
            yield ctx.issue("duplicate-label", "error", f"{other.name} {other.label!r} and {c.name} "
                            f"{c.label!r} have the same label ignoring case", c.loc)


def unknown_geometry(ctx: Ctx) -> Iterator[Issue]:
    for ci in ctx.nl.unknown:
        c = ctx.circ.components[ci]
        yield ctx.issue("unknown-geometry", "info", f"port positions of {c.name} are unknown; its "
                        "connections were not checked", c.loc)


# ------------------------------------------------------------------ placement

def off_grid(ctx: Ctx) -> Iterator[Issue]:
    for c in ctx.circ.components:
        if c.loc[0] % G or c.loc[1] % G:
            yield ctx.issue("off-grid", "warning", f"{c.name} is off the {G}-unit grid", c.loc)
    for w in ctx.circ.wires:
        for end in (w.a, w.b):
            if end[0] % G or end[1] % G:
                yield ctx.issue("off-grid", "warning", "wire end is off the grid", end)


def overlapping_components(ctx: Ctx) -> Iterator[Issue]:
    comps = ctx.circ.components
    by_loc: dict[Point, int] = {}
    for ci, c in enumerate(comps):
        if c.loc in by_loc:
            yield ctx.issue("same-location", "error", f"{ctx.describe(ci)} and {ctx.describe(by_loc[c.loc])} "
                            "share a location; Logisim hangs loading this file", c.loc)
        by_loc.setdefault(c.loc, ci)
    gap = ctx.config.min_component_gap
    items = sorted(ctx.bounds.items())
    for k, (ci, a) in enumerate(items):
        for cj, b in items[k + 1:]:
            if comps[ci].loc == comps[cj].loc:
                continue
            if _overlap(a, b):
                yield ctx.issue("overlap", "warning", f"{ctx.describe(ci)} overlaps {ctx.describe(cj)}",
                                comps[ci].loc)
            elif _overlap(a, b, gap) and not _touching_ports(ctx, ci, cj):
                yield ctx.issue("too-close", "warning", f"{ctx.describe(ci)} is within {gap} of "
                                f"{ctx.describe(cj)}", comps[ci].loc)


def _touching_ports(ctx: Ctx, ci: int, cj: int) -> bool:
    """Components joined port to port (or by one short stub) are meant to sit together."""
    pi = {ctx.nl.port_net.get((ci, p.name)) for p in _ports(ctx, ci)}
    pj = {ctx.nl.port_net.get((cj, p.name)) for p in _ports(ctx, cj)}
    return bool((pi & pj) - {None})


def _ports(ctx: Ctx, ci: int) -> list[geometry.Port]:
    try:
        return geometry.ports(ctx.project, ctx.circ.components[ci])
    except geometry.UnknownGeometry:
        return []


def wires_through_components(ctx: Ctx) -> Iterator[Issue]:
    for ci, box in ctx.bounds.items():
        own = {p.pos for p in _ports(ctx, ci)}
        for w in ctx.circ.wires:
            if w.a in own or w.b in own:
                continue
            if _overlap(_wire_box(w), box):
                yield ctx.issue("wire-through-component", "warning",
                                f"a wire runs through {ctx.describe(ci)}", w.a)


# ------------------------------------------------------------------ drawing

def _wire_nets(ctx: Ctx) -> dict[int, int]:
    return {wi: n.id for n in ctx.nl.nets for wi in n.wires}


def label_overlaps(ctx: Ctx) -> Iterator[Issue]:
    comps = ctx.circ.components
    labels = {ci: geometry.label_box(comps[ci], box) for ci, box in ctx.bounds.items()}
    labels = {ci: b for ci, b in labels.items() if b is not None}
    for ci, lab in labels.items():
        name = comps[ci].label
        for w in ctx.circ.wires:
            if _overlap(_wire_box(w), lab):
                yield ctx.issue("label-overlap", "warning", f"label {name!r} overlaps a wire", w.a)
                break
        for cj, box in ctx.bounds.items():
            if cj != ci and _overlap(lab, box):
                yield ctx.issue("label-overlap", "warning", f"label {name!r} overlaps {ctx.describe(cj)}",
                                comps[ci].loc)
        for cj, other in labels.items():
            if cj > ci and _overlap(lab, other):
                yield ctx.issue("label-overlap", "warning",
                                f"labels {name!r} and {comps[cj].label!r} overlap", comps[ci].loc)


def wire_spacing(ctx: Ctx) -> Iterator[Issue]:
    """Parallel wires of different nets on nearby grid lines, running side by side. Wires
    that both leave ports of one component (a splitter's ends are 10 apart) are exempt."""
    gap = ctx.config.min_wire_gap
    net_of = _wire_nets(ctx)
    wires = ctx.circ.wires
    port_owner = {p.port.pos: p.comp for n in ctx.nl.nets for p in n.ports}

    def owners(w: Wire) -> set[int]:
        return {port_owner[e] for e in (w.a, w.b) if e in port_owner}

    for horizontal in (True, False):
        axis, span = (1, 0) if horizontal else (0, 1)
        group = sorted((i for i, w in enumerate(wires) if w.horizontal == horizontal and w.a != w.b),
                       key=lambda i: wires[i].a[axis])
        for k, i in enumerate(group):
            wi = wires[i]
            lo_i, hi_i = sorted((wi.a[span], wi.b[span]))
            for j in group[k + 1:]:
                wj = wires[j]
                dist = wj.a[axis] - wi.a[axis]
                if dist >= gap:
                    break
                if dist == 0 or net_of.get(i) == net_of.get(j) or owners(wi) & owners(wj):
                    continue
                lo_j, hi_j = sorted((wj.a[span], wj.b[span]))
                if min(hi_i, hi_j) - max(lo_i, lo_j) > G:
                    kind = "horizontal" if horizontal else "vertical"
                    yield ctx.issue("wire-spacing", "warning",
                                    f"{kind} wires of different nets run {dist} apart", wi.a)


def crossings(ctx: Ctx) -> Iterator[Issue]:
    wires = ctx.circ.wires
    hs = [w for w in wires if w.horizontal and not w.vertical]
    vs = [w for w in wires if w.vertical and not w.horizontal]
    count = 0
    for h in hs:
        y = h.a[1]
        x0, x1 = sorted((h.a[0], h.b[0]))
        for v in vs:
            x = v.a[0]
            y0, y1 = sorted((v.a[1], v.b[1]))
            if x0 < x < x1 and y0 < y < y1:
                count += 1
    if count > ctx.config.max_crossings:
        yield ctx.issue("crossings", "warning", f"{count} wire crossings (limit "
                        f"{ctx.config.max_crossings}); consider tunnels or subcircuits")


CHECKS: list[Callable[[Ctx], Iterator[Issue]]] = [
    floating_inputs, unused_outputs, multiple_drivers, width_mismatch, dangling_wires,
    merged_junctions, tunnels, pins, unknown_geometry, off_grid, overlapping_components,
    wires_through_components, label_overlaps, wire_spacing, crossings,
]


def unused_subcircuits(project: Project) -> Iterator[Issue]:
    used = {c.name for circ in project.circuits for c in circ.components if c.lib is None}
    main = project.main_circuit.name if project.circuits else None
    for circ in project.circuits:
        if circ.name != main and circ.name not in used:
            yield Issue("unused-subcircuit", "warning", circ.name,
                        "circuit is neither the main circuit nor used by another circuit")


PROJECT_CHECKS: list[Callable[[Project], Iterator[Issue]]] = [unused_subcircuits]

_ORDER = {"error": 0, "warning": 1, "info": 2}


def lint(project: Project, circuit: str | None = None, config: LintConfig | None = None) -> list[Issue]:
    """All issues in one circuit, or in every circuit plus project-wide checks."""
    config = config or LintConfig()
    circs = [project.circuit(circuit)] if circuit else project.circuits
    issues: list[Issue] = []
    for circ in circs:
        nl = netlist.extract(project, circ)
        ctx = Ctx(project, circ, config, nl)
        for ci, c in enumerate(circ.components):
            if ci not in nl.unknown:
                ctx.bounds[ci] = geometry.bounds(project, c)
        for check in CHECKS:
            issues.extend(check(ctx))
    if circuit is None:
        for check in PROJECT_CHECKS:
            issues.extend(check(project))
    return sorted(issues, key=lambda i: (i.circuit, _ORDER[i.severity], i.code, i.at or (0, 0)))


def lint_file(path: str | os.PathLike, circuit: str | None = None,
              config: LintConfig | None = None) -> list[Issue]:
    return lint(io.load(path), circuit, config)


def report(issues: list[Issue]) -> str:
    if not issues:
        return "no issues"
    counts = Counter(i.severity for i in issues)
    head = ", ".join(f"{counts[s]} {s}{'s' if counts[s] != 1 else ''}" for s in _ORDER if counts[s])
    return "\n".join([head] + [str(i) for i in issues])
