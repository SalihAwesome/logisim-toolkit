"""Netlist extraction.

Connectivity rules, confirmed against the Logisim JAR (see tests/test_jar_rules.py):
  * wires whose endpoints coincide are connected;
  * a wire endpoint or a component port lying on another wire's interior connects to it;
  * two wires that merely cross (interior to interior) are NOT connected;
  * on load Logisim merges collinear wires (WireRepair), so a point where only wires of
    one orientation end, with wire on both sides of it, vanishes: a perpendicular wire
    passing through it only crosses. Two stubs meeting from opposite sides at a point
    inside another wire therefore do not connect to that wire;
  * tunnels with the same label are connected.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from . import geometry
from .geometry import Port
from .model import Circuit, Component, Point, Project


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class PortRef:
    comp: int  # index into circuit.components
    port: Port  # absolute position

    def describe(self, circ: Circuit) -> str:
        c = circ.components[self.comp]
        lab = f" {c.label!r}" if c.label else ""
        return f"{c.name}{lab}.{self.port.name}@({self.port.x},{self.port.y})"


@dataclass
class Net:
    id: int
    points: set[Point] = field(default_factory=set)
    ports: list[PortRef] = field(default_factory=list)
    wires: list[int] = field(default_factory=list)
    tunnels: set[str] = field(default_factory=set)

    def drivers(self) -> list[PortRef]:
        return [p for p in self.ports if p.port.dir == "out"]

    def sinks(self) -> list[PortRef]:
        return [p for p in self.ports if p.port.dir == "in"]

    def widths(self) -> set[int]:
        return {p.port.width for p in self.ports if p.port.width > 0}


@dataclass
class Netlist:
    circuit: Circuit
    nets: list[Net]
    port_net: dict[tuple[int, str], int]  # (component index, port name) -> net id
    unknown: list[int]  # component indices whose geometry could not be determined

    def net_of(self, comp: int, port: str) -> Net | None:
        nid = self.port_net.get((comp, port))
        return None if nid is None else self.nets[nid]

    def signature(self) -> frozenset:
        """Layout-independent description: each net as the set of (component key, port) it joins.

        A component key is (name, label, sorted attrs minus facing) plus an occurrence
        counter, so two drawings of the same circuit compare equal.
        """
        keys = component_keys(self.circuit)
        sig = set()
        for n in self.nets:
            members = frozenset((keys[p.comp], p.port.name) for p in n.ports)
            if members:
                sig.add(members)
        return frozenset(sig)


_LAYOUT_ATTRS = {"facing", "labelloc"}


def component_keys(circ: Circuit) -> list[tuple]:
    seen: dict[tuple, int] = defaultdict(int)
    keys = []
    for c in circ.components:
        base = (c.name, c.label, tuple(sorted((k, v) for k, v in c.attrs.items() if k not in _LAYOUT_ATTRS)))
        seen[base] += 1
        keys.append(base + (seen[base],))
    return keys


def _index_wires(circ: Circuit):
    horiz: dict[int, list[int]] = defaultdict(list)
    vert: dict[int, list[int]] = defaultdict(list)
    for i, w in enumerate(circ.wires):
        if w.horizontal:
            horiz[w.a[1]].append(i)
        if w.vertical:
            vert[w.a[0]].append(i)
    return horiz, vert


def wires_through(circ: Circuit, p: Point, horiz=None, vert=None) -> list[int]:
    """Indices of wires containing point p (endpoints included)."""
    if horiz is None:
        horiz, vert = _index_wires(circ)
    out = [i for i in horiz.get(p[1], ()) if circ.wires[i].contains(p)]
    out += [i for i in vert.get(p[0], ()) if circ.wires[i].contains(p) and i not in out]
    return out


def _merged_orientation(circ: Circuit, pt: Point, ends: list[int], through: list[int]) -> bool | None:
    """If Logisim's collinear merge erases connection point `pt`, the merged wires'
    orientation (True for horizontal), else None. `pt` must carry no component end."""
    if not ends:
        return None
    horizontal = circ.wires[ends[0]].horizontal
    if any(circ.wires[i].horizontal != horizontal for i in ends):
        return None
    axis = 0 if horizontal else 1
    along = [circ.wires[i] for i in through if circ.wires[i].horizontal == horizontal]
    lo = any(min(w.a[axis], w.b[axis]) < pt[axis] for w in along)
    hi = any(max(w.a[axis], w.b[axis]) > pt[axis] for w in along)
    return horizontal if lo and hi else None


def component_ports(project: Project, circ: Circuit) -> tuple[list[list[Port]], list[int]]:
    all_ports: list[list[Port]] = []
    unknown: list[int] = []
    for i, c in enumerate(circ.components):
        try:
            all_ports.append(geometry.ports(project, c))
        except geometry.UnknownGeometry:
            all_ports.append([])
            unknown.append(i)
    return all_ports, unknown


def extract(project: Project, circ: Circuit | str) -> Netlist:
    if isinstance(circ, str):
        circ = project.circuit(circ)
    uf = UnionFind()
    horiz, vert = _index_wires(circ)
    all_ports, unknown = component_ports(project, circ)

    for i, w in enumerate(circ.wires):
        uf.union(("w", i), w.a)
        uf.union(("w", i), w.b)
    ends: dict[Point, list[int]] = defaultdict(list)
    for i, w in enumerate(circ.wires):
        ends[w.a].append(i)
        ends[w.b].append(i)
    port_points = {p.pos for ps in all_ports for p in ps}
    for pt in set(ends) | port_points:
        through = wires_through(circ, pt, horiz, vert)
        merged = None if pt in port_points else _merged_orientation(circ, pt, ends[pt], through)
        for i in through:
            if merged is None or circ.wires[i].horizontal == merged:
                uf.union(("w", i), pt)

    tunnels: dict[str, list[Point]] = defaultdict(list)
    for ci, c in enumerate(circ.components):
        if c.name == "Tunnel" and c.label and all_ports[ci]:
            tunnels[c.label].append(all_ports[ci][0].pos)
    for label, pts in tunnels.items():
        for pt in pts:
            uf.union(("t", label), pt)

    groups: dict = {}
    nets: list[Net] = []

    def net_for(root) -> Net:
        if root not in groups:
            groups[root] = len(nets)
            nets.append(Net(len(nets)))
        return nets[groups[root]]

    port_net: dict[tuple[int, str], int] = {}
    for ci, ps in enumerate(all_ports):
        for p in ps:
            n = net_for(uf.find(p.pos))
            n.points.add(p.pos)
            n.ports.append(PortRef(ci, p))
            port_net[(ci, p.name)] = n.id
    for i, w in enumerate(circ.wires):
        n = net_for(uf.find(("w", i)))
        n.wires.append(i)
        n.points |= {w.a, w.b}
    for label in tunnels:
        root = uf.find(("t", label))
        if root in groups:
            nets[groups[root]].tunnels.add(label)
    return Netlist(circ, nets, port_net, unknown)


def summary(project: Project, circ: Circuit | str) -> list[dict]:
    nl = extract(project, circ)
    out = []
    for n in nl.nets:
        if not n.ports:
            continue
        out.append({
            "net": n.id,
            "widths": sorted(n.widths()),
            "tunnels": sorted(n.tunnels),
            "drivers": [p.describe(nl.circuit) for p in n.drivers()],
            "ports": [p.describe(nl.circuit) for p in n.ports],
        })
    return out
