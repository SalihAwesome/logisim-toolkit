"""Edit existing Logisim .circ projects in memory.

Every operation works on a :class:`logisim.model.Project` and only changes the
circuit drawing (components, wires and attributes).  Saving is deliberately
separate: use :func:`logisim.io.save` (or :func:`save`) when the edit is
finished, so an input file is never overwritten by accident.

The heavy lifting is reused from the other toolkit modules: port geometry comes
from :mod:`logisim.geometry`, connectivity from :mod:`logisim.netlist`, routing
from :mod:`logisim.route` and whole-circuit placement from
:mod:`logisim.layout`.  This module only decides *what* to change.

Every operation that changes a drawing keeps the circuit's connectivity intact:
the ports of a moved, rotated or re-attributed component are put back on the
nets they were on before, and the netlist partition is checked afterwards.

Public API (all functions raise :class:`EditError` on invalid input)::

    list_components(project, circuit) -> list[ComponentInfo]
    find_components(project, circuit, *, name=None, label=None) -> list[ComponentInfo]
    find_component(project, circuit, *, name=None, label=None) -> ComponentInfo

    add_component(project, circuit, name, *, lib=None, attrs=None, loc="auto",
                  connections=None) -> ComponentInfo
    delete_component(project, circuit, target) -> ComponentInfo
    move_component(project, circuit, target, *, loc=None, dx=0, dy=0, force=False) -> ComponentInfo
    rotate_component(project, circuit, target, *, facing=None, turns=1) -> ComponentInfo
    set_attribute(project, circuit, target, key, value, *, force=False) -> ComponentInfo
    set_attributes(project, circuit, target, attrs, *, force=False) -> ComponentInfo

    rename_circuit(project, old, new) -> Circuit
    rename_pin(project, circuit, old_label, new_label) -> ComponentInfo
    add_circuit(project, name, *, attrs=None, main=False) -> Circuit
    add_instance(project, circuit, subcircuit, *, label="", attrs=None,
                 loc="auto", connections=None) -> ComponentInfo

    insert_gate(project, circuit, net, name="NOT Gate", *, lib="gates",
                attrs=None, input_port=None, output_port=None, sinks=None,
                extra_inputs=None, loc="auto") -> ComponentInfo

    delete_wires(project, circuit, *, wires=None, endpoints=None, point=None)
    cleanup_wires(project, circuit) -> list[Wire]

    merge_projects(target, source, *, circuits=None, rename=None,
                   prefix=None) -> dict[str, str]
    tidy(project, circuit, *, spacing="roomy") -> Circuit

    save(project, path, *, in_place=False) -> Path
"""

from __future__ import annotations

import copy
import itertools
import re
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from . import builder, geometry, io, layout, netlist, route
from .model import Circuit, Component, Point, Project, Wire

G = 10
FACINGS = ("east", "north", "west", "south")
_ORTHOGONAL = {"east": (G, 0), "west": (-G, 0), "north": (0, -G), "south": (0, G)}
_GEOMETRY_NEUTRAL = {"label", "labelloc", "labelfont", "labelcolor", "labelvisible"}


class EditError(ValueError):
    """An edit could not be carried out safely."""


# --------------------------------------------------------------------------- info


@dataclass(frozen=True)
class ComponentInfo:
    """Everything needed to identify a component in a circuit.

    ``index`` is its position in ``circuit.components`` at the time of the
    query; ``loc``/``facing`` are the Logisim location and facing attribute
    (a component without a facing attribute faces east).
    """

    index: int
    name: str
    label: str
    loc: Point
    facing: str
    lib: str | None = None
    attrs: dict[str, str] = field(default_factory=dict, compare=False)

    @property
    def location(self) -> Point:
        """Alias for ``loc`` (the wording used in the task description)."""
        return self.loc

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lab = f" {self.label!r}" if self.label else ""
        return f"#{self.index} {self.name}{lab} @({self.loc[0]},{self.loc[1]}) {self.facing}"


def _info(circ: Circuit, index: int) -> ComponentInfo:
    c = circ.components[index]
    facing = c.attrs.get("facing") or "east"
    return ComponentInfo(index, c.name, c.label, c.loc, facing, c.lib, dict(c.attrs))


def _as_circuit(project: Project, circuit: Circuit | str) -> Circuit:
    if isinstance(circuit, Circuit):
        if circuit in project.circuits:
            return circuit
        if project.has_circuit(circuit.name):
            return project.circuit(circuit.name)
        raise EditError(f"circuit {circuit.name!r} is not part of this project")
    try:
        return project.circuit(circuit)
    except KeyError as e:
        raise EditError(str(e)) from None


def _index_of(circ: Circuit, target) -> int:
    if isinstance(target, ComponentInfo):
        i = target.index
        if 0 <= i < len(circ.components):
            return i
        raise EditError(f"component index {i} is out of range in {circ.name!r}")
    if isinstance(target, Component):
        for i, c in enumerate(circ.components):
            if c is target:
                return i
        for i, c in enumerate(circ.components):
            if c == target:
                return i
        raise EditError(f"{target!r} is not in circuit {circ.name!r}")
    if isinstance(target, int):
        if 0 <= target < len(circ.components):
            return target
        raise EditError(f"component index {target} is out of range in {circ.name!r}")
    if isinstance(target, str):
        by_label = [i for i, c in enumerate(circ.components) if c.label == target]
        if len(by_label) == 1:
            return by_label[0]
        if len(by_label) > 1:
            raise EditError(f"several components are labelled {target!r} in {circ.name!r}")
        by_name = [i for i, c in enumerate(circ.components) if c.name == target]
        if len(by_name) == 1:
            return by_name[0]
        if len(by_name) > 1:
            raise EditError(f"several components are named {target!r} in {circ.name!r}; "
                            f"use an index, a label or {target}#N (the Nth one, from 0)")
        occurrence = re.fullmatch(r"(.+)#([0-9]+)", target)
        if occurrence:
            of_type = [i for i, c in enumerate(circ.components) if c.name == occurrence[1]]
            n = int(occurrence[2])
            if n < len(of_type):
                return of_type[n]
            raise EditError(f"{circ.name!r} has {len(of_type)} components named "
                            f"{occurrence[1]!r}; {target!r} is out of range")
        raise EditError(f"no component labelled or named {target!r} in {circ.name!r}")
    raise EditError(f"cannot use {target!r} as a component reference")


def _resolve_port(project: Project, circ: Circuit, target, port: str | None = None):
    """Return ``(component index, port name, Port)`` for a target and optional port."""
    if isinstance(target, tuple):
        if len(target) != 2:
            raise EditError(f"expected (component, port), got {target!r}")
        target, port = target
    ci = _index_of(circ, target)
    comp = circ.components[ci]
    try:
        shape = geometry.shape(project, comp)
    except geometry.UnknownGeometry as e:
        raise EditError(f"cannot determine ports of {comp!r}: {e}") from None
    names = [p.name for p in shape.ports]
    if port is None:
        if len(names) == 1:
            port = names[0]
        elif comp.name == "Pin" and "p" in names:
            port = "p"
        else:
            outs = [p.name for p in shape.ports if p.dir == "out"]
            if len(outs) == 1:
                port = outs[0]
            else:
                raise EditError(f"{comp!r} has ports {names}; specify one")
    if port not in names:
        raise EditError(f"{comp!r} has no port {port!r}; ports are {names}")
    return ci, port, geometry.port(project, comp, port)


def list_components(project: Project, circuit: Circuit | str) -> list[ComponentInfo]:
    """All components of a circuit, in drawing order."""
    circ = _as_circuit(project, circuit)
    return [_info(circ, i) for i in range(len(circ.components))]


def find_components(project: Project, circuit: Circuit | str, *, name: str | None = None,
                    label: str | None = None) -> list[ComponentInfo]:
    """Components matching a type name (e.g. ``"AND Gate"``), a label, or both."""
    circ = _as_circuit(project, circuit)
    return [
        _info(circ, i)
        for i, c in enumerate(circ.components)
        if (name is None or c.name == name) and (label is None or c.label == label)
    ]


def find_component(project: Project, circuit: Circuit | str, *, name: str | None = None,
                   label: str | None = None) -> ComponentInfo:
    """Exactly one component matching the filters; raises if zero or several match."""
    found = find_components(project, circuit, name=name, label=label)
    if len(found) != 1:
        desc = ", ".join(filter(None, [f"name={name!r}" if name else "",
                                       f"label={label!r}" if label else ""]))
        raise EditError(f"expected exactly one component matching {desc or 'the filters'}, "
                        f"found {len(found)}")
    return found[0]


# --------------------------------------------------------------- grid / routing


def _snap(v: float) -> int:
    return int(round(v / G)) * G


def _across(project: Project, comp: Component):
    try:
        return geometry.shape(project, comp), geometry.bounds(project, comp)
    except geometry.UnknownGeometry:
        return None, None


def _rects_overlap(a, b, gap: int = 0) -> bool:
    return (a[0] - gap < b[0] + b[2] and b[0] < a[0] + a[2] + gap and
            a[1] - gap < b[1] + b[3] and b[1] < a[1] + a[3] + gap)


def _segment_hits_rect(a: Point, b: Point, rect) -> bool:
    rx, ry, rw, rh = rect
    if a[1] == b[1]:
        lo, hi = sorted((a[0], b[0]))
        return ry <= a[1] <= ry + rh and max(lo, rx) <= min(hi, rx + rw)
    lo, hi = sorted((a[1], b[1]))
    return rx <= a[0] <= rx + rw and max(lo, ry) <= min(hi, ry + rh)


def _location_ok(project: Project, circ: Circuit, name: str, lib: str | None,
                 attrs: Mapping[str, str], loc: Point, *, ignore: Component | None = None,
                 gap: int = 10, ignore_wires: Iterable[int] = ()) -> bool:
    """True when a component of this type may sit at `loc`.

    The body must not overlap another body, must not touch another wire and must
    not sit on the very location of another component (Logisim hangs on that).
    """
    try:
        shape = geometry.shape(project, Component(name, loc, dict(attrs), lib))
    except geometry.UnknownGeometry:
        return True  # cannot judge; caller will still avoid exact location clashes
    bx, by, bw, bh = shape.bounds
    body = (loc[0] + bx, loc[1] + by, bw, bh)
    for c in circ.components:
        if c is ignore:
            continue
        if c.loc == loc:
            return False
        _, other = _across(project, c)
        if other is not None and _rects_overlap(body, other, gap):
            return False
    skip = set(ignore_wires)
    for i, w in enumerate(circ.wires):
        if i in skip:
            continue
        if _segment_hits_rect(w.a, w.b, body):
            return False
    return True


def _auto_location(project: Project, circ: Circuit, name: str, lib: str | None,
                   attrs: Mapping[str, str], near: Point | None = None,
                   ignore: Component | None = None) -> Point:
    """First grid location whose body is clear of every body and wire."""
    comps = [c for c in circ.components if c is not ignore]
    if near is not None:
        base = near
    elif comps:
        boxes = [b for b in (_across(project, c)[1] for c in comps) if b is not None]
        if boxes:
            base = (max(b[0] + b[2] for b in boxes) + 100, min(b[1] for b in boxes))
        else:
            base = (comps[0].loc[0] + 200, comps[0].loc[1])
    else:
        base = (100, 100)
    step = 20
    for r in range(0, 120):
        for dx, dy in _ring(r):
            loc = (_snap(base[0] + dx * step), _snap(base[1] + dy * step))
            if _location_ok(project, circ, name, lib, attrs, loc, ignore=ignore):
                return loc
    raise EditError(f"no free space for a new {name!r}; move or delete something first")


def _ring(r: int):
    if r == 0:
        yield (0, 0)
        return
    for i in range(-r, r + 1):
        yield (i, -r)
        yield (i, r)
    for j in range(-r + 1, r):
        yield (-r, j)
        yield (r, j)


def _build_grid(project: Project, circ: Circuit):
    """A route.Grid describing the current drawing, plus its netlist and a free net id."""
    nl = netlist.extract(project, circ)
    shapes: dict[int, geometry.Shape] = {}
    bounds: list[tuple[int, int, int, int]] = []
    for i, c in enumerate(circ.components):
        try:
            shapes[i] = geometry.shape(project, c)
            bounds.append(geometry.bounds(project, c))
        except geometry.UnknownGeometry:
            continue
    xs = [b[0] for b in bounds] + [b[0] + b[2] for b in bounds]
    ys = [b[1] for b in bounds] + [b[1] + b[3] for b in bounds]
    for w in circ.wires:
        xs += [w.a[0], w.b[0]]
        ys += [w.a[1], w.b[1]]
    xmin = _snap(min(xs)) - 120 if xs else -100
    ymin = _snap(min(ys)) - 120 if ys else -100
    xmax = _snap(max(xs)) + 120 if xs else 400
    ymax = _snap(max(ys)) + 120 if ys else 400
    grid = route.Grid(xmin, ymin, max(xmax, xmin + 200), max(ymax, ymin + 200))

    next_net = max((n.id for n in nl.nets), default=-1) + 1

    for i, shape in shapes.items():
        keep = {(circ.components[i].loc[0] + p.x, circ.components[i].loc[1] + p.y)
                for p in shape.ports}
        nid = nl.port_net.get((i, shape.ports[0].name)) if shape.ports else None
        for p in shape.ports:
            nid = nl.port_net.get((i, p.name))
            if nid is None:
                nid = next_net
                next_net += 1
            grid.add_hard((circ.components[i].loc[0] + p.x, circ.components[i].loc[1] + p.y), nid)
        bx, by, bw, bh = shape.bounds
        c = circ.components[i]
        grid.block_rect(c.loc[0] + bx, c.loc[1] + by, bw, bh, keep)
        box = geometry.label_box(c, (c.loc[0] + bx, c.loc[1] + by, bw, bh))
        if box is not None:
            grid.block_rect(*box, keep)
        # Ports drawn inside their own box need a clear corridor out.
        for p in shape.ports:
            if bx < p.x < bx + bw and by < p.y < by + bh:
                dx, dy = layout.port_side(shape, p.name)
                q = (c.loc[0] + p.x + dx, c.loc[1] + p.y + dy)
                while (c.loc[0] + bx <= q[0] <= c.loc[0] + bx + bw and
                       c.loc[1] + by <= q[1] <= c.loc[1] + by + bh):
                    grid.blocked.discard(q)
                    q = (q[0] + dx, q[1] + dy)
                grid.blocked.discard(q)

    wire_net = {wi: n.id for n in nl.nets for wi in n.wires}
    for wi, w in enumerate(circ.wires):
        nid = wire_net.get(wi)
        if nid is None:
            nid = next_net
            next_net += 1
        grid.add_path([w.a, w.b], nid)
    return grid, nl, next_net


def _emit_path(circ: Circuit, path: Sequence[Point]) -> None:
    for a, b in route.segments(list(path)):
        circ.add_wire(a, b)


def _free_grid(grid: route.Grid) -> route.Grid:
    """A copy of `grid` in which component bodies do not block the router.

    Used as a last resort: a wire that has to leave a component which stands
    next to (or on top of) another one may cross that body.  Crossing a body is
    ugly but harmless in Logisim: only ports and wire ends connect.
    """
    g = route.Grid(grid.xmin, grid.ymin, grid.xmax, grid.ymax)
    g.hard = dict(grid.hard)
    g.run = {p: {n: set(o) for n, o in nets.items()} for p, nets in grid.run.items()}
    g.arms = {p: {n: set(d) for n, d in nets.items()} for p, nets in grid.arms.items()}
    return g


def _route_to_net(project: Project, circ: Circuit, grid, nl, target_net: int, start: Point,
                  start_dir: tuple[int, int] | None, prefer: Point | None = None):
    """Draw a wire from `start` to the net `target_net`.

    `prefer` is the point the wire should aim for (the old port position or the
    port of the other end); the router may pick any other point of the net when
    that is cheaper.
    """
    net_points = grid.net_points(target_net)
    attempts: list[set[Point]] = []
    if prefer is not None and prefer in net_points:
        attempts.append({prefer})
    attempts.append(net_points)
    if start in net_points:
        grid.add_hard(start, target_net)
        return
    for targets in attempts:
        path = route.route(grid, target_net, start, targets, start_dir=start_dir)
        if path is None:
            path = route.route(grid, target_net, start, targets)
        if path is not None:
            _emit_path(circ, path)
            grid.add_path(path, target_net)
            grid.add_hard(start, target_net)
            return
    # last resort: let the wire cross component bodies
    free = _free_grid(grid)
    for targets in attempts:
        path = route.route(free, target_net, start, targets, start_dir=start_dir)
        if path is None:
            path = route.route(free, target_net, start, targets)
        if path is not None:
            _emit_path(circ, path)
            grid.add_path(path, target_net)
            grid.add_hard(start, target_net)
            grid.hard.update(free.hard)
            grid.run.update(free.run)
            grid.arms.update(free.arms)
            return
    raise EditError(f"could not route a wire from ({start[0]},{start[1]}) "
                    f"to the target net")


def _route_fanout(project: Project, circ: Circuit, grid, net_id: int, source: Point,
                  sinks: Sequence[tuple[Point, geometry.Shape, str]]):
    """Route each sink to the tree grown from ``source`` (same order as layout.py)."""
    grid.add_hard(source, net_id)
    tree = set(grid.net_points(net_id))
    ordered = sorted(sinks, key=lambda s: (abs(s[0][0] - source[0]) + abs(s[0][1] - source[1])))
    for pt, shape, port_name in ordered:
        if pt in tree:
            grid.add_hard(pt, net_id)
            continue
        side = layout.port_side(shape, port_name)
        path = route.route(grid, net_id, pt, tree, start_dir=side)
        if path is None:
            path = route.route(grid, net_id, pt, tree)
        if path is None:
            raise EditError(f"could not route a wire to ({pt[0]},{pt[1]})")
        _emit_path(circ, path)
        grid.add_path(path, net_id)
        grid.add_hard(pt, net_id)
        for a, b in route.segments(path):
            tree |= set(route._segment_points(a, b))


# ---------------------------------------------------------------- add / query


def _lib_id(project: Project, lib: str | None, name: str) -> str | None:
    if lib is None:
        if project.has_circuit(name):
            return None
        raise EditError(f"{name!r} is not a circuit in this project; pass lib= for a "
                        "library component (e.g. 'gates')")
    if project.library(lib) is not None:
        return lib
    return project.lib_id(lib)


def _connection_items(connections) -> list[tuple[str, object]]:
    if connections is None:
        return []
    if isinstance(connections, Mapping):
        return [(str(k), v) for k, v in connections.items()]
    out = []
    for item in connections:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise EditError("connections must map port name -> target or be (port, target) pairs")
        out.append((str(item[0]), item[1]))
    return out


def add_component(project: Project, circuit: Circuit | str, name: str, *, lib: str | None = None,
                  attrs: Mapping[str, object] | None = None, loc="auto",
                  connections=None) -> ComponentInfo:
    """Add a library component (``lib`` + ``name``) or a subcircuit instance.

    ``loc`` may be a point or ``"auto"`` (free space).  ``connections`` maps a
    new component's port name to an existing ``(component, port)``; each is
    routed to the whole net that port belongs to.
    """
    circ = _as_circuit(project, circuit)
    attrs = {k: str(v) for k, v in (attrs or {}).items()}
    lib_id = _lib_id(project, lib, name)
    conns = _connection_items(connections)
    targets = [(pn, _resolve_port(project, circ, t)) for pn, t in conns]
    if loc == "auto":
        near = None
        if targets:
            pts = [t[1][2].pos for t in targets]
            near = (sum(p[0] for p in pts) // len(pts), sum(p[1] for p in pts) // len(pts))
        loc = _auto_location(project, circ, name, lib_id, attrs, near=near)
    loc = (int(loc[0]), int(loc[1]))
    if any(c.loc == loc for c in circ.components):
        raise EditError(f"another component already sits at ({loc[0]},{loc[1]}); "
                        "Logisim hangs loading such a file")
    comp = circ.add(Component(name, loc, attrs, lib_id))
    if targets:
        try:
            shape = geometry.shape(project, comp)
        except geometry.UnknownGeometry as e:
            raise EditError(f"cannot determine ports of {comp!r}: {e}") from None
        grid, nl, _ = _build_grid(project, circ)
        for port_name, (ci, pname, port) in targets:
            if port_name not in {p.name for p in shape.ports}:
                raise EditError(f"{comp!r} has no port {port_name!r}")
            target_net = nl.port_net.get((ci, pname))
            if target_net is None:
                raise EditError(f"cannot find the net of port {pname!r}")
            start = geometry.port(project, comp, port_name).pos
            side = layout.port_side(shape, port_name)
            _route_to_net(project, circ, grid, nl, target_net, start, side)
    return _info(circ, len(circ.components) - 1)


# ------------------------------------------------------- delete / move / rotate


def _port_points(project: Project, comp: Component) -> dict[str, Point]:
    try:
        return {p.name: p.pos for p in geometry.ports(project, comp)}
    except geometry.UnknownGeometry:
        return {}


def _port_anchors(project: Project, circ: Circuit, ci: int) -> dict[str, list[tuple[int, str]]]:
    """For every port of component ``ci``: the other ports that share its net."""
    nl = netlist.extract(project, circ)
    comp = circ.components[ci]
    try:
        ports = geometry.ports(project, comp)
    except geometry.UnknownGeometry:
        return {}
    out: dict[str, list[tuple[int, str]]] = {}
    for p in ports:
        net = nl.net_of(ci, p.name)
        if net is None:
            out[p.name] = []
        else:
            out[p.name] = [(r.comp, r.port.name) for r in net.ports
                           if not (r.comp == ci and r.port.name == p.name)]
    return out


def _partition(nl: netlist.Netlist, exclude: int | None = None) -> frozenset:
    """The nets of a circuit as sets of (component index, port name).

    Unlike :meth:`netlist.Netlist.signature` this does not depend on the
    components' attributes, so it can be compared before and after an edit that
    changes them (a rename, a new input, ...).
    """
    groups = set()
    for n in nl.nets:
        members = frozenset((p.comp, p.port.name) for p in n.ports
                            if exclude is None or p.comp != exclude)
        if members:
            groups.add(members)
    return frozenset(groups)


def _inside_body(comp: Component, shape: geometry.Shape, point: Point) -> bool:
    """True when `point` lies in (or on the edge of) the component's body."""
    bx, by, bw, bh = shape.bounds
    x, y = point
    return comp.loc[0] + bx <= x <= comp.loc[0] + bx + bw and \
        comp.loc[1] + by <= y <= comp.loc[1] + by + bh


def _prefer_point(old: Point | None, new: Point, shape: geometry.Shape, comp: Component,
                  anchors: Sequence[Point]) -> Point | None:
    """The point the new wire should aim for.

    The old port position is preferred (the wire then replaces the old stub in
    place), unless the component now stands on it.  Otherwise the nearest port
    of the other end of the net is used.
    """
    cands = list(anchors)
    if old is not None and old != new and not _inside_body(comp, shape, old):
        cands.append(old)
    if not cands:
        return None
    return min(cands, key=lambda q: (abs(q[0] - new[0]) + abs(q[1] - new[1]), q))


def _reattach(project: Project, circ: Circuit, ci: int,
              anchors: Mapping[str, Sequence[tuple[int, str]]],
              old_points: Mapping[str, Point] | None = None, *,
              preserve: frozenset | None = None, exclude: int | None = None) -> None:
    """Put every port of component ``ci`` back on the net it was on before.

    ``anchors`` maps a port name to the other components' ports that shared its
    net.  The component has already been moved (or turned, or re-attributed) when
    this runs, so the old wires are still in place; they are cut afterwards.

    ``preserve`` is a port partition (see :func:`_partition`) that the cleanup
    must not break: a wire is only removed when the nets stay the same.
    """
    grid, nl, _ = _build_grid(project, circ)
    comp = circ.components[ci]
    try:
        shape = geometry.shape(project, comp)
        ports = geometry.ports(project, comp)
    except geometry.UnknownGeometry as e:
        raise EditError(f"cannot determine ports of {comp!r}: {e}") from None
    for p in ports:
        targets: list[tuple[int, Point]] = []
        for cj, pname in anchors.get(p.name) or []:
            if not (0 <= cj < len(circ.components)):
                continue
            nid = nl.port_net.get((cj, pname))
            if nid is None:
                continue
            pos = geometry.port(project, circ.components[cj], pname).pos
            if all(nid != t[0] for t in targets):
                targets.append((nid, pos))
        if not targets:
            continue
        ids = [t[0] for t in targets]
        here = nl.port_net.get((ci, p.name))
        if here is not None and here in ids:
            continue  # the new port already lies on one of its nets
        if here is not None:
            others = [r for r in nl.nets[here].ports
                      if not (r.comp == ci and r.port.name == p.name)]
            if others:
                raise EditError(
                    f"the {p.name} port of {comp!r} would land on the net of "
                    f"{others[0].describe(circ)}; the two nets would be shorted")
        side = layout.port_side(shape, p.name)
        prefer = _prefer_point((old_points or {}).get(p.name), p.pos, shape, comp,
                                [pos for _, pos in targets])
        for nid in ids:
            _route_to_net(project, circ, grid, nl, nid, p.pos, side, prefer=prefer)
    _trim_stale(project, circ, comp, old_points or {})
    cleanup_wires(project, circ, preserve=preserve, exclude=exclude)


def _trim_stale(project: Project, circ: Circuit, comp: Component,
                old_points: Mapping[str, Point]) -> None:
    """Cut the wires that used to end at the component's old port positions.

    When the new port lands *inside* a wire of its net (the wire ran past the old
    port), that wire is cut at the new port, so the drawing keeps no stub.
    """
    try:
        new_points = _port_points(project, comp)
    except Exception:  # pragma: no cover - geometry is checked by the caller
        return
    for name, old in old_points.items():
        new = new_points.get(name)
        if new is None or new == old:
            continue
        for i, w in enumerate(circ.wires):
            if old not in (w.a, w.b) or new in (w.a, w.b) or not w.contains(new):
                continue
            other = w.b if w.a == old else w.a
            circ.wires[i] = Wire(new, other)


def delete_component(project: Project, circuit: Circuit | str, target) -> ComponentInfo:
    """Delete a component and every wire stub that only served it.

    The wires that only reached the deleted component are removed as well (see
    :func:`cleanup_wires`); the nets that did not touch the component keep
    their shape.
    """
    circ = _as_circuit(project, circuit)
    ci = _index_of(circ, target)
    info = _info(circ, ci)
    circ.components.pop(ci)
    cleanup_wires(project, circ)
    return info


def _snapshot(circ: Circuit):
    """The drawing, so a failed edit can be undone."""
    return ([(c, c.loc, dict(c.attrs)) for c in circ.components], list(circ.wires))


def _restore(circ: Circuit, snap) -> None:
    comps, wires = snap
    for c, loc, attrs in comps:
        c.loc = loc
        c.attrs = attrs
    circ.components[:] = [c for c, _, _ in comps]
    circ.wires[:] = wires


def _overlapped_body(project: Project, circ: Circuit, comp: Component, loc: Point) -> Component | None:
    """The first other component whose body `comp`'s body would overlap at `loc`."""
    shape, _ = _across(project, comp)
    if shape is None:
        return None
    bx, by, bw, bh = shape.bounds
    body = (loc[0] + bx, loc[1] + by, bw, bh)
    for c in circ.components:
        if c is comp:
            continue
        _, other = _across(project, c)
        if other is not None and _rects_overlap(body, other):
            return c
    return None


def move_component(project: Project, circuit: Circuit | str, target, *, loc=None,
                   dx: int = 0, dy: int = 0, force: bool = False) -> ComponentInfo:
    """Move a component; its connected wires are re-routed to the same nets.

    ``loc`` is a point, ``dx``/``dy`` an offset.  A location where the body would
    overlap another component's body is refused unless ``force=True``.  The
    connectivity of the circuit does not change: the ports of the moved
    component are put back on the nets they were on.
    """
    circ = _as_circuit(project, circuit)
    ci = _index_of(circ, target)
    comp = circ.components[ci]
    if loc is not None and loc != "auto":
        new_loc = (int(loc[0]), int(loc[1]))
    elif loc == "auto":
        new_loc = _auto_location(project, circ, comp.name, comp.lib, comp.attrs, ignore=comp)
    else:
        new_loc = (comp.loc[0] + int(dx), comp.loc[1] + int(dy))
    if new_loc == comp.loc:
        return _info(circ, ci)
    if any(c is not comp and c.loc == new_loc for c in circ.components):
        raise EditError(f"another component already sits at ({new_loc[0]},{new_loc[1]})")
    if not force:
        hit = _overlapped_body(project, circ, comp, new_loc)
        if hit is not None:
            raise EditError(f"at ({new_loc[0]},{new_loc[1]}) {comp!r} would overlap {hit!r}; "
                            "pass force=True to allow it")
    anchors = _port_anchors(project, circ, ci)
    old_points = _port_points(project, comp)
    before = _partition(netlist.extract(project, circ))
    snap = _snapshot(circ)
    try:
        circ.components.pop(ci)
        comp.loc = new_loc
        circ.components.insert(ci, comp)
        _reattach(project, circ, ci, anchors, old_points, preserve=before)
        after = _partition(netlist.extract(project, circ))
        if after != before:
            raise EditError(f"moving {comp!r} to ({new_loc[0]},{new_loc[1]}) would change the "
                            "wiring of the circuit")
    except EditError:
        _restore(circ, snap)
        raise
    return _info(circ, ci)


def rotate_component(project: Project, circuit: Circuit | str, target, *, facing: str | None = None,
                     turns: int = 1) -> ComponentInfo:
    """Change a component's facing, re-routing its wires so connectivity is unchanged."""
    circ = _as_circuit(project, circuit)
    ci = _index_of(circ, target)
    comp = circ.components[ci]
    old = comp.attrs.get("facing", "east") or "east"
    if facing is None:
        new = FACINGS[(FACINGS.index(old) + turns) % len(FACINGS)]
    else:
        if facing not in FACINGS:
            raise EditError(f"facing must be one of {FACINGS}, got {facing!r}")
        new = facing
    if new == old:
        return _info(circ, ci)
    anchors = _port_anchors(project, circ, ci)
    old_points = _port_points(project, comp)
    before = _partition(netlist.extract(project, circ))
    snap = _snapshot(circ)
    try:
        circ.components.pop(ci)
        if new == "east":
            comp.attrs.pop("facing", None)
        else:
            comp.attrs["facing"] = new
        circ.components.insert(ci, comp)
        _reattach(project, circ, ci, anchors, old_points, preserve=before)
        after = _partition(netlist.extract(project, circ))
        if after != before:
            raise EditError(f"turning {comp!r} to face {new} would change the wiring "
                            "of the circuit")
    except EditError:
        _restore(circ, snap)
        raise
    return _info(circ, ci)


# ---------------------------------------------------------------- attributes


def _check_width(project: Project, circ: Circuit, ci: int, attrs: Mapping[str, str]) -> None:
    """Reject a width change that would put two widths on one net."""
    comp = circ.components[ci]
    if "width" not in attrs:
        return
    trial = Component(comp.name, comp.loc, {**comp.attrs, **attrs}, comp.lib)
    try:
        new_widths = {p.name: p.width for p in geometry.shape(project, trial).ports}
    except geometry.UnknownGeometry:
        return
    nl = netlist.extract(project, circ)
    for port_name, new_width in new_widths.items():
        net = nl.net_of(ci, port_name)
        if net is None or new_width <= 0:
            continue
        for ref in net.ports:
            if ref.comp == ci or ref.port.width <= 0:
                continue
            if ref.port.width != new_width:
                other = circ.components[ref.comp]
                raise EditError(
                    f"changing width to {new_width} would create a width mismatch on the net "
                    f"shared with {other!r}.{ref.port.name} ({ref.port.width} bits); "
                    "pass force=True to allow it")


def _check_ports(project: Project, circ: Circuit, ci: int, attrs: Mapping[str, str]) -> None:
    """Reject a change that removes a port which is connected to something.

    Dropping an input (for instance ``inputs=1`` on a two-input gate) would
    silently disconnect whatever was wired to it.
    """
    comp = circ.components[ci]
    trial = Component(comp.name, comp.loc, {**comp.attrs, **attrs}, comp.lib)
    try:
        old = {p.name for p in geometry.shape(project, comp).ports}
        new = {p.name for p in geometry.shape(project, trial).ports}
    except geometry.UnknownGeometry:
        return
    gone = sorted(old - new)
    if not gone:
        return
    nl = netlist.extract(project, circ)
    for name in gone:
        net = nl.net_of(ci, name)
        if net is None:
            continue
        others = [r for r in net.ports if not (r.comp == ci and r.port.name == name)]
        if others:
            raise EditError(
                f"changing {sorted(attrs)} would remove the {name!r} port of {comp!r}, which is "
                f"connected to {others[0].describe(circ)}; pass force=True to allow it")


def set_attributes(project: Project, circuit: Circuit | str, target,
                   attrs: Mapping[str, object], *, force: bool = False) -> ComponentInfo:
    """Set one or more attributes; geometry-affecting changes are re-routed.

    A ``width`` change that would mix widths on a connected net is refused,
    and so is a change that removes a port which is connected to something,
    unless ``force=True``.  A ``facing`` change behaves like
    :func:`rotate_component`.
    """
    circ = _as_circuit(project, circuit)
    ci = _index_of(circ, target)
    comp = circ.components[ci]
    values = {k: str(v) for k, v in attrs.items()}
    if not values:
        return _info(circ, ci)
    if not force:
        _check_width(project, circ, ci, values)
        _check_ports(project, circ, ci, values)
    if set(values) <= _GEOMETRY_NEUTRAL:
        comp.attrs.update(values)
        return _info(circ, ci)
    anchors = _port_anchors(project, circ, ci)
    old_points = _port_points(project, comp)
    before = _partition(netlist.extract(project, circ), exclude=ci)
    snap = _snapshot(circ)
    try:
        circ.components.pop(ci)
        comp.attrs.update(values)
        circ.components.insert(ci, comp)
        _reattach(project, circ, ci, anchors, old_points, preserve=before, exclude=ci)
    except EditError:
        _restore(circ, snap)
        raise
    return _info(circ, ci)


def set_attribute(project: Project, circuit: Circuit | str, target, key: str, value, *,
                  force: bool = False) -> ComponentInfo:
    """Set one attribute (see :func:`set_attributes`)."""
    return set_attributes(project, circuit, target, {key: value}, force=force)


# -------------------------------------------------------------------- renames


def _instances_of(project: Project, sub_name: str) -> list[tuple[Circuit, int]]:
    """The (circuit, component index) of every instance of `sub_name`."""
    return [(circ, ci) for circ in project.circuits
            for ci, comp in enumerate(circ.components)
            if comp.lib is None and comp.name == sub_name]


def _reattach_instances(project: Project, saved: Sequence[tuple[Circuit, int, dict, dict]],
                        rename: Mapping[str, str] | None = None) -> None:
    """Re-connect instances whose geometry just changed.

    The width of a subcircuit's box depends on its name and on the labels of
    its pins, so renaming either of them moves the instance's ports; the wires
    that reach them have to be routed again.
    """
    for circ, ci, anchors, old_points in saved:
        if rename:
            anchors = {rename.get(k, k): v for k, v in anchors.items()}
        before = _partition(netlist.extract(project, circ), exclude=ci)
        _reattach(project, circ, ci, anchors, old_points, preserve=before, exclude=ci)


def rename_circuit(project: Project, old: str, new: str) -> Circuit:
    """Rename a circuit and every subcircuit instance that refers to it.

    The instances keep their wiring: their ports may move when the new name
    changes the width of the subcircuit's box, so they are re-connected.
    """
    if not project.has_circuit(old):
        raise EditError(f"no circuit named {old!r} (have: {', '.join(project.circuit_names())})")
    if old == new:
        return project.circuit(old)
    if not builder.valid_circuit_name(new):
        raise EditError(f"bad circuit name {new!r}: Logisim renames it on load; use a letter, "
                        "then letters, digits and single underscores, not ending in '_'")
    if project.has_circuit(new):
        raise EditError(f"circuit {new!r} already exists")
    circ = project.circuit(old)
    saved = [(c, ci, _port_anchors(project, c, ci), _port_points(project, c.components[ci]))
             for c, ci in _instances_of(project, old)]
    circ.name = new
    if circ.attrs.get("circuit") in (None, old):
        circ.attrs["circuit"] = new
    if project.main == old:
        project.main = new
    for other in project.circuits:
        for comp in other.components:
            if comp.lib is None and comp.name == old:
                comp.name = new
    _reattach_instances(project, saved)
    return circ


def rename_pin(project: Project, circuit: Circuit | str, old_label: str, new_label: str) -> ComponentInfo:
    """Rename a pin (its label); neighbouring pins may not share the new label.

    The instances of the circuit keep their wiring: the width of their box may
    change with the label, so their ports are re-connected.
    """
    circ = _as_circuit(project, circuit)
    matches = [i for i, c in enumerate(circ.components) if c.name == "Pin" and c.label == old_label]
    if len(matches) != 1:
        raise EditError(f"expected one pin labelled {old_label!r} in {circ.name!r}, "
                        f"found {len(matches)}")
    if not new_label:
        raise EditError("a pin label may not be empty")
    ci = matches[0]
    for i, c in enumerate(circ.components):
        if i != ci and c.name == "Pin" and c.label == new_label:
            raise EditError(f"another pin in {circ.name!r} is already labelled {new_label!r}")
    saved = [(c, i, _port_anchors(project, c, i), _port_points(project, c.components[i]))
             for c, i in _instances_of(project, circ.name)]
    circ.components[ci].label = new_label
    _reattach_instances(project, saved, rename={old_label: new_label})
    return _info(circ, ci)


# ---------------------------------------------------------------- insert gate


def _net_for(project: Project, circ: Circuit, net):
    nl = netlist.extract(project, circ)
    if isinstance(net, netlist.Net):
        for n in nl.nets:
            if n.id == net.id or n is net:
                return nl, n
        raise EditError("that net does not belong to this circuit")
    ci, port, _ = _resolve_port(project, circ, net)
    nid = nl.port_net.get((ci, port))
    if nid is None:
        raise EditError(f"no net found at component {ci} port {port!r}")
    return nl, nl.nets[nid]


def _step_dir(a: Point, b: Point) -> tuple[int, int]:
    """The grid direction from `a` to `b` (they share a row or a column)."""
    if b[0] != a[0]:
        return (G if b[0] > a[0] else -G, 0)
    return (0, G if b[1] > a[1] else -G)


_FACING_OF_DIR = {(G, 0): "east", (-G, 0): "west", (0, G): "south", (0, -G): "north"}


def _path_direction(circ: Circuit, wire_indices: Sequence[int], start: Point,
                    sinks: Sequence[Point]) -> tuple[int, int]:
    """The direction in which the net's path leaves `start`, towards the sinks."""
    best: tuple[int, tuple[int, int]] | None = None
    for i in wire_indices:
        w = circ.wires[i]
        if not w.contains(start):
            continue
        for far in (w.a, w.b):
            if far == start:
                continue
            dist = min((abs(s[0] - far[0]) + abs(s[1] - far[1]) for s in sinks), default=0)
            if best is None or dist < best[0]:
                best = (dist, _step_dir(start, far))
    return best[1] if best else (G, 0)


def _path_points(circ: Circuit, wire_indices: Sequence[int], start: Point,
                 direction: tuple[int, int]) -> list[Point]:
    """The grid points of the net's path, starting at `start` and following `direction`."""
    pts: list[Point] = [start]
    cur = start
    used: set[int] = set()
    for _ in range(len(wire_indices) + 2):
        nxt: tuple[int, Point] | None = None
        for i in wire_indices:
            if i in used:
                continue
            w = circ.wires[i]
            if not w.contains(cur):
                continue
            far: Point | None = None
            if w.a == cur:
                far = w.b
            elif w.b == cur:
                far = w.a
            else:
                ahead = [e for e in (w.a, w.b) if _step_dir(cur, e) == direction]
                far = ahead[0] if ahead else None
            if far is None:
                continue
            nxt = (i, far)
            break
        if nxt is None:
            break
        i, far = nxt
        used.add(i)
        pts.extend(route._segment_points(cur, far)[1:])
        cur = far
    return pts


def _insert_candidates(project: Project, circ: Circuit, name: str, lib: str | None,
                       attrs: Mapping[str, str], driver_pt: Point, sink_pts: Sequence[Point],
                       wire_indices: Sequence[int]) -> list[tuple[Point, dict[str, str]]]:
    """Places for a component that is inserted into a net, best first.

    The component goes on the net's own path, in the driver's row, between the
    driver and the first sink; it is moved along the path until its body is
    free of the other components and wires.
    """
    direction = _path_direction(circ, wire_indices, driver_pt, sink_pts)
    facing = _FACING_OF_DIR[direction]
    values = dict(attrs)
    if facing != "east":
        values["facing"] = facing
    path = _path_points(circ, wire_indices, driver_pt, direction)
    limit = len(path)
    for s in sink_pts:
        if s in path:
            limit = min(limit, path.index(s))
    out: list[tuple[Point, dict[str, str]]] = []
    for pt in path[1:max(limit, 1)]:
        if _location_ok(project, circ, name, lib, values, pt, gap=10, ignore_wires=wire_indices):
            out.append((pt, dict(values)))
    # last resort: free space next to the driver
    out.append((_auto_location(project, circ, name, lib, values, near=driver_pt), dict(values)))
    return out


def insert_gate(project: Project, circuit: Circuit | str, net, name: str = "NOT Gate", *,
                lib: str | None = "gates", attrs: Mapping[str, object] | None = None,
                input_port: str | None = None, output_port: str | None = None,
                sinks=None, extra_inputs=None, driver=None, loc="auto") -> ComponentInfo:
    """Insert a component into an existing net.

    ``net`` identifies the net through any ``(component, port)`` on it.  The
    original driver is connected to the new component's chosen input, and the
    chosen output drives all sinks (or only the ``sinks`` subset).  All other
    connectivity is left intact.

    With ``loc="auto"`` the component is placed on the net's own path, in the
    driver's row, between the driver and the sinks.
    """
    circ = _as_circuit(project, circuit)
    nl, old_net = _net_for(project, circ, net)
    refs = list(old_net.ports)
    if len(refs) < 2:
        raise EditError("the selected net has fewer than two ports; nothing to insert between")
    if driver is not None:
        dci, dport, _ = _resolve_port(project, circ, driver)
        matches = [p for p in refs if p.comp == dci and p.port.name == dport]
        if not matches:
            raise EditError("the requested driver is not on the selected net")
        src = matches[0]
    else:
        outs = [p for p in refs if p.port.dir == "out"]
        if len(outs) > 1:
            raise EditError("the selected net has several drivers; pass driver=(component, port)")
        src = outs[0] if outs else refs[0]
    consumers = [p for p in refs if not (p.comp == src.comp and p.port.name == src.port.name)]
    selected_keys = None
    if sinks is not None:
        items = sinks if isinstance(sinks, (list, tuple, set)) else [sinks]
        # A bare (component, port) pair is one sink, not a list of two sinks.
        if (isinstance(sinks, tuple) and len(sinks) == 2 and
                not isinstance(sinks[0], (list, tuple, set))):
            items = [sinks]
        selected_keys = set()
        for item in items:
            sci, sport, _ = _resolve_port(project, circ, item)
            if not any(p.comp == sci and p.port.name == sport for p in consumers):
                raise EditError(f"({sci},{sport!r}) is not a sink of the selected net")
            selected_keys.add((sci, sport))

    def chosen(p):
        return selected_keys is None or (p.comp, p.port.name) in selected_keys

    selected = [p for p in consumers if chosen(p)]
    unselected = [p for p in consumers if not chosen(p)]

    attrs = {k: str(v) for k, v in (attrs or {}).items()}
    net_width = max((p.port.width for p in refs if p.port.width > 0), default=1)
    if "width" not in attrs and net_width > 1 and name in geometry.GATE_NAMES | {"NOT Gate", "Buffer"}:
        attrs["width"] = str(net_width)
    lib_id = _lib_id(project, lib, name)
    try:
        shape = geometry.shape(project, Component(name, (0, 0), dict(attrs), lib_id))
    except geometry.UnknownGeometry as e:
        raise EditError(f"cannot determine ports of {name!r}: {e}") from None
    ins = [p.name for p in shape.ports if p.dir == "in"]
    outs = [p.name for p in shape.ports if p.dir == "out"]
    if not ins or not outs:
        raise EditError(f"{name!r} needs at least one input and one output port")
    in_name = input_port or ins[0]
    out_name = output_port or outs[0]
    if in_name not in ins:
        raise EditError(f"{name!r} has no input port {in_name!r}; inputs are {ins}")
    if out_name not in outs:
        raise EditError(f"{name!r} has no output port {out_name!r}; outputs are {outs}")

    doomed = set(old_net.wires)
    src_pt = geometry.port(project, circ.components[src.comp], src.port.name).pos
    sink_pts = [geometry.port(project, circ.components[p.comp], p.port.name).pos for p in selected]
    if loc == "auto":
        places = _insert_candidates(project, circ, name, lib_id, attrs, src_pt, sink_pts,
                                    sorted(doomed))
    else:
        places = [((int(loc[0]), int(loc[1])), dict(attrs))]

    last: EditError | None = None
    for place, values in places:
        if any(c.loc == place for c in circ.components):
            last = EditError(f"another component already sits at ({place[0]},{place[1]})")
            continue
        snap = _snapshot(circ)
        try:
            gate = _insert_at(project, circ, name, place, values, lib_id, doomed, src, selected,
                              unselected, in_name, out_name, extra_inputs)
        except EditError as e:
            last = e
            _restore(circ, snap)
            continue
        return _info(circ, len(circ.components) - 1)
    raise EditError(f"could not insert {name!r} into the net: {last}")


def _insert_at(project: Project, circ: Circuit, name: str, loc: Point, attrs: dict[str, str],
               lib_id: str | None, doomed: set[int], src, selected, unselected,
               in_name: str, out_name: str, extra_inputs) -> Component:
    """Put the new component at `loc` and wire it into the net."""
    gate = circ.add(Component(name, loc, attrs, lib_id))
    try:
        shape = geometry.shape(project, gate)
    except geometry.UnknownGeometry as e:
        circ.components.remove(gate)
        raise EditError(f"cannot determine ports of {gate!r}: {e}") from None
    ins = [p.name for p in shape.ports if p.dir == "in"]
    outs = [p.name for p in shape.ports if p.dir == "out"]
    if not ins or not outs:
        circ.components.remove(gate)
        raise EditError(f"{name!r} needs at least one input and one output port")
    if in_name not in ins or out_name not in outs:
        circ.components.remove(gate)
        raise EditError(f"{name!r} has no port {in_name!r}/{out_name!r}; "
                        f"inputs are {ins}, outputs are {outs}")
    circ.wires = [w for i, w in enumerate(circ.wires) if i not in doomed]
    grid, nl2, _ = _build_grid(project, circ)

    def sink(pt_ref):
        c = circ.components[pt_ref.comp]
        p = geometry.port(project, c, pt_ref.port.name)
        return p.pos, geometry.shape(project, c), pt_ref.port.name

    src_net = nl2.port_net[(src.comp, src.port.name)]
    gate_in_pt = geometry.port(project, gate, in_name).pos
    in_sink = (gate_in_pt, shape, in_name)
    driver_sinks = [in_sink] + [sink(p) for p in unselected]
    _route_fanout(project, circ, grid, src_net, geometry.port(
        project, circ.components[src.comp], src.port.name).pos, driver_sinks)

    gate_out_net = nl2.port_net[(len(circ.components) - 1, out_name)]
    out_pt = geometry.port(project, gate, out_name).pos
    _route_fanout(project, circ, grid, gate_out_net, out_pt, [sink(p) for p in selected])

    group_net = {(src.comp, src.port.name): src_net}
    group_net.update({(p.comp, p.port.name): gate_out_net for p in selected})
    group_net.update({(p.comp, p.port.name): src_net for p in unselected})
    for extra, target in _connection_items(extra_inputs):
        if extra not in ins or extra == in_name:
            raise EditError(f"extra input {extra!r} must be an unused input of {name!r}")
        eci, eport, _ = _resolve_port(project, circ, target)
        target_net = group_net.get((eci, eport), nl2.port_net.get((eci, eport)))
        if target_net is None:
            raise EditError("cannot find the net for an extra input")
        pt = geometry.port(project, gate, extra).pos
        _route_to_net(project, circ, grid, nl2, target_net, pt, layout.port_side(shape, extra))
    return gate


# ------------------------------------------------------------ circuits / instances


def add_circuit(project: Project, name: str, *, attrs: Mapping[str, object] | None = None,
                main: bool = False) -> Circuit:
    """Add a new empty circuit with a Logisim-safe name."""
    if not builder.valid_circuit_name(name):
        raise EditError(f"bad circuit name {name!r}: Logisim renames it on load; use a letter, "
                        "then letters, digits and single underscores, not ending in '_'")
    if project.has_circuit(name):
        raise EditError(f"circuit {name!r} already exists")
    circ = project.add_circuit(Circuit(name, attrs={k: str(v) for k, v in (attrs or {}).items()}))
    if main:
        project.main = name
    return circ


def add_instance(project: Project, circuit: Circuit | str, subcircuit, *, label: str = "",
                 attrs: Mapping[str, object] | None = None, loc="auto",
                 connections=None) -> ComponentInfo:
    """Add an instance of another circuit in the same project."""
    sub_name = subcircuit.name if isinstance(subcircuit, Circuit) else str(subcircuit)
    if not project.has_circuit(sub_name):
        raise EditError(f"no circuit named {sub_name!r} to instance")
    values = {k: str(v) for k, v in (attrs or {}).items()}
    if label:
        values["label"] = label
    return add_component(project, circuit, sub_name, lib=None, attrs=values, loc=loc,
                         connections=connections)


# -------------------------------------------------------------------- wires


def _wire_keys(wires) -> list[Wire]:
    if wires is None:
        return []
    if isinstance(wires, Wire):
        return [wires]
    if (isinstance(wires, tuple) and len(wires) == 2 and
            all(isinstance(p, tuple) and len(p) == 2 for p in wires)):
        return [Wire(wires[0], wires[1])]
    out = []
    for w in wires:
        if isinstance(w, Wire):
            out.append(w)
        elif isinstance(w, (tuple, list)) and len(w) == 2:
            out.append(Wire(w[0], w[1]))
        else:
            raise EditError(f"not a wire: {w!r}")
    return out


def delete_wires(project: Project, circuit: Circuit | str, *, wires=None, endpoints=None,
                 point: Point | None = None) -> list[Wire]:
    """Delete wires by object/endpoints, or every wire touching ``point``.

    Afterwards :func:`cleanup_wires` removes pieces that no longer reach a port
    or another wire and merges collinear neighbours.
    """
    circ = _as_circuit(project, circuit)
    if wires is None and endpoints is None and point is None:
        raise EditError("give wires=, endpoints= or point= to delete_wires")
    wanted = {w.key() for w in _wire_keys(wires)}
    if endpoints is not None:
        wanted |= {w.key() for w in _wire_keys(endpoints)}
    doomed: set[int] = set()
    for i, w in enumerate(circ.wires):
        if w.key() in wanted or (point is not None and w.contains(point)):
            doomed.add(i)
    removed = [w for i, w in enumerate(circ.wires) if i in doomed]
    circ.wires = [w for i, w in enumerate(circ.wires) if i not in doomed]
    cleanup_wires(project, circ)
    return removed


def cleanup_wires(project: Project, circuit: Circuit | str, *, preserve: frozenset | None = None,
                 exclude: int | None = None) -> list[Wire]:
    """Remove dangling wire pieces and merge collinear segments, repeatedly.

    A wire end is dangling when it is not at a component port and no other wire
    touches it.  Collinear neighbours are merged only where Logisim would (two
    wire ends at a point with no port); this never removes a T-junction.

    ``preserve`` is a port partition (see :func:`_partition`): a wire is only
    removed when the nets of the circuit stay the same, so a cleanup never
    breaks a connection.  ``exclude`` leaves one component out of that
    partition (used when the component's own ports are being changed).
    """
    circ = _as_circuit(project, circuit)
    removed: list[Wire] = []

    def port_points() -> set[Point]:
        pts: set[Point] = set()
        for c in circ.components:
            try:
                pts |= {p.pos for p in geometry.ports(project, c)}
            except geometry.UnknownGeometry:
                continue
        return pts

    def may_remove(index: int) -> bool:
        if preserve is None:
            return True
        w = circ.wires.pop(index)
        ok = _partition(netlist.extract(project, circ), exclude) == preserve
        circ.wires.insert(index, w)
        return ok

    changed = True
    while changed:
        changed = False
        # zero-length and duplicate wires
        seen: set[tuple[Point, Point]] = set()
        kept: list[Wire] = []
        for w in circ.wires:
            if w.a == w.b:
                removed.append(w)
                changed = True
                continue
            k = w.key()
            if k in seen:
                removed.append(w)
                changed = True
                continue
            seen.add(k)
            kept.append(w)
        circ.wires = kept

        pts = port_points()
        horiz, vert = netlist._index_wires(circ)
        doomed: list[int] = []
        for i, w in enumerate(circ.wires):
            for end in (w.a, w.b):
                if end in pts:
                    continue
                others = [j for j in netlist.wires_through(circ, end, horiz, vert) if j != i]
                if not others:
                    doomed.append(i)
                    break
        for i in reversed(doomed):
            if not may_remove(i):
                continue
            removed.append(circ.wires.pop(i))
            changed = True
        if changed:
            continue

        # merge two collinear wires whose shared end has exactly two incidences
        ends: dict[Point, list[int]] = defaultdict(list)
        for i, w in enumerate(circ.wires):
            ends[w.a].append(i)
            ends[w.b].append(i)
        consumed: set[int] = set()
        merged: list[Wire] = []
        for pt, at in ends.items():
            if len(at) != 2 or pt in pts:
                continue
            i, j = at
            if i in consumed or j in consumed:
                continue
            a, b = circ.wires[i], circ.wires[j]
            if a.horizontal != b.horizontal:
                continue
            new = _merge_collinear(a, b, pt)
            if new is None:
                continue
            consumed |= {i, j}
            merged.append(new)
            changed = True
        if changed:
            circ.wires = [w for i, w in enumerate(circ.wires) if i not in consumed] + merged
    return removed


def _merge_collinear(a: Wire, b: Wire, pt: Point) -> Wire | None:
    """Union of two collinear wires that leave ``pt`` in opposite directions."""
    axis = 0 if a.horizontal else 1
    a_lo, a_hi = sorted((a.a[axis], a.b[axis]))
    b_lo, b_hi = sorted((b.a[axis], b.b[axis]))
    if a_hi == b_lo == pt[axis]:
        return Wire(a.a if a.a[axis] == a_lo else a.b, b.a if b.a[axis] == b_hi else b.b)
    if b_hi == a_lo == pt[axis]:
        return Wire(b.a if b.a[axis] == b_lo else b.b, a.a if a.a[axis] == a_hi else a.b)
    return None


# --------------------------------------------------------------------- merge


def _unique_circuit_name(base: str, taken: set[str], prefix: str | None) -> str:
    stem = (prefix or "") + base
    if not builder.valid_circuit_name(stem):
        raise EditError(f"bad circuit name {stem!r} from merge prefix {prefix!r}")
    if stem not in taken:
        return stem
    for k in itertools.count(1):
        candidate = stem + ("_copy" if k == 1 else f"_copy{k}")
        if builder.valid_circuit_name(candidate) and candidate not in taken:
            return candidate
    raise EditError("could not find a free circuit name")  # pragma: no cover


def merge_projects(target: Project, source: Project, *, circuits: Iterable[str] | None = None,
                   rename: Mapping[str, str] | None = None,
                   prefix: str | None = None) -> dict[str, str]:
    """Copy circuits (and their subcircuits) from ``source`` into ``target``.

    Name clashes get ``_copy``, ``_copy2``, ... (or an explicit ``rename``
    mapping).  Libraries used by the copied circuits are added to the target.
    Returns the mapping ``{old circuit name: new circuit name}``.
    """
    if target is source:
        raise EditError("cannot merge a project into itself")
    src_names = set(source.circuit_names())
    chosen = list(circuits) if circuits is not None else list(source.circuit_names())
    for name in chosen:
        if name not in src_names:
            raise EditError(f"source has no circuit named {name!r}")

    closure: set[str] = set()

    def visit(name: str) -> None:
        if name in closure:
            return
        closure.add(name)
        for comp in source.circuit(name).components:
            if comp.lib is None and comp.name in src_names:
                visit(comp.name)

    for name in chosen:
        visit(name)
    order = [c.name for c in source.circuits if c.name in closure]

    taken = set(target.circuit_names())
    mapping: dict[str, str] = {}
    for old in order:
        if rename is not None and old in rename:
            new = str(rename[old])
            if not builder.valid_circuit_name(new):
                raise EditError(f"bad circuit name {new!r} in rename mapping")
            if new in taken:
                raise EditError(f"rename target {new!r} already exists in the project")
        else:
            new = old if old not in taken else _unique_circuit_name(old, taken, prefix)
        mapping[old] = new
        taken.add(new)

    lib_map: dict[str, str] = {}
    for lib in source.libraries:
        if not lib.desc:
            continue
        tid = target.lib_id(lib.desc)
        lib_map[lib.id] = tid
        tlib = target.library(tid)
        if tlib is not None and lib.tools:
            existing = {ET.tostring(t, encoding="unicode") for t in tlib.tools}
            for tool in lib.tools:
                key = ET.tostring(tool, encoding="unicode")
                if key not in existing:
                    tlib.tools.append(copy.deepcopy(tool))
                    existing.add(key)

    for old in order:
        new = mapping[old]
        nc = source.circuit(old).copy()
        nc.name = new
        nc.attrs = dict(nc.attrs)
        nc.attrs["circuit"] = new
        for comp in nc.components:
            if comp.lib is None:
                if comp.name in mapping:
                    comp.name = mapping[comp.name]
            elif comp.lib in lib_map:
                comp.lib = lib_map[comp.lib]
        target.circuits.append(nc)

    if target.main is None:
        if source.main in mapping:
            target.main = mapping[source.main]
        elif order:
            target.main = mapping[order[0]]
    return mapping


# ---------------------------------------------------------------------- tidy


def _model_to_builder(project: Project, circ: Circuit, spacing: str = "roomy") -> builder.Circuit:
    """Translate the current drawing into a builder circuit describing the same nets."""
    appearance = circ.attrs.get("appearance", "logisim_evolution")
    bcirc = builder.Circuit(circ.name, spacing=spacing, use_tunnels=False, appearance=appearance)
    builder.Project(bcirc, base=project)  # subcircuit instances need their geometry
    node_of: dict[int, builder.Node] = {}
    for i, comp in enumerate(circ.components):
        desc = project.lib_desc(comp)
        attrs = {k: v for k, v in comp.attrs.items() if not k.startswith("_")}
        if comp.name == "Pin" and "labelloc" not in attrs:
            # Logisim's implicit label side follows the pin's facing, but label_box
            # assumes west.  State the side the label really occupies, otherwise a
            # west-facing output pin's label blocks the only way out of its port.
            attrs["labelloc"] = {"east": "west", "west": "east",
                                 "south": "north", "north": "south"}.get(
                                     attrs.get("facing", "east"), "west")
        try:
            shape = geometry.shape(project, comp)
        except geometry.UnknownGeometry as e:
            raise EditError(f"cannot tidy {circ.name!r}: {e}") from None
        if comp.name == "Pin":
            if comp.get("output", "false") == "true":
                role = "output"
            elif comp.label.lower().rstrip("0123456789_") in builder.CONTROL_NAMES:
                role = "control"
            else:
                role = "input"
            primary = "p"
        elif comp.name == "Clock":
            role, primary = "control", "out"
        elif comp.name == "Constant":
            role, primary = "const", "out"
        else:
            role = "logic"
            outs = [p.name for p in shape.ports if p.dir == "out"]
            primary = next((n for n in ("out", "q", "comb", "p0", "data") if n in outs),
                           outs[0] if outs else (shape.ports[0].name if shape.ports else "out"))
        node = bcirc._add(desc, comp.name, attrs, role=role, primary=primary)
        node_of[i] = node

    nl = netlist.extract(project, circ)
    sources: dict[int, tuple[int, str]] = {}
    for net in nl.nets:
        refs = [(p.comp, p.port.name) for p in net.ports if p.comp in node_of]
        if len(refs) < 2:
            continue
        outs = [(ci, pn) for ci, pn in refs
                if next(q.port.dir for q in net.ports if q.comp == ci and q.port.name == pn) == "out"]
        inouts = [(ci, pn) for ci, pn in refs
                  if next(q.port.dir for q in net.ports if q.comp == ci and q.port.name == pn) == "inout"]
        src = outs[0] if outs else (inouts[0] if inouts else refs[0])
        sources[net.id] = src
        if circ.components[src[0]].name == "Splitter":
            bcirc._splitter_outputs.setdefault(id(node_of[src[0]]), set()).add(src[1])

    for net in nl.nets:
        src = sources.get(net.id)
        if src is None:
            continue
        st = builder.Terminal(node_of[src[0]], src[1])
        for ref in net.ports:
            if ref.comp not in node_of:
                continue
            if (ref.comp, ref.port.name) == src:
                continue
            try:
                bcirc.connect(st, builder.Terminal(node_of[ref.comp], ref.port.name))
            except builder.BuildError as e:
                raise EditError(f"cannot tidy {circ.name!r}: {e}") from None
    return bcirc


def tidy(project: Project, circuit: Circuit | str, *, spacing: str = "roomy") -> Circuit:
    """Re-lay out a circuit from its netlist, preserving connectivity exactly.

    Old wires and tunnels are discarded; every net's driver/sinks are derived
    from :func:`netlist.extract` and drawn again with
    :func:`layout.place_and_route`.  The layout-independent netlist signature
    must be unchanged, otherwise :class:`EditError` is raised.

    ``spacing`` is the layout density ("roomy", "normal" or "compact"); the
    project default is "roomy".
    """
    circ = _as_circuit(project, circuit)
    before = netlist.extract(project, circ).signature()
    bcirc = _model_to_builder(project, circ, spacing)
    try:
        new_circ = layout.place_and_route(bcirc, project)
    except (builder.BuildError, RuntimeError, geometry.UnknownGeometry) as e:
        raise EditError(f"could not tidy {circ.name!r}: {e}") from None
    old_attrs, old_extras = dict(circ.attrs), circ.extras
    new_circ.attrs = {**old_attrs, **new_circ.attrs}
    new_circ.extras = [copy.deepcopy(e) for e in old_extras]
    index = next(i for i, c in enumerate(project.circuits) if c is circ)
    project.circuits[index] = new_circ
    after = netlist.extract(project, new_circ).signature()
    if after != before:
        project.circuits[index] = circ
        missing, extra = before - after, after - before
        detail = ""
        if missing:
            detail += f" lost {len(missing)} net(s)"
        if extra:
            detail += f" gained {len(extra)} net(s)"
        raise EditError(f"tidy of {circ.name!r} would change connectivity:{detail}. "
                        "Tunnels and components with unknown geometry cannot be tidied safely.")
    return new_circ


# ----------------------------------------------------------------------- save


def save(project: Project, path: str | os.PathLike, *, in_place: bool = False) -> Path:
    """Write the project to `path`.

    An existing file is never replaced silently: pass ``in_place=True`` to
    overwrite it; the previous contents are then kept as ``<name>.circ.bak``.
    """
    path = Path(path)
    if path.exists() and not in_place:
        raise FileExistsError(f"{path} exists; pass in_place=True to overwrite it "
                              "(a .bak copy of the old file is kept)")
    return io.save(project, path, overwrite=in_place, backup=True)
