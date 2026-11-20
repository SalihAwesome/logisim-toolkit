"""Orthogonal wire routing on Logisim's 10-unit grid.

Rules come from Logisim's connectivity semantics (see netlist.py): a wire
touching a point connects to it unless it merely crosses another wire. So a
route for net N must never
  * pass through a component body,
  * run along (overlap) a wire of another net,
  * pass through another net's port, wire endpoint or bend,
  * end or bend on another net's wire,
  * end where its own net already has three arms: Logisim merges the two straight
    arms that meet there into one wire, which then only crosses the third (see netlist.py).
Crossing another net's wire at a right angle is allowed but costs extra.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .model import Point

G = 10
DIRS = ((G, 0), (-G, 0), (0, G), (0, -G))


@dataclass
class Costs:
    step: float = 1.0
    bend: float = 6.0
    cross: float = 10.0
    near_wire: float = 4.0  # running parallel one grid line away from another net
    near_body: float = 3.0  # hugging a component body
    vertical_drift: float = 0.01  # per grid step of distance from the start: keeps jogs near the sink


@dataclass
class Grid:
    """Occupancy of the drawing, kept in sync as nets are routed."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int
    blocked: set[Point] = field(default_factory=set)  # component bodies
    near_body: set[Point] = field(default_factory=set)
    # point -> net id for ports, wire endpoints and bends (hard points)
    hard: dict[Point, int] = field(default_factory=dict)
    # point -> {net: orientation set} for wire interiors ("h", "v")
    run: dict[Point, dict[int, set[str]]] = field(default_factory=dict)
    # point -> {net: directions its wires leave the point in}
    arms: dict[Point, dict[int, set[tuple[int, int]]]] = field(default_factory=dict)

    def inside(self, p: Point) -> bool:
        return self.xmin <= p[0] <= self.xmax and self.ymin <= p[1] <= self.ymax

    def block_rect(self, x: int, y: int, w: int, h: int, keep: set[Point]) -> None:
        x0, y0 = -(-x // G) * G, -(-y // G) * G
        for gx in range(x0, x + w + 1, G):
            for gy in range(y0, y + h + 1, G):
                if (gx, gy) not in keep:
                    self.blocked.add((gx, gy))
        for gx in range(x0 - G, x + w + G + 1, G):
            for gy in range(y0 - G, y + h + G + 1, G):
                self.near_body.add((gx, gy))

    def add_hard(self, p: Point, net: int) -> None:
        self.hard[p] = net

    def add_path(self, pts: list[Point], net: int) -> None:
        """Record a routed polyline (corner points) for `net`."""
        for p in (pts[0], pts[-1]):
            self.hard.setdefault(p, net)
        for i in range(1, len(pts) - 1):
            self.hard[pts[i]] = net
        for a, b in zip(pts, pts[1:]):
            orient = "h" if a[1] == b[1] else "v"
            seg = _segment_points(a, b)
            for p in seg:
                self.run.setdefault(p, {}).setdefault(net, set()).add(orient)
            for p, q in zip(seg, seg[1:]):
                step = (q[0] - p[0], q[1] - p[1])
                self.arms.setdefault(p, {}).setdefault(net, set()).add(step)
                self.arms.setdefault(q, {}).setdefault(net, set()).add((-step[0], -step[1]))

    def net_points(self, net: int) -> set[Point]:
        pts = {p for p, n in self.hard.items() if n == net}
        pts |= {p for p, nets in self.run.items() if net in nets}
        return pts


def _segment_points(a: Point, b: Point) -> list[Point]:
    if a[0] == b[0]:
        step = G if b[1] >= a[1] else -G
        return [(a[0], y) for y in range(a[1], b[1] + step, step)]
    step = G if b[0] >= a[0] else -G
    return [(x, a[1]) for x in range(a[0], b[0] + step, step)]


def _foreign_run(grid: Grid, p: Point, net: int) -> set[str]:
    out: set[str] = set()
    for n, o in grid.run.get(p, {}).items():
        if n != net:
            out |= o
    return out


def route(grid: Grid, net: int, start: Point, targets: set[Point], costs: Costs | None = None,
          start_dir: tuple[int, int] | None = None) -> list[Point] | None:
    """Cheapest path from `start` to any point in `targets`, as corner points.

    `start_dir`, if given, forces the first step's direction (leave a port outward).
    """
    costs = costs or Costs()
    targets = set(targets) - {start}
    if not targets:
        return None
    tx = [t[0] for t in targets]
    ty = [t[1] for t in targets]
    # rows/columns a net must use anyway (its own ports); no spacing penalty there
    natural_rows = {start[1], *ty}
    natural_cols = {start[0], *tx}
    bx0, bx1, by0, by1 = min(tx), max(tx), min(ty), max(ty)

    def h(p: Point) -> float:
        dx = max(bx0 - p[0], 0, p[0] - bx1)
        dy = max(by0 - p[1], 0, p[1] - by1)
        return (dx + dy) / G * costs.step

    def ok_hard(p: Point) -> bool:
        owner = grid.hard.get(p)
        return owner is None or owner == net

    def turn_ok(p: Point) -> bool:
        return not _foreign_run(grid, p, net)

    start_state = (start, None)
    best: dict = {start_state: 0.0}
    prev: dict = {}
    heap = [(h(start), 0.0, start, None)]
    while heap:
        f, g, p, d = heapq.heappop(heap)
        if best.get((p, d), float("inf")) < g:
            continue
        if p in targets and p != start:
            # cannot terminate on another net's wire interior, nor make a 4-way junction
            own_arms = grid.arms.get(p, {}).get(net, set())
            four_way = d is not None and len(own_arms | {(-d[0], -d[1])}) == 4
            if (turn_ok(p) or grid.hard.get(p) == net) and not four_way:
                path = [p]
                key = (p, d)
                while key in prev:
                    key = prev[key]
                    path.append(key[0])
                path.reverse()
                return _corners(path)
        for nd in DIRS:
            if d is None and start_dir is not None and nd != start_dir:
                continue
            if d is not None and nd == (-d[0], -d[1]):
                continue
            q = (p[0] + nd[0], p[1] + nd[1])
            if not grid.inside(q) or q in grid.blocked and q not in targets:
                continue
            if not ok_hard(q):
                continue
            if nd in grid.arms.get(p, {}).get(net, ()):
                continue  # would retrace this net's own wire
            orient = "h" if nd[1] == 0 else "v"
            frun = _foreign_run(grid, q, net)
            if orient in frun:
                continue  # would overlap another net's wire
            bend = d is not None and nd != d
            if bend and not turn_ok(p):
                continue  # a bend on another net's wire would connect to it
            cost = costs.step
            if bend:
                cost += costs.bend
            if frun:
                cost += costs.cross
            if q in grid.near_body and q not in targets:
                cost += costs.near_body
            if orient == "v":
                cost += costs.vertical_drift * abs(q[0] - start[0]) / G
            natural = q[1] in natural_rows if orient == "h" else q[0] in natural_cols
            if not natural:
                for side in ((G, 0), (-G, 0)) if orient == "v" else ((0, G), (0, -G)):
                    s = (q[0] + side[0], q[1] + side[1])
                    if orient in _foreign_run(grid, s, net):
                        cost += costs.near_wire
                        break
            ng = g + cost
            key = (q, nd)
            if ng < best.get(key, float("inf")):
                best[key] = ng
                prev[key] = (p, d)
                heapq.heappush(heap, (ng + h(q), ng, q, nd))
    return None


def _corners(path: list[Point]) -> list[Point]:
    if len(path) <= 2:
        return path
    out = [path[0]]
    for a, b, c in zip(path, path[1:], path[2:]):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            out.append(b)
    out.append(path[-1])
    return out


def segments(corners: list[Point]) -> list[tuple[Point, Point]]:
    return [(a, b) for a, b in zip(corners, corners[1:]) if a != b]
