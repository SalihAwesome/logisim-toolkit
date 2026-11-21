"""Placement and routing for builder circuits.

1. Layering: data inputs in column 0, outputs in the last column, every other
   component at its longest-path depth (feedback edges removed), then pulled
   right toward its consumers. Control inputs (clock, reset, enable, select)
   go in a row above the circuit.
2. Ordering: barycenter sweeps reduce crossings in the logic columns while
   pins stay in declaration order.
3. Coordinates: column x from the widest component plus a routing channel
   sized by the nets that jog in it; row y solved per column so each
   component's inputs line up with their drivers (isotonic regression keeps
   the order and the minimum gap).
4. Routing: A* per sink toward the net's existing tree (route.py). A sink that
   cannot be routed gets a labelled tunnel pair instead, so nothing is left
   unconnected.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil

from . import geometry, model, route
from .builder import CONTROL_PORTS, Circuit, Node, Terminal

G = 10

SPACING = {
    #          col_gap track row_gap margin ctrl_gap
    "roomy":   (100, 20, 60, 60, 80),
    "normal":  (80, 20, 40, 50, 60),
    "compact": (50, 20, 20, 40, 40),
}

_DUMMY = geometry.Shape((), (0, 0, 0, 0))

LABEL_CHAR = geometry.LABEL_CHAR


def snap(v: float) -> int:
    return int(round(v / G)) * G


@dataclass(eq=False)
class Placed:
    node: Node | None  # None for a placeholder reserving a long net's corridor
    shape: geometry.Shape
    layer: int
    x: int = 0  # loc
    y: int = 0
    in_links: list = field(default_factory=list)   # (driver Placed, driver port, my port)
    out_links: list = field(default_factory=list)  # (my port, sink Placed, sink port)

    def port_off(self, name: str | None) -> int:
        return 0 if name is None else self.shape.port(name).y

    def port_y(self, name: str | None) -> int:
        return self.y + self.port_off(name)

    @property
    def top(self) -> int:
        return self.shape.bounds[1]

    @property
    def bottom(self) -> int:
        return self.shape.bounds[1] + self.shape.bounds[3]

    def port_abs(self, name: str) -> tuple[int, int]:
        p = self.shape.port(name)
        return self.x + p.x, self.y + p.y


def port_side(shape: geometry.Shape, port: str) -> tuple[int, int]:
    """Direction a wire leaves `port`: away from the body, toward the nearest bounding-box edge."""
    sp = shape.port(port)
    bx, by, bw, bh = shape.bounds
    if sp.x <= bx:
        return (-G, 0)
    if sp.x >= bx + bw:
        return (G, 0)
    if sp.y <= by:
        return (0, -G)
    if sp.y >= by + bh:
        return (0, G)
    dists = {(-G, 0): sp.x - bx, (G, 0): bx + bw - sp.x, (0, -G): sp.y - by, (0, G): by + bh - sp.y}
    return min(dists, key=dists.get)


def logical_port(node: Node, port: str) -> str:
    alias = node.attrs.get("_aliases")
    if alias:
        for pair in alias.split(","):
            k, v = pair.split("=")
            if v == port:
                return k
    return port


def is_control_edge(sink: Node, port: str) -> bool:
    return logical_port(sink, port) in CONTROL_PORTS


def _layers(circ: Circuit) -> dict[int, int]:
    nodes = [n for n in circ.nodes if n.role != "control"]
    ids = {id(n) for n in nodes}
    preds: dict[int, list[Node]] = defaultdict(list)
    succs: dict[int, list[Node]] = defaultdict(list)
    output_drivers: set[int] = set()
    other_drivers: set[int] = set()
    for n in circ.nodes:
        drivers = output_drivers if n.role == "output" else other_drivers
        for port, src in n.inputs.items():
            source = id(src.node)
            drivers.add(source)
            if id(n) in ids and source in ids and not is_control_edge(n, port):
                preds[id(n)].append(src.node)
                succs[source].append(n)

    # Remove feedback: DFS from sources, drop back edges.
    back: set[tuple[int, int]] = set()
    state: dict[int, int] = {}
    order = sorted(nodes, key=lambda n: (0 if n.role == "input" else 1 if not preds[id(n)] else 2, n.index))
    for root in order:
        if id(root) in state:
            continue
        stack = [(root, iter(succs[id(root)]))]
        state[id(root)] = 1
        while stack:
            n, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                state[id(n)] = 2
                stack.pop()
                continue
            s = state.get(id(nxt), 0)
            if s == 1:
                back.add((id(n), id(nxt)))
            elif s == 0:
                state[id(nxt)] = 1
                stack.append((nxt, iter(succs[id(nxt)])))

    fpreds = {k: [p for p in v if (id(p), k) not in back] for k, v in preds.items()}
    fsuccs: dict[int, list[Node]] = defaultdict(list)
    for k, v in fpreds.items():
        for p in v:
            fsuccs[id(p)].append(next(n for n in nodes if id(n) == k))

    layer: dict[int, int] = {}

    def depth(n: Node) -> int:
        if id(n) in layer:
            return layer[id(n)]
        stack = [n]
        while stack:
            cur = stack[-1]
            pending = [p for p in fpreds.get(id(cur), []) if id(p) not in layer]
            if pending:
                stack.extend(pending)
                continue
            stack.pop()
            if id(cur) in layer:
                continue
            if cur.role == "input":
                layer[id(cur)] = 0
            else:
                ps = fpreds.get(id(cur), [])
                layer[id(cur)] = 1 + max((layer[id(p)] for p in ps), default=0)
        return layer[id(n)]

    for n in nodes:
        depth(n)
    logic_max = max((layer[id(n)] for n in nodes if n.role not in ("input", "output")), default=0)
    output_only = output_drivers - other_drivers
    # Pull sources and early nodes right, next to their first consumer.
    # Output-only drivers go immediately before the shared output column.
    for n in sorted(nodes, key=lambda n: -layer[id(n)]):
        if n.role in ("input", "output"):
            continue
        ss = [layer[id(s)] for s in fsuccs.get(id(n), []) if s.role != "output"]
        if ss:
            layer[id(n)] = max(layer[id(n)], min(ss) - 1)
        elif id(n) in output_only:
            layer[id(n)] = logic_max
    for n in nodes:
        if n.role == "output":
            layer[id(n)] = logic_max + 1
        elif n.role != "input":
            layer[id(n)] = max(1, layer[id(n)])
    return layer


def _isotonic(desired: list[float], spacing: list[float]) -> list[float]:
    """Minimise sum (y_i - d_i)^2 subject to y_{i+1} - y_i >= spacing_i (pool adjacent violators)."""
    c = [0.0]
    for s in spacing:
        c.append(c[-1] + s)
    target = [d - ci for d, ci in zip(desired, c)]
    blocks: list[list[float]] = []  # [sum, count]
    for t in target:
        blocks.append([t, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, k = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += k
    z: list[float] = []
    for s, k in blocks:
        z += [s / k] * k
    return [zi + ci for zi, ci in zip(z, c)]


def place_and_route(circ: Circuit, project: model.Project) -> model.Circuit:
    col_gap, track, row_gap, margin, ctrl_gap = SPACING.get(circ.spacing, SPACING["roomy"])
    layer = _layers(circ)
    placed: dict[int, Placed] = {}
    for n in circ.nodes:
        if n.role == "control":
            n.attrs["facing"] = "south"
            if n.name == "Pin":
                n.attrs["labelloc"] = "north"
            placed[id(n)] = Placed(n, n.shape(), -1)
        else:
            placed[id(n)] = Placed(n, n.shape(), layer[id(n)])
    nlayers = max((p.layer for p in placed.values()), default=0) + 1
    cols: list[list[Placed]] = [[] for _ in range(nlayers)]
    for n in circ.nodes:
        p = placed[id(n)]
        if p.layer >= 0:
            cols[p.layer].append(p)

    # ---- links: data edges between consecutive columns; long nets get dummy placeholders
    drivers = circ.driven()
    for (nid, port), sinks in drivers.items():
        src = placed[nid]
        if src.layer < 0:
            continue
        data = [t for t in sinks if not is_control_edge(t.node, t.port) and placed[id(t.node)].layer > src.layer]
        if not data:
            continue
        far = max(placed[id(t.node)].layer for t in data)
        chain: dict[int, tuple[Placed, str | None]] = {src.layer: (src, port)}
        prev: tuple[Placed, str | None] = (src, port)
        for k in range(src.layer + 1, far):
            d = Placed(None, _DUMMY, k)
            cols[k].append(d)
            d.in_links.append((prev[0], prev[1], None))
            prev[0].out_links.append((prev[1], d, None))
            prev = (d, None)
            chain[k] = prev
        for t in data:
            sp = placed[id(t.node)]
            q, qport = chain[sp.layer - 1]
            sp.in_links.append((q, qport, t.port))
            q.out_links.append((qport, sp, t.port))

    def positions() -> dict[int, float]:
        pos = {}
        for col in cols:
            for i, p in enumerate(col):
                pos[id(p)] = (i + 0.5) / max(len(col), 1)
        return pos

    def reorder(col: list[Placed], use_in: bool) -> None:
        pos = positions()

        def key(p: Placed):
            ns = [pos[id(q)] for q, _, _ in p.in_links] if use_in else [pos[id(r)] for _, r, _ in p.out_links]
            return (sum(ns) / len(ns)) if ns else pos[id(p)]
        col.sort(key=key)

    has_outputs = any(n.role == "output" for n in circ.nodes)
    logic_columns = range(1, nlayers - 1 if has_outputs else nlayers)
    for _ in range(4):
        for k in logic_columns:
            reorder(cols[k], True)
        for k in reversed(logic_columns):
            reorder(cols[k], False)
    # Leave the last logic column ordered by its sinks in the fixed output column.
    for k in logic_columns[:-1] if has_outputs else logic_columns:
        reorder(cols[k], True)

    # ---- x coordinates
    controls = [placed[id(n)] for n in circ.nodes if n.role == "control"]
    jogs = [0] * nlayers  # nets that need a vertical segment in the channel left of column k
    for (nid, port), sinks in drivers.items():
        src = placed[nid]
        for k in {placed[id(t.node)].layer for t in sinks}:
            if k > 0:
                jogs[k] += 1 if src.layer >= 0 else 2
    widths = [max((p.shape.bounds[2] for p in col), default=0) for col in cols]
    x = margin + max((LABEL_CHAR * len(p.node.label) for p in cols[0] if p.node and p.node.name == "Pin"),
                     default=0)
    col_left: list[int] = []
    for k in range(nlayers):
        if k > 0:
            x += col_gap + track * max(jogs[k] - 1, 0)
        x = snap(x)
        col_left.append(x)
        for p in cols[k]:
            bx, _, bw, _ = p.shape.bounds
            if p.node is not None and p.node.role == "input":
                p.x = snap(x + widths[k] - (bx + bw))  # right-align pins on the column edge
            else:
                p.x = snap(x - bx)
        x += widths[k]

    # ---- y coordinates
    def solve_column(col: list[Placed], desired: dict[int, float]) -> None:
        if not col:
            return
        d: list[float] = []
        for i, p in enumerate(col):
            want = desired.get(id(p))
            if want is None:
                want = (d[-1] + col[i - 1].bottom - p.top + row_gap) if d else 0.0
            d.append(want)
        sp = [col[i].bottom - col[i + 1].top + row_gap for i in range(len(col) - 1)]
        for p, y in zip(col, _isotonic(d, sp)):
            p.y = snap(y)
        for i in range(1, len(col)):  # snapping can eat into the gap
            need = col[i - 1].y + col[i - 1].bottom + row_gap - col[i].top
            if col[i].y < need:
                col[i].y = snap(need + G / 2)

    def want_from_drivers(p: Placed) -> float | None:
        vals = sorted(q.port_y(qp) - p.port_off(mp) for q, qp, mp in p.in_links if q.layer < p.layer)
        # low median: when inputs disagree, keep the upper one straight instead of bending all of them
        return vals[(len(vals) - 1) // 2] if vals else None

    def want_from_sinks(p: Placed) -> float | None:
        links = [(mp, r, rp) for mp, r, rp in p.out_links if r.layer > p.layer]
        if not links:
            return None
        nearest = min(r.layer for _, r, _ in links)
        vals = [r.port_y(rp) - p.port_off(mp) for mp, r, rp in links if r.layer == nearest]
        return sum(vals) / len(vals)

    def sweep(k: int, fn) -> None:
        solve_column(cols[k], {id(p): w for p in cols[k] if (w := fn(p)) is not None})

    solve_column(cols[0], {})
    for _ in range(4):
        for k in range(1, nlayers):
            sweep(k, want_from_drivers)
        for k in range(nlayers - 2, -1, -1):
            sweep(k, want_from_sinks)
    for k in range(1, nlayers):
        sweep(k, want_from_drivers)

    # normalise so the top of the drawing sits below the control row
    body = [p for col in cols for p in col]
    top = min((p.y + p.top for p in body if p.node is not None), default=0)
    ctrl_space = (ctrl_gap + 40) if controls else 0
    shift = snap(margin + ctrl_space - top)
    for p in body:
        p.y += shift
    for col in cols:
        col[:] = [p for p in col if p.node is not None]

    # ---- control row: control inputs sit above (or below) the channel left of their first sink
    real = [p for col in cols for p in col]
    body_bottom = max((p.y + p.bottom for p in real), default=0)
    control_x: list[float] = []
    control_widths: list[int] = []
    for p in controls:
        sinks = [t for (nid, _), ts in drivers.items() if nid == id(p.node) for t in ts]
        down = 0
        for t in sinks:
            down += 1 if port_side(placed[id(t.node)].shape, t.port) == (0, G) else -1
        row = "bottom" if down > 0 else "top"
        p.node.attrs["facing"] = "north" if row == "bottom" else "south"
        p.node.attrs["labelloc"] = "south" if row == "bottom" else "north"
        p.shape = p.node.shape()
        ks = [placed[id(t.node)].layer for t in sinks if placed[id(t.node)].layer >= 0]
        k = max(min(ks) if ks else 1, 1)
        vertical = [t for t in sinks if port_side(placed[id(t.node)].shape, t.port)[0] == 0]
        if len(sinks) == 1 and vertical:
            want_x = placed[id(vertical[0].node)].port_abs(vertical[0].port)[0]
        else:
            want_x = col_left[k] - col_gap // 2 if k < len(col_left) else x + 40
        control_x.append(want_x)
        control_widths.append(max(40, LABEL_CHAR * len(p.node.label) + 20))
        p.y = snap(margin + 30) if row == "top" else snap(body_bottom + ctrl_gap)

    # Stay near the sink channels while keeping control labels in declaration order.
    control_spacing = [ceil((a + b) / (2 * G)) * G
                       for a, b in zip(control_widths, control_widths[1:])]
    for p, want_x in zip(controls, _isotonic(control_x, control_spacing)):
        p.x = snap(want_x)

    # ---- emit components
    mc = model.Circuit(circ.name, attrs={"circuit": circ.name, "appearance": circ.appearance})
    if circ.appearance == "logisim_evolution":
        mc.attrs["circuitnamedboxfixedsize"] = "false"
    comp_of: dict[int, model.Component] = {}
    for n in circ.nodes:
        p = placed[id(n)]
        attrs = {k: v for k, v in n.attrs.items() if not k.startswith("_")}
        lib = project.lib_id(n.lib) if n.lib else None
        comp = model.Component(n.name, (p.x, p.y), attrs, lib)
        mc.add(comp)
        comp_of[id(n)] = comp
    _check_unique_locations(mc)

    # ---- routing
    allp = list(placed.values())
    if not allp:
        return mc
    xs = [p.x + p.shape.bounds[0] for p in allp] + [p.x + p.shape.bounds[0] + p.shape.bounds[2] for p in allp]
    ys = [p.y + p.shape.bounds[1] for p in allp] + [p.y + p.shape.bounds[1] + p.shape.bounds[3] for p in allp]
    grid = route.Grid(snap(min(xs)) - 100, snap(min(ys)) - 100, snap(max(xs)) + 100, snap(max(ys)) + 100)
    net_of_point: dict[tuple[int, int], int] = {}
    nets: list[tuple[Placed, str, list[Terminal]]] = []
    for (nid, port), sinks in drivers.items():
        src = next(p for p in allp if id(p.node) == nid)
        nets.append((src, port, sinks))
    term_net: dict[tuple[int, str], int] = {}
    for i, (src, port, sinks) in enumerate(nets):
        term_net[(id(src.node), port)] = i
        for t in sinks:
            term_net[(id(t.node), t.port)] = i
    free_net = len(nets)
    def outward(p: Placed, port: str) -> tuple[int, int]:
        return port_side(p.shape, port)

    for p in allp:
        bx, by, bw, bh = p.shape.bounds
        keep = set()
        for sp in p.shape.ports:
            pt = (p.x + sp.x, p.y + sp.y)
            keep.add(pt)
            nid = term_net.get((id(p.node), sp.name))
            if nid is None:
                nid = free_net
                free_net += 1
            grid.add_hard(pt, nid)
            net_of_point[pt] = nid
        grid.block_rect(p.x + bx, p.y + by, bw, bh, keep)
        _block_label(grid, p, keep)
    for p in allp:  # ports drawn inside their own bounding box need a way out
        bx, by, bw, bh = p.shape.bounds
        for sp in p.shape.ports:
            if bx < sp.x < bx + bw and by < sp.y < by + bh:
                dx, dy = outward(p, sp.name)
                q = (p.x + sp.x + dx, p.y + sp.y + dy)
                while p.x + bx <= q[0] <= p.x + bx + bw and p.y + by <= q[1] <= p.y + by + bh:
                    grid.blocked.discard(q)
                    q = (q[0] + dx, q[1] + dy)
                grid.blocked.discard(q)

    def routing_span(item) -> int:
        src, port, sinks = item
        # Route the closely spaced inputs of bus merges together, longest jogs first.
        if any(t.node.name == "Splitter" and t.port != "comb" for t in sinks):
            return 0
        sx = src.port_abs(port)[0]
        return max(abs(placed[id(t.node)].port_abs(t.port)[0] - sx) for t in sinks)

    def rise(item) -> int:
        src, port, sinks = item
        sy = src.port_abs(port)[1]
        return max(abs(placed[id(t.node)].port_abs(t.port)[1] - sy) for t in sinks)

    # Route straight nets first, then reserve the local routes before threading longer
    # nets through multiple columns. Within a span, nest the largest jogs first.
    def nest(item) -> int:
        """Among equal jogs, downward ones route upper-target-first, upward ones lower-first."""
        src, port, sinks = item
        sy = src.port_abs(port)[1]
        ty = max((placed[id(t.node)].port_abs(t.port)[1] for t in sinks), key=lambda y: abs(y - sy))
        return ty if ty > sy else -ty

    order = sorted(range(len(nets)),
                   key=lambda i: (rise(nets[i]) > 0, routing_span(nets[i]), -rise(nets[i]), nest(nets[i]), i))
    costs = route.Costs(cross=14, near_wire=8)
    tunnels = _Tunnels(circ, mc, project.lib_id("wiring"), grid, nets, placed)
    wired = {(i, id(t.node), t.port) for i in order for t in nets[i][2]}
    if circ.use_tunnels:
        # sink tunnels go in before any wire, so wires route around their stubs
        for i in order:
            src, port, sinks = nets[i]
            # a net that stays within the next column is a short wire; mixing wires and
            # tunnels on one net puts stubs beside crossings, so only straight runs stay wired
            local = all(placed[id(t.node)].layer == src.layer + 1 for t in sinks)
            for t in sinks:
                sp = placed[id(t.node)]
                if local or sp.layer == src.layer + 1 and sp.port_abs(t.port)[1] == src.port_abs(port)[1]:
                    continue
                wired.discard((i, id(t.node), t.port))
                tunnels.sink(i, t)
    for i in order:
        src, port, sinks = nets[i]
        start_pt = src.port_abs(port)
        tree = {start_pt}
        # first leave the driver: one step outward so the tree starts with a stub
        sinks_sorted = sorted(sinks, key=lambda t: abs(placed[id(t.node)].port_abs(t.port)[0] - start_pt[0]) +
                              abs(placed[id(t.node)].port_abs(t.port)[1] - start_pt[1]))
        for t in sinks_sorted:
            if (i, id(t.node), t.port) not in wired:
                continue
            sp = placed[id(t.node)]
            s_pt = sp.port_abs(t.port)
            path = route.route(grid, i, s_pt, tree, costs, start_dir=outward(sp, t.port))
            if path is None:
                tunnels.sink(i, t)
                continue
            grid.add_path(path, i)
            for a, b in route.segments(path):
                mc.add_wire(a, b)
                tree |= set(route._segment_points(a, b))

    tunnels.drivers()  # last, so a driver's tunnel can branch off its wire
    errors = connectivity_errors(circ, project, mc)
    if errors:
        raise RuntimeError(f"layout of {circ.name!r} broke connectivity: " + "; ".join(errors))
    return mc


def _block_label(grid: route.Grid, p: Placed, keep: set) -> None:
    n = p.node
    bx, by, bw, bh = p.shape.bounds
    comp = model.Component(n.name, (p.x, p.y), n.attrs)
    box = geometry.label_box(comp, (p.x + bx, p.y + by, bw, bh))
    if box is not None:
        grid.block_rect(*box, keep)


def _check_unique_locations(mc: model.Circuit) -> None:
    seen: dict[tuple[int, int], model.Component] = {}
    for c in mc.components:
        if c.loc in seen:
            raise RuntimeError(f"layout bug: {c} and {seen[c.loc]} share a location; "
                               "Logisim hangs on such files")
        seen[c.loc] = c


class _Tunnels:
    """Labelled tunnel pairs for the sinks the layout does not wire: one tunnel per such
    sink, plus one at each of their drivers."""

    def __init__(self, circ: Circuit, mc: model.Circuit, wiring: str, grid: route.Grid, nets, placed) -> None:
        self.circ, self.mc, self.wiring, self.grid = circ, mc, wiring, grid
        self.nets, self.placed = nets, placed
        self.names: dict[int, str] = {}

    def _name(self, i: int) -> str:
        if i not in self.names:
            label = _net_name(self.circ, *self.nets[i])
            k, cand = 2, label
            while cand in self.names.values():
                cand, k = f"{label}_{k}", k + 1
            self.names[i] = cand
        return self.names[i]

    def sink(self, i: int, t: Terminal) -> None:
        _tunnel_at(self.mc, self.grid, self.wiring, self.placed[id(t.node)], t.port, self._name(i), i,
                   branch=False)

    def drivers(self) -> None:
        for i in list(self.names):
            src, port, _ = self.nets[i]
            _tunnel_at(self.mc, self.grid, self.wiring, src, port, self.names[i], i, branch=True)


def _net_name(circ: Circuit, src: Placed, port: str, sinks: list[Terminal]) -> str:
    """Explicit name, else the driver's label, else the output pin the net feeds, else the
    driver's port name ("q") or kind ("xor")."""
    node = src.node
    label = circ.net_labels.get((id(node), port))
    if not label and node.label:
        label = node.label if node.primary == port else f"{node.label}_{port}"
    if not label:
        label = next((t.node.label for t in sinks if t.node.role == "output" and t.node.label), "")
    if not label:
        label = port if port.isalpha() and port not in ("out", "p") else node.name.split()[0].lower()
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in label)


_TUNNEL_FACING = {(G, 0): "west", (-G, 0): "east", (0, G): "north", (0, -G): "south"}
_TUNNEL_OFFSETS = (0, 2, -2, 3, -3, 4, -4, 6, -6)  # grid steps sideways from the port's line


def _grid_points(x: int, y: int, w: int, h: int) -> set[tuple[int, int]]:
    x0, y0 = -(-x // G) * G, -(-y // G) * G
    return {(gx, gy) for gx in range(x0, x + w + 1, G) for gy in range(y0, y + h + 1, G)}


def _tunnel_at(mc, grid, wiring, p: Placed, port: str, label: str, net: int, *, branch: bool) -> None:
    """Place a tunnel for `net` near p.port, at the first spot whose body touches no
    component, wire or other tunnel and that the router can join with a short stub,
    preferring stubs that cross no wire. A sink stub must end at its port; a driver's
    (branch=True) may join its net's wires. Stubs are marked hard, so later wires do not
    cross them."""
    px, py = p.port_abs(port)
    out = port_side(p.shape, port)
    side = (out[1], out[0])
    attrs = {"facing": _TUNNEL_FACING[out], "label": label}
    width = p.shape.port(port).width
    if width > 1:
        attrs["width"] = str(width)
    bx, by, bw, bh = geometry.FORMULAS[("#Wiring", "Tunnel")]("Tunnel", attrs).bounds
    targets = grid.net_points(net) if branch else {(px, py)}
    costs = route.Costs()
    spots = [(cross, off, along) for cross in (False, True) for off in _TUNNEL_OFFSETS for along in range(2, 9)]
    for allow_cross, off, along in spots:
        t = (px + out[0] * along + side[0] * off, py + out[1] * along + side[1] * off)
        body = _grid_points(t[0] + bx, t[1] + by, bw, bh) | {t}
        if any(q in grid.blocked or q in grid.hard or q in grid.run or not grid.inside(q) for q in body):
            continue
        added = body - {t}
        grid.blocked |= added
        limit = along + abs(off) + 2 * costs.bend + 2 * costs.near_body + 5
        if allow_cross:
            limit += 2 * costs.cross
        path = route.route(grid, net, t, targets, costs, start_dir=(-out[0], -out[1]), limit=limit)
        stub = [q for a, b in route.segments(path or []) for q in route._segment_points(a, b)]
        if path is None or len(path) > 4 or not allow_cross and any(
                route._foreign_run(grid, q, net) for q in stub):
            grid.blocked -= added
            continue
        for a, b in route.segments(path):
            mc.add_wire(a, b)
        grid.add_path(path, net)
        for q in stub:
            grid.hard.setdefault(q, net)
        mc.add(model.Component("Tunnel", t, attrs, wiring))
        grid.block_rect(t[0] + bx, t[1] + by, bw, bh, {t})
        return
    raise RuntimeError(f"no room for a tunnel at {p.node!r}.{port}")


def connectivity_errors(circ: Circuit, project: model.Project, mc: model.Circuit) -> list[str]:
    """Differences between the builder's intended nets and the drawn netlist.

    Components are emitted in node order, so component i is circ.nodes[i].
    """
    from . import netlist

    tmp = model.Project(circuits=[c for c in project.circuits if c.name != mc.name] + [mc],
                        libraries=project.libraries)
    nl = netlist.extract(tmp, mc)
    errors = []
    intended: list[set[tuple[int, str]]] = []
    for (nid, port), sinks in circ.driven().items():
        src = next(n for n in circ.nodes if id(n) == nid)
        intended.append({(src.index, port)} | {(t.node.index, t.port) for t in sinks})
    claimed: dict[tuple[int, str], int] = {}
    for k, members in enumerate(intended):
        nets = {nl.port_net.get(m) for m in members}
        if len(nets) != 1 or None in nets:
            errors.append(f"net {sorted(members)} is split across {len(nets)} drawn nets")
            continue
        drawn = nl.nets[nets.pop()]
        extra = {(p.comp, p.port.name) for p in drawn.ports if p.comp < len(circ.nodes)} - members
        if extra:
            errors.append(f"net {sorted(members)} is shorted to {sorted(extra)}")
        for m in members:
            claimed[m] = k
    return errors
