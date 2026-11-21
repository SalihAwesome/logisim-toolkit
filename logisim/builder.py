"""High-level circuit builder.

    from logisim.builder import Circuit, Project

    c = Circuit("full_adder")
    a, b, cin = c.input("A"), c.input("B"), c.input("Cin")
    x1 = c.xor(a, b)
    c.output("S", c.xor(x1, cin))
    c.output("Cout", c.or_(c.and_(a, b), c.and_(x1, cin)))
    Project(c).save("full_adder.circ")

You describe components and connections; placement and wiring happen when the
project is converted to a model (`Project.to_model()` / `save()`), see layout.py.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Union

from . import geometry
from . import io as circ_io
from . import model

CONTROL_PORTS = {"sel", "en", "clk", "clr", "reset", "preset", "ld", "ud", "ctrl", "nxt", "rst"}
CONTROL_NAMES = {"clk", "clock", "reset", "rst", "clr", "clear", "en", "enable", "sel", "select",
                 "load", "ld", "preset", "set", "we", "oe", "cs", "mode", "op"}


class BuildError(ValueError):
    pass


def valid_circuit_name(name: str) -> bool:
    """Names Logisim 3.9 keeps as written. It renames others on load ("__x" becomes
    "L_x_<hash>"), which breaks <main> and subcircuit references."""
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)) and "__" not in name \
        and not name.endswith("_")


@dataclass(eq=False)
class Node:
    """One component in a builder circuit."""

    circuit: "Circuit"
    lib: str | None  # library descriptor ("#Gates"), None for a subcircuit instance
    name: str
    attrs: dict[str, str]
    role: str = "logic"  # input | output | control | logic | const
    inputs: dict[str, "Terminal"] = field(default_factory=dict)  # sink port -> driving terminal
    primary: str = "out"
    index: int = 0

    def shape(self) -> geometry.Shape:
        return self.circuit._shape(self)

    def port_names(self) -> list[str]:
        return [p.name for p in self.shape().ports]

    def port(self, name: str) -> "Terminal":
        names = self.port_names()
        if name not in names:
            raise BuildError(f"{self.name} has no port {name!r}; ports: {names}")
        return Terminal(self, name)

    def __getattr__(self, name: str) -> "Terminal":
        if name.startswith("_") or name in ("circuit", "lib", "attrs", "inputs"):
            raise AttributeError(name)
        try:
            return self.port(name)
        except BuildError as e:
            raise AttributeError(str(e)) from None

    def __getitem__(self, name: str) -> "Terminal":
        return self.port(name)

    @property
    def label(self) -> str:
        return self.attrs.get("label", "")

    @property
    def width(self) -> int:
        return Terminal(self, self.primary).width

    def __repr__(self) -> str:
        lab = f" {self.label!r}" if self.label else ""
        return f"<Node {self.name}{lab} #{self.index}>"


@dataclass(frozen=True, eq=False)
class Terminal:
    """A specific port of a node; used both as a signal source and as a sink."""

    node: Node
    port: str

    @property
    def spec(self) -> geometry.Port:
        return self.node.shape().port(self.port)

    @property
    def width(self) -> int:
        return self.spec.width

    @property
    def is_output(self) -> bool:
        d = self.spec.dir
        if d == "inout":  # splitters: direction decided by how the builder used them
            return self.port in self.node.circuit._splitter_outputs.get(id(self.node), set())
        return d == "out"

    def __repr__(self) -> str:
        return f"{self.node!r}.{self.port}"


Signal = Union[Terminal, Node]


def _term(sig: Signal) -> Terminal:
    if isinstance(sig, Node):
        return Terminal(sig, sig.primary)
    if isinstance(sig, Terminal):
        return sig
    raise BuildError(f"expected a signal (Node or Terminal), got {sig!r}")


_GATE_NAMES = {"and": "AND Gate", "or": "OR Gate", "xor": "XOR Gate", "nand": "NAND Gate",
               "nor": "NOR Gate", "xnor": "XNOR Gate"}


class Circuit:
    def __init__(self, name: str, *, spacing: str = "roomy", use_tunnels: bool = False,
                 appearance: str = "logisim_evolution") -> None:
        if not valid_circuit_name(name):
            raise BuildError(f"bad circuit name {name!r}: Logisim renames it on load; use a letter, "
                             "then letters, digits and single underscores, not ending in '_'")
        self.name = name
        self.spacing = spacing
        self.use_tunnels = use_tunnels
        self.appearance = appearance
        self.nodes: list[Node] = []
        self.net_labels: dict[int, str] = {}  # id(driver terminal key) -> label
        self._splitter_outputs: dict[int, set[str]] = {}
        self._project: "Project | None" = None
        self._labels: dict[str, str] = {}  # lowercased label -> label as written

    # ------------------------------------------------------------ internals
    def _shape(self, node: Node) -> geometry.Shape:
        if node.lib is None:
            proj = self._project.model_for_geometry() if self._project else None
            if proj is None:
                raise BuildError(f"subcircuit {node.name!r}: add both circuits to a Project first")
            return geometry.subcircuit_shape(proj, node.name, node.attrs)
        f = geometry.FORMULAS.get((node.lib, node.name))
        if f is not None:
            return f(node.name, node.attrs)
        proj = model.Project.new()
        comp = model.Component(node.name, (0, 0), node.attrs, proj.lib_id(node.lib))
        return geometry.shape(proj, comp)

    def _add(self, lib: str | None, name: str, attrs: dict, role: str = "logic", primary: str = "out") -> Node:
        label = str(attrs.get("label", ""))
        if label and name != "Tunnel":
            self._claim_label(label)
        node = Node(self, lib, name, {k: str(v) for k, v in attrs.items()}, role=role,
                    primary=primary, index=len(self.nodes))
        self.nodes.append(node)
        if self._project is not None:
            # the fallback port geometry of the project is built from the nodes
            # of its circuits, so it has to be rebuilt when one of them grows
            self._project._geom_model = None
        return node

    def _claim_label(self, label: str) -> None:
        # Logisim 3.9 clears one of two non-tunnel labels that are equal ignoring case
        other = self._labels.get(label.lower())
        if other == label:
            raise BuildError(f"duplicate label {label!r} in circuit {self.name!r}")
        if other is not None:
            raise BuildError(f"label {label!r} matches {other!r} ignoring case in circuit {self.name!r}; "
                             "Logisim keeps only one of them")
        self._labels[label.lower()] = label

    def connect(self, src: Signal, dst: Terminal | Signal) -> None:
        """Drive sink port `dst` from signal `src`."""
        s = _term(src)
        d = _term(dst) if isinstance(dst, Node) else dst
        if s.node.circuit is not self or d.node.circuit is not self:
            raise BuildError("both ends must belong to this circuit")
        if d.is_output and not s.is_output:
            s, d = d, s
        if d.port in d.node.inputs:
            raise BuildError(f"{d!r} is already driven by {d.node.inputs[d.port]!r}")
        if s.width and d.width and s.width != d.width:
            raise BuildError(f"width mismatch: {s!r} is {s.width} bits, {d!r} is {d.width} bits")
        d.node.inputs[d.port] = s

    def _wire_inputs(self, node: Node, sigs: dict[str, Signal | None]) -> Node:
        for port, sig in sigs.items():
            if sig is not None:
                self.connect(sig, Terminal(node, port))
        return node

    @staticmethod
    def _width_of(*sigs: Signal | None, default: int = 1) -> int:
        for s in sigs:
            if s is not None:
                w = _term(s).width
                if w:
                    return w
        return default

    def name_net(self, sig: Signal, label: str) -> Signal:
        """Give a net a readable name (used for its tunnel labels and in reports)."""
        t = _term(sig)
        self.net_labels[(id(t.node), t.port)] = label
        return sig

    # ------------------------------------------------------------ pins
    def input(self, label: str, width: int = 1, *, control: bool | None = None) -> Node:
        """Input pin. Control inputs (clock, reset, select...) are drawn along the top edge."""
        if control is None:
            control = label.lower().rstrip("0123456789_") in CONTROL_NAMES
        attrs = {"appearance": "classic", "label": label}
        if width != 1:
            attrs["width"] = width
        return self._add("#Wiring", "Pin", attrs, role="control" if control else "input", primary="p")

    def inputs(self, *labels: str, width: int = 1) -> list[Node]:
        return [self.input(lab, width) for lab in labels]

    def output(self, label: str, sig: Signal, width: int | None = None) -> Node:
        w = width or self._width_of(sig)
        attrs = {"appearance": "classic", "facing": "west", "label": label, "labelloc": "east",
                 "output": "true"}
        if w != 1:
            attrs["width"] = w
        node = self._add("#Wiring", "Pin", attrs, role="output", primary="p")
        return self._wire_inputs(node, {"p": sig})

    def const(self, value: int, width: int = 1) -> Node:
        attrs = {"value": hex(value & ((1 << width) - 1))}
        if width != 1:
            attrs["width"] = width
        return self._add("#Wiring", "Constant", attrs, role="const")

    def clock(self, label: str = "clk") -> Node:
        return self._add("#Wiring", "Clock", {"label": label}, role="control")

    # ------------------------------------------------------------ gates
    def gate(self, kind: str, *sigs: Signal, width: int | None = None, negate: Iterable[int] = ()) -> Node:
        kind = kind.lower()
        if kind in ("not", "inv"):
            return self.not_(*sigs)
        if kind not in _GATE_NAMES:
            raise BuildError(f"unknown gate {kind!r}")
        if len(sigs) < 2:
            raise BuildError(f"{kind} needs at least 2 inputs, got {len(sigs)}")
        w = width or self._width_of(*sigs)
        attrs: dict = {"inputs": len(sigs)} if len(sigs) != 2 else {}
        if len(sigs) > 5:
            attrs["size"] = 70  # wider body gives the many inputs room
        if w != 1:
            attrs["width"] = w
        for i in negate:
            attrs[f"negate{i}"] = "true"
        node = self._add("#Gates", _GATE_NAMES[kind], attrs)
        return self._wire_inputs(node, {f"in{i}": s for i, s in enumerate(sigs)})

    def and_(self, *sigs: Signal, **kw) -> Node:
        return self.gate("and", *sigs, **kw)

    def or_(self, *sigs: Signal, **kw) -> Node:
        return self.gate("or", *sigs, **kw)

    def xor(self, *sigs: Signal, **kw) -> Node:
        return self.gate("xor", *sigs, **kw)

    def nand(self, *sigs: Signal, **kw) -> Node:
        return self.gate("nand", *sigs, **kw)

    def nor(self, *sigs: Signal, **kw) -> Node:
        return self.gate("nor", *sigs, **kw)

    def xnor(self, *sigs: Signal, **kw) -> Node:
        return self.gate("xnor", *sigs, **kw)

    def not_(self, sig: Signal, width: int | None = None) -> Node:
        w = width or self._width_of(sig)
        node = self._add("#Gates", "NOT Gate", {"width": w} if w != 1 else {})
        return self._wire_inputs(node, {"in": sig})

    def buffer(self, sig: Signal) -> Node:
        w = self._width_of(sig)
        node = self._add("#Gates", "Buffer", {"width": w} if w != 1 else {})
        return self._wire_inputs(node, {"in": sig})

    # ------------------------------------------------------------ plexers
    def mux(self, sel: Signal, *data: Signal, width: int | None = None) -> Node:
        n = len(data)
        bits = max(1, (n - 1).bit_length())
        if n != 1 << bits:
            raise BuildError(f"mux needs a power-of-two number of data inputs, got {n}")
        w = width or self._width_of(*data)
        attrs = {"enable": "false", "select": bits}
        if w != 1:
            attrs["width"] = w
        node = self._add("#Plexers", "Multiplexer", attrs)
        return self._wire_inputs(node, {"sel": sel, **{f"in{i}": d for i, d in enumerate(data)}})

    def demux(self, sel: Signal, data: Signal, outputs: int = 2) -> Node:
        bits = max(1, (outputs - 1).bit_length())
        w = self._width_of(data)
        attrs = {"enable": "false", "select": bits}
        if w != 1:
            attrs["width"] = w
        node = self._add("#Plexers", "Demultiplexer", attrs, primary="out0")
        return self._wire_inputs(node, {"sel": sel, "in": data})

    def decoder(self, sel: Signal, enable: Signal | None = None) -> Node:
        bits = self._width_of(sel)
        attrs = {"select": bits, "enable": "true" if enable is not None else "false"}
        node = self._add("#Plexers", "Decoder", attrs, primary="out0")
        return self._wire_inputs(node, {"sel": sel, "en": enable})

    def priority_encoder(self, *ins: Signal) -> Node:
        bits = max(1, (len(ins) - 1).bit_length())
        node = self._add("#Plexers", "Priority Encoder", {"select": bits})
        return self._wire_inputs(node, {f"in{i}": s for i, s in enumerate(ins)})

    encoder = priority_encoder

    # ------------------------------------------------------------ arithmetic
    def _arith(self, name: str, a: Signal, b: Signal, extra: dict[str, Signal | None], width: int | None) -> Node:
        w = width or self._width_of(a, b)
        node = self._add("#Arithmetic", name, {"width": w})
        return self._wire_inputs(node, {"a": a, "b": b, **extra})

    def adder(self, a: Signal, b: Signal, cin: Signal | None = None, width: int | None = None) -> Node:
        return self._arith("Adder", a, b, {"cin": cin}, width)

    def subtractor(self, a: Signal, b: Signal, bin: Signal | None = None, width: int | None = None) -> Node:
        return self._arith("Subtractor", a, b, {"bin": bin}, width)

    def multiplier(self, a: Signal, b: Signal, width: int | None = None) -> Node:
        return self._arith("Multiplier", a, b, {}, width)

    def comparator(self, a: Signal, b: Signal, width: int | None = None, signed: bool = False) -> Node:
        node = self._arith("Comparator", a, b, {}, width)
        node.attrs["mode"] = "twosComplement" if signed else "unsigned"
        node.primary = "eq"
        return node

    def negator(self, a: Signal) -> Node:
        node = self._add("#Arithmetic", "Negator", {"width": self._width_of(a)})
        return self._wire_inputs(node, {"in": a})

    # ------------------------------------------------------------ memory
    def _ff(self, name: str, data: dict[str, Signal], clk: Signal, reset: Signal | None,
            preset: Signal | None, trigger: str) -> Node:
        attrs = {"appearance": "logisim_evolution"}
        if trigger != "rising":
            attrs["trigger"] = trigger
        node = self._add("#Memory", name, attrs, primary="q")
        return self._wire_inputs(node, {**data, "clk": clk, "reset": reset, "preset": preset})

    def dff(self, d: Signal, clk: Signal, reset: Signal | None = None, preset: Signal | None = None,
            trigger: str = "rising") -> Node:
        return self._ff("D Flip-Flop", {"D": d}, clk, reset, preset, trigger)

    def tff(self, t: Signal, clk: Signal, reset: Signal | None = None, preset: Signal | None = None,
            trigger: str = "rising") -> Node:
        return self._ff("T Flip-Flop", {"T": t}, clk, reset, preset, trigger)

    def jkff(self, j: Signal, k: Signal, clk: Signal, reset: Signal | None = None,
             preset: Signal | None = None, trigger: str = "rising") -> Node:
        return self._ff("J-K Flip-Flop", {"J": j, "K": k}, clk, reset, preset, trigger)

    def srff(self, s: Signal, r: Signal, clk: Signal, reset: Signal | None = None,
             preset: Signal | None = None, trigger: str = "rising") -> Node:
        return self._ff("S-R Flip-Flop", {"S": s, "R": r}, clk, reset, preset, trigger)

    def register(self, d: Signal, clk: Signal, en: Signal | None = None, clr: Signal | None = None,
                 width: int | None = None, label: str = "") -> Node:
        w = width or self._width_of(d)
        attrs = {"appearance": "logisim_evolution", "width": w}
        if label:
            attrs["label"] = label
        node = self._add("#Memory", "Register", attrs)
        return self._wire_inputs(node, {"in": d, "clk": clk, "en": en, "clr": clr})

    def counter(self, clk: Signal, width: int = 4, en: Signal | None = None, clr: Signal | None = None,
                load: Signal | None = None, d: Signal | None = None, up: Signal | None = None,
                max_value: int | None = None, on_goal: str = "wrap") -> Node:
        attrs = {"appearance": "logisim_evolution", "width": width,
                 "max": hex(max_value if max_value is not None else (1 << width) - 1), "ongoal": on_goal}
        node = self._add("#Memory", "Counter", attrs, primary="p0")
        named = {"clk": clk, "en": en, "clr": clr, "ld": load, "in": d, "ud": up}
        return self._wire_named(node, named, ["out", "in", "clk", "clr", "ld", "ud", "en", "carry"])

    def rom(self, addr: Signal, contents: list[int], data_width: int) -> Node:
        aw = self._width_of(addr)
        attrs = {"appearance": "logisim_evolution", "addrWidth": aw, "dataWidth": data_width,
                 "contents": rom_contents(aw, data_width, contents)}
        node = self._add("#Memory", "ROM", attrs, primary="p1")
        return self._wire_named(node, {"addr": addr}, ["addr", "data"])

    def ram(self, addr: Signal, data_in: Signal, we: Signal, clk: Signal, data_width: int | None = None) -> Node:
        aw = self._width_of(addr)
        dw = data_width or self._width_of(data_in)
        attrs = {"appearance": "logisim_evolution", "addrWidth": aw, "dataWidth": dw,
                 "databus": "bibus"}
        node = self._add("#Memory", "RAM", attrs, primary="p1")
        return self._wire_named(node, {"addr": addr, "din": data_in, "we": we, "clk": clk},
                                ["addr", "data", "din", "we", "clk", "oe"])

    def _wire_named(self, node: Node, named: dict[str, Signal | None], order: list[str]) -> Node:
        """Components from the oracle cache have positional port names p0, p1, ...; map ours."""
        names = node.port_names()
        alias = {n: names[i] for i, n in enumerate(order) if i < len(names)}
        node.attrs.setdefault("_aliases", ",".join(f"{k}={v}" for k, v in alias.items()))
        node.primary = alias.get("out", alias.get("data", node.primary))
        return self._wire_inputs(node, {alias[k]: s for k, s in named.items() if s is not None and k in alias})

    # ------------------------------------------------------------ wiring helpers
    def split(self, bus: Signal, widths: list[int] | None = None) -> list[Terminal]:
        """Split a bus into pieces (LSB first). Default: one wire per bit."""
        total = self._width_of(bus)
        widths = widths or [1] * total
        if sum(widths) != total:
            raise BuildError(f"split widths {widths} do not add up to {total}")
        attrs = {"fanout": len(widths), "incoming": total, "appear": "center"}
        bit = 0
        for i, w in enumerate(widths):
            for _ in range(w):
                attrs[f"bit{bit}"] = i
                bit += 1
        node = self._add("#Wiring", "Splitter", attrs, primary="s0")
        self._splitter_outputs[id(node)] = {f"s{i}" for i in range(len(widths))}
        self._wire_inputs(node, {"comb": bus})
        return [Terminal(node, f"s{i}") for i in range(len(widths))]

    def bits(self, bus: Signal) -> list[Terminal]:
        return self.split(bus)

    def merge(self, parts: list[Signal]) -> Terminal:
        """Combine signals (LSB first) into one bus."""
        widths = [self._width_of(p) for p in parts]
        total = sum(widths)
        attrs = {"fanout": len(parts), "incoming": total, "appear": "center", "facing": "west"}
        bit = 0
        for i, w in enumerate(widths):
            for _ in range(w):
                attrs[f"bit{bit}"] = i
                bit += 1
        node = self._add("#Wiring", "Splitter", attrs, primary="comb")
        self._splitter_outputs[id(node)] = {"comb"}
        self._wire_inputs(node, {f"s{i}": p for i, p in enumerate(parts)})
        return Terminal(node, "comb")

    def probe(self, sig: Signal, label: str = "") -> Node:
        node = self._add("#Wiring", "Probe", {"label": label} if label else {}, role="output", primary="in")
        return self._wire_inputs(node, {"in": sig})

    def led(self, sig: Signal, label: str = "") -> Node:
        attrs = {"facing": "west"}
        if label:
            attrs["label"] = label
        node = self._add("#I/O", "LED", attrs, role="output", primary="in")
        return self._wire_inputs(node, {"in": sig})

    def component(self, lib: str, name: str, attrs: dict | None = None, **connections: Signal) -> Node:
        """Any library component; connect sink ports by name."""
        desc = model.LIB_DESC.get(lib, lib)
        node = self._add(desc, name, attrs or {})
        return self._wire_inputs(node, connections)

    def instance(self, sub: "Circuit | str", **pins: Signal) -> Node:
        """Instance of another circuit in the same Project; connect its input pins by label."""
        name = sub.name if isinstance(sub, Circuit) else sub
        node = self._add(None, name, {})
        outs = [p.name for p in node.shape().ports if p.dir == "out"]
        node.primary = outs[0] if outs else node.port_names()[0]
        return self._wire_inputs(node, pins)

    # ------------------------------------------------------------ results
    def driven(self) -> dict[tuple[int, str], list[Terminal]]:
        """driver (node id, port) -> sinks."""
        out: dict[tuple[int, str], list[Terminal]] = {}
        for n in self.nodes:
            for port, src in n.inputs.items():
                out.setdefault((id(src.node), src.port), []).append(Terminal(n, port))
        return out

    def to_model(self, project: model.Project) -> model.Circuit:
        from .layout import place_and_route
        return place_and_route(self, project)


def rom_contents(addr_width: int, data_width: int, values: list[int]) -> str:
    vals = [v & ((1 << data_width) - 1) for v in values]
    while vals and vals[-1] == 0:
        vals.pop()
    lines = [f"addr/data: {addr_width} {data_width}"]
    for i in range(0, len(vals), 8):
        lines.append(" ".join(format(v, "x") for v in vals[i:i + 8]))
    return "v3.0 hex words addressed\n".replace("v3.0 hex words addressed\n", "") + "\n".join(lines) + "\n"


class Project:
    """A set of builder circuits; the first one added is the main circuit."""

    def __init__(self, *circuits: Circuit, main: str | None = None, base: model.Project | None = None) -> None:
        """`base`: an existing model project (e.g. io.load(...)) whose circuits are kept and can
        be instanced by the new circuits."""
        self.circuits: list[Circuit] = []
        self.base = base
        self.main = main
        self._explicit_main = main is not None
        self._geom_model: model.Project | None = None
        self._building: model.Project | None = None
        for c in circuits:
            self.add(c)

    def add(self, circ: Circuit) -> Circuit:
        if any(c.name == circ.name for c in self.circuits):
            raise BuildError(f"duplicate circuit {circ.name!r}")
        circ._project = self
        self.circuits.append(circ)
        self._geom_model = None
        if self.main is None:
            self.main = circ.name
        return circ

    def model_for_geometry(self) -> model.Project:
        """Model used for subcircuit port geometry.

        During to_model() this is the project being built, so instances see the
        final pin positions of already laid-out subcircuits. Before that, pins are
        stacked in declaration order (good enough for port names and widths).
        """
        if self._building is not None:
            return self._building
        if self.base is not None:
            return self.base
        if self._geom_model is None:
            p = model.Project.new()
            for c in self.circuits:
                mc = p.add_circuit(c.name)
                mc.attrs["appearance"] = c.appearance
                k = 0
                for n in c.nodes:
                    if n.name == "Pin" and n.lib == "#Wiring":
                        mc.add(model.Component("Pin", (0, 10 * k), dict(n.attrs), p.lib_id("wiring")))
                        k += 1
            self._geom_model = p
        return self._geom_model

    def to_model(self) -> model.Project:
        proj = self.base.copy() if self.base is not None else model.Project.new()
        for c in self.circuits:
            if proj.has_circuit(c.name):
                raise BuildError(f"circuit {c.name!r} already exists in the base project")
        kept = len(proj.circuits)
        self._building = proj
        try:
            # subcircuits are laid out first; their final pin positions fix the instance ports
            for c in _dependency_order(self.circuits):
                proj.circuits.append(c.to_model(proj))
        finally:
            self._building = None
        order = {c.name: i for i, c in enumerate(self.circuits)}
        proj.circuits[kept:] = sorted(proj.circuits[kept:], key=lambda mc: order[mc.name])
        if self.base is None or self._explicit_main or not proj.main:
            proj.main = self.main
        return proj

    def save(self, path: str | os.PathLike, *, overwrite: bool = False) -> Path:
        return circ_io.save(self.to_model(), path, overwrite=overwrite)


def _dependency_order(circuits: list[Circuit]) -> list[Circuit]:
    by_name = {c.name: c for c in circuits}
    done: list[Circuit] = []
    seen: set[str] = set()

    def visit(c: Circuit, stack: tuple[str, ...]) -> None:
        if c.name in seen:
            return
        if c.name in stack:
            raise BuildError(f"circular subcircuit use: {' -> '.join(stack + (c.name,))}")
        for n in c.nodes:
            if n.lib is None and n.name in by_name:
                visit(by_name[n.name], stack + (c.name,))
        seen.add(c.name)
        done.append(c)

    for c in circuits:
        visit(c, ())
    return done
