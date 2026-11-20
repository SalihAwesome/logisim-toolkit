"""Port geometry: where each component's connection points are.

Every formula here is ported from the Logisim-evolution 3.9.0 Java sources and
checked against what the real JAR reports (tests/test_geometry.py, fixture
built by tools/make_fixtures.py). Coordinates are relative to the component's
`loc` until `ports()`/`bounds()` add it.

Components without a formula fall back to an oracle cache: their geometry is
read from the JAR once (via the Java probe) and stored in a JSON cache, see
`OracleCache`.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .model import Component, Project

Bounds = tuple[int, int, int, int]  # x, y, width, height


@dataclass(frozen=True)
class Port:
    name: str
    x: int
    y: int
    width: int
    dir: str  # "in", "out" or "inout"

    @property
    def pos(self) -> tuple[int, int]:
        return (self.x, self.y)

    def moved(self, dx: int, dy: int) -> "Port":
        return Port(self.name, self.x + dx, self.y + dy, self.width, self.dir)


@dataclass(frozen=True)
class Shape:
    ports: tuple[Port, ...]
    bounds: Bounds
    exact_bounds: bool = True

    def port(self, name: str) -> Port:
        for p in self.ports:
            if p.name == name:
                return p
        raise KeyError(f"no port {name!r}; ports are {[p.name for p in self.ports]}")


class UnknownGeometry(KeyError):
    pass


# ---------------------------------------------------------------- helpers

def _rot(x: int, y: int, facing: str) -> tuple[int, int]:
    """Rotate an east-facing offset to `facing` (Logisim Location.rotate about 0,0)."""
    if facing == "north":
        return y, -x
    if facing == "west":
        return -x, -y
    if facing == "south":
        return -y, x
    return x, y


def _rot_bounds(b: Bounds, facing: str) -> Bounds:
    x, y, w, h = b
    xs, ys = zip(_rot(x, y, facing), _rot(x + w, y + h, facing))
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def _rotated(ports: list[tuple[str, int, int, int, str]], bounds: Bounds, facing: str) -> Shape:
    return Shape(
        tuple(Port(n, *_rot(x, y, facing), w, d) for n, x, y, w, d in ports),
        _rot_bounds(bounds, facing),
    )


def _int(attrs: dict[str, str], key: str, default: int) -> int:
    v = attrs.get(key)
    if v is None or v == "":
        return default
    v = v.strip()
    return int(v, 16) if v.lower().startswith("0x") else int(v)


def _width(attrs: dict[str, str], key: str = "width") -> int:
    return _int(attrs, key, 1)


def _facing(attrs: dict[str, str], default: str = "east") -> str:
    return attrs.get("facing", default) or default


def _bool(attrs: dict[str, str], key: str, default: bool) -> bool:
    v = attrs.get(key)
    return default if v is None else v == "true"


# ---------------------------------------------------------------- gates

GATE_NAMES = {"AND Gate", "OR Gate", "NAND Gate", "NOR Gate", "XOR Gate", "XNOR Gate",
              "Odd Parity", "Even Parity"}
_GATE_BONUS = {"XOR Gate": 10, "XNOR Gate": 10}
_GATE_NEGATE_OUT = {"NAND Gate", "NOR Gate", "XNOR Gate"}


def gate_input_offset(name: str, size: int, inputs: int, index: int, facing: str,
                      negated: bool = False) -> tuple[int, int]:
    """AbstractGate.getInputOffset."""
    axis = size + _GATE_BONUS.get(name, 0) + (10 if name in _GATE_NEGATE_OUT else 0)
    if inputs <= 3:
        if size < 40:
            start, dist, lower = -5, 10, 10
        elif size < 60 or inputs <= 2:
            start, dist, lower = -10, 20, 20
        else:
            start, dist, lower = -15, 30, 30
    elif inputs == 4 and size >= 60:
        start, dist, lower = -5, 20, 0
    else:
        start, dist, lower = -5, 10, 10
    if inputs % 2 == 1:
        dy = start * (inputs - 1) + dist * index
    else:
        dy = start * inputs + dist * index
        if index >= inputs // 2:
            dy += lower
        if inputs == 4 and size >= 60:
            dy -= 10
    dx = axis + (10 if negated else 0)
    if facing == "north":
        return dy, dx
    if facing == "south":
        return dy, -dx
    if facing == "west":
        return dx, dy
    return -dx, dy


def _gate(name: str, attrs: dict[str, str]) -> Shape:
    facing = _facing(attrs)
    size = _int(attrs, "size", 50)
    inputs = _int(attrs, "inputs", 2)
    width = _width(attrs)
    negs = [_bool(attrs, f"negate{i}", False) for i in range(inputs)]
    ports = [Port("out", 0, 0, width, "out")]
    for i in range(inputs):
        x, y = gate_input_offset(name, size, inputs, i, facing, negs[i])
        ports.append(Port(f"in{i}", x, y, width, "in"))
    n = inputs + 1 if inputs % 2 == 0 else inputs
    w = size + _GATE_BONUS.get(name, 0) + (10 if name in _GATE_NEGATE_OUT else 0)
    if any(negs):
        w += 10
    h = max(10 * n, size)
    if facing == "south":
        b = (-(h // 2), -w, h, w)
    elif facing == "north":
        b = (-(h // 2), 0, h, w)
    elif facing == "west":
        b = (0, -(h // 2), w, h)
    else:
        b = (-w, -(h // 2), w, h)
    return Shape(tuple(ports), b)


def _not_or_buffer(name: str, attrs: dict[str, str]) -> Shape:
    facing = _facing(attrs)
    width = _width(attrs)
    size = 20 if name == "Buffer" else _int(attrs, "size", 30)
    return _rotated([("out", 0, 0, width, "out"), ("in", -size, 0, width, "in")],
                    (-size, -9, size, 18), facing)


def _controlled(name: str, attrs: dict[str, str]) -> Shape:
    facing = _facing(attrs)
    width = _width(attrs)
    cy = -10 if attrs.get("control", "right") == "left" else 10
    ln = 30 if name == "Controlled Inverter" else 20
    return _rotated([("out", 0, 0, width, "out"), ("in", -ln, 0, width, "in"),
                     ("ctrl", 10 - ln, cy, 1, "in")], (-ln, -10, ln, 20), facing)


# ---------------------------------------------------------------- wiring

def _probe_bounds(facing: str, width: int, is_pin: bool) -> Bounds:
    """Probe.getOffsetBounds for radix 2, classic layout."""
    rows = -(-width // 8)
    bw = 20 if width < 2 else (80 if width >= 8 else width * 10)
    bh = 20 if rows < 2 else (160 if rows >= 8 else rows * 20)
    bw = ((bw + 9) // 10) * 10
    if facing == "east":
        return (-bw, -(bh // 2), bw, bh)
    if facing == "west":
        return (0, -(bh // 2), bw, bh)
    if facing == "south":
        return (-(bw // 2), -bh, bw, bh)
    return (-(bw // 2), 0, bw, bh)


def _pin(name: str, attrs: dict[str, str]) -> Shape:
    width = _width(attrs)
    out = _bool(attrs, "output", False)
    # An output pin receives a signal (port is an input end); an input pin drives one.
    port = Port("p", 0, 0, width, "in" if out else "out")
    return Shape((port,), _probe_bounds(_facing(attrs), width, True))


def _probe(name: str, attrs: dict[str, str]) -> Shape:
    # A probe adapts to whatever it is attached to; Logisim reports its end as width 0.
    return Shape((Port("in", 0, 0, 0, "in"),), _probe_bounds(_facing(attrs), 1, False))


def _clock(name: str, attrs: dict[str, str]) -> Shape:
    return Shape((Port("out", 0, 0, 1, "out"),), _probe_bounds(_facing(attrs), 1, False))


def _constant(name: str, attrs: dict[str, str]) -> Shape:
    facing = _facing(attrs)
    width = _width(attrs)
    w = 7 + 7 * ((width + 3) // 4)
    b = {"east": (-w, -8, w, 16), "west": (0, -8, w, 16),
         "south": (-(w // 2), -16, w, 16), "north": (-(w // 2), 0, w, 16)}[facing]
    return Shape((Port("out", 0, 0, width, "out"),), b)


def _tunnel(name: str, attrs: dict[str, str]) -> Shape:
    facing = _facing(attrs, "west")
    width = _width(attrs)
    label = attrs.get("label", "")
    # Tunnel bounds depend on font metrics; this is an estimate (exact_bounds=False).
    wd = max(10, 6 * len(label)) + 20
    b = {"west": (0, -10, wd, 20), "east": (-wd, -10, wd, 20),
         "north": (-wd // 2, 0, wd, 20), "south": (-wd // 2, -20, wd, 20)}[facing]
    return Shape((Port("io", 0, 0, width, "inout"),), b, exact_bounds=False)


def _pull(name: str, attrs: dict[str, str]) -> Shape:
    return _rotated([("io", 0, 0, 0, "inout")], (-42, -6, 42, 12), _facing(attrs, "south"))


def _bit_extender(name: str, attrs: dict[str, str]) -> Shape:
    win = _int(attrs, "in_width", 8)
    wout = _int(attrs, "out_width", 16)
    ports = [Port("out", 0, 0, wout, "out"), Port("in", -40, 0, win, "in")]
    if attrs.get("type", "zero") == "input":
        ports.append(Port("ext", -20, -20, 1, "in"))
    return Shape(tuple(ports), (-40, -20, 40, 40))


def splitter_bit_map(attrs: dict[str, str]) -> list[int | None]:
    """For each bit of the combined end, the split end index (0-based) or None."""
    fanout = _int(attrs, "fanout", 2)
    bits = _int(attrs, "incoming", 2)
    if fanout >= bits:
        dflt = list(range(bits))
    else:
        per, extra = divmod(bits, fanout)
        dflt, cur, left = [], -1, 0
        for _ in range(bits):
            if left == 0:
                cur += 1
                left = per + (1 if extra > 0 else 0)
                extra -= 1 if extra > 0 else 0
            dflt.append(cur)
            left -= 1
    out: list[int | None] = []
    for i in range(bits):
        v = attrs.get(f"bit{i}")
        if v is None:
            out.append(dflt[i])
        elif v == "none":
            out.append(None)
        else:
            out.append(int(v))
    return out


def _splitter(name: str, attrs: dict[str, str]) -> Shape:
    facing = _facing(attrs)
    fanout = _int(attrs, "fanout", 2)
    bits = _int(attrs, "incoming", 2)
    appear = attrs.get("appear", "left")
    justify = 0 if appear in ("center", "legacy") else (1 if appear == "right" else -1)
    gap = _int(attrs, "spacing", 1) * 10
    width = 20
    if facing in ("north", "south"):
        m = 1 if facing == "north" else -1
        x0 = gap * ((fanout + 1) // 2 - 1) if justify == 0 else (
            -10 if m * justify < 0 else 10 + gap * (fanout - 1))
        y0, ddx, ddy = -m * width, -gap, 0
    else:
        m = -1 if facing == "west" else 1
        x0 = m * width
        y0 = -gap * (fanout // 2) if justify == 0 else (
            10 if m * justify > 0 else -(10 + gap * (fanout - 1)))
        ddx, ddy = 0, gap
    bitmap = splitter_bit_map(attrs)
    ports = [Port("comb", 0, 0, bits, "inout")]
    for i in range(fanout):
        w = sum(1 for b in bitmap if b == i)
        ports.append(Port(f"s{i}", x0 + ddx * i, y0 + ddy * i, max(w, 1), "inout"))
    xs = [p.x for p in ports]
    ys = [p.y for p in ports]
    b = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    return Shape(tuple(ports), b, exact_bounds=False)


# ---------------------------------------------------------------- plexers

def _translate(p: tuple[int, int], facing: str, d: int) -> tuple[int, int]:
    dx, dy = {"east": (d, 0), "west": (-d, 0), "north": (0, -d), "south": (0, d)}[facing]
    return p[0] + dx, p[1] + dy


def _mux(name: str, attrs: dict[str, str]) -> Shape:
    facing = _facing(attrs)
    wide = attrs.get("size", "40") == "40" or attrs.get("size") == "wide"
    bottom_left = attrs.get("selloc", "bl") == "bl"
    sm = 1 if bottom_left else -1
    data = _width(attrs)
    sel_bits = _int(attrs, "select", 1)
    enable = _bool(attrs, "enable", True)
    vertical = facing not in ("north", "south")
    n = 1 << sel_bits
    ins: list[tuple[int, int]] = []
    if n == 2:
        w = 30 if wide else 20
        s = 20 if wide else 10
        if facing == "west":
            ins, sel = [(w, -10), (w, 10)], (s, sm * 20)
        elif facing == "north":
            ins, sel = [(-10, w), (10, w)], (sm * -20, s)
        elif facing == "south":
            ins, sel = [(-10, -w), (10, -w)], (sm * -20, -s)
        else:
            ins, sel = [(-w, -10), (-w, 10)], (-s, sm * 20)
    else:
        w = 40 if wide else 20
        s = 20 if wide else 10
        dx = dy = -(n // 2) * 10
        ddx = ddy = 10
        if facing == "west":
            dx, ddx, sel = w, 0, (s, sm * (dy + 10 * n))
        elif facing == "north":
            dy, ddy, sel = w, 0, (sm * dx, s)
        elif facing == "south":
            dy, ddy, sel = -w, 0, (sm * dx, -s)
        else:
            dx, ddx, sel = -w, 0, (-s, sm * (dy + 10 * n))
        for _ in range(n):
            ins.append((dx, dy))
            dx += ddx
            dy += ddy
    if not wide and not vertical and bottom_left and n > 2:
        sel = (sel[0] - 10, sel[1])
    elif not wide and vertical and not bottom_left and n > 2:
        sel = (sel[0], sel[1] - 10)
    en = _translate(sel, facing, 10)
    ports = [Port(f"in{i}", x, y, data, "in") for i, (x, y) in enumerate(ins)]
    ports.append(Port("sel", *sel, sel_bits, "in"))
    if enable:
        ports.append(Port("en", *en, 1, "in"))
    ports.append(Port("out", 0, 0, data, "out"))
    if n == 2:
        bw = 30 if wide else 20
        b = (-bw, -20, bw, 40)
    else:
        bw = 40 if wide else 20
        offs = -(n // 2) * 10 - 10
        if not wide and facing in ("south", "west"):
            offs += 5
        b = (-bw, offs, bw, n * 10 + 20 + (0 if wide else -5))
    return Shape(tuple(ports), _rot_bounds(b, facing))


def _demux(name: str, attrs: dict[str, str]) -> Shape:
    facing = _facing(attrs)
    sm = 1 if attrs.get("selloc", "bl") == "bl" else -1
    data = _width(attrs)
    sel_bits = _int(attrs, "select", 1)
    enable = _bool(attrs, "enable", True)
    n = 1 << sel_bits
    if n == 2:
        if facing == "west":
            outs, sel = [(-30, -10), (-30, 10)], (-20, sm * 20)
        elif facing == "north":
            outs, sel = [(-10, -30), (10, -30)], (sm * -20, -20)
        elif facing == "south":
            outs, sel = [(-10, 30), (10, 30)], (sm * -20, 20)
        else:
            outs, sel = [(30, -10), (30, 10)], (20, sm * 20)
    else:
        dx = dy = -(n // 2) * 10
        ddx = ddy = 10
        if facing == "west":
            dx, ddx, sel = -40, 0, (-20, sm * (dy + 10 * n))
        elif facing == "north":
            dy, ddy, sel = -40, 0, (sm * dx, -20)
        elif facing == "south":
            dy, ddy, sel = 40, 0, (sm * dx, 20)
        else:
            dx, ddx, sel = 40, 0, (20, sm * (dy + 10 * n))
        outs = []
        for _ in range(n):
            outs.append((dx, dy))
            dx += ddx
            dy += ddy
    en = _translate(sel, facing, -10)
    ports = [Port(f"out{i}", x, y, data, "out") for i, (x, y) in enumerate(outs)]
    ports.append(Port("sel", *sel, sel_bits, "in"))
    if enable:
        ports.append(Port("en", *en, 1, "in"))
    ports.append(Port("in", 0, 0, data, "in"))
    b = (0, -25, 30, 50) if n == 2 else (0, -(n // 2) * 10 - 10, 40, n * 10 + 20)
    return Shape(tuple(ports), _rot_bounds(b, facing))


def _decoder(name: str, attrs: dict[str, str]) -> Shape:
    facing = _facing(attrs)
    top_right = attrs.get("selloc", "bl") == "tr"
    sel_bits = _int(attrs, "select", 1)
    enable = _bool(attrs, "enable", True)
    n = 1 << sel_bits
    if n == 2:
        if facing in ("north", "south"):
            y = -10 if facing == "north" else 10
            outs = [(-30, y), (-10, y)] if top_right else [(10, y), (30, y)]
        else:
            x = -10 if facing == "west" else 10
            outs = [(x, 10), (x, 30)] if top_right else [(x, -30), (x, -10)]
    else:
        if facing in ("north", "south"):
            dy, ddy = (-20 if facing == "north" else 20), 0
            dx, ddx = (-10 * n if top_right else 0), 10
        else:
            dx, ddx = (-20 if facing == "west" else 20), 0
            dy, ddy = (0 if top_right else -10 * n), 10
        outs = []
        for _ in range(n):
            outs.append((dx, dy))
            dx += ddx
            dy += ddy
    ports = [Port(f"out{i}", x, y, 1, "out") for i, (x, y) in enumerate(outs)]
    ports.append(Port("sel", 0, 0, sel_bits, "in"))
    if enable:
        ports.append(Port("en", *_translate((0, 0), facing, -10), 1, "in"))
    reversed_ = facing in ("west", "north")
    if top_right:
        reversed_ = not reversed_
    if n == 2:
        y = 0 if reversed_ else -40
        b = (-20, y - 5, 30, 50)
    else:
        y = -10 if reversed_ else -(n * 10 + 10)
        b = (-20, y, 40, n * 10 + 20)
    return Shape(tuple(ports), _rot_bounds(b, facing), exact_bounds=False)


def _priority_encoder(name: str, attrs: dict[str, str]) -> Shape:
    facing = _facing(attrs)
    sel_bits = _int(attrs, "select", 3)
    n = 1 << sel_bits
    if facing in ("north", "south"):
        x = -5 * n + 10
        y = 40 if facing == "north" else -40
        ins = [(x + 10 * i, y) for i in range(n)]
        rest = [("out", 0, 0, sel_bits, "out"), ("en_in", x + 10 * n, y // 2, 1, "in"),
                ("en_out", x - 10, y // 2, 1, "out"), ("gs", 10, 0, 1, "out")]
    else:
        x = -40 if facing == "east" else 40
        y = -5 * n + 10
        ins = [(x, y + 10 * i) for i in range(n)]
        rest = [("out", 0, 0, sel_bits, "out"), ("en_in", x // 2, y + 10 * n, 1, "in"),
                ("en_out", x // 2, y - 10, 1, "out"), ("gs", 0, 10, 1, "out")]
    ports = [Port(f"in{i}", px, py, 1, "in") for i, (px, py) in enumerate(ins)]
    ports += [Port(*r) for r in rest]
    offs, ln = -5 * n, 10 * n + 10
    b = {"north": (offs, 0, ln, 40), "south": (offs, -40, ln, 40),
         "west": (0, offs, 40, ln), "east": (-40, offs, 40, ln)}[facing]
    return Shape(tuple(ports), b)


def _bit_selector(name: str, attrs: dict[str, str]) -> Shape:
    facing = _facing(attrs)
    win = _width(attrs) if "width" in attrs else 8
    group = _int(attrs, "group", 1)
    groups = -(-win // group)
    sel_w = max(1, (groups - 1).bit_length())
    s = _rotated([("out", 0, 0, group, "out"), ("in", -30, 0, win, "in"),
                  ("sel", -10, 10, sel_w, "in")], (-30, -15, 30, 30), facing)
    if facing == "north":  # the select input is mirrored for this one facing
        s = Shape(s.ports[:2] + (Port("sel", -10, 10, sel_w, "in"),), s.bounds)
    return s


# ---------------------------------------------------------------- arithmetic

def _arith(names: tuple[str, ...], extra: dict[str, tuple[int, int, int, str]]):
    def build(name: str, attrs: dict[str, str]) -> Shape:
        w = _width(attrs, "width") if "width" in attrs else 8
        base = {"a": (-40, -10, w, "in"), "b": (-40, 10, w, "in"), "out": (0, 0, w, "out")}
        base.update({k: (x, y, (w if ww == -1 else ww), d) for k, (x, y, ww, d) in extra.items()})
        return Shape(tuple(Port(n, *base[n]) for n in names), (-40, -20, 40, 40))
    return build


_adder = _arith(("a", "b", "out", "cin", "cout"),
                {"cin": (-20, -20, 1, "in"), "cout": (-20, 20, 1, "out")})
_subtractor = _arith(("a", "b", "out", "bin", "bout"),
                     {"bin": (-20, -20, 1, "in"), "bout": (-20, 20, 1, "out")})
_multiplier = _arith(("a", "b", "out", "cin", "cout"),
                     {"cin": (-20, -20, -1, "in"), "cout": (-20, 20, -1, "out")})
_divider = _arith(("a", "b", "out", "upper", "rem"),
                  {"upper": (-20, -20, -1, "in"), "rem": (-20, 20, -1, "out")})
_comparator = _arith(("a", "b", "gt", "eq", "lt"),
                     {"gt": (0, -10, 1, "out"), "eq": (0, 0, 1, "out"), "lt": (0, 10, 1, "out")})


def _negator(name: str, attrs: dict[str, str]) -> Shape:
    w = _int(attrs, "width", 8)
    return Shape((Port("in", -40, 0, w, "in"), Port("out", 0, 0, w, "out")), (-40, -20, 40, 40))


def _shifter(name: str, attrs: dict[str, str]) -> Shape:
    w = _int(attrs, "width", 8)
    sw = max(1, (w - 1).bit_length())
    return Shape((Port("a", -40, -10, w, "in"), Port("dist", -40, 10, sw, "in"),
                  Port("out", 0, 0, w, "out")), (-40, -20, 40, 40))


# ---------------------------------------------------------------- memory

_FF_INPUTS = {"D Flip-Flop": ["D"], "T Flip-Flop": ["T"], "J-K Flip-Flop": ["J", "K"],
              "S-R Flip-Flop": ["S", "R"]}


def _flipflop(name: str, attrs: dict[str, str]) -> Shape:
    data = _FF_INPUTS[name]
    if attrs.get("appearance", "logisim_evolution") == "classic":
        if len(data) == 1:
            din = [(-40, 20)]
            clk = (-40, 0)
        else:
            din = [(-40, 0), (-40, 20)]
            clk = (-40, 10)
        rest = [("clk", *clk, 1, "in"), ("q", 0, 0, 1, "out"), ("nq", 0, 20, 1, "out"),
                ("reset", -10, 30, 1, "in"), ("preset", -30, 30, 1, "in")]
        b: Bounds = (-40, -10, 40, 40)
    else:
        din = [(-10, 10)] if len(data) == 1 else [(-10, 10), (-10, 30)]
        clk = (-10, 50)
        rest = [("clk", *clk, 1, "in"), ("q", 50, 10, 1, "out"), ("nq", 50, 50, 1, "out"),
                ("reset", 20, 60, 1, "in"), ("preset", 20, 0, 1, "in")]
        b = (-10, 0, 60, 60)
    ports = [Port(d, x, y, 1, "in") for d, (x, y) in zip(data, din)] + [Port(*r) for r in rest]
    return Shape(tuple(ports), b)


def _register(name: str, attrs: dict[str, str]) -> Shape:
    w = _int(attrs, "width", 8)
    if attrs.get("appearance", "logisim_evolution") == "classic":
        ports = [("out", 0, 0, w, "out"), ("in", -30, 0, w, "in"), ("clk", -20, 20, 1, "in"),
                 ("clr", -10, 20, 1, "in"), ("en", -30, 10, 1, "in")]
        b: Bounds = (-30, -20, 30, 40)
    else:
        ports = [("out", 60, 30, w, "out"), ("in", 0, 30, w, "in"), ("clk", 0, 70, 1, "in"),
                 ("clr", 30, 90, 1, "in"), ("en", 0, 50, 1, "in")]
        b = (0, 0, 60, 90)
    return Shape(tuple(Port(*p) for p in ports), b, exact_bounds=w <= 8)


# ---------------------------------------------------------------- I/O

def _led(name: str, attrs: dict[str, str]) -> Shape:
    return _rotated([("in", 0, 0, 1, "in")], (-20, -10, 20, 20), _facing(attrs, "west"))


def _button(name: str, attrs: dict[str, str]) -> Shape:
    return _rotated([("out", 0, 0, 1, "out")], (-20, -10, 20, 20), _facing(attrs))


# ---------------------------------------------------------------- subcircuits

def subcircuit_shape(project: Project, circ_name: str, attrs: dict[str, str] | None = None) -> Shape:
    """Ports of an east-facing instance of `circ_name` with an auto-generated appearance.

    `logisim_evolution` (DefaultEvolutionAppearance): inputs on the west edge,
    outputs on the east edge, each sorted by pin (y, x), 20 apart.
    `classic` (DefaultClassicAppearance, also used when the attribute is
    missing): pins go on the edge opposite their facing, 10 apart.
    Ends are listed sorted by (x, y) of the instance port, as Logisim does.
    """
    attrs = attrs or {}
    circ = project.circuit(circ_name)
    if any(e.tag == "appear" for e in circ.extras):
        raise UnknownGeometry(f"{circ_name}: custom appearance")
    appearance = circ.attrs.get("appearance", "classic")
    if appearance not in ("logisim_evolution", "classic") or _facing(attrs) != "east":
        raise UnknownGeometry(f"{circ_name}: appearance {appearance}, facing {_facing(attrs)}")
    pins = circ.pins()
    placed: list[tuple[Component, Port]] = []
    if appearance == "logisim_evolution":
        west = sorted((p for p in pins if p.get("output") != "true"), key=lambda p: (p.loc[1], p.loc[0]))
        east = sorted((p for p in pins if p.get("output") == "true"), key=lambda p: (p.loc[1], p.loc[0]))
        char_w = 8
        max_l = max((len(p.label) * char_w for p in west), default=0)
        max_r = max((len(p.label) * char_w for p in east), default=0)
        fixed = circ.attrs.get("circuitnamedboxfixedsize", "false") == "true"
        text_w = 25 * char_w if fixed else max(max_l + max_r + 35, len(circ_name) * char_w + 15)
        width = (text_w // 10) * 10 + 20
        rows = max(len(east), len(west))
        height = rows * 20 + 20 if rows else 30
        ax, ay = (width, 10) if east else ((0, 10) if west else (0, 0))
        for i, p in enumerate(west):
            placed.append((p, Port(p.label or f"in{i}", -ax, 10 + i * 20 - ay, _width(p.attrs), "in")))
        for i, p in enumerate(east):
            placed.append((p, Port(p.label or f"out{i}", width - ax, 10 + i * 20 - ay, _width(p.attrs), "out")))
        b = (-ax, -ay - 1, width + 1, height + 2)
    else:
        opposite = {"east": "west", "west": "east", "north": "south", "south": "north"}
        edge: dict[str, list[Component]] = {d: [] for d in opposite}
        for p in pins:
            edge[opposite[_facing(p.attrs)]].append(p)
        for d, lst in edge.items():
            lst.sort(key=(lambda p: (p.loc[0], p.loc[1])) if d in ("north", "south")
                     else (lambda p: (p.loc[1], p.loc[0])))
        n_n, n_s, n_e, n_w = (len(edge[d]) for d in ("north", "south", "east", "west"))
        max_v, max_h = max(n_n, n_s), max(n_e, n_w)

        def offset(facing_n: int, opposite_n: int, others: int) -> int:
            m = max(facing_n, opposite_n)
            base = (15 if others == 0 else 10) if m <= 1 else (10 if m == 2 else (5 if others == 0 else 10))
            return base + 10 * ((m - facing_n) // 2)

        def dim(this: int, others: int) -> int:
            return 30 if this < 3 else (10 * this if others == 0 else 10 * this + 10)

        o_n, o_s = offset(n_n, n_s, max_h), offset(n_s, n_n, max_h)
        o_e, o_w = offset(n_e, n_w, max_v), offset(n_w, n_e, max_v)
        width, height = dim(max_v, max_h), dim(max_h, max_v)
        if n_e:
            ax, ay = width, o_e
        elif n_n:
            ax, ay = o_n, 0
        elif n_w:
            ax, ay = 0, o_w
        elif n_s:
            ax, ay = o_s, height
        else:
            ax, ay = 0, 0
        starts = {"west": (0, o_w, 0, 10), "east": (width, o_e, 0, 10),
                  "north": (o_n, 0, 10, 0), "south": (o_s, height, 10, 0)}
        for d, (x0, y0, dx, dy) in starts.items():
            for i, p in enumerate(edge[d]):
                kind = "in" if p.get("output") != "true" else "out"
                placed.append((p, Port(p.label or f"{kind}{i}", x0 + dx * i - ax, y0 + dy * i - ay,
                                       _width(p.attrs), kind)))
        b = (-ax - 1, -ay - 1, width + 2, height + 2)
    if appearance == "logisim_evolution" and not any(p.dir == "in" for _, p in placed):
        b = (b[0] + 9, b[1], b[2] - 9, b[3])  # no west pin stubs: the box starts 9 further right
    placed.sort(key=lambda t: (t[1].x, t[1].y))  # Logisim lists instance ends by position
    return Shape(tuple(pt for _, pt in placed), b)


# ---------------------------------------------------------------- registry

Builder = Callable[[str, dict[str, str]], Shape]

FORMULAS: dict[tuple[str, str], Builder] = {}
for _g in GATE_NAMES:
    FORMULAS[("#Gates", _g)] = _gate
FORMULAS.update({
    ("#Gates", "NOT Gate"): _not_or_buffer,
    ("#Gates", "Buffer"): _not_or_buffer,
    ("#Gates", "Controlled Buffer"): _controlled,
    ("#Gates", "Controlled Inverter"): _controlled,
    ("#Wiring", "Pin"): _pin,
    ("#Wiring", "Probe"): _probe,
    ("#Wiring", "Clock"): _clock,
    ("#Wiring", "Constant"): _constant,
    ("#Wiring", "Tunnel"): _tunnel,
    ("#Wiring", "Splitter"): _splitter,
    ("#Wiring", "Bit Extender"): _bit_extender,
    ("#Wiring", "Pull Resistor"): _pull,
    ("#Plexers", "Multiplexer"): _mux,
    ("#Plexers", "Demultiplexer"): _demux,
    ("#Plexers", "Decoder"): _decoder,
    ("#Plexers", "Priority Encoder"): _priority_encoder,
    ("#Plexers", "BitSelector"): _bit_selector,
    ("#Arithmetic", "Adder"): _adder,
    ("#Arithmetic", "Subtractor"): _subtractor,
    ("#Arithmetic", "Multiplier"): _multiplier,
    ("#Arithmetic", "Divider"): _divider,
    ("#Arithmetic", "Comparator"): _comparator,
    ("#Arithmetic", "Negator"): _negator,
    ("#Arithmetic", "Shifter"): _shifter,
    ("#Memory", "D Flip-Flop"): _flipflop,
    ("#Memory", "T Flip-Flop"): _flipflop,
    ("#Memory", "J-K Flip-Flop"): _flipflop,
    ("#Memory", "S-R Flip-Flop"): _flipflop,
    ("#Memory", "Register"): _register,
    ("#I/O", "LED"): _led,
    ("#I/O", "Button"): _button,
})


# ---------------------------------------------------------------- oracle cache

def _cache_key(desc: str, name: str, attrs: dict[str, str]) -> str:
    skip = {"label", "labelfont", "labelloc", "labelcolor", "labelvisible"}
    items = sorted((k, v) for k, v in attrs.items() if k not in skip)
    return json.dumps([desc, name, items])


class OracleCache:
    """Geometry learned from the JAR, for components without a formula.

    Bundled entries live in logisim/geometry_cache.json; new ones are added to
    a per-user cache file the first time a component is seen (requires Java).
    """

    BUNDLED = Path(__file__).with_name("geometry_cache.json")

    def __init__(self) -> None:
        from .jar import cache_dir
        self.user_file = Path(os.environ.get("LOGISIM_GEOMETRY_CACHE", cache_dir() / "geometry.json"))
        self.data: dict[str, dict] = {}
        for f in (self.BUNDLED, self.user_file):
            if f.exists():
                try:
                    self.data.update(json.loads(f.read_text()))
                except (OSError, ValueError):
                    pass

    def get(self, desc: str, name: str, attrs: dict[str, str]) -> Shape | None:
        e = self.data.get(_cache_key(desc, name, attrs))
        if e is None:
            return None
        return Shape(tuple(Port(*p) for p in e["ports"]), tuple(e["bounds"]))

    def learn(self, items: list[tuple[str, str, str | None, dict[str, str]]]) -> None:
        """Probe the JAR for each (desc, name, lib_id, attrs) and remember the result."""
        from . import io, jar
        from .model import Circuit, Component, Project

        todo = [it for it in items if _cache_key(it[0], it[1], it[3]) not in self.data]
        if not todo:
            return
        proj = Project.new()
        circ = proj.add_circuit(Circuit("probe"))
        placed = []
        for i, (desc, name, _lib, attrs) in enumerate(todo):
            loc = (500 + 600 * (i % 10), 500 + 600 * (i // 10))
            circ.add(Component(name, loc, dict(attrs), proj.lib_id(desc)))
            placed.append((desc, name, attrs, loc))
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "probe.circ"
            io.save(proj, f)
            report = jar.probe_ports(f)
        found = {(c["name"], tuple(c["loc"])): c for c in report["circuits"][0]["components"]}
        for desc, name, attrs, loc in placed:
            c = found.get((name, loc))
            if c is None:
                continue
            ports = [[f"p{k}", e["loc"][0] - loc[0], e["loc"][1] - loc[1], e["width"], e["type"]]
                     for k, e in enumerate(c["ends"])]
            b = c["bounds"]
            self.data[_cache_key(desc, name, attrs)] = {
                "ports": ports, "bounds": [b[0] - loc[0], b[1] - loc[1], b[2], b[3]]}
        try:
            self.user_file.write_text(json.dumps(self.data))
        except OSError:
            pass


_ORACLE: OracleCache | None = None


def oracle() -> OracleCache:
    global _ORACLE
    if _ORACLE is None:
        _ORACLE = OracleCache()
    return _ORACLE


# ---------------------------------------------------------------- public API

def shape(project: Project, comp: Component, *, learn: bool = True) -> Shape:
    """Ports and bounds of `comp`, relative to comp.loc."""
    desc = project.lib_desc(comp)
    if desc is None:
        if not project.has_circuit(comp.name):
            raise UnknownGeometry(f"{comp.name}: not a library component or a circuit in this project")
        try:
            return subcircuit_shape(project, comp.name, comp.attrs)
        except UnknownGeometry:
            raise UnknownGeometry(f"subcircuit {comp.name}: only east-facing default appearance is supported")
    f = FORMULAS.get((desc, comp.name))
    if f is not None:
        return f(comp.name, comp.attrs)
    cached = oracle().get(desc, comp.name, comp.attrs)
    if cached is None and learn:
        from . import jar
        if jar.available():
            oracle().learn([(desc, comp.name, comp.lib, comp.attrs)])
            cached = oracle().get(desc, comp.name, comp.attrs)
    if cached is None:
        raise UnknownGeometry(f"no geometry for {desc} {comp.name} {comp.attrs}")
    return cached


def ports(project: Project, comp: Component) -> list[Port]:
    x, y = comp.loc
    return [p.moved(x, y) for p in shape(project, comp).ports]


def port(project: Project, comp: Component, name: str) -> Port:
    x, y = comp.loc
    return shape(project, comp).port(name).moved(x, y)


def bounds(project: Project, comp: Component) -> Bounds:
    bx, by, bw, bh = shape(project, comp).bounds
    return bx + comp.loc[0], by + comp.loc[1], bw, bh


def supported() -> list[tuple[str, str]]:
    return sorted(FORMULAS)
