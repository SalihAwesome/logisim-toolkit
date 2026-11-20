"""Object model for Logisim-evolution .circ projects.

A Project holds libraries and circuits. A Circuit holds components and wires.
Elements the toolkit does not model (appearance, VHDL, board maps, options)
are kept as raw ElementTree nodes so a load/save round trip preserves them.
"""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

Point = tuple[int, int]

LOGISIM_VERSION = "3.9.0"

# Library descriptors used by Logisim-evolution 3.9.0, keyed by short name.
LIB_DESC = {
    "wiring": "#Wiring",
    "gates": "#Gates",
    "plexers": "#Plexers",
    "arith": "#Arithmetic",
    "memory": "#Memory",
    "io": "#I/O",
    "ttl": "#TTL",
    "tcl": "#TCL",
    "base": "#Base",
    "bfh": "#BFH-Praktika",
    "hdl": "#HDL-IP",
}


@dataclass
class Component:
    name: str
    loc: Point
    attrs: dict[str, str] = field(default_factory=dict)
    lib: str | None = None  # library id ("0", "1", ...); None means a circuit in this file

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.attrs.get(key, default)

    def set(self, key: str, value) -> None:
        self.attrs[key] = str(value)

    @property
    def label(self) -> str:
        return self.attrs.get("label", "")

    @label.setter
    def label(self, value: str) -> None:
        self.attrs["label"] = value

    @property
    def facing(self) -> str:
        return self.attrs.get("facing", "")

    def copy(self) -> "Component":
        return Component(self.name, self.loc, dict(self.attrs), self.lib)

    def __repr__(self) -> str:
        lab = f" {self.label!r}" if self.label else ""
        return f"<{self.name}{lab} @{self.loc[0]},{self.loc[1]}>"


@dataclass
class Wire:
    a: Point
    b: Point

    def __post_init__(self) -> None:
        self.a = (int(self.a[0]), int(self.a[1]))
        self.b = (int(self.b[0]), int(self.b[1]))

    @property
    def horizontal(self) -> bool:
        return self.a[1] == self.b[1]

    @property
    def vertical(self) -> bool:
        return self.a[0] == self.b[0]

    @property
    def length(self) -> int:
        return abs(self.a[0] - self.b[0]) + abs(self.a[1] - self.b[1])

    def key(self) -> tuple[Point, Point]:
        return (self.a, self.b) if self.a <= self.b else (self.b, self.a)

    def contains(self, p: Point) -> bool:
        """True if p lies on the segment, endpoints included."""
        (x0, y0), (x1, y1) = self.a, self.b
        if x0 == x1 == p[0]:
            return min(y0, y1) <= p[1] <= max(y0, y1)
        if y0 == y1 == p[1]:
            return min(x0, x1) <= p[0] <= max(x0, x1)
        return False

    def interior(self, p: Point) -> bool:
        return self.contains(p) and p != self.a and p != self.b


@dataclass
class Library:
    id: str
    desc: str
    tools: list[ET.Element] = field(default_factory=list)


@dataclass
class Circuit:
    name: str
    components: list[Component] = field(default_factory=list)
    wires: list[Wire] = field(default_factory=list)
    attrs: dict[str, str] = field(default_factory=dict)
    extras: list[ET.Element] = field(default_factory=list)  # <appear>, <boardmap>, ...

    def add(self, comp: Component) -> Component:
        self.components.append(comp)
        return comp

    def add_wire(self, a: Point, b: Point) -> Wire | None:
        if a == b:
            return None
        if a[0] != b[0] and a[1] != b[1]:
            raise ValueError(f"wire {a}->{b} is not orthogonal")
        w = Wire(a, b)
        self.wires.append(w)
        return w

    def find(self, name: str | None = None, label: str | None = None) -> list[Component]:
        return [
            c
            for c in self.components
            if (name is None or c.name == name) and (label is None or c.label == label)
        ]

    def pins(self) -> list[Component]:
        return [c for c in self.components if c.name == "Pin"]

    def inputs(self) -> list[Component]:
        return [c for c in self.pins() if c.get("output", "false") != "true"]

    def outputs(self) -> list[Component]:
        return [c for c in self.pins() if c.get("output", "false") == "true"]

    def copy(self) -> "Circuit":
        return Circuit(
            self.name,
            [c.copy() for c in self.components],
            [Wire(w.a, w.b) for w in self.wires],
            dict(self.attrs),
            [copy.deepcopy(e) for e in self.extras],
        )

    def __repr__(self) -> str:
        return f"<Circuit {self.name!r}: {len(self.components)} components, {len(self.wires)} wires>"


@dataclass
class Project:
    circuits: list[Circuit] = field(default_factory=list)
    libraries: list[Library] = field(default_factory=list)
    main: str | None = None
    source: str = LOGISIM_VERSION
    version: str = "1.0"
    options: ET.Element | None = None
    mappings: ET.Element | None = None
    toolbar: ET.Element | None = None
    extras: list[ET.Element] = field(default_factory=list)  # <vhdl>, <message>, ...

    @classmethod
    def new(cls) -> "Project":
        p = cls()
        p.lib_id("wiring")
        p.lib_id("gates")
        return p

    def circuit(self, name: str) -> Circuit:
        for c in self.circuits:
            if c.name == name:
                return c
        raise KeyError(f"no circuit named {name!r} (have: {', '.join(self.circuit_names())})")

    def has_circuit(self, name: str) -> bool:
        return any(c.name == name for c in self.circuits)

    def circuit_names(self) -> list[str]:
        return [c.name for c in self.circuits]

    def add_circuit(self, circ: Circuit | str) -> Circuit:
        if isinstance(circ, str):
            circ = Circuit(circ)
        if self.has_circuit(circ.name):
            raise ValueError(f"circuit {circ.name!r} already exists")
        if not circ.attrs:
            circ.attrs = {"circuit": circ.name}
        self.circuits.append(circ)
        if self.main is None:
            self.main = circ.name
        return circ

    @property
    def main_circuit(self) -> Circuit:
        if self.main and self.has_circuit(self.main):
            return self.circuit(self.main)
        if not self.circuits:
            raise KeyError("project has no circuits")
        return self.circuits[0]

    def library(self, lib_id: str | None) -> Library | None:
        for lib in self.libraries:
            if lib.id == lib_id:
                return lib
        return None

    def lib_desc(self, comp: Component) -> str | None:
        """Library descriptor ("#Gates") of a component, or None for a subcircuit."""
        lib = self.library(comp.lib)
        return lib.desc if lib else None

    def lib_id(self, short_or_desc: str) -> str:
        """Library id for a short name ("gates") or descriptor ("#Gates"), adding it if missing."""
        desc = LIB_DESC.get(short_or_desc, short_or_desc)
        for lib in self.libraries:
            if lib.desc == desc:
                return lib.id
        used = {int(lib.id) for lib in self.libraries if lib.id.isdigit()}
        new_id = str(next(i for i in range(len(used) + 1) if i not in used))
        self.libraries.append(Library(new_id, desc))
        return new_id

    def is_subcircuit(self, comp: Component) -> bool:
        return comp.lib is None and self.has_circuit(comp.name)

    def copy(self) -> "Project":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"<Project {len(self.circuits)} circuits, main={self.main!r}>"
