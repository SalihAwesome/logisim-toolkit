"""Write the reference .circ files (saved by Logisim itself) used by the round-trip tests.

Each circuit is assembled with the low-level model, then the JAR loads and
re-saves it with `-n`, so tests/fixtures/reference/*.circ are genuine
Logisim-evolution 3.9.0 output.

    python tools/make_references.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logisim import geometry, io, jar  # noqa: E402
from logisim.model import Component, Project  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "reference"


def pin(proj, circ, loc, label, width=1, output=False, facing=None):
    a = {"appearance": "classic", "label": label}
    if width != 1:
        a["width"] = str(width)
    if output:
        a["output"] = "true"
        a["facing"] = facing or "west"
    elif facing:
        a["facing"] = facing
    return circ.add(Component("Pin", loc, a, proj.lib_id("wiring")))


def wire_l(circ, a, b):
    """Connect a to b with at most one bend (horizontal first)."""
    if a[0] != b[0] and a[1] != b[1]:
        circ.add_wire(a, (b[0], a[1]))
        circ.add_wire((b[0], a[1]), b)
    else:
        circ.add_wire(a, b)


def comp_with_io(proj, name, lib, attrs, label_prefix=""):
    """One component; every input port gets an input pin, every output an output pin."""
    circ = proj.add_circuit(name.replace(" ", "_").lower() if not proj.circuits else name)
    c = circ.add(Component(attrs.pop("_name"), (400, 300), attrs, proj.lib_id(lib)))
    for p in geometry.ports(proj, c):
        if p.dir == "in":
            src = (p.x - 60, p.y)
            pin(proj, circ, src, label_prefix + p.name.upper(), p.width)
            circ.add_wire(src, p.pos)
        else:
            dst = (p.x + 60, p.y)
            pin(proj, circ, dst, label_prefix + p.name.upper(), p.width, output=True)
            circ.add_wire(p.pos, dst)
    return circ


def build_all() -> dict[str, Project]:
    files: dict[str, Project] = {}

    for gate, n in [("AND Gate", 2), ("AND Gate", 3), ("NOT Gate", None), ("XOR Gate", 2)]:
        proj = Project.new()
        attrs = {"_name": gate}
        if n:
            attrs["inputs"] = str(n)
        comp_with_io(proj, "main", "gates", attrs)
        files[f"{gate.split()[0].lower()}{n or ''}"] = proj

    proj = Project.new()
    for f in ["east", "west", "north", "south"]:
        circ = proj.add_circuit(f"and_{f}")
        circ.add(Component("AND Gate", (300, 300), {"facing": f}, proj.lib_id("gates")))
    files["and_facings"] = proj

    specs = {
        "mux2": ("plexers", {"_name": "Multiplexer", "enable": "false", "width": "4"}),
        "mux4": ("plexers", {"_name": "Multiplexer", "select": "2", "enable": "false"}),
        "decoder3": ("plexers", {"_name": "Decoder", "select": "3", "enable": "false"}),
        "dff": ("memory", {"_name": "D Flip-Flop"}),
        "register8": ("memory", {"_name": "Register", "width": "8"}),
        "adder4": ("arith", {"_name": "Adder", "width": "4"}),
    }
    for fname, (lib, attrs) in specs.items():
        proj = Project.new()
        comp_with_io(proj, "main", lib, dict(attrs))
        files[fname] = proj

    # splitter: 8-bit bus in, split into two nibbles, plus tunnels carrying one nibble
    proj = Project.new()
    circ = proj.add_circuit("main")
    w = proj.lib_id("wiring")
    pin(proj, circ, (100, 200), "BUS", 8)
    sp = circ.add(Component("Splitter", (200, 200), {"fanout": "2", "incoming": "8"}, w))
    circ.add_wire((100, 200), (200, 200))
    s0, s1 = geometry.ports(proj, sp)[1:]
    pin(proj, circ, (400, s0.y), "LO", 4, output=True)
    circ.add_wire(s0.pos, (400, s0.y))
    circ.add(Component("Tunnel", (260, s1.y), {"label": "hi", "width": "4", "facing": "west"}, w))
    circ.add_wire(s1.pos, (260, s1.y))
    circ.add(Component("Tunnel", (340, 300), {"label": "hi", "width": "4", "facing": "east"}, w))
    pin(proj, circ, (400, 300), "HI", 4, output=True)
    circ.add_wire((340, 300), (400, 300))
    files["splitter_tunnel"] = proj

    # subcircuit: half adder used twice by a full adder
    proj = Project.new()
    g = proj.lib_id("gates")
    ha = proj.add_circuit("half_adder")
    pin(proj, ha, (100, 100), "A")
    pin(proj, ha, (100, 200), "B")
    x = ha.add(Component("XOR Gate", (300, 100), {}, g))
    a = ha.add(Component("AND Gate", (300, 200), {}, g))
    xi = [p for p in geometry.ports(proj, x) if p.dir == "in"]
    ai = [p for p in geometry.ports(proj, a) if p.dir == "in"]
    ha.add_wire((100, 100), (150, 100))
    ha.add_wire((150, 100), (150, 180))
    ha.add_wire((150, 80), (150, 100))
    ha.add_wire((150, 80), xi[0].pos)
    ha.add_wire((150, 180), ai[0].pos)
    ha.add_wire((100, 200), (170, 200))
    ha.add_wire((170, 120), (170, 220))
    ha.add_wire((170, 120), xi[1].pos)
    ha.add_wire((170, 220), ai[1].pos)
    pin(proj, ha, (400, 100), "S", output=True)
    pin(proj, ha, (400, 200), "C", output=True)
    ha.add_wire((300, 100), (400, 100))
    ha.add_wire((300, 200), (400, 200))
    main = proj.add_circuit("main")
    proj.main = "main"
    main.add(Component("half_adder", (400, 300), {}))
    files["subcircuit"] = proj
    return files


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for name, proj in build_all().items():
            raw = Path(td) / f"{name}.circ"
            io.save(proj, raw)
            dest = OUT / f"{name}.circ"
            if dest.exists():
                dest.unlink()
            jar.normalize(raw, dest)
            print("wrote", dest.relative_to(ROOT))


if __name__ == "__main__":
    main()
