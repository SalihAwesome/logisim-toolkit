"""Build the geometry oracle: ask the real Logisim JAR where every port is.

Writes one .circ holding a matrix of component configurations (each facing,
size, input count, width, ...), has Logisim re-save it with `-n`, then runs the
Java probe on the result. The output fixture maps each placed component to the
port ends and bounds Logisim reports. tests/test_geometry.py compares the
Python geometry table against it.

    python tools/make_fixtures.py            # rewrite tests/fixtures/geometry_matrix.json
"""

from __future__ import annotations

import itertools
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logisim import io, jar  # noqa: E402
from logisim.model import Circuit, Component, Project  # noqa: E402

FACINGS = ["east", "west", "north", "south"]
GATES = ["AND Gate", "OR Gate", "NAND Gate", "NOR Gate", "XOR Gate", "XNOR Gate",
         "Odd Parity", "Even Parity"]


def variants() -> dict[str, list[tuple[str, str, dict[str, str]]]]:
    v: dict[str, list[tuple[str, str, dict[str, str]]]] = {}
    g = v.setdefault("gates", [])
    for name, n, size, f in itertools.product(GATES, [2, 3, 4, 5, 6, 7, 9], ["30", "50", "70"], FACINGS):
        g.append(("gates", name, {"facing": f, "inputs": str(n), "size": size}))
    for name, size, f in itertools.product(["NOT Gate", "Buffer"], ["20", "30"], FACINGS):
        g.append(("gates", name, {"facing": f, "size": size}))
    for name, f, ctrl in itertools.product(["Controlled Buffer", "Controlled Inverter"], FACINGS, ["right", "left"]):
        g.append(("gates", name, {"facing": f, "control": ctrl}))

    w = v.setdefault("wiring", [])
    for out, f, width in itertools.product(["false", "true"], FACINGS, ["1", "8"]):
        w.append(("wiring", "Pin", {"facing": f, "output": out, "width": width}))
    for name, f in itertools.product(["Constant", "Clock", "Probe", "Tunnel", "Pull Resistor",
                                      "Power", "Ground"], FACINGS):
        w.append(("wiring", name, {"facing": f}))
    for f in FACINGS:
        w.append(("wiring", "Bit Extender", {"in_width": "4", "out_width": "8"}))
    sp = v.setdefault("splitter", [])
    for f, fan, appear, spacing in itertools.product(FACINGS, [2, 3, 4, 8], ["left", "right", "center", "legacy"], ["1", "2"]):
        attrs = {"facing": f, "fanout": str(fan), "incoming": str(fan), "appear": appear}
        if spacing != "1":
            attrs["spacing"] = spacing
        sp.append(("wiring", "Splitter", attrs))

    p = v.setdefault("plexers", [])
    for name, sel, f, loc, en in itertools.product(["Multiplexer", "Demultiplexer"], [1, 2, 3], FACINGS,
                                                    ["bl", "tr"], ["false", "true"]):
        p.append(("plexers", name, {"facing": f, "select": str(sel), "selloc": loc, "enable": en}))
    for sel, f, loc, en in itertools.product([1, 2, 3], FACINGS, ["bl", "tr"], ["false", "true"]):
        p.append(("plexers", "Decoder", {"facing": f, "select": str(sel), "selloc": loc, "enable": en}))
    for sel, f in itertools.product([1, 2, 3], FACINGS):
        p.append(("plexers", "Priority Encoder", {"facing": f, "select": str(sel)}))
    for f in FACINGS:
        p.append(("plexers", "BitSelector", {"facing": f}))

    a = v.setdefault("arith", [])
    for name in ["Adder", "Subtractor", "Multiplier", "Divider", "Negator", "Comparator", "Shifter",
                 "BitAdder", "BitFinder"]:
        for width in ["1", "4"]:
            a.append(("arith", name, {"width": width}))

    m = v.setdefault("memory", [])
    for name in ["D Flip-Flop", "T Flip-Flop", "J-K Flip-Flop", "S-R Flip-Flop"]:
        for appear in ["logisim_evolution", "classic"]:
            m.append(("memory", name, {"appearance": appear}))
    for name in ["Register", "Counter", "Shift Register", "Random"]:
        for appear in ["logisim_evolution", "classic"]:
            m.append(("memory", name, {"appearance": appear, "width": "4"}))
    for appear in ["logisim_evolution", "classic"]:
        m.append(("memory", "ROM", {"appearance": appear, "addrWidth": "4", "dataWidth": "4"}))
        m.append(("memory", "RAM", {"appearance": appear, "addrWidth": "4", "dataWidth": "4"}))

    o = v.setdefault("io", [])
    for name in ["LED", "Button", "Hex Digit Display", "7-Segment Display", "DipSwitch"]:
        for f in FACINGS:
            o.append(("io", name, {"facing": f}))
    return v


def build() -> tuple[Project, list[dict]]:
    proj = Project.new()
    entries: list[dict] = []
    for circ_name, specs in variants().items():
        circ = proj.add_circuit(Circuit(circ_name))
        for i, (lib, name, attrs) in enumerate(specs):
            loc = (200 + 200 * (i % 20), 200 + 200 * (i // 20))
            circ.add(Component(name, loc, dict(attrs), proj.lib_id(lib)))
            entries.append({"circuit": circ_name, "lib": lib, "name": name,
                            "attrs": attrs, "loc": list(loc)})
    return proj, entries


def main() -> None:
    proj, entries = build()
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "matrix.circ"
        norm = Path(td) / "matrix_norm.circ"
        io.save(proj, raw)
        jar.normalize(raw, norm)
        report = jar.probe_ports(norm)
        normalized = io.load(norm)
    seen = {}
    for c in report["circuits"]:
        for comp in c["components"]:
            seen[(c["name"], comp["name"], tuple(comp["loc"]))] = comp
    norm_attrs = {}
    for circ in normalized.circuits:
        for comp in circ.components:
            norm_attrs[(circ.name, comp.name, comp.loc)] = comp.attrs
    missing = 0
    for e in entries:
        key = (e["circuit"], e["name"], tuple(e["loc"]))
        got = seen.get(key)
        if got is None:
            missing += 1
            continue
        e["ends"] = got["ends"]
        e["bounds"] = got["bounds"]
        e["saved_attrs"] = norm_attrs.get(key, {})
    if missing:
        print(f"warning: {missing} components did not survive loading", file=sys.stderr)
    out = ROOT / "tests" / "fixtures" / "geometry_matrix.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"logisim": str(jar.find_jar().name), "entries": entries}, indent=1))
    print(f"wrote {len(entries)} entries to {out}")


if __name__ == "__main__" and "--subcircuits" not in sys.argv:
    main()


def build_subcircuits() -> Project:
    """Subcircuits in both auto appearances, with varied pin counts, order and facing."""
    proj = Project.new()
    w = proj.lib_id("wiring")
    cases = []
    for ap in ["logisim_evolution", "classic", None]:
        for fixed in ["true", "false", None]:
            for nin, nout, lab, outs_first in [(2, 2, "A", False), (3, 1, "LONGNAME", True),
                                               (1, 4, "X", False), (5, 3, "D", True), (0, 2, "K", False)]:
                name = f"s_{ap}_{fixed}_{nin}{nout}{len(lab)}"
                c = proj.add_circuit(name)
                if ap:
                    c.attrs["appearance"] = ap
                if fixed:
                    c.attrs["circuitnamedboxfixedsize"] = fixed
                rows = list(reversed(range(nin))) if outs_first else list(range(nin))
                ins = [Component("Pin", (100, 100 + 40 * rows[i]),
                                 {"label": f"{lab}{i}", "appearance": "classic", "width": str(1 + i % 3)}, w)
                       for i in range(nin)]
                outs = [Component("Pin", (400, 100 + 40 * i), {"label": f"{lab}o{i}", "output": "true",
                                                               "facing": "west", "appearance": "classic"}, w)
                        for i in range(nout)]
                for comp in (outs + ins if outs_first else ins + outs):
                    c.add(comp)
                cases.append(name)
    c = proj.add_circuit("s_classic_nsew")
    c.attrs["appearance"] = "classic"
    for i, f in enumerate(["east", "south", "north", "west", "south", "east"]):
        c.add(Component("Pin", (100 + 60 * i, 100 + 30 * i), {"label": f"p{i}", "facing": f,
                                                              "output": "true" if i % 2 else "false"}, w))
    cases.append("s_classic_nsew")
    m = proj.add_circuit("main")
    proj.main = "main"
    for i, n in enumerate(cases):
        m.add(Component(n, (500, 300 + 300 * i), {}))
    return proj


def main_subcircuits() -> None:
    proj = build_subcircuits()
    dest = ROOT / "tests" / "fixtures" / "subcircuit_matrix.circ"
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "sub.circ"
        io.save(proj, raw)
        if dest.exists():
            dest.unlink()
        jar.normalize(raw, dest)
    report = jar.probe_ports(dest)
    main_c = next(c for c in report["circuits"] if c["name"] == "main")
    out = ROOT / "tests" / "fixtures" / "subcircuit_matrix.json"
    out.write_text(json.dumps(main_c["components"], indent=1))
    print(f"wrote {out}")


if __name__ == "__main__" and "--subcircuits" in sys.argv:
    main_subcircuits()
