"""Port geometry must match what the Logisim JAR itself reports.

The fixture comes from tools/make_fixtures.py (real JAR + Java probe).
"""

import json
from pathlib import Path

import pytest

from logisim import geometry
from logisim.model import Component, Project

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "geometry_matrix.json").read_text())
ENTRIES = [e for e in FIXTURE["entries"] if "ends" in e]
LIB = {"gates": "#Gates", "wiring": "#Wiring", "plexers": "#Plexers", "arith": "#Arithmetic",
       "memory": "#Memory", "io": "#I/O"}
FORMULA = [e for e in ENTRIES if (LIB[e["lib"]], e["name"]) in geometry.FORMULAS]


def _shape(e):
    proj = Project.new()
    comp = Component(e["name"], tuple(e["loc"]), dict(e["attrs"]), proj.lib_id(LIB[e["lib"]]))
    return geometry.shape(proj, comp, learn=False)


def _id(e):
    return f"{e['name']}-" + "-".join(f"{k}={v}" for k, v in sorted(e["attrs"].items()))


@pytest.mark.parametrize("e", FORMULA, ids=[_id(e) for e in FORMULA])
def test_ports_match_logisim(e):
    s = _shape(e)
    got = [(p.x, p.y, p.width, p.dir) for p in s.ports]
    want = [(p["loc"][0] - e["loc"][0], p["loc"][1] - e["loc"][1], p["width"], p["type"]) for p in e["ends"]]
    assert got == want


@pytest.mark.parametrize("e", FORMULA, ids=[_id(e) for e in FORMULA])
def test_bounds_match_logisim(e):
    s = _shape(e)
    if not s.exact_bounds:
        pytest.skip("bounds are an estimate for this component")
    b = e["bounds"]
    assert s.bounds == (b[0] - e["loc"][0], b[1] - e["loc"][1], b[2], b[3])


def test_formula_coverage():
    # every family in the matrix has either a formula or is knowingly left to the oracle cache
    names = {e["name"] for e in ENTRIES}
    assert {"AND Gate", "Pin", "Multiplexer", "Decoder", "Adder", "D Flip-Flop", "Register",
            "Splitter", "Tunnel"} <= {n for (_, n) in geometry.FORMULAS} & names


SUB_PROJ = Path(__file__).parent / "fixtures" / "subcircuit_matrix.circ"
SUB_ORACLE = json.loads((Path(__file__).parent / "fixtures" / "subcircuit_matrix.json").read_text())


@pytest.mark.parametrize("inst", SUB_ORACLE, ids=[f"{c['name']}" for c in SUB_ORACLE])
def test_subcircuit_ports_match_logisim(inst):
    from logisim import io
    proj = io.load(SUB_PROJ)
    s = geometry.subcircuit_shape(proj, inst["name"])
    x, y = inst["loc"]
    assert [(p.x, p.y, p.width, p.dir) for p in s.ports] == \
        [(e["loc"][0] - x, e["loc"][1] - y, e["width"], e["type"]) for e in inst["ends"]]
    b = inst["bounds"]
    assert s.bounds == (b[0] - x, b[1] - y, b[2], b[3])
