"""Builder + layout + routing: the drawing must implement exactly the intended nets."""

import itertools
from pathlib import Path

import pytest

from logisim import io, jar, netlist
from logisim.builder import BuildError, Circuit, Project
from logisim.layout import connectivity_errors
from logisim.model import Component, Project as ModelProject


def full_adder(spacing="roomy", use_tunnels=False) -> Circuit:
    c = Circuit("full_adder", spacing=spacing, use_tunnels=use_tunnels)
    a, b, cin = c.input("A"), c.input("B"), c.input("Cin")
    x1 = c.xor(a, b)
    c.output("S", c.xor(x1, cin))
    c.output("Cout", c.or_(c.and_(a, b), c.and_(x1, cin)))
    return c


def adder4() -> Project:
    fa = full_adder()
    top = Circuit("adder4")
    proj = Project(top, fa)
    a, b, ci = top.input("A", 4), top.input("B", 4), top.input("Cin")
    ab, bb = top.split(a), top.split(b)
    carry, sums = ci, []
    for i in range(4):
        u = top.instance(fa, A=ab[i], B=bb[i], Cin=carry)
        sums.append(u.S)
        carry = u.Cout
    top.output("S", top.merge(sums))
    top.output("Cout", carry)
    return proj


def mux_register() -> Circuit:
    c = Circuit("seq_demo")
    d = c.inputs("D0", "D1", "D2", "D3", width=4)
    sel, clk, en = c.input("sel", 2), c.clock("clk"), c.input("en")
    r = c.register(c.mux(sel, *d), clk, en=en)
    c.output("Q", r)
    c.output("Z", c.not_(c.and_(*c.split(r))))
    return c


CASES = {
    "full_adder": lambda: Project(full_adder()),
    "full_adder_compact": lambda: Project(full_adder("compact")),
    "full_adder_tunnels": lambda: Project(full_adder(use_tunnels=True)),
    "adder4": adder4,
    "mux_register": lambda: Project(mux_register()),
}


@pytest.mark.parametrize("name", CASES)
def test_drawing_matches_intended_nets(name):
    proj = CASES[name]()
    model = proj.to_model()  # place_and_route raises if connectivity breaks
    for bc in proj.circuits:
        assert connectivity_errors(bc, model, model.circuit(bc.name)) == []


@pytest.mark.parametrize("name", CASES)
def test_layout_invariants(name):
    model = CASES[name]().to_model()
    for circ in model.circuits:
        locs = [c.loc for c in circ.components]
        assert len(locs) == len(set(locs)), "two components share a location (Logisim hangs)"
        for w in circ.wires:
            assert w.a[0] % 10 == w.a[1] % 10 == w.b[0] % 10 == w.b[1] % 10 == 0
            assert w.horizontal or w.vertical
        for c in circ.components:
            assert c.loc[0] % 10 == 0 and c.loc[1] % 10 == 0


def test_tunnel_mode_uses_tunnels():
    model = Project(full_adder(use_tunnels=True)).to_model()
    assert model.circuit("full_adder").find("Tunnel")


def test_spacing_presets_change_size():
    def width(spacing):
        circ = Project(full_adder(spacing)).to_model().circuits[0]
        xs = [c.loc[0] for c in circ.components]
        return max(xs) - min(xs)
    assert width("compact") < width("normal") < width("roomy")


def test_width_mismatch_is_rejected():
    c = Circuit("bad")
    a, b = c.input("A", 4), c.input("B", 2)
    with pytest.raises(BuildError, match="width mismatch"):
        c.and_(a, b)


def test_duplicate_pin_label_is_rejected():
    c = Circuit("bad")
    c.input("A")
    with pytest.raises(BuildError, match="duplicate"):
        c.input("A")


# ---- netlist rules (same rules the JAR follows, see test_jar_rules)

def _pins_project(wires, pins):
    proj = ModelProject.new()
    circ = proj.add_circuit("main")
    w = proj.lib_id("wiring")
    for loc, label, out in pins:
        attrs = {"label": label}
        if out:
            attrs.update(output="true", facing="west")
        circ.add(Component("Pin", loc, attrs, w))
    for a, b in wires:
        circ.add_wire(a, b)
    return proj, circ


def test_netlist_t_junction_and_port_on_wire_connect():
    proj, circ = _pins_project(
        [((100, 100), (300, 100)), ((200, 100), (200, 200))],
        [((100, 100), "A", False), ((300, 100), "Y1", True), ((200, 200), "Y2", True),
         ((250, 100), "Y3", True)])
    nl = netlist.extract(proj, circ)
    nets = {nl.port_net[(i, "p")] for i in range(4)}
    assert len(nets) == 1


def test_netlist_crossing_does_not_connect():
    proj, circ = _pins_project(
        [((100, 150), (300, 150)), ((200, 50), (200, 250))],
        [((100, 150), "A", False), ((300, 150), "YA", True), ((200, 50), "B", False), ((200, 250), "YB", True)])
    nl = netlist.extract(proj, circ)
    assert nl.port_net[(0, "p")] == nl.port_net[(1, "p")]
    assert nl.port_net[(2, "p")] == nl.port_net[(3, "p")]
    assert nl.port_net[(0, "p")] != nl.port_net[(2, "p")]


def test_netlist_of_logisim_saved_half_adder():
    proj = io.load(Path(__file__).parent / "fixtures" / "reference" / "subcircuit.circ")
    nl = netlist.extract(proj, "half_adder")
    real = [n for n in nl.nets if n.ports]
    assert len(real) == 4  # A, B, S, C
    assert all(len(n.ports) == (3 if any(p.port.dir == "out" and
               proj.circuit("half_adder").components[p.comp].name == "Pin" for p in n.ports) else 2)
               for n in real)


@pytest.mark.jar
def test_full_adder_truth_table_in_logisim(tmp_path):
    f = tmp_path / "fa.circ"
    Project(full_adder()).save(f)
    out = jar.run_jar(["-tty", "table", str(f)]).stdout.split("\n")
    header = out[0].split()
    rows = [dict(zip(header, line.split())) for line in out[1:] if line.strip()]
    assert len(rows) == 8
    for r in rows:
        a, b, c = int(r["A"]), int(r["B"]), int(r["Cin"])
        assert int(r["S"]) == a ^ b ^ c
        assert int(r["Cout"]) == (a + b + c) // 2
