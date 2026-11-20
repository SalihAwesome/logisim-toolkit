"""Facts about Logisim 3.9.0 that the netlist and router rely on, checked against the JAR."""

import pytest

from logisim import io, jar
from logisim.model import Component, Project

pytestmark = pytest.mark.jar


def _table(tmp_path, wires, pins):
    proj = Project.new()
    circ = proj.add_circuit("main")
    w = proj.lib_id("wiring")
    for loc, label, out in pins:
        attrs = {"appearance": "classic", "label": label}
        if out:
            attrs.update(output="true", facing="west")
        circ.add(Component("Pin", loc, attrs, w))
    for a, b in wires:
        circ.add_wire(a, b)
    f = tmp_path / "t.circ"
    io.save(proj, f)
    lines = jar.run_jar(["-tty", "table", str(f)]).stdout.strip().splitlines()
    head = lines[0].split()
    return [dict(zip(head, ln.split())) for ln in lines[1:]]


def test_wire_end_on_wire_interior_connects(tmp_path):
    rows = _table(tmp_path, [((100, 100), (300, 100)), ((200, 100), (200, 200))],
                  [((100, 100), "A", False), ((300, 100), "Y1", True), ((200, 200), "Y2", True)])
    assert all(r["A"] == r["Y1"] == r["Y2"] for r in rows)


def test_port_on_wire_interior_connects(tmp_path):
    rows = _table(tmp_path, [((100, 100), (300, 100))],
                  [((100, 100), "A", False), ((300, 100), "Y1", True), ((200, 100), "Y2", True)])
    assert all(r["A"] == r["Y2"] for r in rows)


def test_crossing_wires_do_not_connect(tmp_path):
    rows = _table(tmp_path, [((100, 150), (300, 150)), ((200, 50), (200, 250))],
                  [((100, 150), "A", False), ((300, 150), "YA", True),
                   ((200, 50), "B", False), ((200, 250), "YB", True)])
    assert all(r["YA"] == r["A"] and r["YB"] == r["B"] for r in rows)
    assert any(r["A"] != r["B"] for r in rows)


STUB_PINS = [((100, 100), "A", False), ((300, 100), "Y1", True),
             ((200, 50), "Y2", True), ((200, 150), "Y3", True)]


def test_collinear_stubs_meeting_inside_a_wire_do_not_connect_to_it(tmp_path):
    # Logisim merges the two stubs into one wire that merely crosses A's wire
    rows = _table(tmp_path, [((100, 100), (300, 100)), ((200, 50), (200, 100)), ((200, 100), (200, 150))],
                  STUB_PINS)
    assert all(r["A"] == r["Y1"] and r["Y2"] == r["Y3"] == "U" for r in rows)


def test_four_wire_ends_at_one_point_connect(tmp_path):
    rows = _table(tmp_path, [((100, 100), (200, 100)), ((200, 100), (300, 100)),
                             ((200, 50), (200, 100)), ((200, 100), (200, 150))], STUB_PINS)
    assert all(r["A"] == r["Y1"] == r["Y2"] == r["Y3"] for r in rows)
