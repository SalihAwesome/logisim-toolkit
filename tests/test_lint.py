"""Lint: each check on a small circuit, plus the builder's output and Logisim-saved files."""

from pathlib import Path

import pytest

from logisim import io, jar, lint
from logisim.builder import Circuit, Project
from logisim.model import Component, Project as ModelProject

from test_builder import CASES

REFERENCE = Path(__file__).parent / "fixtures" / "reference"


def _circ(pins=(), wires=(), comps=()):
    """pins: (loc, label, is_output, width); comps: (lib, name, loc, attrs)."""
    proj = ModelProject.new()
    circ = proj.add_circuit("main")
    for loc, label, out, width in pins:
        attrs = {"label": label}
        if out:
            attrs.update(output="true", facing="west", labelloc="east")
        if width != 1:
            attrs["width"] = str(width)
        circ.add(Component("Pin", loc, attrs, proj.lib_id("wiring")))
    for lib, name, loc, attrs in comps:
        circ.add(Component(name, loc, attrs, proj.lib_id(lib)))
    for a, b in wires:
        circ.add_wire(a, b)
    return proj


def codes(proj, **kw):
    return {i.code for i in lint.lint(proj, **kw)}


def test_clean_wire_between_pins():
    proj = _circ([((100, 100), "A", False, 1), ((300, 100), "Y", True, 1)], [((100, 100), (300, 100))])
    assert lint.lint(proj) == []


def test_floating_input_and_unused_output():
    proj = _circ([((100, 100), "A", False, 1), ((400, 300), "Y", True, 1)],
                 comps=[("gates", "AND Gate", (300, 100), {})])
    found = codes(proj)
    assert {"floating-input", "unused-output"} <= found
    issues = [i for i in lint.lint(proj) if i.code == "floating-input"]
    assert {i.at for i in issues} == {(250, 80), (250, 120), (400, 300)}


def test_optional_inputs_are_not_floating():
    proj = _circ([((100, 110), "D", False, 1), ((100, 150), "clk", False, 1), ((300, 110), "Q", True, 1)],
                 [((100, 110), (190, 110)), ((100, 150), (190, 150)), ((250, 110), (300, 110))],
                 comps=[("memory", "D Flip-Flop", (200, 100), {"appearance": "logisim_evolution"})])
    assert not [i for i in lint.lint(proj) if i.code == "floating-input"]


def test_multiple_drivers_is_an_error():
    proj = _circ([((100, 100), "A", False, 1), ((100, 200), "B", False, 1), ((300, 100), "Y", True, 1)],
                 [((100, 100), (300, 100)), ((100, 200), (200, 200)), ((200, 200), (200, 100))])
    [issue] = [i for i in lint.lint(proj) if i.code == "multiple-drivers"]
    assert issue.severity == "error" and "'A'" in issue.message and "'B'" in issue.message


def test_width_mismatch_is_an_error():
    proj = _circ([((100, 100), "A", False, 4), ((300, 100), "Y", True, 1)], [((100, 100), (300, 100))])
    [issue] = [i for i in lint.lint(proj) if i.code == "width-mismatch"]
    assert issue.severity == "error" and "=4" in issue.message and "=1" in issue.message


@pytest.mark.jar
@pytest.mark.parametrize("name, build", [
    ("multiple-drivers", lambda: _circ(
        [((100, 100), "A", False, 1), ((100, 200), "B", False, 1), ((300, 100), "Y", True, 1)],
        [((100, 100), (300, 100)), ((100, 200), (200, 200)), ((200, 200), (200, 100))])),
    ("width-mismatch", lambda: _circ(
        [((100, 100), "A", False, 4), ((300, 100), "Y", True, 1)], [((100, 100), (300, 100))])),
])
def test_lint_errors_break_the_circuit_in_logisim(tmp_path, name, build):
    f = tmp_path / "e.circ"
    io.save(build(), f)
    lines = jar.run_jar(["-tty", "table", str(f)]).stdout.split("\n")
    head = lines[0].split()
    rows = [dict(zip(head, ln.split())) for ln in lines[1:] if ln.strip()]
    assert rows and any(not set(r["Y"]) <= {"0", "1"} for r in rows), rows


def test_dangling_wire():
    proj = _circ([((100, 100), "A", False, 1), ((300, 100), "Y", True, 1)],
                 [((100, 100), (300, 100)), ((200, 100), (200, 200))])
    [issue] = [i for i in lint.lint(proj) if i.code == "dangling-wire"]
    assert issue.at == (200, 200)


def test_merged_junction():
    # two stubs meeting in a straight line on A's wire: Logisim merges them (test_jar_rules)
    proj = _circ([((100, 100), "A", False, 1), ((300, 100), "Y1", True, 1),
                  ((200, 50), "Y2", True, 1), ((200, 150), "Y3", True, 1)],
                 [((100, 100), (300, 100)), ((200, 50), (200, 100)), ((200, 100), (200, 150))])
    [issue] = [i for i in lint.lint(proj) if i.code == "merged-junction"]
    assert issue.at == (200, 100)
    split = _circ([((100, 100), "A", False, 1), ((300, 100), "Y1", True, 1),
                   ((200, 50), "Y2", True, 1), ((200, 150), "Y3", True, 1)],
                  [((100, 100), (200, 100)), ((200, 100), (300, 100)),
                   ((200, 50), (200, 100)), ((200, 100), (200, 150))])
    assert "merged-junction" not in codes(split)


def test_tunnels_and_pin_labels():
    proj = _circ([((100, 100), "A", False, 1), ((100, 200), "A", False, 1), ((100, 300), "", False, 1)],
                 comps=[("wiring", "Tunnel", (200, 100), {"label": "solo"})])
    found = {(i.code, i.severity) for i in lint.lint(proj)}
    assert {("duplicate-label", "error"), ("unlabeled-pin", "warning"), ("lonely-tunnel", "warning")} <= found


def test_same_location_and_off_grid():
    proj = _circ(comps=[("gates", "AND Gate", (300, 100), {}), ("gates", "OR Gate", (300, 100), {}),
                        ("gates", "NOT Gate", (505, 100), {})])
    found = {(i.code, i.severity) for i in lint.lint(proj)}
    assert {("same-location", "error"), ("off-grid", "warning")} <= found


def test_overlap_too_close_and_wire_through_component():
    proj = _circ(comps=[("gates", "AND Gate", (300, 100), {}), ("gates", "AND Gate", (320, 120), {}),
                        ("gates", "OR Gate", (300, 200), {}), ("gates", "OR Gate", (300, 255), {})],
                 wires=[((200, 300), (400, 300)), ((240, 100), (240, 30))])
    found = codes(proj)
    assert {"overlap", "too-close"} <= found
    proj = _circ([((100, 100), "A", False, 1), ((400, 100), "Y", True, 1)], [((100, 100), (400, 100))],
                 comps=[("gates", "AND Gate", (300, 110), {})])
    assert "wire-through-component" in codes(proj)


def test_label_overlaps_wire():
    # an output pin with its label on the west, where its own wire arrives
    proj = _circ([((100, 100), "A", False, 1)], [((100, 100), (300, 100))])
    proj.circuit("main").add(Component("Pin", (300, 100), {"label": "OUT", "output": "true",
                                                           "facing": "west"}, proj.lib_id("wiring")))
    assert "label-overlap" in codes(proj)


def test_wire_spacing_and_crossings():
    pins = [((100, 100), "A", False, 1), ((400, 100), "Y", True, 1),
            ((100, 110), "B", False, 1), ((400, 110), "Z", True, 1)]
    proj = _circ(pins, [((100, 100), (400, 100)), ((100, 110), (400, 110))])
    assert "wire-spacing" in codes(proj)
    wires = [((100, 100 + 20 * i), (500, 100 + 20 * i)) for i in range(5)]
    wires += [((150 + 20 * i, 50), (150 + 20 * i, 250)) for i in range(5)]
    assert "crossings" in codes(_circ(wires=wires), config=lint.LintConfig(max_crossings=20))
    assert "crossings" not in codes(_circ(wires=wires), config=lint.LintConfig(max_crossings=25))


def test_unused_subcircuit():
    proj = Project(Circuit("top"), Circuit("spare")).to_model()
    [issue] = [i for i in lint.lint(proj) if i.code == "unused-subcircuit"]
    assert issue.circuit == "spare"


# Layout debt the lint finds in the builder's own output; shrink this as the router improves.
KNOWN_LAYOUT_WARNINGS = {"wire-spacing", "crossings"}


@pytest.mark.parametrize("name", CASES)
def test_builder_output_lints_clean(name):
    issues = lint.lint(CASES[name]().to_model())
    assert {i.code for i in issues} <= KNOWN_LAYOUT_WARNINGS, lint.report(issues)


@pytest.mark.parametrize("path", sorted(REFERENCE.glob("*.circ")), ids=lambda p: p.name)
def test_logisim_saved_files_have_no_errors(path):
    issues = lint.lint_file(path)
    assert not [i for i in issues if i.severity == "error"], lint.report(issues)


def test_report_format():
    proj = _circ([((100, 100), "A", False, 4), ((300, 100), "Y", True, 1)], [((100, 100), (300, 100))])
    text = lint.report(lint.lint(proj))
    assert text.splitlines()[0] == "1 error"
    assert "main: error [width-mismatch] at (100,100): bit widths differ" in text
