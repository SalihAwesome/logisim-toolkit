"""Round trip: load -> save must reproduce every Logisim-written reference file."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from logisim import io
from logisim.model import Component, Project

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE = sorted((FIXTURES / "reference").glob("*.circ")) + [FIXTURES / "subcircuit_matrix.circ"]


def canon(e: ET.Element):
    """Element as nested tuples: tag, attributes, stripped text, children in order."""
    return (e.tag, tuple(sorted(e.attrib.items())), (e.text or "").strip(),
            tuple(canon(c) for c in e))


@pytest.mark.parametrize("path", REFERENCE, ids=lambda p: p.stem)
def test_round_trip_is_lossless(path):
    original = ET.parse(path).getroot()
    again = ET.fromstring(io.dumps(io.load(path)))
    assert canon(again) == canon(original)


@pytest.mark.parametrize("path", REFERENCE, ids=lambda p: p.stem)
def test_round_trip_is_stable(path):
    once = io.dumps(io.load(path))
    assert io.dumps(io.loads(once)) == once


def test_multiline_attribute_uses_text_content():
    proj = Project.new()
    c = proj.add_circuit("main")
    c.add(Component("ROM", (100, 100), {"contents": "addr/data: 4 4\n1 2 3\n"}, proj.lib_id("memory")))
    xml = io.dumps(proj)
    assert '<a name="contents">addr/data: 4 4\n1 2 3\n</a>' in xml
    assert io.loads(xml).circuit("main").components[0].attrs["contents"] == "addr/data: 4 4\n1 2 3\n"


def test_save_refuses_to_overwrite_and_keeps_backup(tmp_path):
    proj = Project.new()
    proj.add_circuit("main")
    f = tmp_path / "x.circ"
    io.save(proj, f)
    with pytest.raises(FileExistsError):
        io.save(proj, f)
    proj.add_circuit("second")
    io.save(proj, f, overwrite=True)
    assert (tmp_path / "x.circ.bak").exists()
    assert io.load(f).circuit_names() == ["main", "second"]
    assert io.load(tmp_path / "x.circ.bak").circuit_names() == ["main"]


def test_bad_input_raises_clear_error():
    with pytest.raises(io.CircFormatError, match="not well-formed"):
        io.loads("<project><circuit")
    with pytest.raises(io.CircFormatError, match="expected <project>"):
        io.loads("<foo/>")
    with pytest.raises(io.CircFormatError, match="bad location"):
        io.loads('<project><circuit name="m"><comp name="Pin" loc="1,2"/></circuit></project>')


def test_new_project_has_logisims_toolbar_so_pins_can_be_poked():
    root = ET.fromstring(io.dumps(Project.new()))
    base = next(lib.get("name") for lib in root.iter("lib") if lib.get("desc") == "#Base")
    toolbar = [(t.get("lib"), t.get("name")) for t in root.find("toolbar").iter("tool")]
    assert (base, "Poke Tool") in toolbar
    assert root.find("mappings").find("tool").get("name") == "Poke Tool"
    assert [c.tag for c in root if c.tag != "lib"] == ["options", "mappings", "toolbar"]
