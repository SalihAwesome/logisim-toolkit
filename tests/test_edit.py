"""Editing existing .circ files: moves, attribute changes, insertions, tidying.

Every operation must leave a drawing that Logisim can load, that lints without
errors and whose connectivity is unchanged (except where the edit is meant to
change it).  The JAR tests check the edited files against a Python reference
with the real Logisim.
"""

import pytest

from logisim import edit, io, lint, netlist, sim
from logisim.builder import Circuit as BCircuit
from logisim.builder import Project as BProject
from logisim.model import Component, Project as ModelProject, Wire

from test_builder import adder4, counter2, full_adder


# --------------------------------------------------------------- helpers


def fa_project() -> ModelProject:
    """The builder's full adder as a model project."""
    return BProject(full_adder()).to_model()


def adder4_project() -> ModelProject:
    """The builder's 4-bit adder as a model project."""
    return adder4().to_model()


def signature(project, circuit) -> frozenset:
    """The layout-independent netlist signature of a circuit."""
    return netlist.extract(project, circuit).signature()


def partition(project, circuit, exclude=()) -> frozenset:
    """The nets as sets of (component index, port name), optionally without some components."""
    exclude = set(exclude)
    groups = set()
    for net in netlist.extract(project, circuit).nets:
        members = frozenset((p.comp, p.port.name) for p in net.ports if p.comp not in exclude)
        if members:
            groups.add(members)
    return frozenset(groups)


def partition_by_object(project, circuit) -> frozenset:
    """The nets as sets of (component, port name), keyed by the component itself.

    Unlike :func:`partition` this survives an insertion or a deletion, which
    shift the component indices.
    """
    comps = project.circuit(circuit).components
    groups = set()
    for net in netlist.extract(project, circuit).nets:
        members = frozenset((id(comps[p.comp]), p.port.name) for p in net.ports)
        if members:
            groups.add(members)
    return frozenset(groups)


def errors(project, circuit=None) -> list[str]:
    return [str(i) for i in lint.lint(project, circuit) if i.severity == "error"]


def warnings(project, circuit=None) -> list[str]:
    return [str(i) for i in lint.lint(project, circuit) if i.severity == "warning"]


def codes(project, circuit=None) -> set[str]:
    return {i.code for i in lint.lint(project, circuit)}


def gates(circuit) -> list[int]:
    return [i for i, c in enumerate(circuit.components) if c.name.endswith("Gate")]


# ------------------------------------------------------------------ query


def test_list_components_identifies_every_component():
    project = fa_project()
    infos = edit.list_components(project, "full_adder")
    circuit = project.circuit("full_adder")
    assert len(infos) == len(circuit.components)
    for info, comp in zip(infos, circuit.components):
        assert info.name == comp.name
        assert info.label == comp.label
        assert info.loc == comp.loc
        assert info.location == comp.loc
        assert info.facing == (comp.attrs.get("facing", "") or "east")


def test_find_components_by_name_label_and_both():
    project = fa_project()
    ands = edit.find_components(project, "full_adder", name="AND Gate")
    assert len(ands) == 2
    assert all(i.name == "AND Gate" for i in ands)
    pins = edit.find_components(project, "full_adder", label="A")
    assert len(pins) == 1 and pins[0].name == "Pin"
    both = edit.find_components(project, "full_adder", name="Pin", label="A")
    assert len(both) == 1
    assert edit.find_components(project, "full_adder", name="Pin", label="Z") == []
    assert edit.find_components(project, "full_adder", name="NAND Gate", label="A") == []


def test_find_component_requires_exactly_one_match():
    project = fa_project()
    assert edit.find_component(project, "full_adder", name="OR Gate").name == "OR Gate"
    with pytest.raises(edit.EditError, match="found 2"):
        edit.find_component(project, "full_adder", name="AND Gate")
    with pytest.raises(edit.EditError, match="found 0"):
        edit.find_component(project, "full_adder", name="NAND Gate")


def test_add_component_in_free_space():
    project = fa_project()
    info = edit.add_component(project, "full_adder", "NOT Gate", lib="gates")
    circuit = project.circuit("full_adder")
    assert info.name == "NOT Gate"
    assert len(circuit.components) == 11
    assert not errors(project, "full_adder")


def test_add_component_connected_to_a_net():
    project = fa_project()
    circuit = project.circuit("full_adder")
    before = partition_by_object(project, "full_adder")
    info = edit.add_component(project, "full_adder", "NOT Gate", lib="gates",
                              connections={"in": (0, "p")})
    nl = netlist.extract(project, circuit)
    # the new gate's input joined the net of pin A
    assert nl.port_net[(info.index, "in")] == nl.port_net[(0, "p")]
    after = partition_by_object(project, "full_adder")
    a_pin = circuit.components[0]
    for group in before:
        if id(a_pin) not in {c for c, _ in group}:
            assert group in after
    assert not errors(project, "full_adder")


def test_add_component_refuses_an_occupied_location():
    project = fa_project()
    with pytest.raises(edit.EditError, match="already sits at"):
        edit.add_component(project, "full_adder", "NOT Gate", lib="gates", loc=(280, 210))


def test_grid_invariants_after_edits():
    """Logisim only loads drawings that are on the grid and have no two components
    at the same location."""
    project = fa_project()
    circuit = project.circuit("full_adder")
    target = edit.find_components(project, circuit, name="AND Gate")[0]
    edit.move_component(project, circuit, target, dx=30, dy=-60)
    nl = netlist.extract(project, circuit)
    s_net = next(n for n in nl.nets if any(p.comp == 5 for p in n.ports))
    edit.insert_gate(project, circuit, s_net, "NOT Gate")
    locs = [c.loc for c in circuit.components]
    assert len(locs) == len(set(locs))
    for c in circuit.components:
        assert c.loc[0] % 10 == 0 and c.loc[1] % 10 == 0
    for w in circuit.wires:
        assert w.a[0] % 10 == w.a[1] % 10 == w.b[0] % 10 == w.b[1] % 10 == 0
        assert w.horizontal or w.vertical


# ------------------------------------------------------------------- move


@pytest.mark.parametrize("dx,dy", [(30, 0), (0, -60), (30, -60), (120, 0), (-60, -60)])
def test_move_keeps_connectivity(dx, dy):
    project = fa_project()
    circuit = project.circuit("full_adder")
    before = signature(project, "full_adder")
    target = edit.find_components(project, circuit, name="AND Gate")[0]
    edit.move_component(project, circuit, target, dx=dx, dy=dy)
    assert signature(project, "full_adder") == before
    assert not errors(project, "full_adder")


def test_move_of_every_gate_keeps_connectivity():
    """The gates of the full adder, each moved a little."""
    for k in range(len(gates(fa_project().circuit("full_adder")))):
        project = fa_project()
        circuit = project.circuit("full_adder")
        before = signature(project, "full_adder")
        index = gates(circuit)[k]
        edit.move_component(project, circuit, index, dx=30 * (k % 3), dy=-60 * (k % 2))
        assert signature(project, "full_adder") == before, k
        assert not errors(project, "full_adder"), k

def test_move_first_and_gate_down_by_200():
    """A move that used to fail with "could not route a wire". The gate would land on the
    XOR's body, so it is refused unless forced."""
    project = fa_project()
    circuit = project.circuit("full_adder")
    before = signature(project, "full_adder")
    target = edit.find_components(project, circuit, name="AND Gate")[0]
    with pytest.raises(edit.EditError, match="would overlap"):
        edit.move_component(project, circuit, target, dy=200)
    assert signature(project, "full_adder") == before
    edit.move_component(project, circuit, target, dy=200, force=True)
    assert signature(project, "full_adder") == before
    assert not errors(project, "full_adder")


def test_move_of_the_xor_that_drives_the_output_pin():
    """A move that used to fail with "could not route a wire from (530,310)"."""
    project = fa_project()
    circuit = project.circuit("full_adder")
    before = signature(project, "full_adder")
    xors = edit.find_components(project, circuit, name="XOR Gate")
    edit.move_component(project, circuit, xors[1], dx=30, dy=0)
    assert signature(project, "full_adder") == before
    assert not errors(project, "full_adder")


def test_move_onto_another_component_is_refused():
    project = fa_project()
    circuit = project.circuit("full_adder")
    target = edit.find_components(project, circuit, name="AND Gate")[0]
    with pytest.raises(edit.EditError, match="already sits at"):
        edit.move_component(project, circuit, target, loc=(490, 200))


# ----------------------------------------------------------------- rotate


def test_rotate_keeps_connectivity():
    project = fa_project()
    circuit = project.circuit("full_adder")
    before = signature(project, "full_adder")
    target = edit.find_components(project, circuit, name="AND Gate")[0]
    edit.rotate_component(project, circuit, target, facing="north")
    assert signature(project, "full_adder") == before
    assert not errors(project, "full_adder")
    assert edit.find_components(project, circuit, name="AND Gate")[0].facing == "north"


def test_rotate_by_turns():
    project = fa_project()
    circuit = project.circuit("full_adder")
    before = signature(project, "full_adder")
    target = edit.find_component(project, circuit, name="OR Gate")
    edit.rotate_component(project, circuit, target, turns=1)
    assert signature(project, "full_adder") == before
    assert not errors(project, "full_adder")


# ----------------------------------------------------------------- delete


def test_delete_component_removes_the_wires_that_only_served_it():
    project = fa_project()
    circuit = project.circuit("full_adder")
    before = partition_by_object(project, "full_adder")
    target = edit.find_components(project, circuit, name="AND Gate")[1]
    deleted = circuit.components[target.index]
    edit.delete_component(project, circuit, target)
    assert len(circuit.components) == 9
    # the nets that did not touch the deleted gate are unchanged
    after = partition_by_object(project, "full_adder")
    for group in before:
        if id(deleted) not in {c for c, _ in group}:
            assert group in after
    assert not errors(project, "full_adder")
    assert "dangling-wire" not in codes(project, "full_adder")


def test_delete_component_reports_what_was_deleted():
    project = fa_project()
    circuit = project.circuit("full_adder")
    info = edit.delete_component(project, circuit, edit.find_component(project, circuit, name="OR Gate"))
    assert info.name == "OR Gate"


# -------------------------------------------------------------- attributes


def test_set_label_is_accepted():
    project = fa_project()
    circuit = project.circuit("full_adder")
    target = edit.find_components(project, circuit, name="AND Gate")[0]
    edit.set_attribute(project, circuit, target, "label", "gate1")
    assert edit.find_components(project, circuit, label="gate1")[0].name == "AND Gate"
    assert not errors(project, "full_adder")


def test_removing_a_connected_input_is_refused():
    """inputs=1 on a gate whose second input is wired would disconnect it."""
    project = fa_project()
    circuit = project.circuit("full_adder")
    target = edit.find_component(project, circuit, name="OR Gate")
    with pytest.raises(edit.EditError, match="in1"):
        edit.set_attribute(project, circuit, target, "inputs", "1")
    # nothing was changed
    assert target.attrs.get("inputs") is None
    assert not errors(project, "full_adder")


def test_removing_a_connected_input_is_possible_with_force():
    project = fa_project()
    circuit = project.circuit("full_adder")
    target = edit.find_component(project, circuit, name="OR Gate")
    edit.set_attribute(project, circuit, target, "inputs", "1", force=True)
    nl = netlist.extract(project, circuit)
    or_gate = [i for i, c in enumerate(circuit.components) if c.name == "OR Gate"][0]
    assert "in1" not in {p.port.name for p in nl.nets[nl.port_net[(or_gate, "in0")]].ports}
    assert not errors(project, "full_adder")


def test_width_change_that_would_mix_widths_is_refused():
    """The A pin drives 1-bit gates: making it 2 bits would mix the widths."""
    project = fa_project()
    circuit = project.circuit("full_adder")
    target = edit.find_component(project, circuit, label="A")
    with pytest.raises(edit.EditError, match="width"):
        edit.set_attribute(project, circuit, target, "width", "2")
    assert target.attrs.get("width") is None


def test_set_attribute_rejects_unknown_component():
    project = fa_project()
    with pytest.raises(edit.EditError):
        edit.set_attribute(project, "full_adder", "no such thing", "label", "x")


# ---------------------------------------------------------------- renames


def test_rename_circuit_updates_main_and_instances():
    project = adder4_project()
    assert "full_adder" in project.circuit_names()
    edit.rename_circuit(project, "full_adder", "fa2")
    assert "fa2" in project.circuit_names()
    assert "full_adder" not in project.circuit_names()
    instances = [c for c in project.circuit("adder4").components if c.lib is None]
    assert instances and all(c.name == "fa2" for c in instances)
    assert not errors(project)


def test_renamed_circuit_keeps_its_instances_wired():
    project = adder4_project()
    instances = {c.name: i for i, c in enumerate(project.circuit("adder4").components)
                 if c.lib is None}
    before = partition(project, "adder4", exclude=set(instances.values()))
    edit.rename_circuit(project, "full_adder", "fa2")
    after = partition(project, "adder4", exclude=set(instances.values()))
    assert before == after
    assert not errors(project)


def test_rename_circuit_rejects_a_bad_name():
    project = adder4_project()
    with pytest.raises(edit.EditError, match="bad circuit name"):
        edit.rename_circuit(project, "full_adder", "1bad")
    with pytest.raises(edit.EditError, match="already exists"):
        edit.rename_circuit(project, "full_adder", "adder4")


def test_rename_the_main_circuit_updates_the_project():
    project = fa_project()
    assert project.main == "full_adder"
    edit.rename_circuit(project, "full_adder", "sum")
    assert project.main == "sum"
    assert project.circuit("sum").attrs["circuit"] == "sum"


def test_rename_pin_changes_the_label_and_keeps_the_instances_wired():
    project = adder4_project()
    instances = {i for i, c in enumerate(project.circuit("adder4").components) if c.lib is None}
    before = partition(project, "adder4", exclude=instances)
    edit.rename_pin(project, "full_adder", "A", "AA")
    assert [c.label for c in project.circuit("full_adder").pins()].count("AA") == 1
    after = partition(project, "adder4", exclude=instances)
    assert before == after
    assert not errors(project)


def test_rename_pin_refuses_duplicates():
    project = adder4_project()
    with pytest.raises(edit.EditError, match="already labelled"):
        edit.rename_pin(project, "full_adder", "A", "B")
    with pytest.raises(edit.EditError, match="expected one pin"):
        edit.rename_pin(project, "full_adder", "nope", "X")


# ------------------------------------------------------------ insert gate


def test_insert_not_into_the_output_net():
    project = fa_project()
    circuit = project.circuit("full_adder")
    nl = netlist.extract(project, circuit)
    s_net = next(n for n in nl.nets if any(p.comp == 5 for p in n.ports))
    before = partition_by_object(project, "full_adder")
    info = edit.insert_gate(project, circuit, s_net, "NOT Gate")
    assert info.name == "NOT Gate"
    # the new gate sits on the net's path, in the driver's row
    driver = next(c for c in circuit.components if c.name == "XOR Gate" and c.loc[0] == 500)
    assert info.loc[1] == driver.loc[1]
    assert driver.loc[0] < info.loc[0] < 790
    # the driver now feeds the new gate, the new gate feeds the output pin
    nl = netlist.extract(project, circuit)
    xor_index = circuit.components.index(driver)
    not_index = info.index
    assert nl.port_net[(xor_index, "out")] == nl.port_net[(not_index, "in")]
    assert nl.port_net[(not_index, "out")] == nl.port_net[(5, "p")]
    # the other nets are untouched
    after = partition_by_object(project, "full_adder")
    out_pin = circuit.components[5]
    for group in before:
        if id(out_pin) not in {c for c, _ in group}:
            assert group in after
    assert not errors(project, "full_adder")


def test_insert_gate_with_an_extra_input():
    project = fa_project()
    circuit = project.circuit("full_adder")
    nl = netlist.extract(project, circuit)
    s_net = next(n for n in nl.nets if any(p.comp == 5 for p in n.ports))
    driver = next(i for i, c in enumerate(circuit.components)
                  if c.name == "XOR Gate" and c.loc[0] == 500)
    info = edit.insert_gate(project, circuit, s_net, "AND Gate", attrs={"inputs": "2"},
                             extra_inputs={"in1": (0, "p")})
    nl = netlist.extract(project, circuit)
    # in0 carries the old net, in1 the extra one, out drives the output pin
    assert nl.port_net[(info.index, "in0")] == nl.port_net[(driver, "out")]
    assert nl.port_net[(info.index, "in1")] == nl.port_net[(0, "p")]
    assert nl.port_net[(info.index, "out")] == nl.port_net[(5, "p")]
    assert not errors(project, "full_adder")


def test_insert_gate_needs_a_net_with_two_ports():
    project = BProject(BCircuit("lonely")).to_model()
    circuit = project.circuit("lonely")
    circuit.add(Component("Pin", (100, 100), {"label": "A"}, project.lib_id("wiring")))
    nl = netlist.extract(project, circuit)
    lone = next(n for n in nl.nets if len(n.ports) == 1)
    with pytest.raises(edit.EditError, match="fewer than two ports"):
        edit.insert_gate(project, circuit, lone, "NOT Gate")


def test_insert_gate_in_front_of_one_sink_only():
    """Only the chosen sink is fed by the new gate; the others stay on the net."""
    project = fa_project()
    circuit = project.circuit("full_adder")
    nl = netlist.extract(project, circuit)
    a_net = next(n for n in nl.nets if len(n.ports) == 3)
    info = edit.insert_gate(project, circuit, a_net, "NOT Gate", sinks=[(3, "in0")])
    nl = netlist.extract(project, circuit)
    # the driver and the other sink keep the old net ...
    assert nl.port_net[(0, "p")] == nl.port_net[(6, "in0")]
    # ... the chosen sink is fed by the new gate
    assert nl.port_net[(3, "in0")] == nl.port_net[(info.index, "out")]
    assert nl.port_net[(info.index, "in")] == nl.port_net[(0, "p")]
    assert not errors(project, "full_adder")


# ------------------------------------------------------------------ wires


def test_delete_wire_by_endpoints():
    project = fa_project()
    circuit = project.circuit("full_adder")
    removed = edit.delete_wires(project, circuit, endpoints=((620, 90), (490, 90)))
    assert len(removed) == 1
    assert (620, 90) not in [w.a for w in circuit.wires]
    assert "dangling-wire" not in codes(project, "full_adder")
    assert not errors(project, "full_adder")


def test_delete_wires_touching_a_point():
    project = fa_project()
    circuit = project.circuit("full_adder")
    removed = edit.delete_wires(project, circuit, point=(500, 310))
    assert removed
    assert all(not w.contains((500, 310)) for w in circuit.wires)
    assert "dangling-wire" not in codes(project, "full_adder")


def test_cleanup_wires_removes_dangling_pieces():
    project = fa_project()
    circuit = project.circuit("full_adder")
    # cut the wire in the middle of the B net: the loose ends must disappear
    edit.delete_wires(project, circuit, endpoints=((200, 110), (200, 230)))
    assert "dangling-wire" not in codes(project, "full_adder")


def test_delete_wires_needs_a_target():
    project = fa_project()
    with pytest.raises(edit.EditError, match="give wires="):
        edit.delete_wires(project, "full_adder")


def test_delete_wires_by_object():
    project = fa_project()
    circuit = project.circuit("full_adder")
    wire = next(w for w in circuit.wires if (w.a, w.b) == ((620, 90), (490, 90)))
    removed = edit.delete_wires(project, circuit, wires=[wire])
    assert removed == [wire]
    assert wire not in circuit.wires
    assert "dangling-wire" not in codes(project, "full_adder")


def test_cleanup_wires_merges_collinear_pieces():
    project = fa_project()
    circuit = project.circuit("full_adder")
    edit.delete_wires(project, circuit, endpoints=((200, 110), (200, 230)))
    # no wire end is left in mid-air
    for w in circuit.wires:
        for end in (w.a, w.b):
            others = [x for x in circuit.wires if x is not w and x.contains(end)]
            assert end in {p.pos for c in circuit.components
                           for p in netlist.geometry.ports(project, c)} or others


# ------------------------------------------------------- circuits / merge


def test_add_circuit_and_instance():
    project = adder4_project()
    edit.add_circuit(project, "extra")
    assert "extra" in project.circuit_names()
    info = edit.add_instance(project, "adder4", "full_adder")
    assert info.name == "full_adder"
    assert any(c.name == "full_adder" for c in project.circuit("adder4").components)
    assert not errors(project)


def test_add_circuit_refuses_a_bad_name():
    project = adder4_project()
    with pytest.raises(edit.EditError, match="bad circuit name"):
        edit.add_circuit(project, "1bad")


def test_add_circuit_can_become_the_main_circuit():
    project = adder4_project()
    edit.add_circuit(project, "extra", main=True)
    assert project.main == "extra"


def test_merge_two_projects():
    target = adder4_project()
    source = fa_project()
    mapping = edit.merge_projects(target, source)
    assert mapping == {"full_adder": "full_adder_copy"}
    assert "full_adder_copy" in target.circuit_names()
    assert not errors(target)
    # the copy is a circuit of its own, the original still works
    assert target.circuit("full_adder_copy").components


def test_merge_renames_clashing_instances():
    target = adder4_project()
    source = fa_project()
    edit.merge_projects(target, source)
    copy = target.circuit("full_adder_copy")
    assert copy.components


def test_merge_with_an_explicit_rename():
    target = adder4_project()
    source = fa_project()
    mapping = edit.merge_projects(target, source, rename={"full_adder": "sum"})
    assert mapping == {"full_adder": "sum"}
    assert "sum" in target.circuit_names()
    assert not errors(target)


def test_merge_with_a_prefix():
    target = adder4_project()
    source = fa_project()
    mapping = edit.merge_projects(target, source, prefix="other_")
    assert mapping == {"full_adder": "other_full_adder"}
    assert "other_full_adder" in target.circuit_names()
    assert not errors(target)


def test_merge_refuses_to_merge_a_project_into_itself():
    project = fa_project()
    with pytest.raises(edit.EditError, match="into itself"):
        edit.merge_projects(project, project)


# ------------------------------------------------------------------ tidy


def test_tidy_keeps_the_signature():
    project = fa_project()
    before = signature(project, "full_adder")
    edit.tidy(project, "full_adder")
    assert signature(project, "full_adder") == before
    assert not errors(project, "full_adder")


def test_tidy_of_a_messy_circuit():
    project = fa_project()
    circuit = project.circuit("full_adder")
    # make a mess first
    for k, index in enumerate(gates(circuit)):
        edit.move_component(project, circuit, index, dx=30 * (k % 3), dy=-60 * (k % 2))
    before = signature(project, "full_adder")
    edit.tidy(project, "full_adder")
    assert signature(project, "full_adder") == before
    assert not errors(project, "full_adder")
    assert codes(project, "full_adder") <= {"wire-spacing", "crossings"}


def test_tidy_uses_the_roomy_spacing_by_default():
    project = fa_project()
    edit.tidy(project, "full_adder")
    roomy = project.circuit("full_adder")
    width = max(c.loc[0] for c in roomy.components) - min(c.loc[0] for c in roomy.components)
    project = fa_project()
    edit.tidy(project, "full_adder", spacing="compact")
    compact = project.circuit("full_adder")
    width2 = max(c.loc[0] for c in compact.components) - min(c.loc[0] for c in compact.components)
    assert width > width2


def test_tidy_of_a_subcircuit_keeps_the_project_consistent():
    project = adder4_project()
    edit.tidy(project, "full_adder")
    assert not errors(project)
    edit.tidy(project, "adder4")
    assert not errors(project)


# ------------------------------------------------------------------ save


def test_save_writes_the_project(tmp_path):
    project = fa_project()
    path = tmp_path / "out.circ"
    edit.save(project, path)
    assert path.exists()
    again = io.load(path)
    assert again.circuit_names() == project.circuit_names()


def test_save_refuses_to_overwrite(tmp_path):
    project = fa_project()
    path = tmp_path / "out.circ"
    edit.save(project, path)
    with pytest.raises(FileExistsError):
        edit.save(project, path)
    assert path.exists()


def test_save_in_place_keeps_a_backup(tmp_path):
    project = fa_project()
    path = tmp_path / "out.circ"
    edit.save(project, path)
    first = path.read_text()
    edit.move_component(project, "full_adder", 0, dx=20)
    edit.save(project, path, in_place=True)
    assert path.read_text() != first
    bak = path.with_suffix(path.suffix + ".bak")
    assert bak.exists()
    assert bak.read_text() == first


# ------------------------------------------------------------------- JAR


@pytest.mark.jar
def test_jar_verify_after_inserting_a_not(tmp_path):
    project = fa_project()
    circuit = project.circuit("full_adder")
    nl = netlist.extract(project, circuit)
    s_net = next(n for n in nl.nets if any(p.comp == 5 for p in n.ports))
    edit.insert_gate(project, circuit, s_net, "NOT Gate")
    path = tmp_path / "fa.circ"
    edit.save(project, path)
    r = sim.verify(path, lambda A, B, Cin: {"S": 1 - (A ^ B ^ Cin),
                                             "Cout": (A + B + Cin) >> 1})
    assert r.ok and r.passed == 8, r.report()


@pytest.mark.jar
def test_jar_verify_after_moving_and_rotating(tmp_path):
    project = fa_project()
    circuit = project.circuit("full_adder")
    target = edit.find_components(project, circuit, name="AND Gate")[0]
    edit.move_component(project, circuit, target, dx=30, dy=-60)
    other = edit.find_component(project, circuit, name="OR Gate")
    edit.rotate_component(project, circuit, other, facing="north")
    path = tmp_path / "fa.circ"
    edit.save(project, path)
    r = sim.verify(path, lambda A, B, Cin: {"S": A ^ B ^ Cin, "Cout": (A + B + Cin) >> 1})
    assert r.ok and r.passed == 8, r.report()


@pytest.mark.jar
def test_jar_verify_after_deleting_a_component(tmp_path):
    project = fa_project()
    circuit = project.circuit("full_adder")
    # the Cout pin goes away: the adder still computes S
    edit.delete_component(project, circuit, edit.find_component(project, circuit, label="Cout"))
    path = tmp_path / "fa.circ"
    edit.save(project, path)
    r = sim.verify(path, lambda A, B, Cin: {"S": A ^ B ^ Cin})
    assert r.ok and r.passed == 8, r.report()


@pytest.mark.jar
def test_jar_verify_after_renaming_a_used_circuit(tmp_path):
    project = adder4_project()
    edit.rename_circuit(project, "full_adder", "fa2")
    path = tmp_path / "adder4.circ"
    edit.save(project, path)
    r = sim.verify(path, lambda A, B, Cin: {"S": (A + B + Cin) % 16,
                                             "Cout": (A + B + Cin) // 16})
    assert r.ok and r.passed == 512, r.report()


@pytest.mark.jar
def test_jar_verify_after_merging_two_files(tmp_path):
    project = adder4_project()
    edit.merge_projects(project, fa_project())
    path = tmp_path / "merged.circ"
    edit.save(project, path)
    r = sim.verify(path, lambda A, B, Cin: {"S": (A + B + Cin) % 16,
                                             "Cout": (A + B + Cin) // 16})
    assert r.ok and r.passed == 512, r.report()


@pytest.mark.jar
def test_jar_verify_after_tidy(tmp_path):
    project = fa_project()
    edit.tidy(project, "full_adder")
    path = tmp_path / "fa.circ"
    edit.save(project, path)
    r = sim.verify(path, lambda A, B, Cin: {"S": A ^ B ^ Cin, "Cout": (A + B + Cin) >> 1})
    assert r.ok and r.passed == 8, r.report()


@pytest.mark.jar
def test_jar_verify_after_renaming_a_pin(tmp_path):
    project = adder4_project()
    edit.rename_pin(project, "full_adder", "A", "AA")
    path = tmp_path / "adder4.circ"
    edit.save(project, path)
    r = sim.verify(path, lambda A, B, Cin: {"S": (A + B + Cin) % 16,
                                             "Cout": (A + B + Cin) // 16})
    assert r.ok and r.passed == 512, r.report()


def test_type_occurrence_targets_and_named_net():
    """"AND Gate#1" is the second AND gate; insert_gate takes (label, port) for its net."""
    project = fa_project()
    circuit = project.circuit("full_adder")
    ands = edit.find_components(project, circuit, name="AND Gate")
    assert len(ands) == 2
    info = edit.rotate_component(project, circuit, "AND Gate#1", facing="south")
    assert info.index == ands[1].index
    with pytest.raises(edit.EditError, match="out of range"):
        edit.rotate_component(project, circuit, "AND Gate#5")
    before = len(circuit.components)
    edit.insert_gate(project, circuit, ("S", "p"))
    assert len(circuit.components) == before + 1
