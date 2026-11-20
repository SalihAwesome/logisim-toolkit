"""Inspection, structured drawing diffs, and structural Verilog exports."""

import pytest

from logisim import edit, generate, io, netlist, report
from logisim.builder import Circuit, Project

from test_builder import counter2, full_adder


def test_describe_generated_project():
    model = generate.ripple_adder(2).to_model()
    main = report.describe(model)
    assert main["name"] == model.main
    assert main["inputs"] == [
        {"label": "A", "width": 2, "direction": "input"},
        {"label": "B", "width": 2, "direction": "input"},
        {"label": "Cin", "width": 1, "direction": "input"},
    ]
    assert {pin["label"]: pin["width"] for pin in main["outputs"]} == {"S": 2, "Cout": 1}
    assert main["subcircuits"] == ["full_adder"]
    assert main["component_counts"]["full_adder"] == 2
    assert main["wire_count"] == len(model.main_circuit.wires)
    assert main["net_count"] == len(netlist.extract(model, model.main_circuit).nets)
    gates = {"by_kind": {"AND Gate": 2, "OR Gate": 1, "XOR Gate": 2}, "total": 5}
    assert report.gate_counts(model, "full_adder") == gates
    assert report.project_summary(model)["circuits"]["full_adder"]["gate_counts"] == gates
    assert report.netlist_summary(model, model.main) == netlist.summary(model, model.main)
    text = report.interface_markdown(model)
    assert f"# {model.main}\n" in text
    assert "| A | 2 | input |" in text
    assert "| S | 2 | output |" in text
    assert "- full_adder" in text and "| full_adder | 2 |" in text


def test_diff_identical_and_tidied_files(tmp_path):
    original = Project(full_adder("compact")).save(tmp_path / "original.circ")
    old, new = io.load(original), io.load(original)
    unchanged = report.diff(old, new)
    assert unchanged["circuits_added"] == unchanged["circuits_removed"] == []
    assert unchanged["circuits"][old.main] == {
        "added": [], "removed": [], "changed": [], "netlist_equal": True,
        "layout_changed": False, "status": "identical",
    }

    edit.tidy(new, new.main)
    edited = io.save(new, tmp_path / "tidied.circ")
    result = report.diff(old, io.load(edited))["circuits"][old.main]
    assert result["status"] == "same netlist, layout differs"
    assert result["netlist_equal"] and result["layout_changed"]
    assert not result["added"] and not result["removed"]
    assert result["changed"] and all(item["layout_only"] for item in result["changed"])


def test_diff_edits_and_rename():
    original = Project(full_adder()).to_model()
    edited = original.copy()
    edit.set_attribute(edited, edited.main, "A", "labelfont", "SansSerif plain 14")
    edit.delete_component(edited, edited.main, "OR Gate")
    edit.add_component(edited, edited.main, "NOT Gate", lib="gates", attrs={"label": "spare"})
    result = report.diff(original, edited)["circuits"][original.main]
    assert [comp["type"] for comp in result["removed"]] == ["OR Gate"]
    assert [comp["label"] for comp in result["added"]] == ["spare"]
    assert not result["netlist_equal"]
    change = next(item for item in result["changed"] if item["before"]["label"] == "A")
    assert change["attribute_changes"]["labelfont"] == {"before": None, "after": "SansSerif plain 14"}
    assert not change["layout_only"]

    edit.rename_circuit(edited, edited.main, "renamed_adder")
    renamed = report.diff(original, edited)
    assert renamed == {"circuits_added": ["renamed_adder"], "circuits_removed": ["full_adder"], "circuits": {}}


@pytest.mark.parametrize("circuit, ports, gates", [
    (full_adder(), ("input A", "input Cin", "output S", "output Cout"), {"xor": 2, "and": 2, "or": 1}),
    (generate.decoder(2, use_tunnels=True), ("input in0", "input in1", "output out3"), {"not": 2, "and": 4}),
])
def test_verilog_combinational(circuit, ports, gates):
    model = Project(circuit).to_model()
    text = report.to_verilog(model, model.main)
    assert text.startswith(f"module {circuit.name}(\n")
    assert all(port in text for port in ports)
    for kind, count in gates.items():
        assert text.count(f"  {kind} gate_") == count
    assert text.endswith("endmodule\n")


def test_verilog_buses_constants_and_splitters():
    circuit = Circuit("bus_logic")
    a = circuit.input("A", 2)
    b = circuit.input("B", 2)
    x = circuit.nand(a, b, circuit.const(1, 2), negate=(1,))
    low, high = circuit.split(x)
    circuit.output("Y", circuit.merge([high, low]))
    model = Project(circuit).to_model()
    text = report.to_verilog(model, "bus_logic")
    assert "input [1:0] A" in text and "output [1:0] Y" in text
    assert " = 2'h1;" in text
    assert text.count("  nand gate_") == 2 and "~net_" in text
    assert any(line.endswith("[0];") for line in text.splitlines() if "assign" in line)
    assert any("] = " in line for line in text.splitlines() if "assign" in line)


def test_verilog_names_unsupported_component():
    model = Project(counter2()).to_model()
    with pytest.raises(report.VerilogError, match="unsupported component .*D Flip-Flop"):
        report.to_verilog(model, model.main)
