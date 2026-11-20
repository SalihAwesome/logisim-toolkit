"""Exercise the command entry point with real projects and Logisim runs."""

import csv
import io as streams
import json
import subprocess
import sys

import pytest

from logisim import cli, generate, io, jar, netlist, sim
from logisim.builder import Circuit, Project
from logisim.model import Component, Project as ModelProject

from test_builder import full_adder


_COMMANDS = "info lint table test gen edit tidy diff export".split()
_GENERATORS = "expr truth fsm block".split()
_EDITS = ("list find add delete move rotate set rename-circuit rename-pin insert delete-wire cleanup "
          "add-circuit add-instance merge").split()


@pytest.mark.parametrize("command", [[]] + [[name] for name in _COMMANDS]
                         + [["gen", name] for name in _GENERATORS]
                         + [["edit", "unused.circ", name] for name in _EDITS])
def test_help(command, capsys):
    assert cli.main([*command, "--help"]) == 0
    captured = capsys.readouterr()
    assert "Example: logisim-tool" in captured.out
    assert not captured.err
    if command == ["gen", "block"]:
        for item in ("adder", "addsub", "mux", "decoder", "comparator", "regfile", "alu", "counter",
                     "--width", "--data-inputs", "--inputs", "--registers", "--read-ports", "--clear"):
            assert item in captured.out


@pytest.fixture
def adder(tmp_path):
    return Project(full_adder("compact")).save(tmp_path / "adder.circ")


@pytest.fixture
def inverter(tmp_path):
    circuit = Circuit("logic")
    circuit.output("Y", circuit.not_(circuit.input("A")))
    return Project(circuit).save(tmp_path / "inverter.circ")


def test_reports_and_exports(adder, tmp_path, capsys):
    assert cli.main(["info", str(adder)]) == 0
    assert "Main circuit: full_adder" in capsys.readouterr().out
    assert cli.main(["info", str(adder), "--circuit", "full_adder", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["main"] == "full_adder"
    assert info["circuits"]["full_adder"]["gate_counts"]["total"] == 5

    assert cli.main(["lint", str(adder), "--errors-only"]) == 0
    assert "no issues" in capsys.readouterr().out
    assert cli.main(["export", str(adder), "--gates"]) == 0
    assert json.loads(capsys.readouterr().out)["full_adder"]["total"] == 5
    assert cli.main(["export", str(adder), "--gates", "--circuit", "full_adder"]) == 0
    assert json.loads(capsys.readouterr().out)["total"] == 5
    assert cli.main(["export", str(adder), "--summary"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["drivers"]
    assert cli.main(["export", str(adder), "--markdown"]) == 0
    assert "| Cin | 1 | input |" in capsys.readouterr().out
    dest = tmp_path / "adder.v"
    assert cli.main(["export", str(adder), "--verilog", "-o", str(dest)]) == 0
    assert dest.read_text(encoding="utf-8").startswith("module full_adder(")
    assert not capsys.readouterr().out


def test_lint_errors_have_check_exit_code(tmp_path, capsys):
    project = ModelProject.new()
    circuit = project.add_circuit("broken")
    circuit.add(Component("Pin", (100, 100), {"output": "true", "label": "Y"}, project.lib_id("wiring")))
    circuit.add(Component("Pin", (200, 100), {"label": "Y"}, project.lib_id("wiring")))
    path = io.save(project, tmp_path / "broken.circ")
    assert cli.main(["lint", str(path), "--json", "--errors-only", "--circuit", "broken"]) == 1
    issues = json.loads(capsys.readouterr().out)
    assert issues and all(i["severity"] == "error" and i["circuit"] == "broken" for i in issues)
    assert set(issues[0]) == {"code", "severity", "circuit", "message", "at"}


def test_tidy_and_diff(adder, capsys):
    assert cli.main(["tidy", str(adder), "--all"]) == 0
    tidied = io.default_output_path(adder)
    assert tidied.exists()
    assert "full_adder: netlist signature unchanged" in capsys.readouterr().out
    assert cli.main(["diff", str(adder), str(tidied)]) == 0
    assert "same netlist, layout differs" in capsys.readouterr().out
    assert cli.main(["diff", str(adder), str(tidied), "--json"]) == 0
    changes = json.loads(capsys.readouterr().out)["circuits"]["full_adder"]
    assert changes["netlist_equal"] and changes["layout_changed"]


def test_edit_query_and_component_mutations(inverter, capsys):
    def run(source, *args):
        assert cli.main(["edit", str(source), *args]) == 0, capsys.readouterr().err
        return io.default_output_path(source)

    run(inverter, "list")
    assert "NOT Gate" in capsys.readouterr().out
    run(inverter, "find", "--label", "A")
    assert "Pin 'A'" in capsys.readouterr().out
    run(inverter, "find", "NOT Gate")
    [match] = capsys.readouterr().out.splitlines()
    gate_index = match.split()[0]
    run(inverter, "find", gate_index)
    assert capsys.readouterr().out.strip() == match
    assert not io.default_output_path(inverter).exists()

    before = io.load(inverter)
    signature = netlist.extract(before, before.main).signature()
    moved = run(inverter, "move", gate_index, "--dy", "150")
    rotated = run(moved, "rotate", "NOT Gate", "--facing", "north")
    model = io.load(rotated)
    assert netlist.extract(model, model.main).signature() == signature
    assert model.main_circuit.find("NOT Gate")[0].facing == "north"
    updated = run(rotated, "set", "NOT Gate", "label=inverter", "labelloc=north")
    updated = run(updated, "rename-pin", "Y", "Z")
    updated = run(updated, "add", "Constant", "--attr", "label=one", "--lib", "wiring")
    updated = run(updated, "delete", "one")
    updated = run(updated, "rename-circuit", "logic", "inverted")
    model = io.load(updated)
    assert model.main == "inverted"
    assert model.main_circuit.find(label="Z") and model.main_circuit.find(label="inverter")
    assert not model.main_circuit.find("Constant")
    assert not capsys.readouterr().err


def test_edit_insert_remove_wire_and_cleanup(inverter, capsys):
    out = io.default_output_path(inverter)
    assert cli.main(["edit", str(inverter), "insert", "Y:p"]) == 0, capsys.readouterr().err
    count = len(io.load(out).main_circuit.wires)
    assert len(io.load(out).main_circuit.find("NOT Gate")) == 2
    assert cli.main(["edit", str(out), "delete-wire", "--index", "0", "--in-place"]) == 0
    assert len(io.load(out).main_circuit.wires) < count
    assert cli.main(["edit", str(out), "cleanup", "--in-place", "--circuit", "logic"]) == 0
    assert "after cleanup" in capsys.readouterr().out


def test_edit_add_circuit_instance_and_merge(inverter, adder, tmp_path, capsys):
    out = tmp_path / "subcircuits.circ"
    file = str(inverter)
    assert cli.main(["edit", file, "-o", str(out), "add-circuit", "empty"]) == 0
    assert cli.main(["edit", str(out), "add-instance", "empty", "--label", "instance0", "--in-place"]) == 0
    assert cli.main(["edit", str(out), "merge", str(adder), "--in-place"]) == 0
    assert cli.main(["edit", str(out), "merge", str(adder), "--prefix", "copy_", "--in-place"]) == 0
    model = io.load(out)
    assert set(model.circuit_names()) == {"logic", "empty", "full_adder", "copy_full_adder"}
    assert model.main_circuit.find("empty", "instance0")
    assert not capsys.readouterr().err
    before = out.read_bytes()
    assert cli.main(["edit", str(out), "merge", str(adder), "--prefix", "bad/", "--in-place"]) == 2
    assert "bad circuit name" in capsys.readouterr().err
    assert out.read_bytes() == before


def test_overwrite_policy_and_backup(inverter, tmp_path, capsys):
    file = str(inverter)
    original = inverter.read_bytes()
    assert cli.main(["edit", file, "set", "A", "labelloc=north"]) == 0
    output = io.default_output_path(inverter)
    first_output = output.read_bytes()
    assert cli.main(["edit", file, "delete", "A"]) == 2
    assert "--force" in capsys.readouterr().err
    assert output.read_bytes() == first_output
    assert cli.main(["edit", file, "delete", "A", "--force"]) == 0
    assert not io.load(output).main_circuit.find(label="A")
    assert cli.main(["edit", file, "delete", "A", "-o", file, "--force"]) == 2
    assert "--in-place" in capsys.readouterr().err
    assert cli.main(["edit", file, "--in-place", "cleanup", "-o", str(tmp_path / "x.circ")]) == 2
    assert "either" in capsys.readouterr().err
    assert inverter.read_bytes() == original
    assert cli.main(["edit", file, "--in-place", "rename-pin", "Y", "Z"]) == 0
    assert inverter.with_suffix(".circ.bak").read_bytes() == original
    assert io.load(inverter).main_circuit.find(label="Z")


@pytest.mark.parametrize("command, message", [
    ([], "required"),
    (["info", "missing.circ"], "missing.circ"),
    (["info", "{file}", "--circuit", "missing"], "no circuit named"),
    (["lint", "{file}", "--circuit", "missing"], "no circuit named"),
    (["table", "{file}", "--circuit", "missing"], "no circuit named"),
    (["test", "{file}", "--expr", "missing = A"], "no output pin"),
    (["test", "{file}", "--vectors", "missing.txt"], "cannot read vector file"),
    (["edit", "{file}", "delete", "absent"], "no component"),
    (["gen", "expr", "A * )", "-o", "{out}"], "column"),
    (["gen", "block", "adder", "--width", "0", "-o", "{out}"], "at least 1"),
    (["gen", "block", "adder", "--inputs", "2", "-o", "{out}"], "not a parameter"),
    (["export", "{file}", "--gates", "--summary"], "not allowed"),
])
def test_input_errors(command, message, inverter, tmp_path, capsys):
    out = tmp_path / "unused.circ"
    assert cli.main([part.format(file=inverter, out=out) for part in command]) == 2
    result = capsys.readouterr()
    assert not result.out and message in result.err
    assert len(result.err.splitlines()) == 1
    assert not out.exists()


def test_missing_jar_diagnostic(inverter, monkeypatch, capsys):
    def missing_jar():
        raise jar.JarNotFound("missing Logisim JAR")

    monkeypatch.setattr(jar, "find_jar", missing_jar)
    assert cli.main(["table", str(inverter)]) == 2
    assert "Set LOGISIM_JAR to the path" in capsys.readouterr().err


def test_module_entry_point():
    result = subprocess.run([sys.executable, "-m", "logisim", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "Example: logisim-tool" in result.stdout


@pytest.mark.parametrize("kind, parameters", [
    ("adder", ["--width", "2"]), ("addsub", ["--width", "2"]),
    ("mux", ["--data-inputs", "2", "--width", "2"]), ("decoder", ["--inputs", "2"]),
    ("comparator", ["--width", "2"]), ("regfile", ["--registers", "2", "--width", "2", "--read-ports", "2"]),
    ("alu", ["--width", "2"]), ("counter", ["--width", "2", "--clear", "async"]),
])
def test_generate_block_kinds(kind, parameters, tmp_path, capsys):
    output = tmp_path / f"{kind}.circ"
    assert cli.main(["gen", "block", kind, *parameters, "-o", str(output)]) == 0, capsys.readouterr().err
    assert io.load(output).main_circuit.outputs()


@pytest.mark.jar
def test_generate_expr_verify_and_render(tmp_path, capsys):
    output, vectors, png = (tmp_path / name for name in ("expr.circ", "gen.txt", "preview.png"))
    expressions = ["S = A ^ B ^ Cin", "Cout = A*B + (A+B)*Cin"]
    assert cli.main(["gen", "expr", *expressions, "--name", "full_adder", "--spacing", "normal",
                     "--tunnels", "--vectors", str(vectors), "--png", str(png), "-o", str(output)]) == 0
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert vectors.read_text(encoding="utf-8").splitlines()[1] == "A B Cin S Cout"
    assert io.load(output).main == "full_adder"
    assert cli.main(["test", str(output), "--circuit", "full_adder",
                     "--expr", expressions[0], "--expr", expressions[1]]) == 0
    assert "8 passed, 0 failed" in capsys.readouterr().out
    assert cli.main(["test", str(output), "--vectors", str(vectors)]) == 0
    assert "8 passed, 0 failed" in capsys.readouterr().out


@pytest.mark.jar
def test_generate_split_expression(tmp_path, capsys):
    output, vectors = (tmp_path / name for name in ("expr.circ", "gen.txt"))
    assert cli.main(["gen", "expr", "AB", "--split-vars", "--vectors", str(vectors), "-o", str(output)]) == 0
    assert cli.main(["test", str(output), "--vectors", str(vectors)]) == 0
    assert "4 passed, 0 failed" in capsys.readouterr().out


@pytest.mark.jar
def test_generate_truth(tmp_path, capsys):
    source, output = tmp_path / "truth.csv", tmp_path / "truth.circ"
    source.write_text("Either,A,Both,B\n0,0,0,0\n1,0,0,1\n1,1,0,0\n1,1,1,1\n", encoding="utf-8")
    assert cli.main(["gen", "truth", str(source), "--outputs", "Both", "Either",
                     "-o", str(output), "--form", "pos"]) == 0
    assert cli.main(["test", str(output), "--expr", "Both = A * B", "--expr", "Either = A + B"]) == 0
    assert "4 passed, 0 failed" in capsys.readouterr().out


@pytest.mark.jar
def test_generate_fsm(tmp_path, capsys):
    source, output = tmp_path / "fsm.txt", tmp_path / "machine.circ"
    source.write_text("""states: OFF, ON
inputs: enable
outputs: Y
initial: OFF
reset: true
transitions:
OFF,1,ON
ON,1,OFF
state_outputs:
OFF,0
ON,1
""", encoding="utf-8")
    assert cli.main(["gen", "fsm", str(source), "--encoding", "onehot", "-o", str(output)]) == 0, capsys.readouterr().err
    steps = [{"rst": 1}, {"enable": 1}, {"enable": 0}, {"enable": 1}, {"enable": 1}, {"rst": 1}]
    expected = [{"Y": y} for y in (0, 1, 1, 0, 1, 0)]
    result = sim.verify_sequence(output, steps, expected)
    assert result.ok and result.passed == len(steps), result.report()


@pytest.mark.jar
def test_generate_and_verify_decoder_and_counter(tmp_path, capsys):
    decoder, counter = tmp_path / "decode.circ", tmp_path / "count.circ"
    assert cli.main(["gen", "block", "decoder", "--inputs", "2", "-o", str(decoder)]) == 0
    assert cli.main(["test", str(decoder), "--expr", "out0 = ~in0 * ~in1", "--expr", "out3 = in0 * in1"]) == 0
    assert "4 passed, 0 failed" in capsys.readouterr().out
    assert cli.main(["gen", "block", "counter", "--width", "2", "-o", str(counter)]) == 0
    steps = [{"en": 1}] * 5 + [{"clr": 1}]
    result = sim.verify_sequence(counter, steps, [{"Q": q} for q in (1, 2, 3, 0, 1, 0)])
    assert result.ok and result.passed == len(steps), result.report()


@pytest.mark.jar
@pytest.mark.parametrize("output_format", ["text", "json", "csv"])
def test_full_adder_table(adder, output_format, capsys):
    args = [] if output_format == "text" else [f"--{output_format}"]
    assert cli.main(["table", str(adder), "--circuit", "full_adder", *args]) == 0
    output = capsys.readouterr().out
    if output_format == "json":
        table = json.loads(output)
        assert set(table["inputs"]) == {"A", "B", "Cin"} and set(table["outputs"]) == {"S", "Cout"}
        rows = table["rows"]
    elif output_format == "csv":
        rows = list(csv.DictReader(streams.StringIO(output)))
    else:
        header, *lines = output.splitlines()
        rows = [dict(zip(header.split(), row.split())) for row in lines]
    assert len(rows) == 8
    for row in rows:
        a, b, cin = (int(row[pin]) for pin in ("A", "B", "Cin"))
        assert int(row["S"]) == (a ^ b ^ cin) and int(row["Cout"]) == (a + b + cin) // 2


@pytest.mark.jar
def test_vector_file_and_failed_check_exit_status(adder, tmp_path, capsys):
    good = generate.expression_vectors(tmp_path / "matches.txt", "A ^ B ^ Cin", output="S")
    assert cli.main(["test", str(adder), "--vectors", str(good)]) == 0
    assert "8 passed, 0 failed" in capsys.readouterr().out
    assert cli.main(["test", str(adder), "--expr", "S = A + B + Cin"]) == 1
    output = capsys.readouterr().out
    assert "5 passed, 3 failed" in output and "expected" in output
