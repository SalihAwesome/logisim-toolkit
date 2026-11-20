"""Direct, stdio and real-Logisim coverage for the MCP tools."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")

from logisim import edit, io, netlist, sim
from logisim.builder import Project
from logisim_mcp.server import ToolError, WorkspaceTools
from test_builder import full_adder

TOOLS = {
    "list_circuits", "describe_circuit", "lint_circuit", "truth_table", "run_tests",
    "verify", "verify_fsm", "build_from_expression", "build_from_truth_table",
    "build_fsm", "build_block", "apply_edits", "tidy_circuit", "render_png",
}
CSV = "A,B,|,Y\n0,0,|,0\n0,1,|,1\n1,0,|,1\n1,1,|,0\n"
FSM = """\
states: IDLE, RUN
inputs: go
outputs: busy
reset: true
transitions:
IDLE,go,RUN
RUN,x,RUN
state_outputs:
IDLE,0
RUN,1
"""


@pytest.fixture
def tools(tmp_path):
    workspace = tmp_path / "workspace"
    Project(full_adder()).save(workspace / "fa.circ")
    return WorkspaceTools(workspace)


def test_workspace_paths(tools):
    workspace = tools.workspace
    assert tools.describe_circuit("fa.circ") == tools.describe_circuit(str(workspace / "fa.circ"))
    assert tools.describe_circuit("inner/../fa.circ")["name"] == "full_adder"
    Project(full_adder()).save(workspace.parent / "outside.circ")
    for path in ("../outside.circ", str(workspace.parent / "outside.circ")):
        with pytest.raises(ToolError, match="outside the workspace"):
            tools.describe_circuit(path)
        with pytest.raises(ToolError, match="outside the workspace"):
            tools.build_from_expression(path, "A", overwrite=True, verify=False)
    (workspace / "wrong.xml").write_text("<project/>", encoding="utf-8")
    with pytest.raises(ToolError, match="disallowed file extension"):
        tools.list_circuits("wrong.xml")
    with pytest.raises(ToolError, match="disallowed file extension"):
        tools.apply_edits("fa.circ", [], output="wrong.xml")
    with pytest.raises(ToolError, match="disallowed file extension"):
        tools.run_tests("fa.circ", "fa.circ")
    with pytest.raises(ToolError, match="disallowed file extension"):
        tools.render_png("fa.circ", output="wrong.svg")
    with pytest.raises(ToolError, match="outside the workspace"):
        tools.build_from_truth_table("out.circ", "../outside.csv", verify=False)
    with pytest.raises(ToolError, match="outside the workspace"):
        tools.build_fsm("out.circ", "../outside.txt", verify=False)
    with pytest.raises(ToolError, match="outside the workspace"):
        tools.apply_edits("fa.circ", [{"op": "merge_projects", "source": "../outside.circ"}])


def dir_link(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if os.name != "nt" or exc.winerror not in (5, 1314):
            raise
        # Windows junctions exercise the same resolved-path boundary without symlink privileges.
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


def test_workspace_link_escape(tools):
    workspace = tools.workspace
    outside = workspace.parent / "outside"
    target = outside / "outside.circ"
    Project(full_adder()).save(target)
    link = workspace / "escape"
    dir_link(link, outside)
    with pytest.raises(ToolError, match="outside the workspace"):
        tools.describe_circuit("escape/outside.circ")
    with pytest.raises(ToolError, match="outside the workspace"):
        tools.list_circuits("escape")
    with pytest.raises(ToolError, match="outside the workspace"):
        tools.build_from_expression("escape/new.circ", "A", verify=False)
    assert tools.list_circuits()["files"] == ["fa.circ"]
    assert list(outside.iterdir()) == [target]


def test_workspace_rejects_redirected_sidecars(tools):
    workspace = tools.workspace
    outside = workspace.parent / "outside.circ"
    outside.write_bytes(b"protected")
    for suffix in (".tmp", ".bak"):
        sidecar = workspace / f"fa.circ{suffix}"
        os.link(outside, sidecar)
        with pytest.raises(ToolError, match="regular, unlinked file"):
            tools.build_from_expression("fa.circ", "A", overwrite=True, verify=False)
        assert outside.read_bytes() == b"protected"
        sidecar.unlink()


def test_startup_requires_workspace():
    env = dict(os.environ)
    env.pop("LOGISIM_WORKSPACE", None)
    result = subprocess.run([sys.executable, "-m", "logisim_mcp.server"], env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "--workspace DIR or LOGISIM_WORKSPACE" in result.stderr
    assert "Traceback" not in result.stderr
    assert not result.stdout


def test_list_describe_and_lint(tools):
    tools.build_from_expression("sub/and.circ", {"Y": "A & B"}, verify=False)
    assert tools.list_circuits() == {"path": ".", "files": ["fa.circ", "sub/and.circ"]}
    assert tools.list_circuits("fa.circ") == {
        "path": "fa.circ", "circuits": ["full_adder"], "main": "full_adder",
    }
    description = tools.describe_circuit("fa.circ", "full_adder")
    for direction, pins in [("input", ["A", "B", "Cin"]), ("output", ["S", "Cout"])]:
        assert description[direction + "s"] == [
            {"label": pin, "width": 1, "direction": direction} for pin in pins]
    assert description["component_counts"]["AND Gate"] == 2
    assert description["net_count"] > 0
    assert description["subcircuits"] == []
    assert tools.lint_circuit("fa.circ", errors_only=True) == []
    project = io.load(tools.workspace / "fa.circ")
    project.main_circuit.inputs()[0].label = ""
    io.save(project, tools.workspace / "unlabeled.circ")
    issues = tools.lint_circuit("unlabeled.circ", "full_adder")
    assert any(issue["code"] == "unlabeled-pin" and issue["severity"] == "warning" for issue in issues)
    assert tools.lint_circuit("unlabeled.circ", errors_only=True) == []
    json.dumps(issues)


def test_apply_edits(tools):
    original = io.load(tools.workspace / "fa.circ")
    gate = edit.find_components(original, original.main_circuit, name="AND Gate")[0]
    result = tools.apply_edits("fa.circ", [
        {"op": "move", "target": "AND Gate#0", "dx": 30},
        {"op": "insert_gate", "net": ["S", "p"]},
        {"op": "rename_pin", "old_label": "S", "new_label": "nS"},
        {"op": "set_attributes", "target": gate.index, "attrs": {"label": "First"}},
    ])
    assert result["path"] == "fa_edited.circ"
    assert result["changes"][0]["result"]["loc"] == [gate.loc[0] + 30, gate.loc[1]]
    assert result["changes"][1]["result"]["name"] == "NOT Gate"
    assert result["changes"][2]["result"]["label"] == "nS"
    edited = io.load(tools.workspace / result["path"])
    assert edited.main_circuit.find(label="nS")
    assert edited.main_circuit.components[gate.index].label == "First"
    assert original.main_circuit.find(label="S")
    json.dumps(result)


@pytest.mark.parametrize("in_place", [False, True])
def test_edit_batch_rolls_back(tools, in_place):
    file = tools.workspace / "fa.circ"
    before = file.read_bytes()
    with pytest.raises(ToolError, match=r"edit 2: .*missing"):
        tools.apply_edits("fa.circ", [
            {"op": "rename_pin", "old_label": "S", "new_label": "Sum"},
            {"op": "delete", "target": "missing"},
        ], in_place=in_place)
    assert file.read_bytes() == before
    assert list(tools.workspace.iterdir()) == [file]


def test_tidy_and_overwrites(tools):
    file = tools.workspace / "fa.circ"
    before = file.read_bytes()
    project = io.load(file)
    signature = netlist.extract(project, project.main_circuit).signature()
    result = tools.tidy_circuit("fa.circ")
    tidy = io.load(tools.workspace / result["path"])
    assert netlist.extract(tidy, tidy.main_circuit).signature() == signature
    assert file.read_bytes() == before
    with pytest.raises(ToolError, match="overwrite=True"):
        tools.tidy_circuit("fa.circ")
    tools.tidy_circuit("fa.circ", in_place=True)
    assert (tools.workspace / "fa.circ.bak").read_bytes() == before

    result = tools.build_from_expression("and.circ", "A & B", verify=False)
    out = tools.workspace / result["path"]
    before = out.read_bytes()
    assert result["verification"] is None
    with pytest.raises(ToolError, match="overwrite=True"):
        tools.build_from_expression("and.circ", "A | B", verify=False)
    assert out.read_bytes() == before
    tools.build_from_expression("and.circ", "A | B", overwrite=True, verify=False)
    assert out.read_bytes() != before
    assert (tools.workspace / "and.circ.bak").read_bytes() == before


def test_invalid_reference_arguments(tools):
    with pytest.raises(ToolError, match="exactly one"):
        tools.verify("fa.circ")
    with pytest.raises(ToolError, match="unknown outputs"):
        tools.verify("fa.circ", {"typo": "A ^ B ^ Cin"})
    with pytest.raises(ToolError, match="unknown inputs"):
        tools.verify("fa.circ", {"S": "Absent"})
    with pytest.raises(ToolError, match="exceed max_rows"):
        tools.truth_table("fa.circ", max_rows=7)
    with pytest.raises(ToolError) as error:
        tools.build_from_expression("bad.circ", "A + (", verify=False)
    assert "\n" not in str(error.value)


def test_stdio(tools):
    env = {key: value for key, value in os.environ.items()
           if key.startswith("LOGISIM_") or key in {"TMP", "TEMP"}}
    env["LOGISIM_WORKSPACE"] = str(tools.workspace / "invalid-default")
    params = mcp.StdioServerParameters(
        command=sys.executable,
        args=["-m", "logisim_mcp.server", "--workspace", str(tools.workspace)],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
    )

    async def run():
        async with mcp.stdio_client(params) as streams:
            async with mcp.ClientSession(*streams) as session:
                await session.initialize()
                listed = await session.list_tools()
                assert {tool.name for tool in listed.tools} == TOOLS
                describe = await session.call_tool("describe_circuit", {"file": "fa.circ"})
                result = describe.model_dump(by_alias=True)
                assert not result["isError"], describe.content
                assert result["structuredContent"]["name"] == "full_adder"
                linted = await session.call_tool("lint_circuit", {"file": "fa.circ", "errors_only": True})
                result = linted.model_dump(by_alias=True)
                assert not result["isError"], linted.content
                if result["structuredContent"] is None:
                    assert not linted.content
                else:
                    assert result["structuredContent"] == {"result": []}
                bad = await session.call_tool("describe_circuit", {"file": "../outside.circ"})
                assert bad.model_dump(by_alias=True)["isError"]
                message = bad.content[0].text
                assert "outside the workspace" in message and "\n" not in message

    asyncio.run(asyncio.wait_for(run(), timeout=30))


@pytest.mark.jar
def test_build_from_expression(tools):
    built = tools.build_from_expression("xor.circ", {"Y": "A ^ B"})
    assert built["verification"]["ok"], built
    assert built["verification"]["passed"] == 4
    assert (tools.workspace / built["path"]).is_file()


@pytest.mark.jar
def test_build_from_truth_table(tools):
    (tools.workspace / "xor.csv").write_text(CSV, encoding="utf-8")
    built = tools.build_from_truth_table("xor.circ", "xor.csv", form="pos")
    assert built["verification"]["ok"], built
    assert tools.verify("xor.circ", truth_table_csv=CSV)["ok"]


@pytest.mark.jar
def test_build_and_verify_fsm(tools):
    (tools.workspace / "machine.txt").write_text(FSM, encoding="utf-8")
    built = tools.build_fsm("fsm.circ", "machine.txt", encoding="onehot")
    assert built["verification"]["ok"], built
    result = tools.verify_fsm("fsm.circ", FSM, [{"go": 1}, {}, {"rst": 1}, {"go": 1}])
    assert result["ok"] and result["passed"] == 4, result


@pytest.mark.jar
@pytest.mark.parametrize(("kind", "options"), [
    ("ripple_adder", {}), ("adder_subtractor", {}), ("mux_tree", {"data_inputs": 3}),
    ("decoder", {"decoder_inputs": 3}), ("comparator", {}), ("alu", {}),
    ("counter", {"clear": "async"}), ("register_file", {"registers": 3, "read_ports": 2}),
])
def test_build_block(tools, kind, options):
    built = tools.build_block(f"{kind}.circ", kind, width=2, **options)
    assert built["verification"]["ok"], built


@pytest.mark.jar
def test_full_adder_truth_table_and_vectors(tools):
    table = tools.truth_table("fa.circ", max_rows=8)
    assert set(table["inputs"]) == {"A", "B", "Cin"}
    assert set(table["outputs"]) == {"S", "Cout"}
    assert len(table["rows"]) == 8
    for row in table["rows"]:
        total = row["A"] + row["B"] + row["Cin"]
        assert (row["Cout"], row["S"]) == divmod(total, 2)
    sim.write_test_vectors(
        tools.workspace / "fa.circ",
        lambda A, B, Cin: {"S": A ^ B ^ Cin, "Cout": (A + B + Cin) >> 1},
        tools.workspace / "fa.txt",
    )
    result = tools.run_tests("fa.circ", "fa.txt")
    assert result["ok"] and result["passed"] == 8, result


@pytest.mark.jar
def test_verify_reports_wrong_expression(tools):
    result = tools.verify("fa.circ", {"S": "A | B | Cin"})
    assert not result["ok"] and result["failed"] == 3, result
    assert all(item["expected"]["S"] != item["actual"]["S"] for item in result["failures"])


@pytest.mark.jar
def test_render_png(tools):
    result = tools.render_png("fa.circ")
    image = tools.workspace / result["path"]
    content = image.read_bytes()
    assert image.name == "fa.png" and content.startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ToolError, match="overwrite=True"):
        tools.render_png("fa.circ")
    tools.render_png("fa.circ", overwrite=True)
    assert image.with_suffix(".png.bak").read_bytes() == content
