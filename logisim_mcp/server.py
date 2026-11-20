"""Workspace-scoped Logisim tools served over MCP stdio."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path, PureWindowsPath
from threading import Lock
from typing import Any, Literal

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ModuleNotFoundError as exc:
    if exc.name != "mcp.server.mcpserver":
        raise
    # The optional dependency also permits the 1.x SDK.
    from mcp.server.fastmcp import FastMCP as MCPServer
    from mcp.server.fastmcp.exceptions import ToolError

from logisim import builder, edit, generate, io, jar, lint, model, netlist, sim

_EXTENSIONS = frozenset({".circ", ".txt", ".csv", ".png"})
_ERRORS = (ValueError, KeyError, OSError, sim.SimError, jar.LogisimError)
_EDIT_ERRORS = _ERRORS + (TypeError, IndexError)


def _message(exc: Exception) -> str:
    detail = exc.args[0] if isinstance(exc, KeyError) and len(exc.args) == 1 else str(exc)
    return " ".join(str(detail).splitlines()).strip() or type(exc).__name__


def _tool(function):
    @wraps(function)
    def call(self, *args, **kwargs):
        try:
            # Sync tools run in worker threads. Serialize their use of the
            # toolkit's shared geometry cache and each read/edit/save transaction.
            with self._lock:
                return function(self, *args, **kwargs)
        except _ERRORS as exc:
            raise ToolError(_message(exc)) from None

    return call


def _select(project: model.Project, circuit: str | None) -> model.Circuit:
    return project.main_circuit if circuit is None else project.circuit(circuit)


def _describe(project: model.Project, circuit: model.Circuit) -> dict:
    return {
        "circuit": circuit.name,
        "pins": [
            {"label": pin.label, "width": int(pin.get("width", "1")), "direction": direction}
            for direction, pins in (("input", circuit.inputs()), ("output", circuit.outputs()))
            for pin in pins
        ],
        "component_counts": dict(Counter(comp.name for comp in circuit.components)),
        "subcircuits": sorted({comp.name for comp in circuit.components if project.is_subcircuit(comp)}),
        "net_count": len(netlist.extract(project, circuit).nets),
    }


def _issues(issues: list[lint.Issue]) -> list[dict]:
    return [{**asdict(issue), "at": list(issue.at) if issue.at is not None else None}
            for issue in issues]


def _verification(result: sim.VerifyResult) -> dict:
    return {
        "ok": result.ok,
        "passed": result.passed,
        "failed": result.failed,
        "failures": [asdict(failure) for failure in result.failures],
        "exhaustive": result.exhaustive,
    }


def _expression_reference(expressions: dict[str, str]):
    if not expressions:
        raise ValueError("provide at least one output expression")
    parsed = {output: generate.parse_expression(expr) for output, expr in expressions.items()}
    inputs = {name for expr in parsed.values() for name in generate.expression_variables(expr)}

    def reference(values, /):
        return {output: generate.evaluate(expr, values) for output, expr in parsed.items()}

    return inputs, parsed.keys(), reference


def _table_reference(table: generate.TruthTable):
    if not table.outputs:
        raise ValueError("provide at least one truth table output")

    def reference(values, /):
        key = tuple(values[name] for name in table.inputs)
        return dict(zip(table.outputs, table.rows[key]))

    return reference


def _check_reference(project: model.Project, circuit: str | None, inputs, outputs,
                     *, binary_inputs: bool = False) -> None:
    if circuit is not None:
        project.circuit(circuit)
    in_pins, out_pins = sim.pin_info(project, circuit)
    input_widths = {pin.name: pin.width for pin in in_pins}
    unknown = set(inputs) - input_widths.keys()
    if unknown:
        raise ValueError(f"reference uses unknown inputs: {', '.join(sorted(unknown))}")
    unknown = set(outputs) - {pin.name for pin in out_pins}
    if unknown:
        raise ValueError(f"reference uses unknown outputs: {', '.join(sorted(unknown))}")
    if binary_inputs and any(input_widths[name] != 1 for name in inputs):
        raise ValueError("truth table reference inputs must be one-bit pins")


def _fsm_steps(fsm: generate.FSM, steps: list[dict[str, int]]) -> list[dict[str, int]]:
    if not steps:
        raise ValueError("provide at least one FSM step")
    inputs = fsm.inputs + (["rst"] if fsm.reset else [])
    for step in steps:
        unknown = step.keys() - set(inputs)
        if unknown:
            raise ValueError(f"FSM step uses unknown inputs: {', '.join(sorted(unknown))}")
        if any(value not in (0, 1) for value in step.values()):
            raise ValueError("FSM step input values must be 0 or 1")
    return [{name: step.get(name, 0) for name in inputs} for step in steps]


def _verify_fsm(path: Path, fsm: generate.FSM, steps: list[dict[str, int]],
                circuit: str | None) -> sim.VerifyResult:
    steps = _fsm_steps(fsm, steps)
    project = io.load(path)
    inputs, _ = sim.pin_info(project, _select(project, circuit).name)
    pin_names = {pin.name for pin in inputs}
    if pin_names != set(steps[0]) | {"clk"} or any(pin.width != 1 for pin in inputs):
        raise ValueError("FSM table inputs plus clk must match the circuit's one-bit inputs")
    _check_reference(project, circuit, steps[0], fsm.outputs)
    return sim.verify_sequence(path, steps, generate.simulate_fsm(fsm, steps), circuit)


def _block_reference(kind: str, width: int, data_inputs: int):
    mask = (1 << width) - 1

    def sum_outputs(total):
        return {"S": total & mask, "Cout": (total >> width) & 1}

    def add(A, B, Cin):
        return sum_outputs(A + B + Cin)

    def add_subtract(A, B, sub):
        return sum_outputs(A + (mask - B + 1 if sub else B))

    def mux(values, /):
        select = values["sel"]
        return {"y": values[f"d{select}"] if select < data_inputs else 0}

    def decode(values, /):
        selected = sum(values[f"in{i}"] << i for i in range(width))
        return {f"out{i}": int(i == selected) for i in range(1 << width)}

    def compare(A, B):
        return {"eq": int(A == B), "lt": int(A < B), "gt": int(A > B)}

    def alu(A, B, op):
        values = (A + B, A + mask - B + 1, A & B, A | B, A ^ B, 0, 0, 0)
        result = values[op] & mask
        return {"Y": result, "Z": int(result == 0), "Cout": values[op] >> width}

    return {
        "ripple_adder": add, "adder_subtractor": add_subtract, "mux_tree": mux,
        "decoder": decode, "comparator": compare, "alu": alu,
    }[kind]


def _verify_block(path: Path, kind: str, width: int, data_inputs: int,
                  registers: int, read_ports: int, clear: str) -> sim.VerifyResult:
    mask = (1 << width) - 1
    if kind == "counter":
        steps = [{"en": int(i % 3 != 0), "clr": int(i == 5)} for i in range(8)]
        expected = []
        count = 0
        for i, step in enumerate(steps):
            count = 0 if step["clr"] else (count + step["en"]) & mask
            next_step = steps[min(i + 1, len(steps) - 1)]
            expected.append({"Q": 0 if clear == "async" and next_step["clr"] else count})
        return sim.verify_sequence(path, steps, expected)
    if kind == "register_file":
        steps = [
            {"we": int(i % 3 != 2), "waddr": i % registers, "wdata": (i + 1) & mask,
             **{f"raddr{port}": (i - 1 + port) % registers for port in range(read_ports)}}
            for i in range(8)
        ]
        memory = [0] * registers
        expected = []
        for i, step in enumerate(steps):
            if step["we"]:
                memory[step["waddr"]] = step["wdata"]
            next_step = steps[min(i + 1, len(steps) - 1)]
            expected.append({f"rdata{p}": memory[next_step[f"raddr{p}"]] for p in range(read_ports)})
        return sim.verify_sequence(path, steps, expected)
    return sim.verify(path, _block_reference(kind, width, data_inputs))


_EDIT_ACTIONS = {
    "add": edit.add_component,
    "delete": edit.delete_component,
    "move": edit.move_component,
    "rotate": edit.rotate_component,
    "set_attributes": edit.set_attributes,
    "rename_pin": edit.rename_pin,
    "insert_gate": edit.insert_gate,
    "delete_wires": edit.delete_wires,
    "cleanup_wires": edit.cleanup_wires,
    "add_instance": edit.add_instance,
}
_BUILTIN_LIBS = frozenset(model.LIB_DESC) | frozenset(model.LIB_DESC.values())
_BLOCKS = {
    "ripple_adder": generate.ripple_adder,
    "adder_subtractor": generate.adder_subtractor,
    "mux_tree": generate.mux_tree,
    "decoder": generate.decoder,
    "comparator": generate.comparator,
    "register_file": generate.register_file,
    "alu": generate.alu,
    "counter": generate.counter,
}


def _target(project: model.Project, circuit: model.Circuit, target: str | int) -> str | int:
    occurrence = re.fullmatch(r"(.+)#([0-9]+)", target) if isinstance(target, str) else None
    if occurrence is None or edit.find_components(project, circuit, label=target):
        return target
    name, index = occurrence[1], int(occurrence[2])
    matches = edit.find_components(project, circuit, name=name)
    if index >= len(matches):
        raise edit.EditError(f"no component {target!r} in {circuit.name!r}")
    return matches[index].index


def _port_ref(project: model.Project, circuit: model.Circuit, ref, *, net: bool = False):
    if not isinstance(ref, (list, tuple)):
        return _target(project, circuit, ref)
    if len(ref) != 2 or not isinstance(ref[1], str):
        raise edit.EditError("use [target, port] for a port reference")
    target = _target(project, circuit, ref[0])
    if net and isinstance(target, str):
        matches = edit.find_components(project, circuit, label=target)
        if not matches:
            matches = edit.find_components(project, circuit, name=target)
        if len(matches) != 1:
            raise edit.EditError(f"expected one component for {target!r}, found {len(matches)}")
        target = matches[0].index
    return target, ref[1]


def _connection_refs(project: model.Project, circuit: model.Circuit, connections) -> dict:
    pairs = connections.items() if isinstance(connections, dict) else connections
    return {port: _port_ref(project, circuit, ref) for port, ref in pairs}


def _wire_endpoints(value):
    # edit.delete_wires accepts one tuple of endpoints or a list of endpoint pairs.
    if len(value) == 2 and all(len(point) == 2 and all(isinstance(v, int) for v in point)
                               for point in value):
        return tuple(tuple(point) for point in value)
    return value


def _check_libraries(project: model.Project) -> None:
    for library in project.libraries:
        if not library.desc.startswith("#"):
            raise ValueError(f"external library {library.desc!r} is unsupported in workspace projects")


class WorkspaceTools:
    """Bind the MCP tools to one workspace. Methods can also be called directly."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError(f"workspace is not a directory: {workspace}")
        self._lock = Lock()

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if os.name == "nt":
            is_reserved = getattr(os.path, "isreserved", lambda part: PureWindowsPath(part).is_reserved())
            if any(":" in part or is_reserved(part) for part in candidate.parts[bool(candidate.anchor):]):
                raise ValueError(f"invalid workspace path: {path}")
            if candidate.anchor and not candidate.is_absolute():
                raise ValueError(f"use a workspace-relative or absolute path: {path}")
            if str(candidate).startswith(("\\\\?\\", "\\\\.\\")):
                raise ValueError(f"invalid workspace path: {path}")
        try:
            resolved = (self.workspace / candidate).resolve()
        except RuntimeError as exc:
            raise ValueError(f"cannot resolve path {path!r}: {exc}") from None
        if not resolved.is_relative_to(self.workspace):
            raise ValueError(f"path is outside the workspace: {path}")
        return resolved

    def _file(self, path: str | Path, extensions: frozenset[str] = _EXTENSIONS,
              *, writing: bool = False) -> Path:
        resolved = self._resolve(path)
        if any(p.suffix.lower() not in extensions for p in (Path(path), resolved)):
            raise ValueError(f"disallowed file extension for {path!r}; expected {', '.join(sorted(extensions))}")
        if not writing and not resolved.is_file():
            raise FileNotFoundError(f"file does not exist: {path}")
        return resolved

    def _display(self, path: Path) -> str:
        return path.relative_to(self.workspace).as_posix()

    def _output(self, path: str | Path, extension: str, overwrite: bool) -> Path:
        output = self._file(path, frozenset({extension}), writing=True)
        if output.exists():
            if not output.is_file():
                raise ValueError(f"output is not a file: {path}")
            if not overwrite:
                raise FileExistsError(f"{path} exists; pass overwrite=True to keep a .bak copy")
        if extension == ".circ":
            self._sidecar(output, ".tmp")
        if overwrite and output.exists():
            self._sidecar(output, ".bak")
        return output

    def _sidecar(self, output: Path, suffix: str) -> Path:
        path = output.with_suffix(output.suffix + suffix)
        resolved = self._resolve(path)
        if resolved != path or path.is_symlink():
            raise ValueError(f"output sidecar must not be a link: {path.name}")
        if path.exists() and (not path.is_file() or path.stat().st_nlink > 1):
            raise ValueError(f"output sidecar must be a regular, unlinked file: {path.name}")
        return path

    def _circ_file(self, path: str | Path) -> Path:
        return self._file(path, frozenset({".circ"}))

    def _project(self, path: Path) -> model.Project:
        project = io.load(path)
        _check_libraries(project)
        return project

    def _source(self, source: str, extensions: frozenset[str], *, inline_marker: str) -> str:
        if "\n" in source or "\r" in source or inline_marker in source:
            return source
        return self._file(source, extensions).read_text(encoding="utf-8")

    def _read_csv(self, csv: str, outputs: list[str] | None = None) -> generate.TruthTable:
        content = self._source(csv, frozenset({".csv"}), inline_marker="|")
        # The toolkit also accepts filenames. Force the text path even for a one-line file.
        return generate.truth_table_from_csv(content + "\n", outputs=outputs)

    def _read_fsm(self, table: str) -> generate.FSM:
        content = self._source(table, frozenset({".txt", ".csv"}), inline_marker="states:")
        return generate.fsm_from_table(content)

    def _save_build(self, project: builder.Project, output: Path, overwrite: bool,
                    check=None) -> dict:
        project.save(output, overwrite=overwrite)
        issues = _issues(lint.lint(self._project(output)))
        return {
            "path": self._display(output),
            "lint": issues,
            "verification": _verification(check(output)) if check is not None else None,
        }

    def _edit_output(self, source: Path, output: str | None, in_place: bool,
                     overwrite: bool) -> Path:
        if in_place:
            if output is not None:
                raise ValueError("choose either output or in_place=True")
            return self._output(source, ".circ", True)
        destination = output if output is not None else io.default_output_path(source)
        result = self._output(destination, ".circ", overwrite)
        if result == source:
            raise ValueError("pass in_place=True to replace the input file")
        return result

    @_tool
    def list_circuits(self, path: str = ".") -> dict[str, Any]:
        """Recursively list workspace-relative .circ files in a folder, or list circuit names and main in one .circ file."""
        resolved = self._resolve(path)
        if resolved.is_dir():
            files = []
            for folder, directories, names in os.walk(resolved, followlinks=False):
                parent = Path(folder)
                # Skip directory aliases, including Windows junctions.
                for directory in directories[:]:
                    child = parent / directory
                    try:
                        if self._resolve(child) == child:
                            continue
                    except (ValueError, OSError):
                        pass
                    directories.remove(directory)
                for name in names:
                    child = parent / name
                    if child.suffix.lower() != ".circ":
                        continue
                    try:
                        self._circ_file(child)
                    except (ValueError, OSError):
                        continue
                    files.append(self._display(child))
            return {"path": self._display(resolved), "files": sorted(files)}
        file = self._circ_file(path)
        project = io.load(file)
        return {"path": self._display(file), "circuits": project.circuit_names(), "main": project.main}

    @_tool
    def describe_circuit(self, file: str, circuit: str | None = None) -> dict[str, Any]:
        """Describe one circuit (main by default). Returns pin labels, widths and directions, component counts, subcircuits and net count."""
        project = self._project(self._circ_file(file))
        return _describe(project, _select(project, circuit))

    @_tool
    def lint_circuit(self, file: str, circuit: str | None = None, errors_only: bool = False) -> list[dict]:
        """Lint a named circuit or the whole project. Returns issues with severity, code, circuit, message and optional coordinates."""
        project = self._project(self._circ_file(file))
        if circuit is not None:
            project.circuit(circuit)
        return _issues([issue for issue in lint.lint(project, circuit)
                        if not errors_only or issue.severity == "error"])

    @_tool
    def truth_table(self, file: str, circuit: str | None = None, max_rows: int = 1024) -> dict[str, Any]:
        """Run Logisim's exhaustive truth table for a combinational circuit (main by default). Returns input/output columns and rows. Rejects tables larger than max_rows before simulation."""
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        path = self._circ_file(file)
        project = self._project(path)
        name = _select(project, circuit).name
        inputs, _ = sim.pin_info(project, name)
        bits = sum(pin.width for pin in inputs)
        if bits >= max_rows.bit_length():
            raise ValueError(f"{bits} input bits exceed max_rows={max_rows}; increase max_rows to run")
        return asdict(sim.truth_table(path, name))

    @_tool
    def run_tests(self, file: str, vectors_file: str, circuit: str | None = None) -> dict[str, Any]:
        """Run a Logisim test-vector .txt file against a circuit (main by default). Returns ok, pass/fail counts, failures and whether the run was exhaustive."""
        path = self._circ_file(file)
        vectors = self._file(vectors_file, frozenset({".txt"}))
        name = _select(self._project(path), circuit).name
        return _verification(sim.run_vectors(path, vectors, name))

    @_tool
    def verify(self, file: str, expressions: dict[str, str] | None = None,
               truth_table_csv: str | None = None, circuit: str | None = None) -> dict[str, Any]:
        """Compare Logisim with one Python reference. Supply output-to-boolean expressions or CSV text/a .csv path with a | column marking outputs. Returns ok, counts, mismatches and whether inputs were exhaustive."""
        if (expressions is None) == (truth_table_csv is None):
            raise ValueError("provide exactly one of expressions or truth_table_csv")
        path = self._circ_file(file)
        project = self._project(path)
        name = _select(project, circuit).name
        if expressions is not None:
            inputs, outputs, reference = _expression_reference(expressions)
            _check_reference(project, name, inputs, outputs)
        else:
            table = self._read_csv(truth_table_csv)
            reference = _table_reference(table)
            _check_reference(project, name, table.inputs, table.outputs, binary_inputs=True)
        return _verification(sim.verify(path, reference, name))

    @_tool
    def verify_fsm(self, file: str, fsm_table: str, steps: list[dict[str, int]],
                   circuit: str | None = None) -> dict[str, Any]:
        """Compare Logisim's clocked outputs against an FSM table (.txt/.csv path or inline text). Each step maps input names to 0/1, including rst for a resettable FSM. Missing inputs are zero. Returns ok, counts and mismatches."""
        path = self._circ_file(file)
        name = _select(self._project(path), circuit).name
        return _verification(_verify_fsm(path, self._read_fsm(fsm_table), steps, name))

    @_tool
    def build_from_expression(self, output: str, expressions: dict[str, str] | str,
                              name: str = "expr", spacing: str = "roomy", use_tunnels: bool = False,
                              overwrite: bool = False, verify: bool = True) -> dict[str, Any]:
        """Build and save boolean expressions as a .circ. A single expression drives out. A dict maps output labels to expressions. Returns the path, lint issues and optional Logisim verification."""
        path = self._output(output, ".circ", overwrite)
        circuit = generate.expression_circuit(expressions, name=name, spacing=spacing, use_tunnels=use_tunnels)
        reference = None
        if verify:
            mapping = {"out": expressions} if isinstance(expressions, str) else expressions
            _, _, reference = _expression_reference(mapping)
        check = (lambda file: sim.verify(file, reference)) if verify else None
        return self._save_build(builder.Project(circuit), path, overwrite, check)

    @_tool
    def build_from_truth_table(self, output: str, csv: str, form: Literal["sop", "pos"] = "sop",
                               name: str = "truth_table", spacing: str = "roomy",
                               use_tunnels: bool = False, overwrite: bool = False, verify: bool = True,
                               outputs: list[str] | None = None) -> dict[str, Any]:
        """Generate a .circ from CSV text or a .csv path using sum-of-products or product-of-sums. Supply output column names if the CSV lacks a | separator. Returns path, lint issues and optional Logisim verification."""
        path = self._output(output, ".circ", overwrite)
        table = self._read_csv(csv, outputs)
        reference = _table_reference(table)
        circuit = generate.truth_table_circuit(table, name=name, form=form, spacing=spacing,
                                              use_tunnels=use_tunnels)
        check = (lambda file: sim.verify(file, reference)) if verify else None
        return self._save_build(builder.Project(circuit), path, overwrite, check)

    @_tool
    def build_fsm(self, output: str, table: str, encoding: Literal["binary", "onehot", "gray"] = "binary",
                  name: str = "fsm", spacing: str = "roomy", use_tunnels: bool = False,
                  overwrite: bool = False, verify: bool = True) -> dict[str, Any]:
        """Build and save a clocked FSM from inline table text or a .txt/.csv path. Returns path, lint issues and optional Logisim verification using a short input sequence."""
        path = self._output(output, ".circ", overwrite)
        fsm = replace(self._read_fsm(table), encoding=encoding)
        project = generate.fsm_circuit(fsm, name=name, spacing=spacing, use_tunnels=use_tunnels)
        pins = [sim.PinInfo(pin, 1) for pin in fsm.inputs]
        steps, _ = sim.input_space(pins, max_exhaustive_bits=3, samples=8)
        if fsm.reset:
            steps.append({"rst": 1})
        check = (lambda file: _verify_fsm(file, fsm, steps, name)) if verify else None
        return self._save_build(project, path, overwrite, check)

    @_tool
    def build_block(
        self, output: str,
        kind: Literal["ripple_adder", "adder_subtractor", "mux_tree", "decoder", "comparator",
                      "register_file", "alu", "counter"],
        width: int = 4, spacing: str = "roomy", use_tunnels: bool = False,
        overwrite: bool = False, verify: bool = True, data_inputs: int = 4,
        decoder_inputs: int | None = None, registers: int = 4, read_ports: int = 1,
        clear: Literal["sync", "async"] = "sync",
    ) -> dict[str, Any]:
        """Build a standard circuit block. width is the data width, or decoder input count unless decoder_inputs is set. data_inputs configures mux_tree, registers/read_ports configure register_file, and clear configures counter. Returns path, lint issues and optional Logisim verification."""
        path = self._output(output, ".circ", overwrite)
        if kind not in _BLOCKS:
            raise ValueError(f"unknown block kind {kind!r}; choose from {', '.join(_BLOCKS)}")
        if kind == "decoder" and decoder_inputs is not None:
            width = decoder_inputs
        options = {
            "mux_tree": {"data_inputs": data_inputs, "width": width},
            "decoder": {"inputs": width},
            "register_file": {"registers": registers, "width": width, "read_ports": read_ports},
            "counter": {"width": width, "clear": clear},
        }.get(kind, {"width": width})
        built = _BLOCKS[kind](spacing=spacing, use_tunnels=use_tunnels, **options)
        project = built if isinstance(built, builder.Project) else builder.Project(built)
        check = (lambda file: _verify_block(file, kind, width, data_inputs,
                                           registers, read_ports, clear)) if verify else None
        return self._save_build(project, path, overwrite, check)

    def _apply_edit(self, project: model.Project, operation: dict) -> dict:
        args = dict(operation)
        op = args.pop("op")
        if op == "rename_circuit":
            result = edit.rename_circuit(project, **args)
        elif op == "add_circuit":
            result = edit.add_circuit(project, **args)
        elif op == "merge_projects":
            source = self._project(self._circ_file(args.pop("source")))
            result = edit.merge_projects(project, source, **args)
        elif op in _EDIT_ACTIONS:
            circuit = _select(project, args.pop("circuit", None))
            if "lib" in args and args["lib"] is not None:
                library = args["lib"]
                if project.library(library) is None and library not in _BUILTIN_LIBS:
                    raise ValueError(f"unknown built-in library: {library!r}")
            if "target" in args:
                args["target"] = _target(project, circuit, args["target"])
            for key in ("net", "driver"):
                if args.get(key) is not None:
                    args[key] = _port_ref(project, circuit, args[key], net=key == "net")
            if args.get("sinks") is not None:
                args["sinks"] = [_port_ref(project, circuit, ref) for ref in args["sinks"]]
            for key in ("connections", "extra_inputs"):
                if args.get(key) is not None:
                    args[key] = _connection_refs(project, circuit, args[key])
            if op == "delete_wires":
                if args.get("point") is not None:
                    args["point"] = tuple(args["point"])
                for key in ("wires", "endpoints"):
                    if args.get(key) is not None:
                        args[key] = _wire_endpoints(args[key])
            result = _EDIT_ACTIONS[op](project, circuit, **args)
        else:
            raise edit.EditError(f"unknown operation {op!r}")
        if isinstance(result, edit.ComponentInfo):
            detail = {**asdict(result), "loc": list(result.loc)}
        elif isinstance(result, model.Circuit):
            detail = {"circuit": result.name}
        elif isinstance(result, list):
            detail = [{"a": list(wire.a), "b": list(wire.b)} for wire in result]
        else:
            detail = result
        return {"op": op, "result": detail}

    @_tool
    def apply_edits(self, file: str, edits: list[dict], output: str | None = None,
                    in_place: bool = False, overwrite: bool = False) -> dict[str, Any]:
        """Apply a batch of edits in order and save only if every edit succeeds. Returns the output path, per-edit results and lint issues.

        Each edit is an object with op and the fields below. Circuit-specific edits accept an optional circuit name (default main). A target is a label, unique component type, zero-based component index, or zero-based type occurrence such as AND Gate#0. A port reference is [target, port] or a target with an unambiguous port. A point is [x, y].

        - add: name, lib (built-in name or project ID, optional for subcircuits), optional attrs, loc (point or "auto"), connections ({new port: port reference}).
        - delete: target.
        - move: target, optional dx/dy or loc, and force.
        - rotate: target, optional facing or turns.
        - set_attributes: target, attrs, optional force.
        - rename_pin: old_label, new_label.
        - rename_circuit: old, new.
        - insert_gate: net (port reference), optional name (default NOT Gate), lib, attrs, input_port, output_port, sinks (list of port references), extra_inputs (port map), driver (port reference), loc.
        - delete_wires: point, endpoints or wires (one endpoint pair or a list of pairs).
        - cleanup_wires: circuit only.
        - add_circuit: name, optional attrs and main.
        - add_instance: subcircuit, optional label, attrs, loc and connections.
        - merge_projects: source (.circ path), optional circuits (list of names), rename (name map) and prefix.

        The default destination is <stem>_edited.circ. in_place=True replaces the input with a backup. Use overwrite=True to replace a separate output with a backup.
        """
        source = self._circ_file(file)
        path = self._edit_output(source, output, in_place, overwrite)
        project = self._project(source).copy()
        changes = []
        for index, operation in enumerate(edits, start=1):
            try:
                changes.append(self._apply_edit(project, operation))
            except _EDIT_ERRORS as exc:
                raise edit.EditError(f"edit {index}: {_message(exc)}") from None
        issues = _issues(lint.lint(project))
        io.save(project, path, overwrite=in_place or overwrite)
        return {"path": self._display(path), "changes": changes, "lint": issues}

    @_tool
    def tidy_circuit(self, file: str, circuit: str | None = None, output: str | None = None,
                     in_place: bool = False, overwrite: bool = False) -> dict[str, Any]:
        """Re-lay out one circuit (main by default) while preserving its netlist. Returns path, circuit and lint issues. Defaults to <stem>_edited.circ. Replacements keep a backup."""
        source = self._circ_file(file)
        path = self._edit_output(source, output, in_place, overwrite)
        project = self._project(source)
        name = _select(project, circuit).name
        edit.tidy(project, name)
        issues = _issues(lint.lint(project, name))
        io.save(project, path, overwrite=in_place or overwrite)
        return {"path": self._display(path), "circuit": name, "lint": issues}

    @_tool
    def render_png(self, file: str, circuit: str | None = None, output: str | None = None,
                   overwrite: bool = False) -> dict[str, Any]:
        """Render one circuit (main by default) with Logisim and return its workspace-relative PNG path and circuit name. Defaults to <stem>.png next to the input. Replacements keep a backup."""
        source = self._circ_file(file)
        name = _select(self._project(source), circuit).name
        path = self._output(output if output is not None else source.with_suffix(".png"), ".png", overwrite)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent) as scratch:
            png = jar.render_png(source, name, Path(scratch) / "preview.png")
            if overwrite:
                if path.exists():
                    shutil.copy2(path, self._sidecar(path, ".bak"))
                os.replace(png, path)
            else:
                # Publish the completed preview atomically without clobbering an existing file.
                os.link(png, path)
        return {"path": self._display(path), "circuit": name}


def create_server(workspace: str | Path) -> MCPServer:
    """Register the workspace tools with the official MCP SDK."""
    tools = WorkspaceTools(workspace)
    server = MCPServer("logisim-toolkit", instructions="Use workspace-relative file paths.")
    for tool in (
        tools.list_circuits, tools.describe_circuit, tools.lint_circuit, tools.truth_table,
        tools.run_tests, tools.verify, tools.verify_fsm, tools.build_from_expression,
        tools.build_from_truth_table, tools.build_fsm, tools.build_block, tools.apply_edits,
        tools.tidy_circuit, tools.render_png,
    ):
        server.add_tool(tool)
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", help="allowed directory (defaults to LOGISIM_WORKSPACE)")
    args = parser.parse_args()
    workspace = args.workspace if args.workspace is not None else os.environ.get("LOGISIM_WORKSPACE")
    if not workspace:
        parser.error("set --workspace DIR or LOGISIM_WORKSPACE")
    try:
        server = create_server(workspace)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(_message(exc))
    # The toolkit discovers LOGISIM_JAR when a tool needs Logisim.
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
