"""Command-line entry point for the Logisim toolkit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

from . import builder, edit, generate, geometry, io, jar, lint, report, sim


_SPACING = ("compact", "normal", "roomy")


class _InputError(ValueError):
    pass


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _InputError(f"{self.prog}: {message}")


def _sub(parsers, name, description, example, **kwargs):
    return parsers.add_parser(name, help=description, description=description,
                              formatter_class=argparse.RawDescriptionHelpFormatter,
                              epilog=f"Example: logisim-tool {example}", **kwargs)


def _file(parser, *, circuit=True):
    parser.add_argument("file", type=Path, metavar="FILE", help="input .circ project")
    if circuit:
        parser.add_argument("--circuit", metavar="C", help="circuit name")


def _output(parser, *, required=False):
    parser.add_argument("-o", "--output", required=required, type=Path, metavar="OUT")


def _edit_options(parser, *, circuit=True):
    # Suppressed defaults let edit options appear on either side of the operation.
    output = parser.add_mutually_exclusive_group()
    output.add_argument("-o", "--output", type=Path, default=argparse.SUPPRESS, metavar="OUT",
                        help="write to OUT (default: FILE with _edited appended)")
    output.add_argument("--in-place", action="store_true", default=argparse.SUPPRESS,
                        help="replace FILE and keep its original contents in FILE.bak")
    parser.add_argument("--force", action="store_true", default=argparse.SUPPRESS,
                        help="allow replacement of an existing output file")
    if circuit:
        parser.add_argument("--circuit", default=argparse.SUPPRESS, metavar="C",
                            help="circuit to edit (default: main)")


def _json(data):
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _write(text, path=None):
    if path is None:
        sys.stdout.write(text)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _info(args):
    project = io.load(args.file)
    if args.circuit is None:
        data = report.project_summary(project)
    else:
        data = {"main": project.main_circuit.name,
                "circuits": {args.circuit: report.describe(project, args.circuit)}}
    if args.json:
        _write(_json(data))
        return 0
    print(f"Main circuit: {data['main'] or '(none)'}")
    print(f"Circuits: {', '.join(project.circuit_names()) or '(none)'}")
    for name, details in data["circuits"].items():
        print(f"\n{name}: {details['net_count']} nets, {details['wire_count']} wires")
        for pins in ("inputs", "outputs"):
            labels = [f"{pin['label'] or '(unlabelled)'} [{pin['width']}]" for pin in details[pins]]
            print(f"  {pins.capitalize()}: {', '.join(labels) or '(none)'}")
        counts = ", ".join(f"{kind}={count}" for kind, count in details["component_counts"].items())
        gates = ", ".join(f"{kind}={count}" for kind, count in details["gate_counts"]["by_kind"].items())
        print(f"  Components: {counts or '(none)'}")
        print(f"  Gates: {details['gate_counts']['total']}" + (f" ({gates})" if gates else ""))
        print(f"  Subcircuits: {', '.join(details['subcircuits']) or '(none)'}")
    return 0


def _lint(args):
    if args.circuit == "":
        raise _InputError("circuit name cannot be empty")
    issues = lint.lint_file(args.file, args.circuit)
    errors = [issue for issue in issues if issue.severity == "error"]
    visible = errors if args.errors_only else issues
    _write(_json([asdict(issue) for issue in visible]) if args.json else lint.report(visible) + "\n")
    return int(bool(errors))


def _table(args):
    project = io.load(args.file)
    if args.circuit is not None:
        project.circuit(args.circuit)
    table = sim.truth_table(args.file, args.circuit)
    if args.json:
        _write(_json(asdict(table)))
    elif args.csv:
        columns = table.inputs + table.outputs
        writer = csv.DictWriter(sys.stdout, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(table.rows)
    else:
        print(table.to_text())
    return 0


def _expression_map(items, *, split_vars=False):
    expressions = {}
    for text in items:
        label, delimiter, expression = text.partition("=")
        if delimiter:
            label = label.strip()
            if not label:
                raise _InputError("an expression assignment needs an output name")
        elif len(items) == 1:
            label, expression = "out", text
        else:
            raise _InputError("use NAME = EXPR for each of multiple expressions")
        if label in expressions:
            raise _InputError(f"duplicate output expression for {label!r}")
        expressions[label] = generate.parse_expression(expression.strip(), split_vars=split_vars)
    return expressions


def _reference(expressions):
    def expected(values, /):
        return {label: generate.evaluate(expr, values) for label, expr in expressions.items()}

    return expected


def _test(args):
    project = io.load(args.file)
    circuit = args.circuit if args.circuit is not None else project.main_circuit.name
    project.circuit(circuit)
    if args.vectors is not None:
        if not args.vectors.is_file():
            raise _InputError(f"cannot read vector file {args.vectors}")
        result = sim.run_vectors(args.file, args.vectors, circuit)
    else:
        expressions = _expression_map(args.expr)
        inputs, outputs = sim.pin_info(project, circuit)
        input_widths = {pin.name: pin.width for pin in inputs}
        output_widths = {pin.name: pin.width for pin in outputs}
        for output, expr in expressions.items():
            if output not in output_widths:
                raise _InputError(f"circuit {circuit!r} has no output pin {output!r}")
            if output_widths[output] != 1:
                raise _InputError(f"expression output {output!r} must be a one-bit pin")
            for name in generate.expression_variables(expr):
                if name not in input_widths:
                    raise _InputError(f"circuit {circuit!r} has no input pin {name!r}")
                if input_widths[name] != 1:
                    raise _InputError(f"expression variable {name!r} must be a one-bit pin")
        result = sim.verify(args.file, _reference(expressions), circuit)
    print(result.report())
    return 0 if result.ok else 1


_BLOCKS = {
    "adder": (generate.ripple_adder, ("width",)),
    "addsub": (generate.adder_subtractor, ("width",)),
    "mux": (generate.mux_tree, ("data_inputs", "width")),
    "decoder": (generate.decoder, ("inputs",)),
    "comparator": (generate.comparator, ("width",)),
    "regfile": (generate.register_file, ("registers", "width", "read_ports")),
    "alu": (generate.alu, ("width",)),
    "counter": (generate.counter, ("width", "clear")),
}


def _generate_block(args, options):
    generator, parameters = _BLOCKS[args.kind]
    supplied = {key: getattr(args, key) for key in ("width", "data_inputs", "inputs", "registers", "read_ports", "clear")
                if getattr(args, key) is not None}
    if args.kind == "decoder" and "width" in supplied and "inputs" not in supplied:
        supplied["inputs"] = supplied.pop("width")
    if args.kind == "mux" and "inputs" in supplied and "data_inputs" not in supplied:
        supplied["data_inputs"] = supplied.pop("inputs")
    for key in supplied.keys() - set(parameters):
        raise _InputError(f"--{key.replace('_', '-')} is not a parameter of {args.kind}")
    return generator(**options, **supplied)


def _same_path(a: Path, b: Path) -> bool:
    return a.resolve() == b.resolve() or (a.exists() and b.exists() and a.samefile(b))


def _gen(args):
    vectors = getattr(args, "vectors", None)
    outputs = [path for path in (args.output, vectors, args.png) if path is not None]
    for i, path in enumerate(outputs):
        if any(_same_path(path, other) for other in outputs[:i]):
            raise _InputError(f"generation output paths must be distinct: {path}")
        if path.exists():
            raise _InputError(f"{path} already exists")

    options = {"spacing": args.spacing, "use_tunnels": args.tunnels}
    if args.generator == "expr":
        expressions = _expression_map(args.expressions, split_vars=args.split_vars)
        result = generate.expression_circuit(expressions, name=args.name, **options)
    elif args.generator == "truth":
        table = generate.truth_table_from_csv(args.table, outputs=args.outputs)
        result = generate.truth_table_circuit(table, form=args.form, **options)
    elif args.generator == "fsm":
        machine = generate.fsm_from_table(args.table.read_text(encoding="utf-8"))
        if args.encoding is not None:
            machine = replace(machine, encoding=args.encoding)
        result = generate.fsm_circuit(machine, **options)
    else:
        result = _generate_block(args, options)

    project = builder.Project(result) if isinstance(result, builder.Circuit) else result
    try:
        path = project.save(args.output)
    except RuntimeError as exc:
        raise _InputError(f"could not lay out generated circuit: {exc}") from exc
    model = io.load(path)
    if vectors is not None:
        vectors.parent.mkdir(parents=True, exist_ok=True)
        if len(expressions) == 1:
            label, expr = next(iter(expressions.items()))
            generate.expression_vectors(vectors, expr, output=label)
        else:
            sim.write_test_vectors(path, _reference(expressions), vectors)
    issues = lint.lint(model)
    for issue in issues:
        if issue.severity in ("warning", "error"):
            print(issue, file=sys.stderr)
    if args.png is not None:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        jar.render_png(path, model.main_circuit.name, args.png)
    print(f"Wrote {path}")
    return int(any(issue.severity == "error" for issue in issues))


def _assignment(text):
    key, sep, value = text.partition("=")
    if not sep or not key.strip():
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    return key.strip(), value


def _point(text):
    try:
        return io.parse_point(text if text.startswith("(") else f"({text})")
    except io.CircFormatError:
        raise argparse.ArgumentTypeError(f"expected a point X,Y, got {text!r}") from None


def _location(text):
    return text if text == "auto" else _point(text)


def _target(text):
    index = text.removeprefix("#")
    digits = index.removeprefix("-")
    if digits.isascii() and digits.isdigit():
        try:
            return int(index)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"invalid component index {text!r}") from e
    return text


def _endpoint(text):
    component, separator, port = text.rpartition(":")
    if not separator:
        return _target(text)
    if not component or not port:
        raise argparse.ArgumentTypeError(f"expected TARGET:PORT, got {text!r}")
    return _target(component), port


def _connection(text):
    port, target = _assignment(text)
    if not target:
        raise argparse.ArgumentTypeError(f"expected PORT=TARGET[:PORT], got {text!r}")
    return port, _endpoint(target)


def _find_components(project, circuit, target):
    if isinstance(target, int):
        items = edit.list_components(project, circuit)
        return items[target:target + 1] if target >= 0 else []
    return (edit.find_components(project, circuit, label=target)
            or edit.find_components(project, circuit, name=target))


def _edit_destination(args, *, other_input=None):
    output = getattr(args, "output", None)
    in_place = getattr(args, "in_place", False)
    force = getattr(args, "force", False)
    if in_place and output is not None:
        raise _InputError("use either --output or --in-place")
    target = args.file if in_place else output or io.default_output_path(args.file)
    if not in_place and _same_path(target, args.file):
        raise _InputError("use --in-place to replace the input project")
    if target.exists() and not (in_place or force):
        raise _InputError(f"{target} already exists; use --force to replace it")
    sidecars = [target.with_suffix(target.suffix + suffix) for suffix in (".bak", ".tmp")]
    if any(_same_path(sidecar, args.file) for sidecar in sidecars):
        raise _InputError(f"{target} has an output sidecar that refers to the input project")
    if other_input is not None and any(_same_path(path, other_input) for path in [target, *sidecars]):
        raise _InputError(f"{target} would replace the merge source {other_input}")
    return target, in_place or force


def _apply_edit(args, project, selected):
    op = args.operation
    if op == "rename-circuit":
        return edit.rename_circuit(project, args.old, args.new).name
    if op == "add-circuit":
        return edit.add_circuit(project, args.name, attrs=dict(args.attr), main=args.main).name
    if op == "merge":
        return edit.merge_projects(project, io.load(args.other), circuits=args.include,
                                   prefix=args.prefix, rename=dict(args.rename))

    circuit = selected if selected is not None else project.main_circuit.name
    if op == "add":
        library = args.lib
        if library is None and not project.has_circuit(args.name):
            matches = [lib for lib, name in geometry.supported() if name == args.name]
            if len(matches) == 1:
                library = matches[0]
        return edit.add_component(project, circuit, args.name, lib=library, attrs=dict(args.attr),
                                  loc=args.loc, connections=args.connect)
    if op == "delete":
        return edit.delete_component(project, circuit, args.target)
    if op == "move":
        if args.position is not None and args.loc is not None:
            raise _InputError("give the new location once")
        loc = args.loc if args.loc is not None else args.position
        if loc is not None and (args.dx is not None or args.dy is not None):
            raise _InputError("use a location or --dx/--dy")
        if loc is None and args.dx is None and args.dy is None:
            raise _InputError("move requires a location or --dx/--dy")
        return edit.move_component(project, circuit, args.target, loc=loc, dx=args.dx or 0,
                                   dy=args.dy or 0, force=args.force_edit)
    if op == "rotate":
        return edit.rotate_component(project, circuit, args.target, facing=args.facing, turns=args.turns)
    if op == "set":
        return edit.set_attributes(project, circuit, args.target, dict(args.attributes), force=args.force_edit)
    if op == "rename-pin":
        return edit.rename_pin(project, circuit, args.old, args.new)
    if op == "insert":
        return edit.insert_gate(project, circuit, args.net, args.name,
                                lib=args.lib, attrs=dict(args.attr), loc=args.loc,
                                input_port=args.input_port, output_port=args.output_port,
                                sinks=args.sink, driver=args.driver, extra_inputs=args.extra_input)
    if op == "delete-wire":
        points = args.points
        if len(points) > 2 or (points and (args.point is not None or args.index is not None)):
            raise _InputError("give one point, two endpoints, --point, or --index")
        if args.index is not None:
            wires = project.circuit(circuit).wires
            if not 0 <= args.index < len(wires):
                raise _InputError(f"wire index {args.index} is out of range")
            removed = edit.delete_wires(project, circuit, wires=wires[args.index])
        elif len(points) == 2:
            removed = edit.delete_wires(project, circuit, endpoints=tuple(points))
        else:
            removed = edit.delete_wires(project, circuit, point=points[0] if points else args.point)
        return f"Removed {len(removed)} wire(s)"
    if op == "cleanup":
        edit.cleanup_wires(project, circuit)
        return f"{len(project.circuit(circuit).wires)} wire(s) after cleanup"
    if op == "add-instance":
        return edit.add_instance(project, circuit, args.subcircuit, label=args.label, attrs=dict(args.attr),
                                 loc=args.loc, connections=args.connect)
    raise AssertionError(f"unknown edit operation {op}")


def _edit(args):
    project = io.load(args.file)
    selected = getattr(args, "circuit", None)
    if selected is not None:
        project.circuit(selected)
    if args.operation in ("list", "find"):
        if any(hasattr(args, key) for key in ("output", "in_place", "force")):
            raise _InputError(f"edit {args.operation} does not write a project")
        circuit = selected if selected is not None else project.main_circuit.name
        if args.operation == "list":
            components = edit.list_components(project, circuit)
        elif args.target is not None:
            if args.name is not None or args.label is not None:
                raise _InputError("use a target or --type/--label filters")
            components = _find_components(project, circuit, args.target)
        else:
            components = edit.find_components(project, circuit, name=args.name, label=args.label)
        for component in components:
            print(component)
        if not components:
            print("No components found")
        return 0

    path, overwrite = _edit_destination(args, other_input=getattr(args, "other", None))
    result = _apply_edit(args, project, selected)
    edit.save(project, path, in_place=overwrite)
    print(result)
    print(f"Wrote {path}")
    return 0


def _tidy(args):
    project = io.load(args.file)
    if args.all:
        names = project.circuit_names()
    else:
        names = [args.circuit if args.circuit is not None else project.main_circuit.name]
    path, overwrite = _edit_destination(args)
    for name in names:
        edit.tidy(project, project.circuit(name), spacing=args.spacing)
    edit.save(project, path, in_place=overwrite)
    for name in names:
        print(f"{name}: netlist signature unchanged")
    print(f"Wrote {path}")
    return 0


def _component_text(component):
    label = f" {component['label']!r}" if component["label"] else ""
    return f"#{component['index']} {component['type']}{label} @{io.format_point(component['location'])}"


def _diff(args):
    data = report.diff(io.load(args.before), io.load(args.after))
    if args.json:
        _write(_json(data))
        return 0
    for key, verb in (("circuits_added", "Added"), ("circuits_removed", "Removed")):
        for circuit in data[key]:
            print(f"{verb} circuit {circuit}")
    for circuit, changes in data["circuits"].items():
        print(f"{circuit}: {changes['status']}")
        for key, verb in (("added", "Added"), ("removed", "Removed")):
            for component in changes[key]:
                print(f"  {verb} {_component_text(component)}")
        for change in changes["changed"]:
            before, after = change["before"], change["after"]
            status = "Layout only" if change["layout_only"] else "Changed"
            print(f"  {status}: {_component_text(before)} -> {_component_text(after)}")
            if before["library"] != after["library"]:
                print(f"    library: {before['library']!r} -> {after['library']!r}")
            for key, values in change["attribute_changes"].items():
                print(f"    {key}: {values['before']!r} -> {values['after']!r}")
    return 0


def _export(args):
    project = io.load(args.file)
    if args.format == "gates" and args.circuit is None:
        text = _json({circ.name: report.gate_counts(project, circ) for circ in project.circuits})
    else:
        circuit = args.circuit if args.circuit is not None else project.main_circuit.name
        if args.format == "summary":
            text = _json(report.netlist_summary(project, circuit))
        elif args.format == "gates":
            text = _json(report.gate_counts(project, circuit))
        elif args.format == "markdown":
            text = report.interface_markdown(project, circuit)
        else:
            text = report.to_verilog(project, circuit)
    _write(text, args.output)
    return 0


def _generator_parser(commands):
    gen = _sub(commands, "gen", "Generate a circuit project.", 'gen expr "S = A ^ B" -o sum.circ')
    gen.set_defaults(handler=_gen)
    choices = gen.add_subparsers(dest="generator", required=True, metavar="KIND")
    expr = _sub(choices, "expr", "Generate gates from Boolean expressions.",
                'gen expr "S = A ^ B" "C = A * B" -o half_adder.circ')
    expr.add_argument("expressions", nargs="+", metavar="EXPR", help="one expression or NAME=EXPR assignments")
    expr.add_argument("--name", default="expr", help="generated circuit name (default: expr)")
    expr.add_argument("--split-vars", action="store_true", help="treat each letter of an identifier as a variable")
    expr.add_argument("--vectors", type=Path, metavar="VEC.txt", help="also write test vectors")

    truth = _sub(choices, "truth", "Generate minimized logic from a truth table.",
                 "gen truth table.csv --outputs Y --form sop -o logic.circ")
    truth.add_argument("table", type=Path, metavar="TABLE.csv")
    truth.add_argument("--form", choices=("sop", "pos"), default="sop", help="minimization form (default: sop)")
    truth.add_argument("--outputs", nargs="+", metavar="PIN", help="output columns if the table has no | separator")

    fsm = _sub(choices, "fsm", "Generate a clocked finite state machine.",
               "gen fsm states.txt --encoding onehot -o machine.circ")
    fsm.add_argument("table", type=Path, metavar="TABLE.txt")
    fsm.add_argument("--encoding", choices=("binary", "onehot", "gray"), help="override the table's state encoding")

    block = _sub(choices, "block", "Generate a standard logic block.",
                 "gen block regfile --registers 4 --width 8 --read-ports 2 -o registers.circ")
    block.add_argument("kind", choices=_BLOCKS)
    block.add_argument("--width", type=int, metavar="N",
                       help="bus width for adder, addsub, mux, comparator, regfile, alu, counter")
    block.add_argument("--data-inputs", type=int, metavar="N", help="number of mux data inputs (default: 4)")
    block.add_argument("--inputs", type=int, metavar="N", help="decoder input bits (default: 3), or mux data inputs")
    block.add_argument("--registers", type=int, metavar="N", help="number of regfile registers (default: 4)")
    block.add_argument("--read-ports", type=int, choices=(1, 2), help="regfile read ports (default: 1)")
    block.add_argument("--clear", choices=("sync", "async"), help="counter clear mode (default: sync)")
    for parser in (expr, truth, fsm, block):
        _output(parser, required=True)
        parser.add_argument("--spacing", choices=_SPACING, default="roomy", help="layout spacing (default: roomy)")
        parser.add_argument("--tunnels", action="store_true", help="use tunnels for routing")
        parser.add_argument("--png", type=Path, metavar="OUT.png", help="render the main circuit with Logisim")


def _edit_parser(commands):
    edit_parser = _sub(commands, "edit", "Inspect or modify a project.", 'edit design.circ set S labelloc=north')
    edit_parser.set_defaults(handler=_edit)
    _file(edit_parser, circuit=False)
    _edit_options(edit_parser)
    operations = edit_parser.add_subparsers(dest="operation", required=True, metavar="OP")

    def operation(name, description, example="", *, writes=True):
        parser = _sub(operations, name, description, f"edit design.circ {name} {example}".rstrip())
        if writes:
            _edit_options(parser)
        else:
            parser.add_argument("--circuit", default=argparse.SUPPRESS, metavar="C")
        return parser

    operation("list", "List components and their indices.", writes=False)
    find = operation("find", "Find components by index, label or type.", '--type "AND Gate"', writes=False)
    find.add_argument("target", nargs="?", type=_target, metavar="TARGET")
    find.add_argument("--type", "--name", dest="name", metavar="TYPE", help="component type")
    find.add_argument("--label", help="component label")

    add = operation("add", "Add a library component.", '"AND Gate" --attr label=carry')
    add.add_argument("name", metavar="TYPE")
    add.add_argument("--lib", metavar="LIB", help="library name or ID (inferred where possible)")

    def component_operation(name, description, example):
        parser = operation(name, description, example)
        parser.add_argument("target", type=_target, metavar="TARGET", help="component index, label, type, or TYPE#N for the Nth of a type (from 0)")
        return parser

    component_operation("delete", "Delete a component and clean up its wires.", "carry")
    move = component_operation("move", "Move a component and reconnect its wires.", "carry --dx 100")
    move.add_argument("position", nargs="?", type=_location, metavar="X,Y")
    move.add_argument("--loc", "--to", type=_location, metavar="X,Y", help="new location or auto")
    move.add_argument("--dx", type=int, help="horizontal offset")
    move.add_argument("--dy", type=int, help="vertical offset")

    rotate = component_operation("rotate", "Rotate a component and reconnect its wires.", "carry --facing north")
    rotation = rotate.add_mutually_exclusive_group()
    rotation.add_argument("--facing", choices=edit.FACINGS)
    rotation.add_argument("--turns", type=int, default=1, help="quarter turns (default: 1)")

    set_attrs = component_operation("set", "Set one or more component attributes.", "carry label=CarryOut")
    set_attrs.add_argument("attributes", nargs="+", type=_assignment, metavar="KEY=VALUE")

    for parser in (move, set_attrs):
        parser.add_argument("--force-edit", action="store_true", help="allow an edit the library normally rejects")

    for name, example in (("rename-circuit", "old_name new_name"), ("rename-pin", "S Sum")):
        rename = operation(name, f"Rename a {name.removeprefix('rename-')}.", example)
        rename.add_argument("old", metavar="OLD")
        rename.add_argument("new", metavar="NEW")

    insert = operation("insert", "Insert a gate between a net driver and its sinks.", 'S:p "NOT Gate"')
    insert.add_argument("net", type=_endpoint, metavar="TARGET[:PORT]", help="a component port on the net")
    insert.add_argument("name", nargs="?", default="NOT Gate", metavar="TYPE")
    insert.add_argument("--lib", default="gates", metavar="LIB")
    insert.add_argument("--input-port", help="input port of the inserted gate")
    insert.add_argument("--output-port", help="output port of the inserted gate")
    insert.add_argument("--sink", action="append", type=_endpoint, metavar="TARGET[:PORT]",
                        help="reroute only this sink (repeatable)")
    insert.add_argument("--driver", type=_endpoint, metavar="TARGET[:PORT]", help="choose the driver on a shared net")
    insert.add_argument("--extra-input", action="append", type=_connection, metavar="PORT=TARGET[:PORT]",
                        help="connect another input (repeatable)")

    delete_wire = operation("delete-wire", "Delete wires by point, endpoints or index.", "100,100 200,100")
    delete_wire.add_argument("points", nargs="*", type=_point, metavar="X,Y")
    wire = delete_wire.add_mutually_exclusive_group()
    wire.add_argument("--point", type=_point, metavar="X,Y", help="remove all wires through this point")
    wire.add_argument("--index", type=int, metavar="N", help="remove a wire by its zero-based index")
    operation("cleanup", "Clean up the circuit's wires.", "--in-place")

    add_circuit = operation("add-circuit", "Add an empty circuit to the project.", "control")
    add_circuit.add_argument("name", metavar="NAME")
    add_circuit.add_argument("--main", action="store_true", help="make it the main circuit")

    instance = operation("add-instance", "Place a subcircuit instance.", "half_adder --label fa0")
    instance.add_argument("subcircuit", metavar="SUBCIRCUIT")
    instance.add_argument("--label", default="", help="instance label")

    merge = operation("merge", "Copy circuits and dependencies from another project.", "blocks.circ --rename alu=imported_alu")
    merge.add_argument("other", type=Path, metavar="OTHER.circ")
    merge.add_argument("--include", action="append", metavar="C", help="copy this circuit and its dependencies (repeatable)")
    merge.add_argument("--prefix", help="prefix to use when a circuit name is already taken")
    merge.add_argument("--rename", action="append", type=_assignment, default=[], metavar="OLD=NEW")

    for parser in (add, insert, instance, add_circuit):
        parser.add_argument("--attr", "-a", action="append", type=_assignment, default=[], metavar="KEY=VALUE",
                            help="set a component or circuit attribute (repeatable)")
    for parser in (add, insert, instance):
        parser.add_argument("--loc", type=_location, default="auto", metavar="X,Y", help="placement (default: auto)")
    for parser in (add, instance):
        parser.add_argument("--connect", action="append", type=_connection, metavar="PORT=TARGET[:PORT]",
                            help="connect a new port to an existing net (repeatable)")


def _build_parser():
    parser = _Parser(prog="logisim-tool", description="Build, inspect, edit and verify Logisim-evolution projects.",
                     epilog="Example: logisim-tool info design.circ --json")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    info = _sub(commands, "info", "Describe a project and its circuits.", "info design.circ --json")
    info.set_defaults(handler=_info)
    _file(info)
    info.add_argument("--json", action="store_true", help="write JSON")

    check = _sub(commands, "lint", "Check a project for wiring and layout issues.", "lint design.circ --errors-only")
    check.set_defaults(handler=_lint)
    _file(check)
    check.add_argument("--json", action="store_true", help="write JSON issues")
    check.add_argument("--errors-only", action="store_true", help="omit warnings and informational issues")

    table = _sub(commands, "table", "Run Logisim and print a circuit's truth table.", "table design.circ --csv")
    table.set_defaults(handler=_table)
    _file(table)
    formats = table.add_mutually_exclusive_group()
    formats.add_argument("--json", action="store_true", help="write JSON rows and column names")
    formats.add_argument("--csv", action="store_true", help="write CSV rows with a header")

    test = _sub(commands, "test", "Check a circuit against vectors or Boolean expressions.",
                'test design.circ --expr "S = A ^ B ^ Cin"')
    test.set_defaults(handler=_test)
    _file(test)
    checks = test.add_mutually_exclusive_group(required=True)
    checks.add_argument("--vectors", type=Path, metavar="VEC.txt", help="run a Logisim vector file")
    checks.add_argument("--expr", action="append", metavar="NAME=EXPR", help="check a one-bit output (repeatable)")

    _generator_parser(commands)
    _edit_parser(commands)

    tidy = _sub(commands, "tidy", "Re-route a circuit while preserving its netlist.", "tidy design.circ --all -o tidy.circ")
    tidy.set_defaults(handler=_tidy)
    _file(tidy, circuit=False)
    selection = tidy.add_mutually_exclusive_group()
    selection.add_argument("--circuit", metavar="C", help="tidy this circuit (default: main)")
    selection.add_argument("--all", action="store_true", help="tidy every circuit")
    tidy.add_argument("--spacing", choices=_SPACING, default="roomy", help="layout spacing (default: roomy)")
    _edit_options(tidy, circuit=False)

    compare = _sub(commands, "diff", "Compare circuits, components and netlists.", "diff old.circ new.circ --json")
    compare.set_defaults(handler=_diff)
    compare.add_argument("before", type=Path, metavar="A.circ")
    compare.add_argument("after", type=Path, metavar="B.circ")
    compare.add_argument("--json", action="store_true", help="write JSON changes")

    export = _sub(commands, "export", "Export circuit data, Markdown or structural Verilog.",
                  "export design.circ --verilog -o design.v")
    export.set_defaults(handler=_export)
    _file(export)
    formats = export.add_mutually_exclusive_group(required=True)
    for name, help_text in (("summary", "netlist summary as JSON"), ("gates", "gate counts as JSON (all circuits by default)"),
                            ("markdown", "circuit interface as Markdown"), ("verilog", "flat combinational Verilog module")):
        formats.add_argument(f"--{name}", dest="format", action="store_const", const=name, help=help_text)
    _output(export)
    return parser


def main(argv=None) -> int:
    """Run a command and return 0 for success, 1 for failed checks or 2 for input errors."""
    try:
        args = _build_parser().parse_args(argv)
        return args.handler(args)
    except SystemExit as exc:
        return int(exc.code)
    except jar.JarNotFound as exc:
        message = str(exc)
        if "set LOGISIM_JAR" not in message:
            message += ". Set LOGISIM_JAR to the path of the Logisim-evolution JAR."
    except (OSError, ValueError, KeyError, jar.LogisimError, sim.SimError) as exc:
        message = str(exc.args[0]) if isinstance(exc, KeyError) else str(exc)
    print(f"error: {' '.join(line.strip() for line in message.splitlines() if line.strip())}", file=sys.stderr)
    return 2
