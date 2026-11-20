"""Data and text exports for Logisim projects."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from . import geometry, netlist
from .model import Circuit, Component, Project


def _circuit(project: Project, circuit: Circuit | str | None) -> Circuit:
    if isinstance(circuit, Circuit):
        return circuit
    return project.main_circuit if circuit is None else project.circuit(circuit)


def gate_counts(project: Project, circuit: Circuit | str) -> dict:
    """Count components in the Gates library, without expanding subcircuits."""
    circ = _circuit(project, circuit)
    by_kind = Counter(c.name for c in circ.components if project.lib_desc(c) == "#Gates")
    return {"by_kind": dict(sorted(by_kind.items())), "total": sum(by_kind.values())}


def describe(project: Project, circuit: Circuit | str | None = None) -> dict:
    """Describe one circuit's interface, components, and drawing size."""
    circ = _circuit(project, circuit)
    pins = {
        direction + "s": [
            {"label": pin.label, "width": int(pin.get("width", "1")), "direction": direction}
            for pin in components
        ]
        for direction, components in [("input", circ.inputs()), ("output", circ.outputs())]
    }
    counts = Counter(c.name for c in circ.components)
    return {
        "name": circ.name,
        **pins,
        "component_counts": dict(sorted(counts.items())),
        "gate_counts": gate_counts(project, circ),
        "subcircuits": sorted({c.name for c in circ.components if project.is_subcircuit(c)}),
        "net_count": len(netlist.extract(project, circ).nets),
        "wire_count": len(circ.wires),
    }


def project_summary(project: Project) -> dict:
    """Return the main circuit name and descriptions in project order."""
    return {
        "main": project.main_circuit.name if project.circuits else None,
        "circuits": {circ.name: describe(project, circ) for circ in project.circuits},
    }


def netlist_summary(project: Project, circuit: Circuit | str) -> list[dict]:
    """Return the toolkit's port and driver summary for a circuit's nets."""
    return netlist.summary(project, circuit)


def _markdown_cell(text: str) -> str:
    return " ".join(text.splitlines()).replace("\\", "\\\\").replace("|", "\\|")


def interface_markdown(project: Project, circuit: Circuit | str | None = None) -> str:
    """Format a circuit's interface as Markdown."""
    description = describe(project, circuit)
    lines = [f"# {_markdown_cell(description['name'])}", "", "## Pins", "",
             "| Label | Width | Direction |", "| --- | ---: | --- |"]
    for pin in description["inputs"] + description["outputs"]:
        label = _markdown_cell(pin["label"]) or "(unlabelled)"
        lines.append(f"| {label} | {pin['width']} | {pin['direction']} |")
    lines += ["", "## Subcircuits", ""]
    lines += [f"- {_markdown_cell(name)}" for name in description["subcircuits"]] or ["None"]
    lines += ["", "## Components", "", "| Type | Count |", "| --- | ---: |"]
    lines += [f"| {_markdown_cell(name)} | {count} |"
              for name, count in description["component_counts"].items()]
    return "\n".join(lines) + "\n"


def _component_data(project: Project, circ: Circuit, index: int) -> dict:
    comp = circ.components[index]
    return {"index": index, "type": comp.name, "label": comp.label, "location": list(comp.loc),
            "library": project.lib_desc(comp) or comp.lib, "attributes": dict(comp.attrs)}


def _match_components(left: list[dict], right: list[dict], keys_a: list[tuple], keys_b: list[tuple]):
    """Pair unchanged identities first, then match edits by type and label."""
    remaining_a, remaining_b = set(range(len(left))), set(range(len(right)))
    matches = []
    identities_a = [(*key[:-1], comp["library"]) for key, comp in zip(keys_a, left)]
    identities_b = [(*key[:-1], comp["library"]) for key, comp in zip(keys_b, right)]

    for ids_a, ids_b, by_location, unique in (
        (identities_a, identities_b, True, False),
        (identities_a, identities_b, False, False),
        ([(c["type"], c["label"]) for c in left], [(c["type"], c["label"]) for c in right], True, True),
        ([(c["type"], c["label"]) for c in left], [(c["type"], c["label"]) for c in right], False, True),
    ):
        groups = []
        for components, identities, remaining in ((left, ids_a, remaining_a), (right, ids_b, remaining_b)):
            grouped = defaultdict(list)
            for i in sorted(remaining):
                key = (identities[i], tuple(components[i]["location"])) if by_location else identities[i]
                grouped[key].append(i)
            groups.append(grouped)
        for key, indices_a in groups[0].items():
            indices_b = groups[1].get(key, ())
            if unique and (len(indices_a) != 1 or len(indices_b) != 1):
                continue
            for i, j in zip(indices_a, indices_b):
                matches.append((i, j))
                remaining_a.remove(i)
                remaining_b.remove(j)
    return sorted(matches), sorted(remaining_a), sorted(remaining_b)


def _circuit_diff(project_a: Project, project_b: Project, name: str) -> dict:
    a, b = project_a.circuit(name), project_b.circuit(name)
    left = [_component_data(project_a, a, i) for i in range(len(a.components))]
    right = [_component_data(project_b, b, i) for i in range(len(b.components))]
    matches, removed, added = _match_components(left, right, netlist.component_keys(a), netlist.component_keys(b))
    layout_attrs = {"facing", "labelloc"}
    layout_changed = Counter(w.key() for w in a.wires) != Counter(w.key() for w in b.wires)
    changed = []
    for i, j in matches:
        before, after = left[i], right[j]
        old, new = before["attributes"], after["attributes"]
        attrs = {k: {"before": old.get(k), "after": new.get(k)}
                 for k in sorted(old.keys() | new.keys()) if old.get(k) != new.get(k)}
        moved = before["location"] != after["location"]
        library_changed = before["library"] != after["library"]
        if moved or attrs or library_changed:
            layout_changed |= moved or bool(attrs.keys() & layout_attrs)
            changed.append({"before": before, "after": after, "attribute_changes": attrs,
                            "layout_only": not library_changed and attrs.keys() <= layout_attrs})

    equal = netlist.extract(project_a, a).signature() == netlist.extract(project_b, b).signature()
    if not equal:
        status = "netlist differs"
    elif added or removed or any(not item["layout_only"] for item in changed):
        status = "same netlist, components differ"
    elif layout_changed:
        status = "same netlist, layout differs"
    else:
        status = "identical"
    return {"added": [right[i] for i in added], "removed": [left[i] for i in removed],
            "changed": changed, "netlist_equal": equal, "layout_changed": layout_changed,
            "status": status}


def diff(project_a: Project, project_b: Project) -> dict:
    """Compare named circuits, component edits and layout-independent netlist signatures.

    A circuit rename appears as a removed name and an added name. Component indices
    refer to the respective before and after drawings.
    """
    names_a, names_b = set(project_a.circuit_names()), set(project_b.circuit_names())
    return {"circuits_added": sorted(names_b - names_a), "circuits_removed": sorted(names_a - names_b),
            "circuits": {name: _circuit_diff(project_a, project_b, name) for name in sorted(names_a & names_b)}}


class VerilogError(ValueError):
    """A circuit cannot be represented by the structural exporter."""


_PRIMITIVES = {
    "AND Gate": "and", "OR Gate": "or", "XOR Gate": "xor", "NAND Gate": "nand",
    "NOR Gate": "nor", "XNOR Gate": "xnor", "NOT Gate": "not", "Buffer": "buf",
}
_PASSIVE = {"Probe", "Splitter", "Tunnel"}
_VERILOG_KEYWORDS = set("""
    always and assign automatic begin buf bufif0 bufif1 case casex casez cell cmos config
    deassign default defparam design disable edge else end endcase endconfig endfunction
    endgenerate endmodule endprimitive endspecify endtable endtask event for force forever
    fork function generate genvar highz0 highz1 if ifnone incdir include initial inout input
    instance integer join large liblist library localparam macromodule medium module nand
    negedge nmos nor noshowcancelled not notif0 notif1 or output parameter pmos posedge
    primitive pull0 pull1 pulldown pullup pulsestyle_ondetect pulsestyle_onevent rcmos real
    realtime reg release repeat rnmos rpmos rtran rtranif0 rtranif1 scalared showcancelled
    signed small specify specparam strong0 strong1 supply0 supply1 table task time tran
    tranif0 tranif1 tri tri0 tri1 triand trior trireg unsigned use vectored wait wand weak0
    weak1 while wire wor xnor xor
""".split())


def _verilog_name(label: str, fallback: str, used: set[str]) -> str:
    name = re.sub(r"[^A-Za-z0-9_$]", "_", label) or fallback
    if not re.match(r"[A-Za-z_]", name) or name in _VERILOG_KEYWORDS:
        name = "_" + name
    unique = name
    suffix = 2
    while unique in used:
        unique = f"{name}_{suffix}"
        suffix += 1
    used.add(unique)
    return unique


def _bus(width: int) -> str:
    return f"[{width - 1}:0] " if width > 1 else ""


def _bit(name: str, width: int, bit: int) -> str:
    return f"{name}[{bit}]" if width > 1 else name


def _splitter_assignments(nl: netlist.Netlist, wires: dict[int, str], widths: dict[int, int]) -> list[str]:
    # A splitter joins individual bits in either direction. Drive each joined group
    # from the net bit that carries the real driver, even when splitters are chained.
    joined = netlist.UnionFind()
    for i, comp in enumerate(nl.circuit.components):
        if comp.name != "Splitter":
            continue
        combined = nl.port_net[i, "comb"]
        split_bits: Counter = Counter()
        for bit, side in enumerate(geometry.splitter_bit_map(comp.attrs)):
            if side is None:
                continue
            split = nl.net_of(i, f"s{side}")
            if split is None:
                raise VerilogError(f"unsupported component {comp!r}: invalid splitter bit mapping")
            joined.union((combined, bit), (split.id, split_bits[side]))
            split_bits[side] += 1

    groups = defaultdict(list)
    for bit_ref in sorted(joined.parent):
        groups[joined.find(bit_ref)].append(bit_ref)

    def ref(bit_ref: tuple[int, int]) -> str:
        nid, bit = bit_ref
        return _bit(wires[nid], widths[nid], bit)

    lines = []
    driven = {n.id for n in nl.nets if n.drivers()}
    for group in groups.values():
        drivers = [part for part in group if part[0] in driven]
        if len(drivers) > 1:
            raise VerilogError("unsupported Splitter connection: more than one driver on a joined bit")
        source = drivers[0] if drivers else group[0]
        lines.extend(f"assign {ref(part)} = {ref(source)};" for part in group if part != source)
    return lines


def _gate_lines(comp: Component, primitive: str, instance: str, out: str, inputs: list[str]) -> list[str]:
    # Logisim's default multi-input XOR/XNOR detects exactly one active input.
    # Two-input XOR/XNOR has the same truth table in both modes.
    if primitive in {"xor", "xnor"} and len(inputs) > 2 and comp.get("xor", "1") != "odd":
        terms = ["(" + " & ".join(value if i == active else f"~({value})"
                                  for i, value in enumerate(inputs)) + ")"
                 for active in range(len(inputs))]
        result = " | ".join(terms)
        if primitive == "xnor":
            result = f"~({result})"
        return [f"assign {out} = {result};"]
    return [f"{primitive} {instance}({', '.join([out, *inputs])});"]


def to_verilog(project: Project, circuit: Circuit | str) -> str:
    """Export one flat combinational circuit as a structural Verilog module.

    Pins become ports. Tunnels and splitters become internal connectivity, and
    probes have no effect on the module. Unsupported components raise VerilogError.
    """
    circ = _circuit(project, circuit)
    for comp in circ.components:
        library = project.lib_desc(comp)
        allowed = ({"Pin", "Constant"} | _PASSIVE) if library == "#Wiring" else ()
        if comp.name not in allowed and (library != "#Gates" or comp.name not in _PRIMITIVES):
            raise VerilogError(f"unsupported component {comp!r} in circuit {circ.name!r}")
    nl = netlist.extract(project, circ)
    if nl.unknown:
        raise VerilogError(f"unsupported geometry for component {circ.components[nl.unknown[0]]!r}")

    module = _verilog_name(circ.name, "circuit", set())
    used: set[str] = set()
    ports = {i: _verilog_name(c.label, f"pin_{i}", used)
             for i, c in enumerate(circ.components) if c.name == "Pin"}
    widths, wires = {}, {}
    for n in nl.nets:
        if not n.ports:
            continue
        sizes = n.widths()
        if len(sizes) > 1:
            raise VerilogError(f"net {n.id} in circuit {circ.name!r} has conflicting widths {sorted(sizes)}")
        widths[n.id] = next(iter(sizes), 1)
        wires[n.id] = _verilog_name(f"net_{n.id}", "net", used)

    def signal(index: int, port: str) -> tuple[str, int]:
        nid = nl.port_net[index, port]
        return wires[nid], widths[nid]

    declarations, logic = [], []
    for i, comp in enumerate(circ.components):
        if comp.name in _PASSIVE:
            continue
        if comp.name == "Pin":
            wire, width = signal(i, "p")
            is_output = comp.get("output", "false") == "true"
            name = ports[i]
            direction = "output" if is_output else "input"
            declarations.append(f"{direction} {_bus(width)}{name}")
            dst, src = (name, wire) if is_output else (wire, name)
            logic.append(f"assign {dst} = {src};")
            continue

        out, width = signal(i, "out")
        if comp.name == "Constant":
            value = comp.get("value", "0x1")
            try:
                number = int(value, 0) & ((1 << width) - 1)
            except ValueError as e:
                raise VerilogError(f"unsupported value {value!r} on component {comp!r}") from e
            logic.append(f"assign {out} = {width}'h{number:x};")
            continue

        primitive = _PRIMITIVES[comp.name]
        if comp.get("out", "01") != "01":
            raise VerilogError(f"unsupported output mode on component {comp!r}")
        unary = primitive in {"not", "buf"}
        count = 1 if unary else int(comp.get("inputs", "2"))
        if count < 1:
            raise VerilogError(f"unsupported input count on component {comp!r}")
        for bit in range(width):
            inputs = []
            for j in range(count):
                port = "in" if unary else f"in{j}"
                value = _bit(*signal(i, port), bit)
                if comp.get(f"negate{j}", "false") == "true":
                    value = "~" + value
                inputs.append(value)
            instance = _verilog_name(f"gate_{i}_{bit}", "gate", used)
            logic.extend(_gate_lines(comp, primitive, instance, _bit(out, width, bit), inputs))

    header = f"module {module}(\n    " + ",\n    ".join(declarations) + "\n);" if declarations else f"module {module}();"
    body = [f"wire {_bus(widths[nid])}{wire};" for nid, wire in wires.items()]
    body += _splitter_assignments(nl, wires, widths) + logic
    return "\n".join([header, *(f"  {line}" for line in body), "endmodule", ""])
