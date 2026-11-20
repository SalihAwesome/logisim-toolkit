"""Read and write .circ files."""

from __future__ import annotations

import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from .model import LOGISIM_VERSION, Circuit, Component, Library, Project, Wire

_POINT = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
HEADER_TEXT = (
    f"This file is intended to be loaded by Logisim-evolution v{LOGISIM_VERSION}"
    "(https://github.com/logisim-evolution/)."
)
# Children of <circuit> written before the components; anything else unmodelled
# (e.g. <boardmap>) goes after the wires. Logisim itself writes comps, then wires.
_BEFORE_WIRES = {"appear"}


class CircFormatError(ValueError):
    pass


def parse_point(s: str) -> tuple[int, int]:
    m = _POINT.fullmatch(s.strip())
    if not m:
        raise CircFormatError(f"bad location {s!r}")
    return int(m.group(1)), int(m.group(2))


def format_point(p: tuple[int, int]) -> str:
    return f"({p[0]},{p[1]})"


def _read_attrs(elt: ET.Element) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for a in elt.findall("a"):
        name = a.get("name")
        if name is None:
            continue
        attrs[name] = a.get("val") if a.get("val") is not None else (a.text or "")
    return attrs


def _write_attrs(parent: ET.Element, attrs: dict[str, str]) -> None:
    for name, val in attrs.items():
        a = ET.SubElement(parent, "a", {"name": name})
        if "\n" in val:
            a.text = val
        else:
            a.set("val", val)


def loads(text: str | bytes) -> Project:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise CircFormatError(f"not well-formed XML: {e}") from e
    if root.tag != "project":
        raise CircFormatError(f"root element is <{root.tag}>, expected <project>")
    proj = Project(source=root.get("source", LOGISIM_VERSION), version=root.get("version", "1.0"))
    for child in root:
        tag = child.tag
        if tag == "lib":
            proj.libraries.append(
                Library(child.get("name", ""), child.get("desc", ""), [t for t in child])
            )
        elif tag == "main":
            proj.main = child.get("name")
        elif tag == "options":
            proj.options = child
        elif tag == "mappings":
            proj.mappings = child
        elif tag == "toolbar":
            proj.toolbar = child
        elif tag == "circuit":
            proj.circuits.append(_read_circuit(child))
        else:
            proj.extras.append(child)
    return proj


def _read_circuit(elt: ET.Element) -> Circuit:
    name = elt.get("name")
    if not name:
        raise CircFormatError("<circuit> without a name")
    circ = Circuit(name, attrs=_read_attrs(elt))
    for child in elt:
        if child.tag == "a":
            continue
        if child.tag == "wire":
            try:
                circ.wires.append(Wire(parse_point(child.get("from", "")), parse_point(child.get("to", ""))))
            except CircFormatError as e:
                raise CircFormatError(f"circuit {name!r}: wire: {e}") from e
        elif child.tag == "comp":
            cname = child.get("name")
            if cname is None:
                raise CircFormatError(f"circuit {name!r}: <comp> without a name")
            loc = parse_point(child.get("loc", ""))
            circ.components.append(Component(cname, loc, _read_attrs(child), child.get("lib")))
        else:
            circ.extras.append(child)
    return circ


def load(path: str | os.PathLike) -> Project:
    data = Path(path).read_bytes()
    try:
        return loads(data)
    except CircFormatError as e:
        raise CircFormatError(f"{path}: {e}") from e


def to_element(proj: Project) -> ET.Element:
    root = ET.Element("project", {"source": proj.source, "version": proj.version})
    root.text = HEADER_TEXT
    for lib in proj.libraries:
        le = ET.SubElement(root, "lib", {"desc": lib.desc, "name": lib.id})
        le.extend(lib.tools)
    if proj.main is not None:
        ET.SubElement(root, "main", {"name": proj.main})
    for elt in (proj.options, proj.mappings, proj.toolbar):
        if elt is not None:
            root.append(elt)
    if proj.options is None:
        opts = ET.SubElement(root, "options")
        _write_attrs(opts, {"gateUndefined": "ignore", "simlimit": "1000", "simrand": "0"})
    if proj.mappings is None:
        ET.SubElement(root, "mappings")
    if proj.toolbar is None:
        ET.SubElement(root, "toolbar")
    root.extend(proj.extras)
    for circ in proj.circuits:
        root.append(_circuit_element(circ))
    return root


def _circuit_element(circ: Circuit) -> ET.Element:
    ce = ET.Element("circuit", {"name": circ.name})
    attrs = dict(circ.attrs)
    attrs.setdefault("circuit", circ.name)
    _write_attrs(ce, attrs)
    for e in circ.extras:
        if e.tag in _BEFORE_WIRES:
            ce.append(e)
    for c in circ.components:
        ae = {"lib": c.lib} if c.lib is not None else {}
        ae.update(loc=format_point(c.loc), name=c.name)
        ce_comp = ET.SubElement(ce, "comp", ae)
        _write_attrs(ce_comp, c.attrs)
    for w in circ.wires:
        ET.SubElement(ce, "wire", {"from": format_point(w.a), "to": format_point(w.b)})
    for e in circ.extras:
        if e.tag not in _BEFORE_WIRES:
            ce.append(e)
    return ce


def dumps(proj: Project) -> str:
    root = to_element(proj)
    ET.indent(root, space="  ")
    # ET.indent replaces the header text's trailing whitespace; restore Logisim's layout.
    root.text = "\n  " + HEADER_TEXT + "\n\n  "
    body = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    return '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n' + body + "\n"


def save(proj: Project, path: str | os.PathLike, *, overwrite: bool = False, backup: bool = True) -> Path:
    """Write the project. Refuses to replace an existing file unless overwrite=True.

    When overwriting, the previous file is kept as <name>.bak unless backup=False.
    """
    path = Path(path)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True (a .bak copy is kept)")
        if backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(dumps(proj), encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


def default_output_path(path: str | os.PathLike, suffix: str = "_edited") -> Path:
    p = Path(path)
    return p.with_name(p.stem + suffix + p.suffix)
