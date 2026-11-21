from pathlib import Path

import pytest

from examples.build import EXAMPLES, build_example
from logisim import io, lint, netlist


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.jar
@pytest.mark.parametrize("name", EXAMPLES)
def test_build_example(name, tmp_path):
    result = build_example(name, tmp_path)
    assert result.ok, result.report()
    assert (tmp_path / f"{name}.png").is_file()
    issues = lint.lint_file(tmp_path / f"{name}.circ")
    assert not [issue for issue in issues if issue.severity == "error"], lint.report(issues)


@pytest.mark.parametrize("name", EXAMPLES)
def test_committed_example_matches_generator(name):
    committed = io.load(EXAMPLE_DIR / f"{name}.circ")
    fresh = EXAMPLES[name]().project.to_model()
    assert committed.main == fresh.main
    assert {c.name for c in committed.circuits} == {c.name for c in fresh.circuits}
    for circuit in fresh.circuits:
        saved_netlist = netlist.extract(committed, circuit.name)
        fresh_netlist = netlist.extract(fresh, circuit)
        assert not saved_netlist.unknown and not fresh_netlist.unknown, circuit.name
        assert saved_netlist.signature() == fresh_netlist.signature(), circuit.name
