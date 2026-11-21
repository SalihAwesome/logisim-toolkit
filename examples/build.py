"""Generate, verify and render every example in this directory."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from shutil import copyfile
import sys
from tempfile import TemporaryDirectory
from typing import Callable

# Use the source in this worktree when run directly, even with a shared venv.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from logisim import generate, io, jar, lint, sim
from logisim.builder import Project


@dataclass
class Example:
    project: Project
    verify: Callable[[Path], sim.VerifyResult]


def full_adder() -> Example:
    circuit = generate.expression_circuit(
        {"S": "A ^ B ^ Cin", "Cout": "(A & B) | ((A ^ B) & Cin)"},
        name="full_adder", spacing="compact",
    )

    def reference(A, B, Cin):
        carry, low = divmod(A + B + Cin, 2)
        return {"S": low, "Cout": carry}

    return Example(Project(circuit), partial(sim.verify, expected_fn=reference))


def ripple_adder() -> Example:
    def reference(A, B, Cin):
        carry, low = divmod(A + B + Cin, 16)
        return {"S": low, "Cout": carry}

    return Example(generate.ripple_adder(4),
                   partial(sim.verify, expected_fn=reference))


def mux_2_to_1() -> Example:
    circuit = generate.mux_tree(2, name="mux_2_to_1", spacing="compact")
    return Example(Project(circuit), partial(sim.verify, expected_fn=mux_reference))


def mux_4_to_1() -> Example:
    circuit = generate.mux_tree(4, name="mux_4_to_1", spacing="compact")
    return Example(Project(circuit), partial(sim.verify, expected_fn=mux_reference))


def mux_reference(**inputs):
    return {"y": inputs[f"d{inputs['sel']}"]}


def decoder() -> Example:
    circuit = generate.decoder(3, spacing="normal", use_tunnels=True)

    def reference(in0, in1, in2):
        selected = in0 + 2 * in1 + 4 * in2
        return {f"out{code}": int(code == selected) for code in range(8)}

    return Example(Project(circuit), partial(sim.verify, expected_fn=reference))


def alu() -> Example:
    def reference(A, B, op):
        if op == 0:
            carry, value = divmod(A + B, 16)
        elif op == 1:
            value, carry = (A - B) % 16, int(A >= B)
        else:
            value, carry = {2: A & B, 3: A | B, 4: A ^ B}.get(op, 0), 0
        return {"Y": value, "Z": int(value == 0), "Cout": carry}

    return Example(generate.alu(4),
                   partial(sim.verify, expected_fn=reference))


TRAFFIC_LIGHT_TABLE = """\
states: green, yellow, red
inputs: timer
outputs: green, yellow, red
initial: green
moore: true
transitions:
green,timer,yellow
yellow,timer,red
red,timer,green
state_outputs:
green,1,0,0
yellow,0,1,0
red,0,0,1
"""


def traffic_light() -> Example:
    machine = generate.fsm_from_table(TRAFFIC_LIGHT_TABLE)
    project = generate.fsm_circuit(machine, name="traffic_light", spacing="compact")
    steps = [{"timer": tick} for tick in (0, 1, 0, 1, 0, 1, 1)]
    expected_lights = ("green", "yellow", "yellow", "red", "red", "green", "yellow")
    expected = [{light: int(light == active) for light in ("green", "yellow", "red")}
                for active in expected_lights]
    return Example(project, partial(sim.verify_sequence, steps=steps, expected=expected))


def counter() -> Example:
    project = generate.counter(4, spacing="normal")
    # Count from 1 through 15 to 0, then check hold and synchronous clear.
    count = {"en": 1, "clr": 0}
    steps = [count] * 17 + [{"en": 0, "clr": 0},
                            {"en": 1, "clr": 1}, count]
    expected = [{"Q": value} for value in (*range(1, 16), 0, 1, 1, 0, 1)]
    return Example(project, partial(sim.verify_sequence, steps=steps, expected=expected))


EXAMPLES = {
    "full_adder": full_adder,
    "ripple_adder": ripple_adder,
    "mux_2_to_1": mux_2_to_1,
    "mux_4_to_1": mux_4_to_1,
    "decoder": decoder,
    "alu": alu,
    "traffic_light": traffic_light,
    "counter": counter,
}


def build_example(name: str, directory: Path) -> sim.VerifyResult:
    example = EXAMPLES[name]()
    with TemporaryDirectory(dir=directory) as staging:
        path = example.project.save(Path(staging) / f"{name}.circ")
        project = io.load(path)
        errors = [issue for issue in lint.lint(project) if issue.severity == "error"]
        if errors:
            raise RuntimeError(f"{name}: {lint.report(errors)}")
        result = example.verify(path)
        if not result.ok:
            raise RuntimeError(f"{name}: {result.report()}")
        preview = jar.render_png(path, project.main_circuit.name, path.with_suffix(".png"))
        copyfile(path, directory / path.name)
        copyfile(preview, directory / preview.name)
    return result


def main() -> None:
    directory = Path(__file__).resolve().parent
    for name in EXAMPLES:
        result = build_example(name, directory)
        print(f"{name}: {result.passed} checks passed, no lint errors", flush=True)


if __name__ == "__main__":
    main()
