"""Generators: boolean expressions, truth tables, parameterised blocks and FSMs.

The pure Python tests cover the parser, the Quine-McCluskey minimizer, the
CSV reader and the FSM reference model.  The tests marked `jar` build the
circuits, save them, run them in the real Logisim and compare the result with
the Python reference.
"""

import itertools
import random
from pathlib import Path

import pytest

from logisim import generate, lint, sim
from logisim.builder import Project

# The router still leaves some wires close together and some crossings; those
# are the only warnings a generated circuit may have.
ALLOWED_WARNINGS = {"wire-spacing", "crossings"}


def assert_lint_clean(path: Path) -> None:
    issues = lint.lint_file(path)
    assert not [i for i in issues if i.severity == "error"], lint.report(issues)
    assert {i.code for i in issues} <= ALLOWED_WARNINGS, lint.report(issues)


def save(project, path: Path) -> Path:
    """Save a project (or a circuit) and check that the result lints clean."""
    if isinstance(project, Project):
        project.save(path)
    else:
        Project(project).save(path)
    assert_lint_clean(path)
    return path


# ================================================================== parsing

def eval_expr(text, values, **kw):
    return generate.evaluate(generate.parse_expression(text, **kw), values)


@pytest.mark.parametrize("text", [
    "A&B|C", "A and B or C", "A*B+C", "A . B | C", "(A&B)|C", "A&B + C",
])
def test_syntaxes_agree(text):
    for combo in itertools.product((0, 1), repeat=3):
        values = dict(zip("ABC", combo))
        assert eval_expr(text, values) == eval_expr("A&B|C", values), (text, values)


@pytest.mark.parametrize("text", ["~A*B|C", "!A and B or C", "not A and B or C"])
def test_not_syntaxes_agree(text):
    for combo in itertools.product((0, 1), repeat=3):
        values = dict(zip("ABC", combo))
        assert eval_expr(text, values) == eval_expr("~A&B|C", values), (text, values)


def test_precedence_not_and_xor_or():
    v = {"A": 1, "B": 1, "C": 0}
    # OR is the loosest, then XOR, then AND, then NOT
    assert eval_expr("A | B & C", v) == 1
    assert eval_expr("A ^ B & C", v) == 1      # A ^ (B & C)
    assert eval_expr("(A ^ B) & C", v) == 0
    assert eval_expr("A & B | C", v) == 1
    # a prefix NOT binds to the operand that follows, not to the whole product
    assert eval_expr("~A & B", v) == 0
    assert eval_expr("~(A & B)", v) == 0
    # postfix NOT has the same precedence
    assert eval_expr("A'B", {"A": 1, "B": 1}) == 0
    assert eval_expr("A''B", {"A": 1, "B": 1}) == 1


def test_juxtaposition_is_and():
    # without split_vars "AB" is one variable
    assert eval_expr("AB", {"AB": 1}) == 1
    # with split_vars every letter is a variable and juxtaposition is an AND
    assert eval_expr("A'B", {"A": 0, "B": 1}, split_vars=True) == 1
    assert eval_expr("A'B", {"A": 1, "B": 1}, split_vars=True) == 0
    assert eval_expr("(A+B)(C+D)", {"A": 1, "B": 0, "C": 0, "D": 1}, split_vars=True) == 1


def test_identifiers_are_one_variable_by_default():
    expr = generate.parse_expression("AB + C")
    assert generate.expression_variables(expr) == ["AB", "C"]
    assert eval_expr("AB", {"AB": 1}) == 1


def test_split_vars_makes_every_letter_a_variable():
    expr = generate.parse_expression("AB+C", split_vars=True)
    assert generate.expression_variables(expr) == ["A", "B", "C"]
    values = {"A": 1, "B": 1, "C": 0}
    assert generate.evaluate(expr, values) == 1
    # the postfix NOT belongs to the letter before it
    expr = generate.parse_expression("AB'", split_vars=True)
    assert generate.evaluate(expr, {"A": 1, "B": 0}) == 1
    assert generate.evaluate(expr, {"A": 1, "B": 1}) == 0


def test_constants():
    for text, value in [("0", 0), ("1", 1), ("True", 1), ("False", 0)]:
        assert eval_expr(text, {}) == value
    assert eval_expr("A + 1", {"A": 0}) == 1


def test_variable_order_is_first_appearance():
    assert generate.expression_variables(generate.parse_expression("C & (B + A)")) == ["C", "B", "A"]


@pytest.mark.parametrize("text,column", [
    ("A +", 4),          # missing operand
    ("A $ B", 3),        # unknown character
    ("(A + B", 7),       # missing ')'
    ("A B )", 5),        # trailing token
    ("", 1),             # empty
    ("A + * B", 5),      # missing operand
    ("2", 1),            # only 0 and 1 are constants
])
def test_syntax_errors_point_at_the_column(text, column):
    with pytest.raises(generate.ExprSyntaxError) as exc:
        generate.parse_expression(text)
    assert exc.value.column == column
    assert f"column {column}" in str(exc.value)


def test_evaluate_needs_every_variable():
    with pytest.raises(generate.GenerateError):
        generate.evaluate(generate.parse_expression("A & B"), {"A": 1})


# ============================================================ minimization

def _covers(value, mask, minterm):
    return (minterm & ~mask) == (value & ~mask)


def _is_implicant(value, mask, ones, zeros, n):
    hits_one = False
    for m in range(1 << n):
        if not _covers(value, mask, m):
            continue
        if m in zeros:
            return False
        if m in ones:
            hits_one = True
    return hits_one


def _is_prime(value, mask, ones, zeros, n):
    for bit in range(n):
        if (mask >> bit) & 1:
            continue
        if _is_implicant(value & ~(1 << bit), mask | (1 << bit), ones, zeros, n):
            return False
    return True


def _brute_force_primes(ones, zeros, n):
    primes = []
    for value in range(1 << n):
        for mask in range(1 << n):
            if value & mask:
                continue
            if _is_implicant(value, mask, ones, zeros, n) and _is_prime(value, mask, ones, zeros, n):
                primes.append((value, mask))
    return primes


def _brute_force_min_cover(ones, zeros, n):
    """Exact minimum cover by exhaustive search (independent of Petrick)."""
    primes = _brute_force_primes(ones, zeros, n)
    best = [len(ones) + 1]

    def search(chosen, covered):
        if len(chosen) >= best[0]:
            return
        remaining = ones - covered
        if not remaining:
            best[0] = len(chosen)
            return
        target = min(remaining, key=lambda m: sum(1 for v, mk in primes if _covers(v, mk, m)))
        for v, mk in primes:
            if _covers(v, mk, target):
                search(chosen + [(v, mk)], covered | {m for m in ones if _covers(v, mk, m)})

    search([], set())
    return best[0]


def test_minimize_matches_brute_force_on_random_functions():
    rng = random.Random(20240517)
    for n in (2, 3, 4):
        for _ in range(25):
            zeros = set()
            ones = set()
            dcs = set()
            for m in range(1 << n):
                roll = rng.random()
                if roll < 0.15:
                    dcs.add(m)
                elif roll < 0.5:
                    ones.add(m)
                else:
                    zeros.add(m)
            cubes = generate.minimize_function(ones, dcs, n)
            # the cover is valid
            for m in ones:
                assert any(c.covers(m) for c in cubes), (ones, dcs, cubes)
            for m in zeros:
                assert not any(c.covers(m) for c in cubes), (ones, dcs, cubes)
            # and it is as small as the best possible cover
            assert len(cubes) == _brute_force_min_cover(ones, zeros, n)


def test_minimize_uses_the_dont_cares():
    # two minterms that only differ in two bits: without a don't care they
    # need two cubes, with one they fit in a single cube
    assert len(generate.minimize_function({0b000, 0b011}, set(), 3)) == 2
    cubes = generate.minimize_function({0b000, 0b011}, {0b001, 0b010}, 3)
    assert len(cubes) == 1
    assert all(c.mask for c in cubes)


def test_minimized_expression_matches_the_function():
    """The expression built from the cover is the function on every care minterm."""
    rng = random.Random(4711)
    for n in (2, 3, 4):
        names = [f"v{i}" for i in range(n)]
        for _ in range(20):
            ones, zeros, dcs = set(), set(), set()
            for m in range(1 << n):
                roll = rng.random()
                if roll < 0.15:
                    dcs.add(m)
                elif roll < 0.5:
                    ones.add(m)
                else:
                    zeros.add(m)
            sop = generate.cube_expression(generate.minimize_function(ones, dcs, n), names)
            pos = generate.cube_expression(generate.minimize_function(zeros, dcs, n), names,
                                           form="pos")
            for m in range(1 << n):
                if m in dcs:
                    continue
                values = {name: (m >> (n - 1 - i)) & 1 for i, name in enumerate(names)}
                want = 1 if m in ones else 0
                assert generate.evaluate(sop, values) == want, (ones, dcs, m)
                assert generate.evaluate(pos, values) == want, (ones, dcs, m)


def test_prime_implicants_are_all_of_them():
    primes = generate.prime_implicants({0b001, 0b011}, set(), 3)
    assert len(primes) == len(_brute_force_primes({0b001, 0b011}, set(), 3))
    for cube in primes:
        assert cube.covers(0b001) or cube.covers(0b011)


def test_minimal_cover_picks_the_smallest_set():
    primes = generate.prime_implicants({0b000, 0b001, 0b010, 0b011}, set(), 2)
    cover = generate.minimal_cover({0b000, 0b001, 0b010, 0b011}, primes)
    assert len(cover) == 1


# ============================================================== truth tables

def test_truth_table_from_dict_with_tuples():
    table = generate.truth_table_from_dict({(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
                                           inputs=["A", "B"])
    assert table.inputs == ["A", "B"]
    assert table.outputs == ["Y"]
    ones, zeros, dcs = table.value_sets("Y")
    assert ones == {0b01, 0b10}
    assert zeros == {0b00, 0b11}
    assert dcs == set()


def test_truth_table_from_dict_with_minterm_numbers():
    table = generate.truth_table_from_dict({1: 1, 2: 1}, inputs=["A", "B"])
    ones, zeros, _ = table.value_sets("Y")
    assert ones == {1, 2}
    assert zeros == {0, 3}


def test_truth_table_from_dict_with_several_outputs():
    table = generate.truth_table_from_dict({(0,): (0, 1), (1,): (1, 1)}, inputs=["A"],
                                           outputs=["X", "Y"])
    assert table.outputs == ["X", "Y"]
    assert table.value_sets("X")[0] == {0b1}
    assert table.value_sets("Y")[0] == {0b0, 0b1}


def test_truth_table_from_function():
    table = generate.truth_table_from_function(lambda a, b: a & b, ["A", "B"])
    assert table.value_sets("Y")[0] == {0b11}
    table = generate.truth_table_from_function(lambda ins: ins["A"], ["A"])
    assert table.value_sets("Y")[0] == {0b1}


def test_truth_table_from_csv_with_separator():
    table = generate.truth_table_from_csv("A B | Y\n0 0 | 0\n0 1 | 1\n1 0 | 1\n1 1 | 0\n")
    assert table.inputs == ["A", "B"]
    assert table.outputs == ["Y"]
    assert table.value_sets("Y")[0] == {0b01, 0b10}


def test_truth_table_from_csv_with_output_argument():
    table = generate.truth_table_from_csv("A,B,Y\n0,0,0\n0,1,1\n1,0,1\n1,1,0\n", outputs=["Y"])
    assert table.inputs == ["A", "B"]
    assert table.outputs == ["Y"]
    assert table.value_sets("Y")[0] == {0b01, 0b10}


def test_truth_table_from_csv_reads_dont_cares():
    table = generate.truth_table_from_csv("A B | Y\n0 0 | 0\n0 1 | 1\n1 0 | x\n1 1 | -\n")
    ones, zeros, dcs = table.value_sets("Y")
    assert ones == {0b01}
    assert zeros == {0b00}
    assert dcs == {0b10, 0b11}


def test_truth_table_from_csv_needs_output_columns():
    with pytest.raises(generate.TruthTableError):
        generate.truth_table_from_csv("A B\n0 0\n1 1\n")


def test_truth_table_from_csv_rejects_bad_cells():
    with pytest.raises(generate.TruthTableError):
        generate.truth_table_from_csv("A | Y\n0 | 2\n")


def test_minimize_table_sop_and_pos():
    table = generate.truth_table_from_csv("A B | Y\n0 0 | 0\n0 1 | 1\n1 0 | 1\n1 1 | 0\n")
    sop = generate.minimize_table(table, form="sop")["Y"]
    pos = generate.minimize_table(table, form="pos")["Y"]
    assert len(sop) == 2
    assert len(pos) == 2
    ones, zeros, _ = table.value_sets("Y")
    # the SOP covers the 1s of the function, the POS the 0s (its maxterms)
    assert all(any(c.covers(m) for c in sop) for m in ones)
    assert all(any(c.covers(m) for c in pos) for m in zeros)


def test_minimize_table_pos_is_the_complement():
    table = generate.truth_table_from_csv("A | Y\n0 | 0\n1 | 1\n")
    pos = generate.minimize_table(table, form="pos")["Y"]
    # the only zero of Y is A=0, so the only maxterm is A
    assert len(pos) == 1
    assert pos[0].covers(0)


# ==================================================================== FSM

MOORE_TABLE = """
states: IDLE, RUN, DONE
inputs: go
outputs: busy
initial: IDLE
moore: true
transitions:
IDLE,0,IDLE
IDLE,1,RUN
RUN,0,RUN
RUN,1,DONE
DONE,x,DONE
state_outputs:
IDLE,0
RUN,0
DONE,1
"""

MEALY_TABLE = """
states: IDLE, RUN, DONE
inputs: go
outputs: busy
initial: IDLE
moore: false
transitions:
IDLE,0,IDLE,0
IDLE,1,RUN,1
RUN,0,RUN,0
RUN,1,DONE,1
DONE,x,DONE,1
"""


def test_fsm_from_table_reads_the_sections():
    fsm = generate.fsm_from_table(MOORE_TABLE)
    assert fsm.states == ["IDLE", "RUN", "DONE"]
    assert fsm.inputs == ["go"]
    assert fsm.outputs == ["busy"]
    assert fsm.initial == "IDLE"
    assert fsm.moore
    assert len(fsm.transitions) == 5
    assert fsm.state_outputs["DONE"] == {"busy": 1}


def test_fsm_from_dict():
    fsm = generate.fsm_from_dict({
        "states": ["A", "B"],
        "inputs": ["x"],
        "outputs": ["y"],
        "initial": "A",
        "moore": True,
        "transitions": [{"state": "A", "condition": "x", "to": "B"},
                        {"state": "B", "condition": "x", "to": "B"}],
        "state_outputs": {"A": {"y": 0}, "B": {"y": 1}},
    })
    assert fsm.states == ["A", "B"]
    assert fsm.next_state("A", {"x": 1}) == "B"
    assert fsm.next_state("A", {"x": 0}) == "A"


def test_fsm_encodings():
    fsm = generate.fsm_from_table(MOORE_TABLE)
    assert [fsm.encode(s) for s in fsm.states] == [0, 1, 2]
    fsm.encoding = "onehot"
    assert [fsm.encode(s) for s in fsm.states] == [0b100, 0b010, 0b001]
    fsm.encoding = "gray"
    assert [fsm.encode(s) for s in fsm.states] == [0b00, 0b01, 0b11]
    assert fsm.state_bits == 2


def test_fsm_reference_model_moore():
    fsm = generate.fsm_from_table(MOORE_TABLE)
    model = generate.FSMModel(fsm)
    assert model.state == "IDLE"
    assert model.outputs({"go": 0}) == {"busy": 0}
    assert model.step({"go": 0}) == {"busy": 0}
    assert model.state == "IDLE"
    assert model.step({"go": 1}) == {"busy": 0}
    assert model.state == "RUN"
    assert model.step({"go": 1}) == {"busy": 1}
    assert model.state == "DONE"


def test_fsm_reference_model_mealy():
    fsm = generate.fsm_from_table(MEALY_TABLE)
    model = generate.FSMModel(fsm)
    assert model.outputs({"go": 0}) == {"busy": 0}
    model.step({"go": 1})
    assert model.state == "RUN"
    assert model.outputs({"go": 0}) == {"busy": 0}
    assert model.outputs({"go": 1}) == {"busy": 1}


def test_simulate_fsm_follows_the_harness():
    fsm = generate.fsm_from_table(MOORE_TABLE)
    steps = [{"go": 0}, {"go": 1}, {"go": 1}]
    rows = generate.simulate_fsm(fsm, steps)
    assert rows == [{"busy": 0}, {"busy": 0}, {"busy": 1}]


def test_fsm_rejects_overlapping_conditions():
    fsm = generate.fsm_from_dict({
        "states": ["A", "B"], "inputs": ["x"], "outputs": ["y"],
        "transitions": [{"state": "A", "condition": "x", "to": "B"},
                        {"state": "A", "condition": "x & 1", "to": "A"}],
    })
    with pytest.raises(generate.FSMError):
        generate.fsm_circuit(fsm)


def test_fsm_rejects_unknown_states():
    with pytest.raises(generate.FSMError):
        generate.FSM(states=["A"], transitions=[generate.Transition("A", "x", "B")])


# ============================================================ jar tests

@pytest.mark.jar
def test_expression_circuit_in_logisim(tmp_path):
    text = "A'B + C"
    circ = generate.expression_circuit(text)
    path = save(circ, tmp_path / "expr.circ")
    r = sim.verify(path, lambda A, B, C: {"out": eval_expr(text, {"A": A, "B": B, "C": C})})
    assert r.ok, r.report()


@pytest.mark.jar
def test_expression_circuit_with_several_outputs(tmp_path):
    circ = generate.expression_circuit({"Y": "A & B", "Z": "A | B", "W": "~A"})
    path = save(circ, tmp_path / "exprs.circ")
    r = sim.verify(path, lambda A, B: {"Y": A & B, "Z": A | B, "W": 1 - A})
    assert r.ok, r.report()


@pytest.mark.jar
def test_expression_circuit_with_explicit_input_order(tmp_path):
    circ = generate.expression_circuit("B + A", inputs=["A", "B"])
    path = save(circ, tmp_path / "expr_order.circ")
    r = sim.verify(path, lambda A, B: {"out": A | B})
    assert r.ok, r.report()


@pytest.mark.jar
def test_expression_circuit_with_an_unused_input(tmp_path):
    # an input that the expression does not use still gets a pin
    circ = generate.expression_circuit("A & B", inputs=["A", "B", "C"])
    path = save(circ, tmp_path / "expr_unused.circ")
    r = sim.verify(path, lambda A, B, C: {"out": A & B})
    assert r.ok, r.report()


@pytest.mark.jar
def test_expression_vectors_run_in_logisim(tmp_path):
    text = "A'B + C"
    circ = generate.expression_circuit(text)
    path = save(circ, tmp_path / "expr.circ")
    vectors = generate.expression_vectors(tmp_path / "vectors.txt", text)
    lines = vectors.read_text().splitlines()
    assert lines[1].split() == ["A", "B", "C", "out"]
    assert len(lines) == 2 + 8
    r = sim.run_vectors(path, vectors)
    assert r.ok and r.passed == 8, r.report()


@pytest.mark.parametrize("form", ["sop", "pos"])
@pytest.mark.jar
def test_truth_table_circuit_in_logisim(tmp_path, form):
    table = generate.truth_table_from_csv(
        "A B | Y Z\n0 0 | 0 1\n0 1 | 1 1\n1 0 | 1 1\n1 1 | 0 0\n")
    circ = generate.truth_table_circuit(table, form=form)
    path = save(circ, tmp_path / f"tt_{form}.circ")

    def reference(A, B):
        return {"Y": A ^ B, "Z": 1 if not (A and B) else 0}

    r = sim.verify(path, reference)
    assert r.ok, r.report()


@pytest.mark.jar
def test_truth_table_circuit_with_dont_cares(tmp_path):
    table = generate.truth_table_from_csv("A B | Y\n0 0 | 0\n0 1 | 1\n1 0 | x\n1 1 | 0\n")
    circ = generate.truth_table_circuit(table)
    path = save(circ, tmp_path / "tt_dc.circ")
    # the don't care may not change the value of the other rows
    r = sim.verify(path, lambda A, B: {"Y": 1 if (not A and B) else 0})
    assert r.ok, r.report()


@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.jar
def test_ripple_adder_in_logisim(tmp_path, width):
    proj = generate.ripple_adder(width)
    path = save(proj, tmp_path / f"adder{width}.circ")

    def reference(A, B, Cin):
        total = A + B + Cin
        return {"S": total & ((1 << width) - 1), "Cout": total >> width}

    r = sim.verify(path, reference)
    assert r.ok, r.report()


@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.jar
def test_adder_subtractor_in_logisim(tmp_path, width):
    proj = generate.adder_subtractor(width)
    path = save(proj, tmp_path / f"addsub{width}.circ")

    def reference(A, B, sub):
        if sub:
            total = A + ((~B) & ((1 << width) - 1)) + 1
        else:
            total = A + B
        return {"S": total & ((1 << width) - 1), "Cout": (total >> width) & 1}

    r = sim.verify(path, reference)
    assert r.ok, r.report()


@pytest.mark.parametrize("count,width", [(4, 4), (5, 2)])
@pytest.mark.jar
def test_mux_tree_in_logisim(tmp_path, count, width):
    circ = generate.mux_tree(count, width)
    path = save(circ, tmp_path / "mux.circ")
    names = [f"d{i}" for i in range(count)]
    sel_bits = max(1, (count - 1).bit_length())

    def reference(**values):
        sel = values["sel"]
        chosen = values[names[sel]] if sel < count else 0
        return {"y": chosen & ((1 << width) - 1)}

    r = sim.verify(path, reference, max_exhaustive_bits=14)
    assert r.ok, r.report()
    assert sel_bits


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.jar
def test_decoder_in_logisim(tmp_path, count):
    circ = generate.decoder(count)
    path = save(circ, tmp_path / "decoder.circ")

    def reference(**values):
        code = sum(values[f"in{i}"] << i for i in range(count))
        return {f"out{k}": int(k == code) for k in range(1 << count)}

    r = sim.verify(path, reference)
    assert r.ok, r.report()


@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.jar
def test_comparator_in_logisim(tmp_path, width):
    proj = generate.comparator(width)
    path = save(proj, tmp_path / f"cmp{width}.circ")
    r = sim.verify(path, lambda A, B: {"eq": int(A == B), "lt": int(A < B), "gt": int(A > B)})
    assert r.ok, r.report()


@pytest.mark.parametrize("registers,width,read_ports", [(4, 4, 1), (4, 2, 2)])
@pytest.mark.jar
def test_register_file_in_logisim(tmp_path, registers, width, read_ports):
    proj = generate.register_file(registers, width, read_ports=read_ports)
    path = save(proj, tmp_path / f"regfile{read_ports}.circ")
    steps = []
    for k in range(6):
        step = {"we": int(k % 2 == 0), "waddr": k % registers,
                "wdata": (k * 3 + 1) & ((1 << width) - 1)}
        for p in range(read_ports):
            step[f"raddr{p}"] = (k + p) % registers
        steps.append(step)

    # the sequence harness shows the circuit while the next step is applied
    mem = [0] * registers
    snaps = []
    for step in steps:
        if step["we"]:
            mem[step["waddr"]] = step["wdata"]
        snaps.append(list(mem))
    expected = []
    for k in range(len(steps)):
        nxt = steps[k + 1] if k + 1 < len(steps) else steps[-1]
        expected.append({f"rdata{p}": snaps[k][nxt[f"raddr{p}"]] for p in range(read_ports)})

    r = sim.verify_sequence(path, steps, expected, "register_file")
    assert r.ok, r.report()


@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.jar
def test_alu_in_logisim(tmp_path, width):
    proj = generate.alu(width)
    path = save(proj, tmp_path / f"alu{width}.circ")

    def reference(A, B, op):
        mask = (1 << width) - 1
        if op == 0:
            total = A + B
            value, carry = total & mask, (total >> width) & 1
        elif op == 1:
            total = A + ((~B) & mask) + 1
            value, carry = total & mask, (total >> width) & 1
        elif op == 2:
            value, carry = A & B, 0
        elif op == 3:
            value, carry = A | B, 0
        elif op == 4:
            value, carry = A ^ B, 0
        else:
            value, carry = 0, 0
        return {"Y": value, "Z": int(value == 0), "Cout": carry}

    r = sim.verify(path, reference, max_exhaustive_bits=13)
    assert r.ok, r.report()


def counter_reference(steps, width, clear):
    """The counter as the sequence harness sees it.

    The clear wins over the count at the clock edge; an asynchronous clear
    also forces the output to 0 while it is high.
    """
    q = 0
    rows = []
    for k, step in enumerate(steps):
        if step["clr"]:
            q = 0
        elif step["en"]:
            q = (q + 1) % (1 << width)
        nxt = steps[k + 1] if k + 1 < len(steps) else steps[-1]
        shown = 0 if (clear == "async" and nxt["clr"]) else q
        rows.append({"Q": shown})
    return rows


@pytest.mark.parametrize("clear", ["sync", "async"])
@pytest.mark.parametrize("width", [4, 8])
@pytest.mark.jar
def test_counter_in_logisim(tmp_path, clear, width):
    proj = generate.counter(width, clear=clear)
    path = save(proj, tmp_path / f"counter_{clear}{width}.circ")
    steps = []
    for k in range(8):
        steps.append({"en": int(k % 3 != 0), "clr": int(k == 5)})
    expected = counter_reference(steps, width, clear)
    r = sim.verify_sequence(path, steps, expected, "counter")
    assert r.ok, r.report()


@pytest.mark.parametrize("encoding", ["binary", "onehot", "gray"])
@pytest.mark.jar
def test_fsm_moore_in_logisim(tmp_path, encoding):
    fsm = generate.fsm_from_table(MOORE_TABLE)
    fsm.encoding = encoding
    proj = generate.fsm_circuit(fsm)
    path = save(proj, tmp_path / f"fsm_moore_{encoding}.circ")
    steps = [{"go": 0}, {"go": 1}, {"go": 1}, {"go": 0}, {"go": 0}]
    expected = generate.simulate_fsm(fsm, steps)
    r = sim.verify_sequence(path, steps, expected, "fsm")
    assert r.ok, r.report()


@pytest.mark.parametrize("encoding", ["binary", "onehot", "gray"])
@pytest.mark.jar
def test_fsm_mealy_in_logisim(tmp_path, encoding):
    fsm = generate.fsm_from_table(MEALY_TABLE)
    fsm.encoding = encoding
    proj = generate.fsm_circuit(fsm)
    path = save(proj, tmp_path / f"fsm_mealy_{encoding}.circ")
    steps = [{"go": 0}, {"go": 1}, {"go": 1}, {"go": 0}, {"go": 0}]
    expected = generate.simulate_fsm(fsm, steps)
    r = sim.verify_sequence(path, steps, expected, "fsm")
    assert r.ok, r.report()


@pytest.mark.parametrize("encoding", ["binary", "onehot", "gray"])
@pytest.mark.jar
def test_fsm_starts_in_its_initial_state(tmp_path, encoding):
    """After power-on the machine is in the initial state, with no reset pulse."""
    table = """
states: OFF, ON
inputs: sw
outputs: lamp
initial: OFF
moore: true
transitions:
OFF,0,OFF
OFF,1,ON
ON,x,ON
state_outputs:
OFF,0
ON,1
"""
    fsm = generate.fsm_from_table(table)
    fsm.encoding = encoding
    proj = generate.fsm_circuit(fsm)
    path = save(proj, tmp_path / f"fsm_start_{encoding}.circ")
    # the switch stays off, so the lamp may only stay off if the machine
    # really started in OFF
    steps = [{"sw": 0}, {"sw": 0}, {"sw": 0}]
    expected = generate.simulate_fsm(fsm, steps)
    assert expected == [{"lamp": 0}] * 3
    r = sim.verify_sequence(path, steps, expected, "fsm")
    assert r.ok, r.report()


@pytest.mark.jar
def test_fsm_with_reset_in_logisim(tmp_path):
    table = """
states: IDLE, RUN
inputs: go
outputs: busy
initial: IDLE
moore: true
reset: true
transitions:
IDLE,0,IDLE
IDLE,1,RUN
RUN,x,RUN
state_outputs:
IDLE,0
RUN,1
"""
    fsm = generate.fsm_from_table(table)
    proj = generate.fsm_circuit(fsm)
    path = save(proj, tmp_path / "fsm_reset.circ")
    steps = [{"go": 1, "rst": 0}, {"go": 0, "rst": 0}, {"go": 0, "rst": 1}, {"go": 0, "rst": 0}]
    expected = []
    state = fsm.initial
    for k, step in enumerate(steps):
        if step["rst"]:
            state = fsm.initial
        else:
            state = fsm.next_state(state, {"go": step["go"]})
        expected.append({"busy": fsm.state_outputs[state]["busy"]})
    r = sim.verify_sequence(path, steps, expected, "fsm")
    assert r.ok, r.report()


@pytest.mark.jar
def test_fsm_with_constant_outputs_keeps_its_pins(tmp_path):
    """Outputs that never depend on the state still get pins (driven by constants)."""
    fsm = generate.fsm_from_dict({
        "states": ["A", "B"], "inputs": ["go"], "outputs": ["one", "zero"], "initial": "A",
        "moore": False,
        "transitions": [{"from": a, "condition": c, "to": b, "outputs": {"one": 1, "zero": 0}}
                        for a, c, b in (("A", "1", "B"), ("A", "0", "A"), ("B", "x", "A"))],
    })
    proj = generate.fsm_circuit(fsm)
    path = save(proj, tmp_path / "fsm_const.circ")
    steps = [{"go": 1}, {"go": 0}, {"go": 1}]
    assert generate.simulate_fsm(fsm, steps) == [{"one": 1, "zero": 0}] * 3
    r = sim.verify_sequence(path, steps, [{"one": 1, "zero": 0}] * 3, "fsm")
    assert r.ok, r.report()
