"""Circuit generators: expressions, truth tables, blocks and finite state machines.

Everything here builds on logisim.builder (stages 1-3); no netlist, layout,
routing, simulation or linting logic is duplicated.

Four families of generators:

1. Boolean expressions
     parse_expression(text, *, split_vars=False) -> Expr
     evaluate(expr, values) -> int
     expression_circuit(expression, ...) -> Circuit
     expression_vectors(path, expression, ...) -> Path

2. Truth tables -> minimized logic (Quine-McCluskey with don't-cares)
     truth_table_from_dict / truth_table_from_function / truth_table_from_csv
     prime_implicants / minimal_cover / minimize_function / minimize_table
     cube_expression, truth_table_circuit(table, ..., form="sop"|"pos") -> Circuit

3. Parameterized blocks (each returns a builder Project or Circuit)
     ripple_adder, adder_subtractor, mux_tree, decoder, comparator,
     register_file, alu, counter

4. Finite state machines
     FSM / Transition, fsm_from_dict, fsm_from_table
     fsm_circuit(fsm, ...) -> Project
     FSMModel, simulate_fsm(fsm, steps) -> list[dict]

Numbering convention
--------------------
A *minterm number* is a bit mask over the variables of a function.  Variable 0
is the most significant bit, so for the variables (A, B, C) the minterm 5 means
A=1, B=0, C=1 -- the usual textbook numbering.  `Cube.value`/`Cube.mask` use
the same numbering: bit n-1-i of a cube belongs to variables[i].

Transition conditions
---------------------
A condition of a state machine is "x" (always), an input vector like "01" (a
"x" in it is a don't care), "name=value" or a boolean expression over the
inputs.

Example
-------
    from logisim.builder import Project
    from logisim import generate

    Project(generate.expression_circuit("A'B + C")).save("expr.circ")
    generate.alu(width=4).save("alu.circ")
    fsm = generate.fsm_from_table('''
        states: IDLE, RUN
        inputs: go
        outputs: busy
        moore: true
        transitions:
        IDLE,0,IDLE
        IDLE,1,RUN
        RUN,x,RUN
        state_outputs:
        IDLE,0
        RUN,1
    ''')
    generate.fsm_circuit(fsm).save("fsm.circ")
"""

from __future__ import annotations

import inspect
import itertools
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from . import sim
from .builder import Circuit, Node, Project, Signal, Terminal

__all__ = [
    "GenerateError", "ExprSyntaxError", "TruthTableError", "FSMError",
    "Var", "Const", "Not", "Op", "Expr",
    "parse_expression", "evaluate", "expression_variables", "format_expression",
    "expression_circuit", "expression_vectors",
    "TruthTable", "truth_table_from_dict", "truth_table_from_function",
    "truth_table_from_csv", "Cube", "prime_implicants", "minimal_cover",
    "minimize_function", "minimize_table", "cube_expression", "truth_table_circuit",
    "ripple_adder", "adder_subtractor", "mux_tree", "decoder", "comparator",
    "register_file", "alu", "counter",
    "Transition", "FSM", "fsm_from_dict", "fsm_from_table", "fsm_circuit",
    "FSMModel", "simulate_fsm",
]


# ====================================================================== errors

class GenerateError(ValueError):
    """Base class for every error this module raises."""


class ExprSyntaxError(GenerateError):
    """A boolean expression could not be parsed.  `column` is 1-based."""

    def __init__(self, message: str, column: int, text: str = "") -> None:
        self.column = column
        self.text = text
        detail = f"{message} at column {column}"
        if text:
            caret = " " * max(0, column - 1) + "^"
            super().__init__(f"{detail}\n  {text}\n  {caret}")
        else:
            super().__init__(detail)


class TruthTableError(GenerateError):
    """A truth table could not be read or is inconsistent."""


class FSMError(GenerateError):
    """A state machine description is invalid."""


# ================================================================ expressions

@dataclass(frozen=True)
class Var:
    """A variable of a boolean expression."""
    name: str


@dataclass(frozen=True)
class Const:
    """The constant 0 or 1."""
    value: int


@dataclass(frozen=True)
class Not:
    """Negation of `child`."""
    child: "Expr"


@dataclass(frozen=True)
class Op:
    """An n-ary AND, OR or XOR of `args`."""
    op: str
    args: tuple["Expr", ...]


Expr = Var | Const | Not | Op

_KEYWORDS = {"and", "or", "xor", "not"}
_CONST_WORDS = {"true": 1, "false": 0}
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class _Tok:
    kind: str  # ident | const | op | lparen | rparen
    text: str
    col: int


def _tokenize(text: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            low = word.lower()
            if low in _KEYWORDS:
                toks.append(_Tok("op", low, i + 1))
            elif low in _CONST_WORDS:
                toks.append(_Tok("const", str(_CONST_WORDS[low]), i + 1))
            else:
                toks.append(_Tok("ident", word, i + 1))
            i = j
        elif ch.isdigit():
            j = i
            while j < n and text[j].isdigit():
                j += 1
            word = text[i:j]
            if word not in ("0", "1"):
                raise ExprSyntaxError(f"only the constants 0 and 1 are supported, got {word!r}",
                                     i + 1, text)
            toks.append(_Tok("const", word, i + 1))
            i = j
        elif ch == "'":
            toks.append(_Tok("op", "'", i + 1))
            i += 1
        elif ch in "+|":
            toks.append(_Tok("op", "or", i + 1))
            i += 1
        elif ch == "^":
            toks.append(_Tok("op", "xor", i + 1))
            i += 1
        elif ch in "*&.":
            toks.append(_Tok("op", "and", i + 1))
            i += 1
        elif ch in "~!":
            toks.append(_Tok("op", "not", i + 1))
            i += 1
        elif ch == "(":
            toks.append(_Tok("lparen", "(", i + 1))
            i += 1
        elif ch == ")":
            toks.append(_Tok("rparen", ")", i + 1))
            i += 1
        else:
            raise ExprSyntaxError(f"unexpected character {ch!r}", i + 1, text)
    return toks


class _Parser:
    """Recursive descent parser for boolean expressions.

    Precedence (loosest first): OR < XOR < AND < NOT.
    Juxtaposition is an AND, so `A'B` is `~A & B` and `(A+B)(C+D)` is an AND.
    """

    def __init__(self, text: str, split_vars: bool) -> None:
        self.text = text
        self.split_vars = split_vars
        self.toks = _tokenize(text)
        self.pos = 0

    # -- token helpers
    def peek(self) -> _Tok | None:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def take(self) -> _Tok:
        tok = self.peek()
        if tok is None:
            raise ExprSyntaxError("unexpected end of expression", len(self.text) + 1, self.text)
        self.pos += 1
        return tok

    def error(self, message: str, tok: _Tok | None = None) -> ExprSyntaxError:
        col = tok.col if tok is not None else len(self.text) + 1
        return ExprSyntaxError(message, col, self.text)

    # -- grammar
    def parse(self) -> Expr:
        if not self.toks:
            raise ExprSyntaxError("empty expression", 1, self.text)
        expr = self.parse_or()
        rest = self.peek()
        if rest is not None:
            raise self.error(f"unexpected {rest.text!r}", rest)
        return expr

    def parse_or(self) -> Expr:
        left = self.parse_xor()
        while True:
            tok = self.peek()
            if tok is None or tok.kind != "op" or tok.text != "or":
                return left
            self.take()
            left = Op("or", (left, self.parse_xor()))

    def parse_xor(self) -> Expr:
        left = self.parse_and()
        while True:
            tok = self.peek()
            if tok is None or tok.kind != "op" or tok.text != "xor":
                return left
            self.take()
            left = Op("xor", (left, self.parse_and()))

    def parse_and(self) -> Expr:
        left = self.parse_not()
        while True:
            tok = self.peek()
            if tok is None:
                return left
            if tok.kind == "op" and tok.text == "and":
                self.take()
                left = Op("and", (left, self.parse_not()))
                continue
            if self._starts_operand(tok):
                # juxtaposition means AND
                left = Op("and", (left, self.parse_not()))
                continue
            return left

    def _starts_operand(self, tok: _Tok) -> bool:
        if tok.kind in ("ident", "const", "lparen"):
            return True
        if tok.kind == "op" and tok.text == "not":
            return False  # `not` is a prefix operator, never an operand
        return False

    def parse_not(self) -> Expr:
        tok = self.peek()
        if tok is not None and tok.kind == "op" and tok.text == "not":
            self.take()
            return Not(self.parse_not())
        if self.split_vars and tok is not None and tok.kind == "ident" and tok.text.isalpha():
            return self.parse_split_ident()
        return self.parse_postfix()

    def parse_split_ident(self) -> Expr:
        """Textbook mode: every letter of an identifier is a variable of its own."""
        word = self.take().text
        parts: list[Expr] = [Var(ch) for ch in word]
        # a postfix NOT belongs to the last letter of the group: AB' is A and not B
        while True:
            tok = self.peek()
            if tok is None or tok.kind != "op" or tok.text != "'":
                break
            self.take()
            parts[-1] = Not(parts[-1])
        return parts[0] if len(parts) == 1 else Op("and", tuple(parts))

    def parse_postfix(self) -> Expr:
        expr = self.parse_primary()
        while True:
            tok = self.peek()
            if tok is None or tok.kind != "op" or tok.text != "'":
                return expr
            self.take()
            expr = Not(expr)

    def parse_primary(self) -> Expr:
        tok = self.peek()
        if tok is None:
            raise self.error("expected a variable, a constant or '('")
        if tok.kind == "const":
            self.take()
            return Const(int(tok.text))
        if tok.kind == "ident":
            self.take()
            return Var(tok.text)
        if tok.kind == "lparen":
            self.take()
            expr = self.parse_or()
            close = self.peek()
            if close is None or close.kind != "rparen":
                raise self.error("missing ')'", close)
            self.take()
            return expr
        raise self.error(f"expected a variable, constant or '(' but found {tok.text!r}", tok)


def parse_expression(text: str, *, split_vars: bool = False) -> Expr:
    """Parse a boolean expression into an expression tree.

    Accepted syntax (textbook and Python-ish):

    ===========  =========================================================
    NOT          postfix `'` (`A'`), prefix `~`, `!` or the word `not`
    AND          juxtaposition (`A'B`), `*`, `&`, `.` or the word `and`
    OR           `+`, `|` or the word `or`
    XOR          `^` or the word `xor`
    constants    `0`, `1`, `True`, `False`
    grouping     `( ... )`
    ===========  =========================================================

    Precedence is NOT > AND > XOR > OR.  Identifiers are
    `[A-Za-z_][A-Za-z0-9_]*`, so `AB` is one variable; with
    `split_vars=True` every letter of an identifier is a variable of its own
    (`AB+C` then means `A and B or C`).

    Keywords are lower case only, so `AND` stays an identifier.

    Raises ExprSyntaxError (with a `column` attribute) on bad input.
    """
    return _Parser(text, split_vars).parse()


def evaluate(expr: Expr | str, values: Mapping[str, int]) -> int:
    """Evaluate an expression for one assignment of its variables -> 0 or 1."""
    if isinstance(expr, str):
        expr = parse_expression(expr)
    if isinstance(expr, Var):
        if expr.name not in values:
            raise GenerateError(f"no value for variable {expr.name!r}")
        return 1 if values[expr.name] else 0
    if isinstance(expr, Const):
        return 1 if expr.value else 0
    if isinstance(expr, Not):
        return 1 - evaluate(expr.child, values)
    if isinstance(expr, Op):
        parts = [evaluate(a, values) for a in expr.args]
        if expr.op == "and":
            return 1 if all(parts) else 0
        if expr.op == "or":
            return 1 if any(parts) else 0
        if expr.op == "xor":
            return sum(parts) & 1
        raise GenerateError(f"unknown operator {expr.op!r}")
    raise GenerateError(f"not an expression: {expr!r}")


def expression_variables(expr: Expr | str) -> list[str]:
    """The variables of an expression, in order of first appearance."""
    if isinstance(expr, str):
        expr = parse_expression(expr)
    out: list[str] = []

    def walk(e: Expr) -> None:
        if isinstance(e, Var):
            if e.name not in out:
                out.append(e.name)
        elif isinstance(e, Not):
            walk(e.child)
        elif isinstance(e, Op):
            for a in e.args:
                walk(a)

    walk(expr)
    return out


def format_expression(expr: Expr) -> str:
    """Render an expression in the canonical syntax (`*`, `+`, `^`, `~`)."""
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Const):
        return str(expr.value)
    if isinstance(expr, Not):
        return f"~{format_expression(expr.child)}"
    if isinstance(expr, Op):
        op = {"and": "*", "or": "+", "xor": "^"}[expr.op]
        return "(" + op.join(format_expression(a) for a in expr.args) + ")"
    raise GenerateError(f"not an expression: {expr!r}")


# ------------------------------------------------------------- gate emitter

def _expr_key(expr: Expr) -> tuple:
    if isinstance(expr, Var):
        return ("v", expr.name)
    if isinstance(expr, Const):
        return ("c", expr.value)
    if isinstance(expr, Not):
        return ("n", _expr_key(expr.child))
    if isinstance(expr, Op):
        return (expr.op, tuple(_expr_key(a) for a in expr.args))
    raise GenerateError(f"not an expression: {expr!r}")


class _Emitter:
    """Turns expression trees into gates in a circuit.

    Identical sub-expressions share one gate, so there is exactly one inverter
    per variable and one gate per distinct sub-expression.  Associative chains
    are flattened into a single n-input gate, split into a balanced tree when
    there are more terms than `max_fanin`.
    """

    def __init__(self, circ: Circuit, env: Mapping[str, Signal], max_fanin: int = 8) -> None:
        self.circ = circ
        self.env = env
        self.max_fanin = max(2, max_fanin)
        self.cache: dict[tuple, Signal] = {}

    def emit(self, expr: Expr) -> Signal:
        key = _expr_key(expr)
        if key in self.cache:
            return self.cache[key]
        if isinstance(expr, Var):
            if expr.name not in self.env:
                raise GenerateError(f"no signal for variable {expr.name!r}")
            sig: Signal = self.env[expr.name]
        elif isinstance(expr, Const):
            sig = self.circ.const(expr.value)
        elif isinstance(expr, Not):
            sig = self.circ.not_(self.emit(expr.child))
        elif isinstance(expr, Op):
            terms = [self.emit(a) for a in expr.args]
            if expr.op == "and":
                sig = self._combine("and", terms)
            elif expr.op == "or":
                sig = self._combine("or", terms)
            elif expr.op == "xor":
                sig = self._combine("xor", terms)
            else:
                raise GenerateError(f"unknown operator {expr.op!r}")
        else:
            raise GenerateError(f"not an expression: {expr!r}")
        self.cache[key] = sig
        return sig

    def _combine(self, op: str, terms: list[Signal]) -> Signal:
        if len(terms) == 1:
            return terms[0]
        if len(terms) <= self.max_fanin:
            return getattr(self.circ, {"and": "and_", "or": "or_", "xor": "xor"}[op])(*terms)
        # split into a balanced tree of gates that each stay within the fan-in
        n = len(terms)
        groups = -(-n // self.max_fanin)
        parts = []
        for g in range(groups):
            chunk = terms[g * n // groups:(g + 1) * n // groups]
            parts.append(getattr(self.circ, {"and": "and_", "or": "or_", "xor": "xor"}[op])(*chunk))
        return self._combine(op, parts)


def _and_expr(parts: Sequence[Expr]) -> Expr:
    if not parts:
        return Const(1)
    if len(parts) == 1:
        return parts[0]
    return Op("and", tuple(parts))


def _or_expr(parts: Sequence[Expr]) -> Expr:
    if not parts:
        return Const(0)
    if len(parts) == 1:
        return parts[0]
    return Op("or", tuple(parts))


def _probe_unread(circ: Circuit) -> None:
    """Give every otherwise unread output a probe.

    Logisim's linter warns about outputs nothing reads; a probe is the
    cheapest way to say "this signal is deliberate but unused".
    """
    driven = circ.driven()
    for node in circ.nodes:
        try:
            names = node.port_names()
        except Exception:
            continue
        outs = []
        for port in names:
            try:
                term = Terminal(node, port)
            except Exception:
                continue
            if term.is_output:
                outs.append(term)
        if not outs:
            continue
        if any(driven.get((id(node), t.port)) for t in outs):
            continue
        circ.probe(outs[0])


# ------------------------------------------------------- expression circuits

def expression_circuit(expression: str | Expr | Mapping[str, str | Expr], *,
                       name: str = "expr", output: str = "out", inputs: Sequence[str] | None = None,
                       spacing: str = "roomy", use_tunnels: bool = False,
                       max_fanin: int = 8) -> Circuit:
    """Build a circuit from one expression or a dict of {output: expression}.

    `expression` may be

    * a string or expression tree -> one output pin named `output`;
    * a mapping ``{output name: expression}`` -> one output pin per entry.

    Input pins are created for the variables that occur, ordered by first
    appearance or, when `inputs` is given, in that order (a variable that is
    listed but never used still gets a pin, terminated with a probe).
    """
    if isinstance(expression, Mapping):
        outputs = {str(k): _as_expr(v) for k, v in expression.items()}
    else:
        outputs = {output: _as_expr(expression)}
    if not outputs:
        raise GenerateError("no expression to build")

    used: list[str] = []
    for expr in outputs.values():
        for v in expression_variables(expr):
            if v not in used:
                used.append(v)
    if inputs is not None:
        order = [str(i) for i in inputs]
        missing = [v for v in used if v not in order]
        if missing:
            raise GenerateError(f"variables {missing} are not listed in `inputs`")
        ordered = order
    else:
        ordered = used

    circ = Circuit(name, spacing=spacing, use_tunnels=use_tunnels)
    env: dict[str, Signal] = {}
    for var in ordered:
        env[var] = circ.input(var)
    emitter = _Emitter(circ, env, max_fanin)
    for out_name, expr in outputs.items():
        circ.output(out_name, emitter.emit(expr))
    _probe_unread(circ)
    return circ


def _as_expr(value: str | Expr) -> Expr:
    return parse_expression(value) if isinstance(value, str) else value


def expression_vectors(path: str | os.PathLike, expression: str | Expr, *,
                       inputs: Sequence[str] | None = None, output: str = "out") -> Path:
    """Write a Logisim test-vector file for an expression.

    No .circ file is needed: the vectors hold the input combinations and the
    values the expression gives them.  The file can be used with Logisim's
    `-w` mode or its Test Vector window.
    """
    expr = _as_expr(expression)
    variables = expression_variables(expr)
    if inputs is not None:
        order = [str(i) for i in inputs]
        missing = [v for v in variables if v not in order]
        if missing:
            raise GenerateError(f"variables {missing} are not listed in `inputs`")
        variables = order
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", output):
        raise GenerateError(f"output name {output!r} cannot be a test-vector column")
    in_pins = [sim.PinInfo(v, 1) for v in variables]
    out_pins = [sim.PinInfo(output, 1)]
    rows = []
    for combo in itertools.product((0, 1), repeat=len(variables)):
        values = dict(zip(variables, combo))
        rows.append((values, {output: evaluate(expr, values)}))
    return sim.write_vectors(path, in_pins, out_pins, rows)


# ================================================================= truth tables

@dataclass
class TruthTable:
    """A truth table: input names, output names and one row per input assignment.

    `rows` maps an input assignment (a tuple of 0/1 in the order of `inputs`)
    to the value of each output; `None` marks a don't care.
    """
    inputs: list[str]
    outputs: list[str]
    rows: dict[tuple[int, ...], list[int | None]] = field(default_factory=dict)

    @property
    def n_inputs(self) -> int:
        return len(self.inputs)

    def value_sets(self, output: str | int) -> tuple[set[int], set[int], set[int]]:
        """(minterms, maxterms, don't-cares) of one output, as minterm numbers."""
        if isinstance(output, int):
            idx = output
        else:
            if output not in self.outputs:
                raise TruthTableError(f"no output {output!r}; outputs: {self.outputs}")
            idx = self.outputs.index(output)
        ones: set[int] = set()
        zeros: set[int] = set()
        dcs: set[int] = set()
        n = self.n_inputs
        for assign, vals in self.rows.items():
            num = _assign_number(assign)
            v = vals[idx] if idx < len(vals) else None
            if v is None:
                dcs.add(num)
            elif v:
                ones.add(num)
            else:
                zeros.add(num)
        return ones, zeros, dcs


def _assign_number(assign: Sequence[int]) -> int:
    """Minterm number of an assignment: variable 0 is the most significant bit."""
    n = len(assign)
    return sum(int(v) << (n - 1 - i) for i, v in enumerate(assign))


def _number_assign(num: int, n: int) -> tuple[int, ...]:
    return tuple((num >> (n - 1 - i)) & 1 for i in range(n))


def _complete_rows(rows: dict[tuple[int, ...], list[int | None]], inputs: Sequence[str],
                   outputs: Sequence[str]) -> dict[tuple[int, ...], list[int | None]]:
    """Fill in the assignments the table does not mention; they count as 0."""
    n = len(inputs)
    full: dict[tuple[int, ...], list[int | None]] = {}
    for combo in itertools.product((0, 1), repeat=n):
        full[combo] = rows.get(combo, [0] * len(outputs))
    return full


def truth_table_from_dict(data: Mapping, *, inputs: Sequence[str] | None = None,
                          outputs: Sequence[str] | None = None) -> TruthTable:
    """Build a truth table from a dict.

    Keys are either an input assignment (a tuple of 0/1) or a minterm number
    (an int; variable 0 is the most significant bit).  Values are a single
    0/1 (or None for a don't care) or a sequence with one entry per output.
    Assignments that are not listed count as 0.
    """
    if not data:
        raise TruthTableError("the truth table is empty")
    keys = list(data)
    if inputs is not None:
        in_names = [str(i) for i in inputs]
    elif all(isinstance(k, (tuple, list)) for k in keys):
        in_names = [f"in{i}" for i in range(len(keys[0]))]
    else:
        raise TruthTableError("`inputs` is required when the keys are minterm numbers")
    n = len(in_names)

    values = [data[k] for k in keys]
    if outputs is not None:
        out_names = [str(o) for o in outputs]
    elif all(isinstance(v, (tuple, list)) for v in values):
        out_names = [f"Y{i}" for i in range(len(values[0]))]
    else:
        out_names = ["Y"]
    n_out = len(out_names)

    rows: dict[tuple[int, ...], list[int | None]] = {}
    for key in keys:
        if isinstance(key, (tuple, list)):
            if len(key) != n:
                raise TruthTableError(f"assignment {key!r} does not have {n} inputs")
            assign = tuple(int(v) for v in key)
        elif isinstance(key, int):
            if key < 0 or key >= (1 << n):
                raise TruthTableError(f"minterm {key} is out of range for {n} inputs")
            assign = _number_assign(key, n)
        else:
            raise TruthTableError(f"cannot use {key!r} as a truth-table key")
        value = data[key]
        if isinstance(value, (tuple, list)):
            if len(value) != n_out:
                raise TruthTableError(f"value {value!r} does not have {n_out} outputs")
            vals: list[int | None] = [None if v is None else int(v) for v in value]
        else:
            vals = [None if value is None else int(value)]
            while len(vals) < n_out:
                vals.append(0)
        rows[assign] = vals
    return TruthTable(in_names, out_names, _complete_rows(rows, in_names, out_names))


def truth_table_from_function(fn: Callable, inputs: Sequence[str], *,
                              outputs: Sequence[str] | None = None) -> TruthTable:
    """Build a truth table from a Python function of the inputs.

    `fn` is called either with the input values as positional arguments, with
    them as keyword arguments (when the parameter names match the input names)
    or with a single dict -- whichever fits its signature.  It returns a 0/1
    value or a sequence with one entry per output; None means don't care.
    """
    in_names = [str(i) for i in inputs]
    n = len(in_names)
    values = []
    for combo in itertools.product((0, 1), repeat=n):
        values.append(_call_table_function(fn, in_names, combo))
    n_out = len(outputs) if outputs is not None else (
        len(values[0]) if isinstance(values[0], (tuple, list)) else 1)
    if outputs is not None:
        out_names = [str(o) for o in outputs]
    elif n_out == 1:
        out_names = ["Y"]
    else:
        out_names = [f"Y{i}" for i in range(n_out)]
    rows: dict[tuple[int, ...], list[int | None]] = {}
    for combo, value in zip(itertools.product((0, 1), repeat=n), values):
        if isinstance(value, (tuple, list)):
            vals = [None if v is None else int(v) for v in value]
        else:
            vals = [None if value is None else int(value)]
        while len(vals) < n_out:
            vals.append(0)
        rows[combo] = vals
    return TruthTable(in_names, out_names, _complete_rows(rows, in_names, out_names))


def _call_table_function(fn: Callable, names: Sequence[str], combo: Sequence[int]):
    """Call a truth-table function with the input values.

    The function is called with the values as keyword arguments when it takes
    **kwargs or when its parameter names are the input names, with the values
    as positional arguments when it takes as many parameters as there are
    inputs, and with a single dict when it takes a single parameter that is
    not named after an input.
    """
    params = inspect.signature(fn).parameters
    positional = [p for p in params.values()
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_kwargs or (positional and {p.name for p in positional} == set(names)):
        return fn(**dict(zip(names, combo)))
    if len(positional) == len(combo) and len(positional) != 1:
        return fn(*combo)
    if len(positional) == 1:
        return fn(dict(zip(names, combo)))
    return fn(*combo)


def _split_cells(line: str) -> list[str]:
    if "," in line:
        return [c.strip() for c in line.split(",")]
    return line.split()


def _cell_value(text: str, where: str) -> int | None:
    t = text.strip().lower()
    if t == "0":
        return 0
    if t == "1":
        return 1
    if t in ("x", "-", "dc", "don't care"):
        return None
    raise TruthTableError(f"{where}: {text.strip()!r} is not 0, 1 or a don't care (x/-)")


def truth_table_from_csv(source: str | os.PathLike, *,
                         outputs: Sequence[str] | None = None) -> TruthTable:
    """Read a truth table from CSV text or a CSV file.

    The first row names the columns.  The output columns are either given with
    `outputs` or separated from the input columns by a `|` column.  Cells are
    `0`, `1` or a don't care (`x`, `X` or `-`).  Columns may be separated by
    commas or by whitespace; `#` starts a comment line.
    """
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    elif isinstance(source, str) and "\n" not in source and os.path.exists(source):
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = str(source)

    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        raise TruthTableError("the truth table is empty")
    header = _split_cells(lines[0])
    if "|" in header:
        cut = header.index("|")
        in_names = header[:cut]
        out_names = header[cut + 1:]
    elif outputs is not None:
        out_names = [str(o) for o in outputs]
        in_names = [h for h in header if h not in out_names]
    else:
        raise TruthTableError("say which columns are outputs: pass `outputs=` or use a '|' column")
    if not in_names:
        raise TruthTableError("the truth table has no input columns")
    if outputs is not None and "|" not in header:
        out_names = [str(o) for o in outputs]
    n_in, n_out = len(in_names), len(out_names)
    rows: dict[tuple[int, ...], list[int | None]] = {}
    for lineno, line in enumerate(lines[1:], start=2):
        cells = _split_cells(line)
        if "|" in cells:
            cells = [c for c in cells if c != "|"]
        if len(cells) != n_in + n_out:
            raise TruthTableError(f"row {lineno}: expected {n_in + n_out} cells, got {len(cells)}")
        assign = []
        for i, name in enumerate(in_names):
            v = _cell_value(cells[i], f"row {lineno}, column {name}")
            if v is None:
                raise TruthTableError(f"row {lineno}, column {name}: an input cannot be a don't care")
            assign.append(v)
        vals = [_cell_value(cells[n_in + i], f"row {lineno}, column {out_names[i]}")
                for i in range(n_out)]
        rows[tuple(assign)] = vals
    return TruthTable(in_names, out_names, _complete_rows(rows, in_names, out_names))


# ============================================================ Quine-McCluskey

@dataclass(frozen=True)
class Cube:
    """An implicant: the variables it fixes (`value`) and the ones it ignores (`mask`).

    Bit n-1-i of `value`/`mask` belongs to variables[i]; a set mask bit means
    "this variable does not matter here".
    """
    value: int
    mask: int

    def covers(self, minterm: int) -> bool:
        return (minterm & ~self.mask) == (self.value & ~self.mask)

    def literals(self, variables: Sequence[str]) -> list[str]:
        """The literals of the cube in the SOP sense (a 0 bit is negated)."""
        n = len(variables)
        out = []
        for i, name in enumerate(variables):
            bit = n - 1 - i
            if (self.mask >> bit) & 1:
                continue
            out.append(name if (self.value >> bit) & 1 else f"~{name}")
        return out

    def __str__(self) -> str:
        n = self.mask.bit_length()
        bits = "".join("-" if (self.mask >> i) & 1 else ("1" if (self.value >> i) & 1 else "0")
                       for i in reversed(range(n)))
        return bits


# Petrick's method can blow up; above these limits the greedy cover is used.
MAX_PETRICK_TERMS = 20000
MAX_PRIMES = 512


def prime_implicants(minterms: Iterable[int], dontcares: Iterable[int] = (),
                     nvars: int = 0) -> list[Cube]:
    """All prime implicants of a function given by its minterms and don't-cares.

    An implicant may cover don't-cares but never a 0 of the function, so the
    don't-cares are used while combining and only care minterms have to be
    covered afterwards.
    """
    if nvars < 0:
        raise TruthTableError("nvars must not be negative")
    ones = {m for m in minterms}
    dcs = {m for m in dontcares}
    limit = 1 << nvars if nvars else 1
    for m in ones | dcs:
        if m < 0 or m >= limit:
            raise TruthTableError(f"minterm {m} is out of range for {nvars} variables")
    current = {(m, 0) for m in (ones | dcs)}
    primes: set[tuple[int, int]] = set()
    while current:
        used: set[tuple[int, int]] = set()
        nxt: set[tuple[int, int]] = set()
        items = sorted(current)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                (a, ma), (b, mb) = items[i], items[j]
                if ma != mb:
                    continue
                diff = a ^ b
                if diff == 0 or (diff & (diff - 1)):
                    continue  # more than one bit differs
                if ma & diff:
                    continue  # the differing bit is already a don't care
                nxt.add((a & ~diff, ma | diff))
                used.add((a, ma))
                used.add((b, mb))
        primes |= (current - used)
        current = nxt
    out = [Cube(v, mk) for (v, mk) in primes if any(Cube(v, mk).covers(m) for m in ones)]
    out.sort(key=lambda c: (bin(c.mask).count("1"), c.value, c.mask))
    return out


def minimal_cover(minterms: Iterable[int], primes: Sequence[Cube]) -> list[Cube]:
    """A minimum set of implicants that covers every minterm.

    The cover is exact (Petrick's method); for very large charts a greedy
    cover is used instead, which is documented in the module docstring of
    `minimize_function`.
    """
    care = sorted(set(minterms))
    if not care:
        return []
    primes = list(primes)
    chart = {m: [i for i, p in enumerate(primes) if p.covers(m)] for m in care}
    for m, options in chart.items():
        if not options:
            raise TruthTableError(f"minterm {m} is covered by no implicant")
    if len(primes) > MAX_PRIMES:
        return _greedy_cover(care, primes)
    terms: list[frozenset[int]] = [frozenset()]
    for m in care:
        grown: set[frozenset[int]] = set()
        for term in terms:
            for i in chart[m]:
                grown.add(term | {i})
        terms = _prune_terms(grown)
        if len(terms) > MAX_PETRICK_TERMS:
            return _greedy_cover(care, primes)
    best = min(terms, key=lambda t: (len(t), sorted(t)))
    return [primes[i] for i in sorted(best)]


def _prune_terms(terms: Iterable[frozenset[int]]) -> list[frozenset[int]]:
    """Drop the products that contain another product (they can never be cheapest)."""
    keep: list[frozenset[int]] = []
    for term in sorted(terms, key=lambda t: (len(t), sorted(t))):
        if any(other <= term for other in keep):
            continue
        keep.append(term)
    return keep


def _greedy_cover(care: Sequence[int], primes: Sequence[Cube]) -> list[Cube]:
    """Greedy minimum-ish cover, used when Petrick's method would explode."""
    remaining = set(care)
    chosen: list[Cube] = []
    while remaining:
        best = None
        for i, prime in enumerate(primes):
            covered = {m for m in remaining if prime.covers(m)}
            if not covered:
                continue
            key = (-len(covered), i)
            if best is None or key < best[0]:
                best = (key, i, covered)
        if best is None:
            raise TruthTableError(f"minterm {sorted(remaining)[0]} is covered by no implicant")
        chosen.append(primes[best[1]])
        remaining -= best[2]
    return chosen


def minimize_function(minterms: Iterable[int], dontcares: Iterable[int] = (),
                      nvars: int = 0) -> list[Cube]:
    """Minimize a function: the implicants of an exact minimum cover.

    The cover is minimum in the number of implicants.  When the chart is too
    large for Petrick's method (see MAX_PETRICK_TERMS) a greedy cover is
    returned instead, which is still correct but not necessarily minimal.
    """
    ones = set(minterms)
    primes = prime_implicants(ones, dontcares, nvars)
    return minimal_cover(ones, primes)


def minimize_table(table: TruthTable, *, output: str | None = None,
                   form: str = "sop") -> dict[str, list[Cube]]:
    """Minimize every output of a truth table.

    With `form="sop"` the returned cubes are the implicants of the function
    itself; with `form="pos"` they are the implicants of its complement, one
    cube per maxterm.
    """
    if form not in ("sop", "pos"):
        raise TruthTableError(f"unknown form {form!r}; use 'sop' or 'pos'")
    names = [output] if output is not None else list(table.outputs)
    result: dict[str, list[Cube]] = {}
    for name in names:
        ones, zeros, dcs = table.value_sets(name)
        if form == "sop":
            result[name] = minimize_function(ones, dcs, table.n_inputs) if ones else []
        else:
            result[name] = minimize_function(zeros, dcs, table.n_inputs) if zeros else []
    return result


def _cube_expr(cube: Cube, variables: Sequence[str], *, complement: bool) -> Expr:
    """The product (SOP) or sum (POS) of the literals of a cube."""
    n = len(variables)
    parts: list[Expr] = []
    for i, name in enumerate(variables):
        bit = n - 1 - i
        if (cube.mask >> bit) & 1:
            continue
        one = bool((cube.value >> bit) & 1)
        if complement:
            one = not one
        parts.append(Var(name) if one else Not(Var(name)))
    return _and_expr(parts) if not complement else _or_expr(parts)


def cube_expression(cubes: Sequence[Cube], variables: Sequence[str], *,
                    form: str = "sop") -> Expr:
    """Render a cover of implicants as a boolean expression.

    With `form="sop"` the cubes are the implicants of the function itself and
    the result is a sum of products; with `form="pos"` they are the implicants
    of its complement and the result is a product of sums.
    """
    if form == "sop":
        return _or_expr([_cube_expr(c, variables, complement=False) for c in cubes])
    if form == "pos":
        return _and_expr([_cube_expr(c, variables, complement=True) for c in cubes])
    raise TruthTableError(f"unknown form {form!r}; use 'sop' or 'pos'")


def _table_expression(table: TruthTable, output: str, form: str) -> Expr:
    ones, zeros, dcs = table.value_sets(output)
    variables = list(table.inputs)
    if not ones:
        return Const(0)
    if form == "sop":
        cubes = minimize_function(ones, dcs, table.n_inputs)
        return cube_expression(cubes, variables, form="sop")
    if not zeros:
        return Const(1)
    cubes = minimize_function(zeros, dcs, table.n_inputs)
    return cube_expression(cubes, variables, form="pos")


def truth_table_circuit(table: TruthTable, *, name: str = "truth_table", form: str = "sop",
                        spacing: str = "roomy", use_tunnels: bool = False,
                        max_fanin: int = 8) -> Circuit:
    """Build a circuit from a truth table.

    `form="sop"` builds a sum of products, `form="pos"` a product of sums.
    Every output gets its own minimized expression; the logic is shared
    between the outputs where it is the same.
    """
    if form not in ("sop", "pos"):
        raise TruthTableError(f"unknown form {form!r}; use 'sop' or 'pos'")
    circ = Circuit(name, spacing=spacing, use_tunnels=use_tunnels)
    env = {var: circ.input(var) for var in table.inputs}
    emitter = _Emitter(circ, env, max_fanin)
    for out_name in table.outputs:
        expr = _table_expression(table, out_name, form)
        circ.output(out_name, emitter.emit(expr))
    _probe_unread(circ)
    return circ


# ================================================================== blocks

def _full_adder(spacing: str = "roomy", use_tunnels: bool = False) -> Circuit:
    """The classic one-bit full adder (A, B, Cin -> S, Cout)."""
    c = Circuit("full_adder", spacing=spacing, use_tunnels=use_tunnels)
    a, b, cin = c.input("A"), c.input("B"), c.input("Cin")
    x1 = c.xor(a, b)
    c.output("S", c.xor(x1, cin))
    c.output("Cout", c.or_(c.and_(a, b), c.and_(x1, cin)))
    return c


def ripple_adder(width: int = 4, *, spacing: str = "roomy", use_tunnels: bool = False) -> Project:
    """An N-bit ripple-carry adder built from full-adder subcircuits.

    Ports: A, B, Cin (inputs), S and Cout (outputs).
    """
    if width < 1:
        raise GenerateError("width must be at least 1")
    fa = _full_adder(spacing, use_tunnels)
    top = Circuit("ripple_adder", spacing=spacing, use_tunnels=use_tunnels)
    proj = Project(fa, top, main="ripple_adder")
    a = top.input("A", width)
    b = top.input("B", width)
    cin = top.input("Cin")
    ab, bb = top.split(a), top.split(b)
    carry: Signal = cin
    sums: list[Signal] = []
    for i in range(width):
        unit = top.instance(fa, A=ab[i], B=bb[i], Cin=carry)
        sums.append(unit.S)
        carry = unit.Cout
    top.output("S", top.merge(sums), width=width)
    top.output("Cout", carry)
    return proj


def adder_subtractor(width: int = 4, *, spacing: str = "roomy",
                     use_tunnels: bool = False) -> Project:
    """An N-bit adder/subtractor.

    With `sub` low the circuit adds A and B; with `sub` high it computes A-B
    (B is inverted and the carry-in is set).  Cout is the carry out of that
    addition, so for a subtraction it is the inverse of the borrow.
    """
    if width < 1:
        raise GenerateError("width must be at least 1")
    fa = _full_adder(spacing, use_tunnels)
    top = Circuit("adder_subtractor", spacing=spacing, use_tunnels=use_tunnels)
    proj = Project(fa, top, main="adder_subtractor")
    a = top.input("A", width)
    b = top.input("B", width)
    sub = top.input("sub")
    ab, bb = top.split(a), top.split(b)
    eff = [top.xor(bb[i], sub) for i in range(width)]
    carry: Signal = sub
    sums: list[Signal] = []
    for i in range(width):
        unit = top.instance(fa, A=ab[i], B=eff[i], Cin=carry)
        sums.append(unit.S)
        carry = unit.Cout
    top.output("S", top.merge(sums), width=width)
    top.output("Cout", carry)
    return proj


def mux_tree(data_inputs: int = 4, width: int = 1, *, name: str = "mux_tree",
             spacing: str = "roomy", use_tunnels: bool = False) -> Circuit:
    """A multiplexer built from a tree of 2:1 multiplexers.

    Ports: d0 .. dN-1 (data), sel (select) and y (the result).  `sel` is
    ceil(log2(N)) bits wide; inputs beyond N are padded with zeros.
    """
    if data_inputs < 1:
        raise GenerateError("data_inputs must be at least 1")
    if width < 1:
        raise GenerateError("width must be at least 1")
    c = Circuit(name, spacing=spacing, use_tunnels=use_tunnels)
    data = [c.input(f"d{i}", width) for i in range(data_inputs)]
    sel_bits = max(1, (data_inputs - 1).bit_length())
    sel = c.input("sel", sel_bits)
    sel_bits_sig = c.split(sel)
    items: list[Signal] = list(data)
    while len(items) < (1 << sel_bits):
        items.append(c.const(0, width))

    def build(level: int, group: list[Signal]) -> Signal:
        if len(group) == 1:
            return group[0]
        half = len(group) // 2
        low = build(level + 1, group[:half])
        high = build(level + 1, group[half:])
        # the least significant select bit chooses inside a pair, the more
        # significant ones choose between the pairs
        return c.mux(sel_bits_sig[sel_bits - 1 - level], low, high, width=width)

    c.output("y", build(0, items), width=width)
    _probe_unread(c)
    return c


def decoder(inputs: int = 3, *, name: str = "decoder", spacing: str = "roomy",
            use_tunnels: bool = False) -> Circuit:
    """An n-to-2^n decoder built from gates.

    Ports: in0 .. in(n-1) and out0 .. out(2^n-1); exactly one output is high.
    """
    if inputs < 1:
        raise GenerateError("inputs must be at least 1")
    c = Circuit(name, spacing=spacing, use_tunnels=use_tunnels)
    ins = [c.input(f"in{i}") for i in range(inputs)]
    inv = [c.not_(ins[i]) for i in range(inputs)]
    for code in range(1 << inputs):
        if inputs == 1:
            sig: Signal = ins[0] if code else inv[0]
        else:
            literals = [ins[i] if (code >> i) & 1 else inv[i] for i in range(inputs)]
            sig = c.and_(*literals)
        c.output(f"out{code}", sig)
    _probe_unread(c)
    return c


def comparator(width: int = 4, *, spacing: str = "roomy", use_tunnels: bool = False) -> Project:
    """An N-bit unsigned comparator built from gates.

    Ports: A, B (inputs) and eq, lt, gt (outputs).
    """
    if width < 1:
        raise GenerateError("width must be at least 1")
    bit = Circuit("cmp_bit", spacing=spacing, use_tunnels=use_tunnels)
    top = Circuit("comparator", spacing=spacing, use_tunnels=use_tunnels)
    proj = Project(bit, top, main="comparator")
    a, b, carry = bit.input("a"), bit.input("b"), bit.input("cin")
    nb = bit.not_(b)
    # carry of a + ~b + 1: it is 1 exactly when a >= b
    cout = bit.or_(bit.and_(a, nb), bit.and_(carry, bit.or_(a, nb)))
    ne = bit.xor(a, b)
    bit.output("cout", cout)
    bit.output("ne", ne)

    av = top.input("A", width)
    bv = top.input("B", width)
    ab, bb = top.split(av), top.split(bv)
    carry: Signal = top.const(1)  # a + ~b + 1
    nes: list[Signal] = []
    for i in range(width):
        unit = top.instance(bit, a=ab[i], b=bb[i], cin=carry)
        carry = unit.cout
        nes.append(unit.ne)
    eq = top.nor(*nes)
    ge = carry
    lt = top.not_(ge)
    gt = top.and_(ge, top.not_(eq))
    top.output("eq", eq)
    top.output("lt", lt)
    top.output("gt", gt)
    return proj


def register_file(registers: int = 4, width: int = 8, *, read_ports: int = 1,
                  spacing: str = "roomy", use_tunnels: bool = False) -> Project:
    """A register file: `registers` registers of `width` bits.

    One write port (we, waddr, wdata) and one or two read ports (raddr0,
    raddr1 -> rdata0, rdata1).  The address is wide enough for the registers
    and rounded up to a power of two.
    """
    if registers < 1:
        raise GenerateError("registers must be at least 1")
    if width < 1:
        raise GenerateError("width must be at least 1")
    if read_ports not in (1, 2):
        raise GenerateError("read_ports must be 1 or 2")
    addr_bits = max(1, (registers - 1).bit_length())
    padded = 1 << addr_bits

    top = Circuit("register_file", spacing=spacing, use_tunnels=use_tunnels)
    clk = top.input("clk")
    we = top.input("we")
    waddr = top.input("waddr", addr_bits)
    wdata = top.input("wdata", width)
    raddrs = [top.input(f"raddr{i}", addr_bits) for i in range(read_ports)]

    dec = top.decoder(waddr)
    values: list[Signal] = []
    for i in range(registers):
        en = top.and_(we, dec[f"out{i}"])
        reg = top.register(wdata, clk, en=en, width=width, label=f"R{i}")
        values.append(reg)
    for port in range(read_ports):
        items = list(values)
        if padded > registers:
            items += [top.const(0, width)] * (padded - registers)
        bits = top.split(raddrs[port])

        def build(level: int, group: list[Signal]) -> Signal:
            if len(group) == 1:
                return group[0]
            half = len(group) // 2
            return top.mux(bits[addr_bits - 1 - level], build(level + 1, group[:half]),
                           build(level + 1, group[half:]), width=width)

        top.output(f"rdata{port}", build(0, items), width=width)
    return Project(top, main="register_file")


def _adder_circuit(name: str, spacing: str, use_tunnels: bool) -> Circuit:
    """An empty ripple-carry adder subcircuit (filled in by _fill_adder)."""
    return Circuit(name, spacing=spacing, use_tunnels=use_tunnels)


def _fill_adder(c: Circuit, width: int, fa: Circuit) -> None:
    """Fill an adder subcircuit: A, B, Cin -> S, Cout."""
    a = c.input("A", width)
    b = c.input("B", width)
    cin = c.input("Cin")
    ab, bb = c.split(a), c.split(b)
    carry: Signal = cin
    sums: list[Signal] = []
    for i in range(width):
        unit = c.instance(fa, A=ab[i], B=bb[i], Cin=carry)
        sums.append(unit.S)
        carry = unit.Cout
    c.output("S", c.merge(sums), width=width)
    c.output("Cout", carry)


def _inverter_circuit(name: str, width: int, spacing: str, use_tunnels: bool) -> Circuit:
    """A bus inverter as a subcircuit: B -> ~B."""
    c = Circuit(name, spacing=spacing, use_tunnels=use_tunnels)
    b = c.input("B", width)
    bits = c.split(b)
    c.output("nB", c.merge([c.not_(bits[i]) for i in range(width)]), width=width)
    return c


def alu(width: int = 4, *, spacing: str = "roomy", use_tunnels: bool = False) -> Project:
    """A simple N-bit ALU.

    `op` selects the operation: 000 add, 001 subtract, 010 and, 011 or,
    100 xor (the other codes give 0).  Outputs are Y (the result), Z (the
    zero flag) and Cout (the carry out of the add or the subtract).
    """
    if width < 1:
        raise GenerateError("width must be at least 1")
    fa = _full_adder(spacing, use_tunnels)
    adder = _adder_circuit("alu_adder", spacing, use_tunnels)
    inv = _inverter_circuit("alu_inv", width, spacing, use_tunnels)
    top = Circuit("alu", spacing=spacing, use_tunnels=use_tunnels)
    proj = Project(top, fa, adder, inv, main="alu")
    _fill_adder(adder, width, fa)
    a = top.input("A", width)
    b = top.input("B", width)
    op = top.input("op", 3)
    op_bits = top.split(op)

    add = top.instance(adder, A=a, B=b, Cin=top.const(0))
    # a subtraction is a + ~b + 1
    sub = top.instance(adder, A=a, B=top.instance(inv, B=b).nB, Cin=top.const(1))
    and_res = top.and_(a, b)
    or_res = top.or_(a, b)
    xor_res = top.xor(a, b)
    zero = top.const(0, width)
    options = [add.S, sub.S, and_res, or_res, xor_res, zero, zero, zero]
    y = top.mux(op, *options, width=width)
    top.output("Y", y, width=width)
    y_bits = top.split(y)
    top.output("Z", top.nor(*y_bits))
    # the carry out belongs to the add and the subtract only
    cout = top.mux(op_bits[0], add.Cout, sub.Cout)
    cout = top.mux(op_bits[1], cout, top.const(0))
    cout = top.mux(op_bits[2], cout, top.const(0))
    top.output("Cout", cout)
    return proj


def counter(width: int = 4, *, clear: str = "sync", spacing: str = "roomy",
            use_tunnels: bool = False) -> Project:
    """An N-bit counter built from flip-flops.

    Ports: clk, en and clr (inputs) and Q (the count).  With `clear="sync"`
    the clear takes effect at the next clock edge, with `clear="async"` it
    clears the flip-flops at once.
    """
    if width < 1:
        raise GenerateError("width must be at least 1")
    if clear not in ("sync", "async"):
        raise GenerateError("clear must be 'sync' or 'async'")
    bit = Circuit("counter_bit", spacing=spacing, use_tunnels=use_tunnels)
    top = Circuit("counter", spacing=spacing, use_tunnels=use_tunnels)
    proj = Project(bit, top, main="counter")
    q = bit.input("q")
    carry = bit.input("carry")
    clrn = bit.input("clrn")
    d = bit.and_(bit.xor(q, carry), clrn)
    bit.output("d", d)
    bit.output("cout", bit.and_(carry, q))

    clk = top.input("clk")
    en = top.input("en")
    clr = top.input("clr")
    if clear == "sync":
        clrn: Signal = top.not_(clr)
        ffs = [top.dff(None, clk) for _ in range(width)]
    else:
        clrn = top.const(1)
        ffs = [top.dff(None, clk, reset=clr) for _ in range(width)]
    chain: Signal = en
    for i in range(width):
        unit = top.instance(bit, q=ffs[i].q, carry=chain, clrn=clrn)
        top.connect(unit.d, ffs[i].D)
        chain = unit.cout
    top.output("Q", top.merge([f.q for f in ffs]), width=width)
    return proj


# ====================================================================== FSM

@dataclass
class Transition:
    """One transition: from `state`, while `condition` holds, to `next`.

    `condition` is a boolean expression over the machine's inputs ("x" means
    always).  `outputs` are the Mealy outputs of this transition.
    """
    state: str
    condition: str
    next: str
    outputs: dict[str, int] = field(default_factory=dict)


@dataclass
class FSM:
    """A finite state machine.

    `transitions` lists the transitions of every state; a state without a
    matching transition stays where it is.  For a Moore machine the outputs
    of a state are in `state_outputs`, for a Mealy machine they belong to the
    transitions.  `encoding` is "binary", "onehot" or "gray"; `reset` adds a
    synchronous reset input named "rst".
    """
    states: list[str]
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    initial: str = ""
    encoding: str = "binary"
    moore: bool = True
    reset: bool = False
    state_outputs: dict[str, dict[str, int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states = [str(s) for s in self.states]
        self.inputs = [str(i) for i in self.inputs]
        self.outputs = [str(o) for o in self.outputs]
        self.transitions = [t if isinstance(t, Transition) else Transition(*t)
                            for t in self.transitions]
        if not self.states:
            raise FSMError("the state machine has no states")
        if len(set(self.states)) != len(self.states):
            raise FSMError("the state names are not unique")
        if not self.initial:
            self.initial = self.states[0]
        if self.initial not in self.states:
            raise FSMError(f"initial state {self.initial!r} is not one of {self.states}")
        if self.encoding not in ("binary", "onehot", "gray"):
            raise FSMError(f"unknown encoding {self.encoding!r}; use binary, onehot or gray")
        for t in self.transitions:
            if t.state not in self.states:
                raise FSMError(f"transition from unknown state {t.state!r}")
            if t.next not in self.states:
                raise FSMError(f"transition to unknown state {t.next!r}")
        self._conditions: list[Expr] = []

    # -- encoding
    @property
    def state_bits(self) -> int:
        """How many bits the state needs."""
        n = len(self.states)
        if self.encoding == "onehot":
            return n
        return max(1, (n - 1).bit_length())

    def encode(self, state: str) -> int:
        """The code of a state; bit len(state_bits)-1-i belongs to state bit i."""
        index = self.states.index(state)
        if self.encoding == "onehot":
            return 1 << (self.state_bits - 1 - index)
        if self.encoding == "gray":
            return index ^ (index >> 1)
        return index

    def decode(self, code: int) -> str | None:
        """The state a code stands for, or None for an unused code."""
        for state in self.states:
            if self.encode(state) == code:
                return state
        return None

    # -- conditions
    def condition_expressions(self) -> list[Expr]:
        """The parsed transition conditions."""
        if not self._conditions or len(self._conditions) != len(self.transitions):
            self._conditions = [_parse_condition(t.condition, self.inputs) for t in self.transitions]
        return self._conditions

    def matching_transition(self, state: str, values: Mapping[str, int]) -> Transition | None:
        """The first transition of `state` whose condition holds."""
        for t, expr in zip(self.transitions, self.condition_expressions()):
            if t.state == state and evaluate(expr, values):
                return t
        return None

    def outputs_of(self, state: str, values: Mapping[str, int]) -> dict[str, int]:
        """The outputs during a cycle in `state` with the given inputs."""
        if self.moore:
            return {o: int(self.state_outputs.get(state, {}).get(o, 0)) for o in self.outputs}
        tr = self.matching_transition(state, values)
        return {o: int(tr.outputs.get(o, 0)) if tr else 0 for o in self.outputs}

    def next_state(self, state: str, values: Mapping[str, int], reset: bool = False) -> str:
        """The state the machine moves to after a clock edge."""
        if reset:
            return self.initial
        tr = self.matching_transition(state, values)
        return tr.next if tr else state


_EQUALS_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([01])")


def _parse_condition(text: str, inputs: Sequence[str]) -> Expr:
    """Parse a transition condition.

    * ``name=value``  the input `name` has that value
    * ``0101``        the inputs have those values; ``x``/``-`` is a don't care
    * ``x``           always (also ``-``, ``*``, "always"); an input named x wins
    * anything else   a boolean expression over the inputs

    A bit string is only read as an input vector when it is as long as the
    list of inputs, so a one-input machine can use plain ``0`` and ``1``.
    """
    t = (text or "").strip()
    if not t:
        return Const(1)
    if "=" in t:
        m = _EQUALS_RE.fullmatch(t)
        if not m:
            raise FSMError(f"bad condition {text!r}; use 'name=0' or 'name=1'")
        name, value = m.group(1), int(m.group(2))
        if name not in inputs:
            raise FSMError(f"condition {text!r} uses the unknown input {name!r}")
        return Var(name) if value else Not(Var(name))
    if t.lower() in ("-", "*", "always") or (t.lower() == "x" and t not in inputs):
        return Const(1)
    if t in inputs:
        return Var(t)
    if len(t) == len(inputs) and all(ch in "01xX-" for ch in t):
        parts: list[Expr] = []
        for name, ch in zip(inputs, t):
            if ch in "xX-":
                continue
            parts.append(Var(name) if ch == "1" else Not(Var(name)))
        return _and_expr(parts)
    try:
        expr = parse_expression(t)
    except ExprSyntaxError as e:
        raise FSMError(f"bad transition condition {text!r}: {e}") from None
    if expr == Const(0):
        raise FSMError(f"the condition {text!r} never holds; write 'x' for 'always'")
    return expr


def fsm_from_dict(spec: Mapping) -> FSM:
    """Build an FSM from a plain dict.

    Keys: `states`, `inputs`, `outputs`, `transitions`, `initial`, `encoding`,
    `moore`, `reset` and `state_outputs`.  A transition is a dict with `state`
    (or `from`), `condition`, `next` (or `to`) and optional `outputs`.
    """
    data = dict(spec)
    transitions = []
    for t in data.get("transitions", []):
        if isinstance(t, Transition):
            transitions.append(t)
            continue
        t = dict(t)
        state = t.get("state", t.get("from"))
        nxt = t.get("next", t.get("to"))
        if state is None or nxt is None:
            raise FSMError(f"transition {t!r} needs a state and a next state")
        outs = t.get("outputs", {})
        transitions.append(Transition(str(state), str(t.get("condition", "x")), str(nxt),
                                      {str(k): int(v) for k, v in dict(outs).items()}))
    state_outputs = {str(k): {str(ok): int(ov) for ok, ov in dict(v).items()}
                     for k, v in dict(data.get("state_outputs", {})).items()}
    return FSM(states=[str(s) for s in data.get("states", [])],
               inputs=[str(i) for i in data.get("inputs", [])],
               outputs=[str(o) for o in data.get("outputs", [])],
               transitions=transitions,
               initial=str(data.get("initial", "")),
               encoding=str(data.get("encoding", "binary")),
               moore=bool(data.get("moore", True)),
               reset=bool(data.get("reset", False)),
               state_outputs=state_outputs)


def fsm_from_table(text: str) -> FSM:
    """Build an FSM from a simple text table.

    ::

        states: IDLE, RUN, DONE
        inputs: go
        outputs: busy
        initial: IDLE
        encoding: binary
        moore: true
        reset: false
        transitions:
        IDLE,0,RUN
        IDLE,1,IDLE
        RUN,1,DONE
        RUN,0,RUN
        state_outputs:
        IDLE,0
        RUN,0
        DONE,1

    The `transitions` rows are `state, condition, next[, output ...]`, the
    `state_outputs` rows are `state, output ...`.  A condition may be a
    boolean expression over the inputs; "x" means "always".
    """
    header: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":") and ":" in line and "," not in line:
            current = line[:-1].strip().lower()
            sections.setdefault(current, [])
            continue
        if current is None:
            if ":" not in line:
                raise FSMError(f"cannot read {line!r}; expected 'key: value'")
            key, value = line.split(":", 1)
            header[key.strip().lower()] = value.strip()
            continue
        sections[current].append(line)
    if "states" not in header:
        raise FSMError("the table needs a 'states:' line")

    def split_list(value: str) -> list[str]:
        return [v.strip() for v in value.replace(";", ",").split(",") if v.strip()]

    states = split_list(header.get("states", ""))
    inputs = split_list(header.get("inputs", ""))
    outputs = split_list(header.get("outputs", ""))
    transitions: list[Transition] = []
    for line in sections.get("transitions", []):
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 3:
            raise FSMError(f"transition row {line!r} needs at least state, condition and next state")
        outs = {}
        for i, name in enumerate(outputs):
            if i + 3 < len(cells):
                outs[name] = int(cells[i + 3])
        transitions.append(Transition(cells[0], cells[1], cells[2], outs))
    state_outputs: dict[str, dict[str, int]] = {}
    for line in sections.get("state_outputs", []):
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 1:
            continue
        outs = {}
        for i, name in enumerate(outputs):
            if i + 1 < len(cells):
                outs[name] = int(cells[i + 1])
        state_outputs[cells[0]] = outs
    moore = header.get("moore", "true").strip().lower() not in ("0", "false", "no")
    return FSM(states=states, inputs=inputs, outputs=outputs, transitions=transitions,
               initial=header.get("initial", "").strip(),
               encoding=header.get("encoding", "binary").strip().lower(),
               moore=moore,
               reset=header.get("reset", "false").strip().lower() in ("1", "true", "yes"),
               state_outputs=state_outputs)


class FSMModel:
    """A Python model of an FSM, usable like the circuit.

    `step()` applies one clock edge; `outputs()` gives the outputs of the
    current cycle.
    """
    def __init__(self, fsm: FSM) -> None:
        self.fsm = fsm
        self.state = fsm.initial

    def reset(self) -> None:
        """Go back to the initial state."""
        self.state = self.fsm.initial

    def outputs(self, inputs: Mapping[str, int] | None = None) -> dict[str, int]:
        """The outputs of the current cycle."""
        values = {name: 0 for name in self.fsm.inputs}
        if inputs:
            values.update({k: int(v) for k, v in inputs.items()})
        return self.fsm.outputs_of(self.state, values)

    def step(self, inputs: Mapping[str, int] | None = None, *, reset: bool = False) -> dict[str, int]:
        """Apply one clock edge and return the outputs of the new state."""
        values = {name: 0 for name in self.fsm.inputs}
        if inputs:
            values.update({k: int(v) for k, v in inputs.items()})
        self.state = self.fsm.next_state(self.state, values, reset)
        return self.fsm.outputs_of(self.state, values)


def simulate_fsm(fsm: FSM, steps: Sequence[Mapping[str, int]]) -> list[dict[str, int]]:
    """Run an FSM through a list of input steps.

    The returned list has one entry per step and matches what
    `sim.verify_sequence` observes in Logisim: entry k holds the outputs of
    the cycle that *follows* step k, that is the state the machine reached
    with step k while step k+1's inputs are applied (the last step is
    repeated at the end).  Moore outputs depend on the state only, so they
    are the outputs of the state reached by step k.
    """
    if not steps:
        return []
    states: list[str] = []
    state = fsm.initial
    for step in steps:
        values = {k: int(v) for k, v in step.items()}
        state = fsm.next_state(state, values, reset=fsm.reset and bool(values.get("rst", 0)))
        states.append(state)
    out = []
    for k, state in enumerate(states):
        nxt = steps[k + 1] if k + 1 < len(steps) else steps[-1]
        out.append(fsm.outputs_of(state, {kk: int(vv) for kk, vv in nxt.items()}))
    return out


def _fsm_next_tables(fsm: FSM) -> tuple[list[set[int]], set[int], list[set[int]], list[set[int]]]:
    """Build the truth tables of the next-state and output functions.

    Returns (minterms per state bit, don't-cares, Moore minterms per output,
    Mealy minterms per output).  The don't-cares are the rows of the unused
    state codes.
    """
    k = fsm.state_bits
    m = len(fsm.inputs)
    has_rst = fsm.reset
    nvars = k + m + (1 if has_rst else 0)
    codes = {s: fsm.encode(s) for s in fsm.states}
    valid = set(codes.values())
    next_ones: list[set[int]] = [set() for _ in range(k)]
    dcs: set[int] = set()
    moore_ones: list[set[int]] = [set() for _ in fsm.outputs]
    mealy_ones: list[set[int]] = [set() for _ in fsm.outputs]
    for assign in itertools.product((0, 1), repeat=nvars):
        num = _assign_number(assign)
        code = _assign_number(assign[:k])
        state = fsm.decode(code)
        if state is None:
            dcs.add(num)
            continue
        values = {fsm.inputs[j]: assign[k + j] for j in range(m)}
        rst = bool(assign[k + m]) if has_rst else False
        tr = None if rst else fsm.matching_transition(state, values)
        nxt = fsm.initial if rst else (tr.next if tr else state)
        ncode = codes[nxt]
        for j in range(k):
            if (ncode >> (k - 1 - j)) & 1:
                next_ones[j].add(num)
        if fsm.moore:
            outs = fsm.state_outputs.get(state, {})
            for oi, name in enumerate(fsm.outputs):
                if int(outs.get(name, 0)):
                    moore_ones[oi].add(code)
        else:
            outs = tr.outputs if tr else {}
            for oi, name in enumerate(fsm.outputs):
                if int(outs.get(name, 0)):
                    mealy_ones[oi].add(num)
    return next_ones, dcs, moore_ones, mealy_ones


def fsm_circuit(fsm: FSM, *, name: str = "fsm", spacing: str = "roomy",
                use_tunnels: bool = False, max_fanin: int = 8) -> Project:
    """Build the circuit of a state machine.

    The project holds the machine (`name`), the next-state logic (`fsm_next`)
    and the output logic (`fsm_out`).  The clock input is named "clk"; with
    `fsm.reset` there is a synchronous reset input "rst".

    The state register holds the state code XOR-ed with the code of the
    initial state, so that all flip-flops start at 0 and the machine still
    starts in its initial state (one-hot codes are not all zero).
    """
    if not isinstance(fsm, FSM):
        raise FSMError("fsm_circuit needs an FSM")
    _check_conditions(fsm)
    k = fsm.state_bits
    m = len(fsm.inputs)
    has_rst = fsm.reset
    svars = [f"s{j}" for j in range(k)]
    ivars = [f"i{j}" for j in range(m)]
    rvar = "rst" if has_rst else None
    allvars = svars + ivars + ([rvar] if rvar else [])
    nvars = len(allvars)
    codes = {s: fsm.encode(s) for s in fsm.states}
    mask = codes[fsm.initial]

    next_ones, dcs, moore_ones, mealy_ones = _fsm_next_tables(fsm)

    # -- next-state logic
    next_exprs: list[Expr] = []
    for j in range(k):
        ones = next_ones[j]
        if not ones:
            next_exprs.append(Const(0))
            continue
        cubes = minimize_function(ones, dcs, nvars)
        next_exprs.append(_or_expr([_cube_expr(c, allvars, complement=False) for c in cubes]))

    # -- output logic
    out_exprs: list[Expr] = []
    for oi in range(len(fsm.outputs)):
        if fsm.moore:
            ones = moore_ones[oi]
            dcs_o = {c for c in range(1 << k) if fsm.decode(c) is None}
            if not ones:
                out_exprs.append(Const(0))
                continue
            cubes = minimize_function(ones, dcs_o, k)
            out_exprs.append(_or_expr([_cube_expr(c, svars, complement=False) for c in cubes]))
        else:
            ones = mealy_ones[oi]
            if not ones:
                out_exprs.append(Const(0))
                continue
            cubes = minimize_function(ones, dcs, nvars)
            out_exprs.append(_or_expr([_cube_expr(c, allvars, complement=False) for c in cubes]))

    used_next = _expr_vars(next_exprs, allvars)
    used_out = _expr_vars(out_exprs, allvars)
    used_state = {v for v in used_next | used_out if v in svars}

    next_circ = Circuit("fsm_next", spacing=spacing, use_tunnels=use_tunnels)
    out_circ = Circuit("fsm_out", spacing=spacing, use_tunnels=use_tunnels)
    top = Circuit(name, spacing=spacing, use_tunnels=use_tunnels)
    # without variables the outputs are constant: they need no logic module
    circuits = [next_circ, out_circ, top] if (fsm.outputs and used_out) else [next_circ, top]
    proj = Project(*circuits, main=name)

    next_env = {v: next_circ.input(v) for v in allvars if v in used_next}
    next_em = _Emitter(next_circ, next_env, max_fanin)
    for j in range(k):
        next_circ.output(f"n{j}", next_em.emit(next_exprs[j]))

    if used_out:
        out_env = {v: out_circ.input(v) for v in allvars if v in used_out}
        out_em = _Emitter(out_circ, out_env, max_fanin)
        for oi, name in enumerate(fsm.outputs):
            out_circ.output(f"o{oi}", out_em.emit(out_exprs[oi]))

    # -- the machine itself
    in_pins = {fsm.inputs[j]: top.input(fsm.inputs[j]) for j in range(m)}
    rst_pin = top.input("rst") if has_rst else None
    clk = top.input("clk")
    ffs = [top.dff(None, clk) for _ in range(k)]

    decoded: list[Signal | None] = []
    for j in range(k):
        if svars[j] in used_state:
            sig: Signal = ffs[j].q
            if (mask >> (k - 1 - j)) & 1:
                sig = top.not_(sig)
            decoded.append(sig)
        else:
            decoded.append(None)

    def signal_for(var: str) -> Signal:
        if var in svars:
            j = int(var[1:])
            sig = decoded[j]
            if sig is None:
                raise FSMError(f"state bit {var} is needed but was not built")
            return sig
        if var in ivars:
            j = int(var[1:])
            return in_pins[fsm.inputs[j]]
        if var == "rst":
            assert rst_pin is not None
            return rst_pin
        raise FSMError(f"unknown variable {var!r}")

    next_inst = top.instance(next_circ, **{v: signal_for(v) for v in used_next})
    for j in range(k):
        d = next_inst[f"n{j}"]
        if (mask >> (k - 1 - j)) & 1:
            d = top.not_(d)
        top.connect(d, ffs[j].D)
    if used_out:
        out_inst = top.instance(out_circ, **{v: signal_for(v) for v in used_out})
        for oi, name in enumerate(fsm.outputs):
            top.output(name, out_inst[f"o{oi}"])
    else:
        # every output is constant: drive the pins from constants
        for name, expr in zip(fsm.outputs, out_exprs):
            top.output(name, top.const(evaluate(expr, {})))
    for circ in circuits:
        _probe_unread(circ)
    return proj


def _expr_vars(exprs: Iterable[Expr], variables: Sequence[str]) -> set[str]:
    """The variables that occur in the expressions."""
    used: set[str] = set()
    for expr in exprs:
        for v in expression_variables(expr):
            if v in variables:
                used.add(v)
    return used


def _check_conditions(fsm: FSM) -> None:
    """Refuse a machine whose transition conditions of one state overlap."""
    m = len(fsm.inputs)
    if m == 0 or m > 10:
        return
    conditions = fsm.condition_expressions()
    for state in fsm.states:
        rows: list[tuple[int, Expr]] = []
        for i, (t, expr) in enumerate(zip(fsm.transitions, conditions)):
            if t.state != state:
                continue
            rows.append((i, expr))
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i][1], rows[j][1]
                for combo in itertools.product((0, 1), repeat=m):
                    values = {fsm.inputs[k]: combo[k] for k in range(m)}
                    if evaluate(a, values) and evaluate(b, values):
                        raise FSMError(
                            f"the conditions of the transitions from {state!r} overlap "
                            f"({fsm.transitions[rows[i][0]].condition!r} and "
                            f"{fsm.transitions[rows[j][0]].condition!r})")
