"""Calculator tool — safe arithmetic via AST whitelist.

We deliberately avoid ``eval()`` and ``sympy.sympify`` directly. The whitelist
covers the operators a calculator actually needs: + - * / // % **, unary +/-,
parentheses, int/float literals, and a fixed set of math functions.
"""

from __future__ import annotations

import ast
import math
import operator as op

_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "ln": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "max": max,
    "min": min,
}

_CONSTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}


class Calculator:
    name = "calculator"
    description = (
        "Safely evaluate an arithmetic expression. "
        "Supports + - * / // % ** parentheses and functions "
        "sqrt, log, exp, sin/cos/tan, abs, round, floor, ceil, max, min. "
        "Constants: pi, e. Example: '47 * 1337'."
    )

    def run(self, input_str: str) -> str:
        expr = input_str.strip()
        if not expr:
            return "Error: empty expression"
        try:
            tree = ast.parse(expr, mode="eval")
            value = _eval(tree.body)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"
        return _fmt(value)


def _eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported literal: {node.value!r}")
    if isinstance(node, ast.BinOp):
        left = _eval(node.left)
        right = _eval(node.right)
        op_fn = _BIN_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op_fn(left, right)
    if isinstance(node, ast.UnaryOp):
        op_fn = _UNARY_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op_fn(_eval(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("only simple function calls allowed")
        name = node.func.id
        if name not in _FUNCS:
            raise ValueError(f"unknown function: {name}")
        args = [_eval(a) for a in node.args]
        return _FUNCS[name](*args)
    if isinstance(node, ast.Name):
        if node.id not in _CONSTS:
            raise ValueError(f"unknown identifier: {node.id}")
        return _CONSTS[node.id]
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def _fmt(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)
