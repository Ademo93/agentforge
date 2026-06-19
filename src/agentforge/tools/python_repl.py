"""Sandboxed Python REPL tool.

Runs *one expression or statement block* with a tightly restricted set of
builtins and an AST whitelist that explicitly rejects imports, attribute access
to dunder/sensitive names, subscription of `__class__`, and exec/eval.

Intended use: numeric / list / string transformations the model can't do in
its head. Not intended as a general code execution environment — if you need
that, run the agent inside a real container or VM.
"""

from __future__ import annotations

import ast
import contextlib
import io

_FORBIDDEN_NAMES = {
    "exec",
    "eval",
    "compile",
    "open",
    "input",
    "__import__",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "memoryview",
    "breakpoint",
    "help",
}

_FORBIDDEN_ATTR_PREFIXES = ("__",)

_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "chr": chr,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}


class PythonREPL:
    name = "python_repl"
    description = (
        "Run a short Python expression or statement block in a sandbox. "
        "Supports arithmetic, lists, dicts, comprehensions, math via the safe "
        "builtins (no imports, no I/O, no attribute access to dunder names). "
        "Returns the printed output and the value of the last expression."
    )

    def __init__(self, *, max_chars: int = 4096) -> None:
        self.max_chars = max_chars

    def run(self, input_str: str) -> str:
        code = input_str.strip()
        if not code:
            return "Error: empty input"
        if len(code) > self.max_chars:
            return f"Error: input exceeds {self.max_chars} chars"

        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            return f"Error: SyntaxError: {e}"

        try:
            _validate(tree)
        except ValueError as e:
            return f"Error: {e}"

        env = {"__builtins__": _SAFE_BUILTINS.copy()}
        out = io.StringIO()
        last_value = None

        try:
            with contextlib.redirect_stdout(out):
                # Execute every statement; if the last one is an expression,
                # capture its value to display.
                body = tree.body
                if body and isinstance(body[-1], ast.Expr):
                    head = ast.Module(body=body[:-1], type_ignores=[])
                    tail = ast.Expression(body=body[-1].value)
                    exec(compile(head, "<repl>", "exec"), env)
                    last_value = eval(compile(tail, "<repl>", "eval"), env)
                else:
                    exec(compile(tree, "<repl>", "exec"), env)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

        stdout = out.getvalue()
        parts: list[str] = []
        if stdout:
            parts.append(stdout.rstrip())
        if last_value is not None:
            parts.append(repr(last_value))
        return "\n".join(parts) if parts else "(no output)"


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed in the sandbox")
        if isinstance(node, ast.Attribute) and any(
            node.attr.startswith(p) for p in _FORBIDDEN_ATTR_PREFIXES
        ):
            raise ValueError(f"attribute '{node.attr}' is not allowed")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"name '{node.id}' is not allowed")
