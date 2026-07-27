"""Safe expression evaluator and ``{{ }}`` param templating.

Vendored verbatim from Imagira's ``core/expression.py``. Two distinct jobs
live here:

1. :class:`ExprParams` — what every node's ``self.params`` actually is. It
   resolves ``{{ $json.path }}`` templates on read, which is why this module
   has to ship with the contract rather than stay in the app: a node tested
   standalone against the SDK must behave exactly as it does inside Imagira.
2. :func:`evaluate` / :func:`make_env` — the condition evaluator the host
   app's control-flow nodes (If / If-Else / Switch / While) are built on.
   Exported for host apps and control-flow node authors; not something a
   typical image-processing node needs.

The module-level ContextVar below is the reason there must be exactly ONE copy
of this module in a process — see docs/versioning.md.

The grammar covers what users actually want in workflow conditions:

  - Numeric / string literals, booleans, None
  - Arithmetic: + - * / % **  (// integer division)
  - Comparisons: < > <= >= == != in
  - Boolean: and or not
  - Function calls (whitelist): abs, min, max, round, len, int, float, bool, str
  - Variable lookup: $data, $mask_coverage, $iteration, $item, ...
  - Attribute / item access on dicts: $data.score, $item["x"]

It explicitly forbids attribute access on arbitrary objects, dunder access,
and function definitions/lambdas. The AST is whitelisted node-by-node — any
node not in the allowlist raises ``ExpressionError``.

This is deliberately tiny — about 100 LOC — and avoids the simpleeval/
RestrictedPython dependency.
"""

from __future__ import annotations

import ast
import contextvars
import json
import re
import operator as _op

_ALLOWED_FUNCS = {
    "abs":   abs,
    "min":   min,
    "max":   max,
    "round": round,
    "len":   len,
    "int":   int,
    "float": float,
    "bool":  bool,
    "str":   str,
    "sum":   sum,
}

# Bounds for the ** operator (S12): unbounded `ast.Pow` lets `9**9**9**9` pin a
# CPU and balloon memory. Normal workflow conditions never need huge powers, so
# we cap both the exponent and the base magnitude and raise a clear error past
# them. `2 ** 8` and similar small powers keep working.
_MAX_POW_EXPONENT = 64
_MAX_POW_BASE_ABS = 2 ** 64


def _guarded_pow(base, exp):
    """``operator.pow`` with a magnitude cap to prevent a CPU/memory DoS (S12)."""
    if isinstance(exp, bool) or isinstance(base, bool):
        # bool is an int subclass; allow it (True/False → 1/0) without the check.
        return _op.pow(base, exp)
    if isinstance(exp, (int, float)) and abs(exp) > _MAX_POW_EXPONENT:
        raise ExpressionError(
            f"** exponent too large (|{exp}| > {_MAX_POW_EXPONENT}) — refused to avoid DoS"
        )
    if isinstance(base, (int, float)) and abs(base) > _MAX_POW_BASE_ABS:
        raise ExpressionError(
            f"** base too large (|base| > {_MAX_POW_BASE_ABS}) — refused to avoid DoS"
        )
    return _op.pow(base, exp)


_BIN_OPS = {
    ast.Add:      _op.add,
    ast.Sub:      _op.sub,
    ast.Mult:     _op.mul,
    ast.Div:      _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod:      _op.mod,
    ast.Pow:      _guarded_pow,
}

_CMP_OPS = {
    ast.Eq:    _op.eq,
    ast.NotEq: _op.ne,
    ast.Lt:    _op.lt,
    ast.LtE:   _op.le,
    ast.Gt:    _op.gt,
    ast.GtE:   _op.ge,
    ast.In:    lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_UNARY_OPS = {
    ast.USub: _op.neg,
    ast.UAdd: _op.pos,
    ast.Not:  _op.not_,
}


class ExpressionError(ValueError):
    """Raised when an expression is malformed or uses a disallowed feature."""


def evaluate(expr: str, variables: dict | None = None) -> object:
    """Evaluate *expr* in the context of *variables* and return the result.

    *variables* keys are referenced in the expression as ``$key`` (the dollar
    sign is stripped before AST parsing). Unknown variables raise
    ExpressionError; treat ``$missing or 0`` for defaults.
    """
    if not isinstance(expr, str):
        raise ExpressionError("expression must be a string")
    expr = expr.strip()
    if not expr:
        raise ExpressionError("empty expression")
    # Replace $name → __name__var (avoid dunder, keep parseable identifier).
    # We pick a prefix that is unlikely to collide with user variables.
    py_expr = _strip_dollars(expr)
    try:
        tree = ast.parse(py_expr, mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"syntax error: {e}") from e
    env = {f"__v_{k}__": v for k, v in (variables or {}).items()}
    return _eval(tree.body, env)


def _strip_dollars(expr: str) -> str:
    """Rewrite ``$name`` → ``__v_name__`` so Python's parser accepts it."""
    out: list[str] = []
    i = 0
    while i < len(expr):
        c = expr[i]
        if c == "$":
            j = i + 1
            while j < len(expr) and (expr[j].isalnum() or expr[j] == "_"):
                j += 1
            if j > i + 1:
                out.append("__v_")
                out.append(expr[i+1:j])
                out.append("__")
                i = j
                continue
        out.append(c)
        i += 1
    return "".join(out)


def _eval(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in _ALLOWED_FUNCS:
            return _ALLOWED_FUNCS[node.id]
        raise ExpressionError(f"unknown variable: {node.id.replace('__v_', '$').rstrip('_')}")
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"unsupported unary op: {type(node.op).__name__}")
        return op(_eval(node.operand, env))
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"unsupported binary op: {type(node.op).__name__}")
        return op(_eval(node.left, env), _eval(node.right, env))
    if isinstance(node, ast.BoolOp):
        values = [_eval(v, env) for v in node.values]
        if isinstance(node.op, ast.And):
            result = True
            for v in values:
                result = result and v
                if not result:
                    return result
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for v in values:
                result = result or v
                if result:
                    return result
            return result
        raise ExpressionError(f"unsupported bool op: {type(node.op).__name__}")
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op_node, right_node in zip(node.ops, node.comparators):
            op = _CMP_OPS.get(type(op_node))
            if op is None:
                raise ExpressionError(f"unsupported comparison: {type(op_node).__name__}")
            right = _eval(right_node, env)
            if not op(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        func = _eval(node.func, env)
        if func not in _ALLOWED_FUNCS.values():
            raise ExpressionError("only whitelisted functions may be called")
        args = [_eval(a, env) for a in node.args]
        if node.keywords:
            raise ExpressionError("keyword arguments are not supported")
        return func(*args)
    if isinstance(node, ast.Attribute):
        obj = _eval(node.value, env)
        if node.attr.startswith("_"):
            raise ExpressionError("attribute access on dunder/private names is forbidden")
        if isinstance(obj, dict):
            if node.attr not in obj:
                raise ExpressionError(f"key not found: {node.attr}")
            return obj[node.attr]
        raise ExpressionError("attribute access only allowed on dicts")
    if isinstance(node, ast.Subscript):
        obj = _eval(node.value, env)
        key = _eval(node.slice, env)
        try:
            return obj[key]
        except (KeyError, IndexError, TypeError) as e:
            raise ExpressionError(f"subscript failed: {e}") from e
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e, env) for e in node.elts]
    raise ExpressionError(f"unsupported expression node: {type(node).__name__}")


# ── Convenience: pre-built variable env from a node-execution context ────────

def make_env(*, data=None, mask=None, iteration=None, item=None, extras=None) -> dict:
    """Assemble the standard variable set for control-flow conditions."""
    env: dict = {}
    if data is not None:
        env["data"] = data
    env["mask_coverage"] = _mask_coverage(mask)
    if iteration is not None:
        env["iteration"] = iteration
    if item is not None:
        env["item"] = item
    if extras:
        env.update(extras)
    return env


def _mask_coverage(mask) -> float:
    if mask is None:
        return 0.0
    try:
        import numpy as np
        a = np.asarray(mask)
        if a.size == 0:
            return 0.0
        return float((a > 0).mean())
    except Exception:
        return 0.0


# ── n8n-style {{ }} templates in node params ────────────────────────────────
#
# Lets ANY param on ANY node reference either:
#   - the node's incoming data-port payload (the green wire), as
#     ``{{ $json.article }}``/``{{ $data.items[0].name }}`` — not just the
#     handful of params a node's own code has hand-wired to read a specific
#     data key (video_sink's fps/bitrate, etc.); or
#   - ANY other node's own *configured param value*, by id, as
#     ``{{ $nodes["chroma_key_1"].tolerance }}`` — regardless of which wire
#     (or no wire at all) connects the two nodes. A param value is static
#     configuration, not a runtime-computed payload, so reading it has none
#     of the ordering/concurrency hazards real data-flow would: it's known
#     the moment the graph is defined, before any execution happens, which
#     is also what makes live-preview-without-a-run possible (see
#     the host app's expression-preview endpoint).
#
# Reuses `evaluate` above (already a safe, AST-whitelisted evaluator, no new
# attack surface) rather than a second bespoke path-parser; `$json` is just
# an alias for the existing `$data` variable name so either reads naturally.

_TEMPLATE_RE = re.compile(r"\{\{(.*?)\}\}", re.DOTALL)


def resolve_template(value, data=None, all_nodes=None):
    """Resolve ``{{ expression }}`` placeholders in *value*.

    *data* is bound as ``$json``/``$data`` — the current node's incoming
    data-port payload. *all_nodes* is bound as ``$nodes``: a
    ``{node_id: {param_key: raw_value}}`` mapping of every node currently in
    the graph, letting an expression reach across to another node's own
    configured value regardless of what (if anything) connects the two —
    e.g. ``{{ $nodes["chroma_key_1"].tolerance }}`` to sync a Despill
    setting to whatever a Chroma Key node upstream is set to. Values there
    are the RAW stored param value, not recursively resolved — if the
    referenced node's own param is itself a ``{{ }}`` expression, you get
    that literal template text back, not a further-resolved value. This is
    deliberate: recursively resolving would mean evaluating another node's
    expression against ITS OWN data context (not this one's), which is
    ambiguous at best and can cycle (A references B references A) at worst.
    Getting the raw template text back instead is honestly wrong-looking
    (obviously still a `{{ }}`) rather than silently wrong.

    A value that is *exactly* one ``{{ ... }}`` expression (no other text
    around it) returns that expression's raw Python result unchanged — so a
    number param wired to ``{{ $json.fps }}`` stays a real number rather than
    a stringified one. A value with the placeholder embedded in other text
    (or more than one placeholder) stringifies each result in place instead:
    dicts/lists via ``json.dumps(..., indent=2)`` — the practical equivalent
    of n8n's ``JSON.stringify(...)`` — everything else via ``str()``.

    Values that aren't strings, or strings with no ``{{`` at all, pass
    through completely unchanged without even a regex scan — this has to
    stay cheap since it runs on every param read of every node.
    """
    if not isinstance(value, str) or "{{" not in value:
        return value

    variables = {"data": data, "json": data, "nodes": all_nodes or {}}

    # "Whole field" means exactly one {{ }} block spanning the ENTIRE trimmed
    # string — checked by finding every non-overlapping placeholder (the same
    # non-greedy regex used for substitution below) rather than a separate
    # greedy ^{{...}}$ pattern, which would wrongly treat two adjacent blocks
    # with nothing between them (e.g. "{{ a }}{{ b }}") as one single
    # expression spanning from the first `{{` to the last `}}`, swallowing
    # the middle `}}{{` as if it were expression text.
    stripped = value.strip()
    matches = list(_TEMPLATE_RE.finditer(stripped))
    if len(matches) == 1 and matches[0].group(0) == stripped:
        return _eval_template_expr(matches[0].group(1), variables)

    def _sub(m: "re.Match[str]") -> str:
        result = _eval_template_expr(m.group(1), variables)
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, default=str)
        return "" if result is None else str(result)

    return _TEMPLATE_RE.sub(_sub, value)


def _eval_template_expr(inner: str, variables: dict):
    try:
        return evaluate(inner, variables)
    except ExpressionError as e:
        raise ExpressionError(f"invalid expression {{{{{inner}}}}}: {e}") from e


# Per-call "current node context" used by ExprParams below, set by the
# pipeline engine right before invoking a node's execute() and reset in a
# `finally` right after (see the host app's pipeline engine). A plain
# module-level variable
# would be unsafe — the engine runs one thread per image and reuses the same
# node instance across all of them — but
# contextvars.ContextVar is naturally thread-isolated: a new OS thread starts
# with its own empty Context, so concurrent worker threads setting this
# never see each other's value, with no lock needed. Holds both `data` (the
# per-call, per-image incoming data-port payload) and `nodes` (the whole
# graph's raw params by id — constant for the run, but still threaded through
# here rather than a separate global, since a new OS thread has no other safe
# way to receive it either).
_CURRENT_CTX: "contextvars.ContextVar[dict | None]" = contextvars.ContextVar(
    "_current_node_ctx", default=None)


def set_current_data(data, all_nodes=None):
    """Bind *data* (``$json``/``$data``) and *all_nodes* (``$nodes``) as the
    template source for the duration of the current thread's node.execute()
    call. Returns a token — pass it to :func:`reset_current_data` in a
    ``finally`` block.
    """
    return _CURRENT_CTX.set({"data": data, "nodes": all_nodes or {}})


def reset_current_data(token) -> None:
    _CURRENT_CTX.reset(token)


class ExprParams(dict):
    """A node's ``self.params`` dict, transparently resolving ``{{ }}``
    templates on read against whatever context the current execute() call
    bound via :func:`set_current_data`.

    Only ``.get()``/``__getitem__`` resolve — ``.items()``/``.values()``/
    ``dict(params)`` (used by ``BaseNode.to_dict()`` for workflow
    save/serialize, and by cache-signature/debug code) read the underlying
    dict directly without going through either override, so a saved workflow
    still stores the raw ``{{ ... }}`` text, not whatever it last resolved
    to. This means no per-node code has to change at all: every node already
    reads params via ``self.params.get("key", default)`` per the project's
    own node-authoring convention.
    """

    def get(self, key, default=None):
        raw = dict.get(self, key, default)
        ctx = _CURRENT_CTX.get() or {}
        return resolve_template(raw, ctx.get("data"), ctx.get("nodes"))

    def __getitem__(self, key):
        raw = dict.__getitem__(self, key)
        ctx = _CURRENT_CTX.get() or {}
        return resolve_template(raw, ctx.get("data"), ctx.get("nodes"))
