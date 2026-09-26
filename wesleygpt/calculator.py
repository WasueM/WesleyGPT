# Wesley wrote this
"""The calculator tool the SFT model learned to call (<|python_start|>expr<|python_end|>).

Same contract as nanochat.engine.use_calculator -- arithmetic without `**`, plus
"string".count("x") -- but it walks the parsed syntax tree instead of calling eval(),
so it is safe on a public server and needs no SIGALRM timeout (signals only work on
the main thread, and web servers generate on worker threads).
"""
import ast
import operator

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
# Bounds every intermediate value, which also bounds the work a single expression can cause.
MAX_MAGNITUDE = 1e30


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        value = node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        value = _BINARY[type(node.op)](_eval(node.left), _eval(node.right))
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        value = _UNARY[type(node.op)](_eval(node.operand))
    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "count"
          and isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str)
          and len(node.args) == 1 and not node.keywords
          and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
        return node.func.value.value.count(node.args[0].value)
    else:
        raise ValueError(f"unsupported expression: {ast.dump(node)[:80]}")
    if abs(value) > MAX_MAGNITUDE:
        raise ValueError("result too large")
    return value


def calculate(expr):
    """The value of `expr`, or None when it isn't a supported calculation (the model sees no output)."""
    try:
        return _eval(ast.parse(expr.replace(",", ""), mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
