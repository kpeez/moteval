import ast
import configparser
import json
import re
from collections.abc import Callable
from typing import TypeVar

from moteval import _moteval

T = TypeVar("T")

_JSON_MESSAGE = re.compile(r"(?s)(.*): line (\d+) column (\d+) \(char (\d+)\)")


def _key(message: str) -> object:
    try:
        return ast.literal_eval(message)
    except (ValueError, SyntaxError):
        if message in ("nan", "inf", "-inf"):
            return float(message)
        return message


def _json_error(message: str) -> json.JSONDecodeError:
    err = json.JSONDecodeError.__new__(json.JSONDecodeError)
    ValueError.__init__(err, message)
    match = _JSON_MESSAGE.fullmatch(message)
    if match is not None:
        err.msg = match.group(1)
        err.lineno, err.colno, err.pos = (int(match.group(i)) for i in (2, 3, 4))
    else:
        err.msg, err.lineno, err.colno, err.pos = message, 0, 0, 0
    err.doc = None  # pyrefly: ignore[bad-assignment]
    return err


def python_error(exception: str, message: str) -> Exception:
    if exception == "OSError":
        return OSError(message)
    if exception == "KeyError":
        return KeyError(_key(message))
    if exception == "TypeError":
        return TypeError(message)
    if exception == "IndexError":
        return IndexError(message)
    if exception == "JSONDecodeError":
        return _json_error(message)
    if exception == "InterpolationError":
        return configparser.InterpolationError("", "", message)
    return ValueError(message)


def call(fn: Callable[..., T], *args: object) -> T:
    try:
        return fn(*args)
    except _moteval.RustError as err:
        raise python_error(*err.args) from None
