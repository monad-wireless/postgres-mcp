"""Apache AGE ``agtype`` values, parsed from the text form the server prints.

agtype text is JSON with two additions (observed on AGE 1.8.0 / PostgreSQL 18):

1. A type suffix after a value: ``{...}::vertex``, ``{...}::edge``, ``[...]::path`` and
   ``1.5::numeric`` (also ``NaN::numeric``).
2. Bare float tokens ``NaN``, ``Infinity`` and ``-Infinity``.

Strings use JSON escapes, so a ``::vertex`` inside a string is data, not a suffix. A Cypher
``null`` reaches the client as SQL NULL, so the loader never sees it at the top level.

``parse_agtype`` returns ``GraphVertex``, ``GraphEdge``, ``GraphPath``, ``Decimal`` for numeric,
and plain ``dict`` / ``list`` / ``str`` / ``int`` / ``float`` / ``bool`` / ``None`` otherwise.
Text it cannot parse raises ``AgtypeSyntaxError``; the loader then keeps the raw text, so one
odd value never fails a whole result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal
from decimal import InvalidOperation
from typing import Any

_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
_SUFFIX = re.compile(r"::([a-z_]+)")
_WORDS: dict[str, Any] = {"true": True, "false": False, "null": None}
_FLOAT_WORDS = {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": float("-inf")}


class AgtypeSyntaxError(ValueError):
    """The text is not agtype as this parser knows it."""


@dataclass(frozen=True, slots=True)
class GraphVertex:
    id: int
    label: str
    properties: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GraphEdge:
    id: int
    label: str
    start_id: int
    end_id: int
    properties: dict[str, Any]


@dataclass(frozen=True, slots=True)
class GraphPath:
    """Alternating vertices and edges, as AGE prints a path."""

    elements: tuple[GraphVertex | GraphEdge, ...]


def parse_agtype(text: str) -> Any:
    parser = _Parser(text)
    value = parser.value()
    parser.skip_ws()
    if parser.pos != len(text):
        raise AgtypeSyntaxError(f"trailing text at {parser.pos}")
    return value


class _Parser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t\r\n":
            self.pos += 1

    def value(self) -> Any:
        self.skip_ws()
        raw, value = self._bare_value()
        match = _SUFFIX.match(self.text, self.pos)
        if match is None:
            return value
        self.pos = match.end()
        return _annotate(match.group(1), raw, value)

    def _bare_value(self) -> tuple[str | None, Any]:
        """The value at ``pos``; ``raw`` is the number token when the value is a number."""
        text, pos = self.text, self.pos
        if pos >= len(text):
            raise AgtypeSyntaxError("unexpected end of text")
        ch = text[pos]
        if ch == "{":
            return None, self._object()
        if ch == "[":
            return None, self._array()
        if ch == '"':
            try:
                string, self.pos = json.decoder.scanstring(text, pos + 1)  # type: ignore[attr-defined]
            except json.JSONDecodeError as e:
                raise AgtypeSyntaxError(str(e)) from e
            return None, string
        for word, word_value in _FLOAT_WORDS.items():
            if text.startswith(word, pos):
                self.pos = pos + len(word)
                return word, word_value
        for word, word_value in _WORDS.items():
            if text.startswith(word, pos):
                self.pos = pos + len(word)
                return None, word_value
        match = _NUMBER.match(text, pos)
        if match is None:
            raise AgtypeSyntaxError(f"unexpected {ch!r} at {pos}")
        self.pos = match.end()
        token = match.group(0)
        is_int = not any(c in token for c in ".eE")
        return token, int(token) if is_int else float(token)

    def _expect(self, ch: str) -> None:
        self.skip_ws()
        if self.pos >= len(self.text) or self.text[self.pos] != ch:
            raise AgtypeSyntaxError(f"expected {ch!r} at {self.pos}")
        self.pos += 1

    def _peek(self, ch: str) -> bool:
        self.skip_ws()
        return self.pos < len(self.text) and self.text[self.pos] == ch

    def _object(self) -> dict[str, Any]:
        self._expect("{")
        result: dict[str, Any] = {}
        if self._peek("}"):
            self.pos += 1
            return result
        while True:
            self.skip_ws()
            key = self.value()
            if not isinstance(key, str):
                raise AgtypeSyntaxError(f"object key is not a string at {self.pos}")
            self._expect(":")
            result[key] = self.value()
            if self._peek(","):
                self.pos += 1
                continue
            self._expect("}")
            return result

    def _array(self) -> list[Any]:
        self._expect("[")
        result: list[Any] = []
        if self._peek("]"):
            self.pos += 1
            return result
        while True:
            result.append(self.value())
            if self._peek(","):
                self.pos += 1
                continue
            self._expect("]")
            return result


def _annotate(suffix: str, raw: str | None, value: Any) -> Any:
    if suffix == "numeric":
        if raw is None:
            raise AgtypeSyntaxError("::numeric after a non-number")
        try:
            return Decimal(raw)
        except InvalidOperation as e:
            raise AgtypeSyntaxError(f"bad numeric {raw!r}") from e
    if suffix == "vertex":
        return _vertex(value)
    if suffix == "edge":
        return _edge(value)
    if suffix == "path":
        if not isinstance(value, list) or not all(isinstance(v, (GraphVertex, GraphEdge)) for v in value):
            raise AgtypeSyntaxError("::path is not a list of vertices and edges")
        return GraphPath(tuple(value))
    return value  # a suffix this parser does not know: keep the value, drop the tag


def _vertex(value: Any) -> GraphVertex:
    try:
        return GraphVertex(id=int(value["id"]), label=str(value["label"]), properties=dict(value["properties"]))
    except (KeyError, TypeError, ValueError) as e:
        raise AgtypeSyntaxError(f"malformed vertex: {e}") from e


def _edge(value: Any) -> GraphEdge:
    try:
        return GraphEdge(
            id=int(value["id"]),
            label=str(value["label"]),
            start_id=int(value["start_id"]),
            end_id=int(value["end_id"]),
            properties=dict(value["properties"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        raise AgtypeSyntaxError(f"malformed edge: {e}") from e
