"""Read-only check of the Cypher text passed to Apache AGE's ``cypher()`` (restricted mode).

pglast parses the SQL around ``cypher('graph', $$ ... $$)`` but sees the Cypher as one string
constant, so the SQL validator alone cannot judge it. Probes on AGE 1.8.0 / PostgreSQL 18 set
what this guard must stop:

- **Writes.** ``CREATE``, ``MERGE``, ``SET``, ``DELETE`` and ``REMOVE`` all fail inside
  ``BEGIN TRANSACTION READ ONLY`` (AGE runs DDL or DML that PostgreSQL refuses), so the forced
  read-only transaction already blocks them. The guard rejects them first anyway, with a clear
  message, and that keeps two barriers in place.
- **SQL functions.** Cypher calls a SQL function when the name is schema-qualified
  (``RETURN pg_catalog.pg_backend_pid()`` ran) and also by bare name: AGE first looks for
  ``ag_catalog.age_<name>``, then for ``<name>`` on ``search_path``, where a user's
  VOLATILE function was found and run. A read-only transaction does not stop a function such
  as ``pg_terminate_backend``.

So the guard admits a function call only when the name is unqualified and either
``age_<name>`` is in the extension's read-only allowlist (``extension_functions``, the same
IMMUTABLE/STABLE rule as SQL) or the name is in ``CYPHER_BUILTINS``. The built-ins are the
functions that AGE handles with no ``age_`` function behind them (checked one by one on 1.8.0).

The scan drops string literals, backtick-quoted names, comments and numbers first, so a keyword
inside a string or a quoted name never matches. A name directly after ``.`` (property) or ``:``
(label, relationship type) is a name, not a clause. A map key that is also a write keyword
(``{set: 1}``) is refused; quote it with backticks to pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CYPHER_BUILTINS: frozenset[str] = frozenset({"count", "coalesce", "any", "all", "none", "single", "reduce", "rand"})

# Clauses that write, call procedures or run the statement (EXPLAIN ANALYZE).
WRITE_KEYWORDS: frozenset[str] = frozenset({"create", "merge", "set", "delete", "detach", "remove", "call", "load", "foreach", "analyze"})

# Keywords that may stand directly before "(" without being a function call.
_PAREN_KEYWORDS: frozenset[str] = frozenset(
    {
        "match",
        "optional",
        "where",
        "return",
        "with",
        "unwind",
        "and",
        "or",
        "xor",
        "not",
        "in",
        "as",
        "case",
        "when",
        "then",
        "else",
        "end",
        "union",
        "distinct",
        "order",
        "by",
        "skip",
        "limit",
        "is",
        "contains",
        "starts",
        "ends",
        "asc",
        "desc",
        "ascending",
        "descending",
        "explain",
        "verbose",
        "yield",
    }
)

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER = re.compile(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_PARAM = re.compile(r"\$[A-Za-z0-9_]+")


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str  # "word" (unquoted identifier), "quoted" (backtick name), "punct"
    text: str


def _tokens(query: str) -> list[_Token]:
    tokens: list[_Token] = []
    i, n = 0, len(query)
    while i < n:
        ch = query[i]
        if ch.isspace():
            i += 1
        elif query.startswith("//", i):
            end = query.find("\n", i)
            i = n if end < 0 else end + 1
        elif query.startswith("/*", i):
            end = query.find("*/", i + 2)
            if end < 0:
                raise ValueError("unterminated /* comment in Cypher")
            i = end + 2
        elif ch in "'\"":
            i = _skip_string(query, i)
        elif ch == "`":
            j = i + 1
            name: list[str] = []
            while True:
                end = query.find("`", j)
                if end < 0:
                    raise ValueError("unterminated `name` in Cypher")
                name.append(query[j:end])
                if query.startswith("``", end):  # a doubled backtick is a literal backtick
                    name.append("`")
                    j = end + 2
                    continue
                i = end + 1
                break
            tokens.append(_Token("quoted", "".join(name)))
        elif (m := _IDENT.match(query, i)) is not None:
            tokens.append(_Token("word", m.group(0)))
            i = m.end()
        elif (m := _NUMBER.match(query, i)) is not None:
            i = m.end()
        elif (m := _PARAM.match(query, i)) is not None:
            i = m.end()
        else:
            tokens.append(_Token("punct", ch))
            i += 1
    return tokens


def _skip_string(query: str, start: int) -> int:
    quote = query[start]
    i = start + 1
    while i < len(query):
        if query[i] == "\\":
            i += 2
        elif query[i] == quote:
            return i + 1
        else:
            i += 1
    raise ValueError("unterminated string literal in Cypher")


class CypherGuard:
    """Accepts read-only Cypher; ``check`` raises ``ValueError`` naming the first violation."""

    def __init__(self, extension_functions: frozenset[str]):
        self.extension_functions = extension_functions

    def function_allowed(self, name: str) -> bool:
        lowered = name.lower()
        return lowered in CYPHER_BUILTINS or f"age_{lowered}" in self.extension_functions

    def check(self, query: str) -> None:
        tokens = _tokens(query)
        for i, tok in enumerate(tokens):
            if tok.kind == "punct":
                continue
            prev = tokens[i - 1].text if i > 0 and tokens[i - 1].kind == "punct" else None
            nxt = tokens[i + 1] if i + 1 < len(tokens) else None
            calls = nxt is not None and nxt.kind == "punct" and nxt.text == "("
            word = tok.text.lower()
            if calls and prev == ".":
                raise ValueError(f"Cypher calls the schema-qualified function {tok.text}(); restricted mode admits only AGE built-in functions")
            if prev in (".", ":"):
                continue  # property key, label or relationship type
            if tok.kind == "word" and word in WRITE_KEYWORDS:
                raise ValueError(f"Cypher clause {tok.text.upper()} is not allowed in restricted (read-only) mode")
            if calls and not (tok.kind == "word" and word in _PAREN_KEYWORDS) and not self.function_allowed(tok.text):
                raise ValueError(
                    f"Cypher function {tok.text}() is not an AGE built-in that restricted mode admits "
                    "(a bare name can resolve to any SQL function on search_path)"
                )
