"""Tiny stdlib HTML/SVG builders used by `render_html`.

Output is escaped by default so renderer code does not have to remember
when to call `html.escape`. Children that are pre-built strings (e.g. SVG
fragments) opt out of escaping by passing `Raw(...)`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
from typing import Union

_VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "source",
        "track",
        "wbr",
    }
)


@dataclass(frozen=True)
class Raw:
    """Pre-built HTML/SVG that should not be re-escaped."""

    body: str


Child = Union[str, "Raw", None]


def _attrs(attrs: dict[str, object] | None) -> str:
    if not attrs:
        return ""
    parts: list[str] = []
    for key, value in attrs.items():
        if value is None or value is False:
            continue
        attr = key.rstrip("_").replace("_", "-")
        if value is True:
            parts.append(f" {attr}")
        else:
            parts.append(f' {attr}="{escape(str(value), quote=True)}"')
    return "".join(parts)


def _render_child(child: Child) -> str:
    if child is None:
        return ""
    if isinstance(child, Raw):
        return child.body
    return escape(child)


def tag(
    name: str,
    attrs: dict[str, object] | None = None,
    *children: Child | Iterable[Child],
) -> str:
    """Render an element, escaping string children and serializing attrs."""
    flat: list[Child] = []
    for c in children:
        if isinstance(c, (str, Raw)) or c is None:
            flat.append(c)
        else:
            flat.extend(c)
    attr_str = _attrs(attrs)
    if name in _VOID_TAGS:
        return f"<{name}{attr_str}>"
    body = "".join(_render_child(c) for c in flat)
    return f"<{name}{attr_str}>{body}</{name}>"


def raw(body: str) -> Raw:
    return Raw(body)
