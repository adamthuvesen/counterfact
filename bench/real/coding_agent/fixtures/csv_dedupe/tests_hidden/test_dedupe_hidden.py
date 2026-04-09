"""Hidden tests — one per normalization rule from spec.md."""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from dedupe import dedupe  # noqa: E402


# Rule 1 — outer-whitespace strip
def test_whitespace_rule_strips_outer_whitespace() -> None:
    assert dedupe(["a", " a "]) == ["a"]
    assert dedupe(["hello", "hello\n", "  hello  "]) == ["hello"]


# Rule 2 — case fold
def test_case_rule_is_case_insensitive() -> None:
    assert dedupe(["A", "a"]) == ["A"]
    assert dedupe(["Hello", "HELLO", "hello"]) == ["Hello"]


# Rule 3 — Unicode NFC
def test_nfc_rule_treats_nfc_and_nfd_as_equal() -> None:
    nfc = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", "café")
    assert nfc != nfd, "test fixture invariant: NFC and NFD differ as raw strings"
    assert dedupe([nfc, nfd]) == [nfc]


# Rule 4 — BOM strip
def test_bom_rule_strips_leading_byte_order_mark() -> None:
    bom = "﻿"
    assert dedupe([bom + "foo", "foo"]) == [bom + "foo"]


# Combined-rules sanity: all four rules at once on a single pair of duplicates
def test_combined_rules_treat_full_normalization_as_equal() -> None:
    bom = "﻿"
    nfd = unicodedata.normalize("NFD", "Café")
    a = bom + "  " + nfd + "  "
    b = "café"
    assert dedupe([a, b]) == [a]


# Negative case — accent-stripping is OUT of scope per spec.md.
def test_accent_difference_is_not_a_duplicate() -> None:
    assert dedupe(["café", "cafe"]) == ["café", "cafe"]
