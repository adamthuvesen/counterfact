"""Hidden tests — semantic Unicode cases named in spec.md."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from unicode_normalize import dedupe_normalized  # noqa: E402


def test_canonical_equivalence_dedupes_composed_and_decomposed_accents() -> None:
    assert dedupe_normalized(["Cafe\u0301", "café"]) == ["Cafe\u0301"]


def test_casefolding_dedupes_german_sharp_s() -> None:
    assert dedupe_normalized(["Straße", "STRASSE"]) == ["Straße"]


def test_combining_mark_from_casefolding_is_ignored() -> None:
    assert dedupe_normalized(["İstanbul", "istanbul"]) == ["İstanbul"]


def test_non_ascii_letters_and_combining_marks_share_identity() -> None:
    assert dedupe_normalized(["Μάϊος", "ΜΑΙΟΣ"]) == ["Μάϊος"]


def test_leading_bom_is_ignored_and_removed_from_output() -> None:
    assert dedupe_normalized(["\ufeffalpha", "alpha"]) == ["alpha"]


def test_stable_duplicate_behavior_keeps_first_cleaned_label() -> None:
    labels = ["  Café  ", "cafe\u0301", "CAFÉ", "beta"]
    assert dedupe_normalized(labels) == ["Café", "beta"]
