# csv_dedupe — specification

## Function under specification

```python
def dedupe(rows: list[str]) -> list[str]:
    """Return rows with duplicates removed, preserving the order of first
    occurrence."""
```

## Duplicate equivalence

Two rows `a` and `b` are duplicates iff `normalize(a) == normalize(b)`. The
output keeps the row whose first occurrence comes earlier (its original
spelling — normalization is for comparison, not rewriting).

## Normalization rules

`normalize(s)` applies four rules, in order:

1. **BOM strip.** Remove a leading byte-order mark (`U+FEFF`) if present.
   BOM is a file-level marker, so it is removed before any other rule
   has a chance to be confused by it.
2. **Outer-whitespace strip.** Strip leading and trailing whitespace
   characters (`str.strip()`-equivalent).
3. **Case fold.** Convert to a case-insensitive form (`str.casefold()`).
4. **Unicode NFC.** Normalize to Unicode Normalization Form C
   (`unicodedata.normalize("NFC", ...)`).

## Worked examples

| Input rows | Output | Reason |
|---|---|---|
| `["a", "a"]` | `["a"]` | Exact duplicates. |
| `["a", " a "]` | `["a"]` | Outer-whitespace rule. |
| `["A", "a"]` | `["A"]` | Case-fold rule (first occurrence wins). |
| `["café", "café"]` | `["café"]` | NFC rule (NFC vs NFD). |
| `["﻿foo", "foo"]` | `["﻿foo"]` | BOM rule. |
| `[" Café́ ", "cafe"]` | `[" Café́ ", "cafe"]` | Different even after all rules — `é` (with accent) vs plain `e`. |

## Out of scope

- Inner whitespace collapse, accent stripping, locale-specific case folding.
- Anything beyond the four rules above.
