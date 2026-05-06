# unicode_normalize — specification

## Function under specification

```python
def dedupe_normalized(labels: list[str]) -> list[str]:
    """Return labels deduplicated by Unicode-normalized identity."""
```

## Semantics

The function receives user-visible labels and returns the first occurrence of
each normalized label identity. Output order is stable: the first label for a
normalized identity wins, and later duplicates are skipped.

The normalized identity for each label is computed by:

1. Removing a leading Unicode byte-order mark (`\ufeff`) if present.
2. Stripping leading and trailing whitespace.
3. Applying Unicode case folding, not ASCII-only lowercase.
4. Decomposing Unicode characters so combining marks are explicit.
5. Removing combining marks, making accent-composed and accent-decomposed
   spellings compare the same.
6. Re-composing the remaining text to canonical form.

The returned label should be the cleaned first occurrence after BOM removal and
surrounding whitespace stripping, not the normalized identity string.

## Worked examples

| Input | Output | Reason |
|---|---|---|
| `["alpha", "beta", "alpha"]` | `["alpha", "beta"]` | Exact duplicate skipped. |
| `["  alpha  ", "alpha"]` | `["alpha"]` | Surrounding whitespace is ignored for identity and output. |
| `["Cafe\u0301", "café"]` | `["Cafe\u0301"]` | Canonical and accent variants share identity. |
| `["Straße", "STRASSE"]` | `["Straße"]` | Case folding treats `ß` like `ss`. |
| `["İstanbul", "istanbul"]` | `["İstanbul"]` | Combining dot from case folding is ignored. |
| `["Μάϊος", "ΜΑΙΟΣ"]` | `["Μάϊος"]` | Non-ASCII letters and combining marks are normalized. |

## Out of scope

- Locale-specific sorting or collation.
- Transliteration between unrelated scripts.
- Mutating the input list.
- Treating internal whitespace as insignificant.
