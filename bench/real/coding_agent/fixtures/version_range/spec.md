# version_range — specification

## Function under specification

```python
def satisfies(version: str, constraints: list[str]) -> bool:
    """Return whether `version` satisfies all semantic-version constraints."""
```

## Version syntax

A version has the form `MAJOR.MINOR.PATCH`, where all three parts are
non-negative integers with no leading sign.

A prerelease suffix may be present after a hyphen, for example `1.2.0-rc1` or
`1.2.0-rc.2`. Prerelease versions are valid but compare lower than the
corresponding final release.

Prerelease suffixes are compared by dot-separated identifiers from left to
right. Numeric identifiers compare numerically, numeric identifiers compare
lower than non-numeric identifiers, and when all shared identifiers are equal,
the shorter prerelease list compares lower. Empty prerelease identifiers are
malformed.

Malformed versions must raise `ValueError`.

## Constraint syntax

Each constraint is one operator followed by one version:

- `>=`
- `>`
- `<=`
- `<`
- `==`

All constraints must be satisfied. An empty constraint list accepts any valid
version.

Malformed constraints must raise `ValueError`.

## Worked examples

| Input | Output | Reason |
|---|---|---|
| `version="1.4.0", constraints=[">=1.2.0", "<2.0.0"]` | `True` | Inside the half-open range. |
| `version="1.2.0", constraints=[">1.2.0"]` | `False` | Strict lower bound excludes equality. |
| `version="1.2.0-rc1", constraints=[">=1.2.0"]` | `False` | Prerelease compares below the final release. |
| `version="1.2.0-rc.10", constraints=[">1.2.0-rc.2", "<1.2.0"]` | `True` | Numeric prerelease identifiers compare numerically. |
| `version="2.0.0", constraints=["==2.0.0"]` | `True` | Exact match. |

## Out of scope

- Wildcards.
- Caret or tilde ranges.
- Build metadata after `+`.
- Missing version components.
