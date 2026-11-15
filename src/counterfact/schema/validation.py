"""Pure trace identity checks shared by analysis and corpus IO."""

from collections.abc import Iterable

from counterfact.schema.models import Run


def validate_runs(runs: Iterable[Run]) -> list[Run]:
    """Revalidate and detach traces without changing their order."""
    validated: list[Run] = []
    seen: set[str] = set()
    for run in runs:
        snapshot = Run.model_validate(run.model_dump())
        if snapshot.run_id in seen:
            raise ValueError(f"duplicate run_id {snapshot.run_id!r} in corpus")
        seen.add(snapshot.run_id)
        validated.append(snapshot)
    return validated


def require_focal(focal: Run, corpus: Iterable[Run]) -> None:
    """Require the focal trace's identity and contents to match the corpus."""
    match = next((run for run in corpus if run.run_id == focal.run_id), None)
    if match is None:
        raise ValueError(f"focal run {focal.run_id!r} is not present in the supplied corpus")
    if match.model_dump() != focal.model_dump():
        raise ValueError(f"focal run {focal.run_id!r} differs from its corpus trace")
