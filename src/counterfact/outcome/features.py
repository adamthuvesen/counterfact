"""Feature keys shared by outcome fitting and interventions."""

from __future__ import annotations

from typing import Any

from counterfact.schema import Decision
from counterfact.taxonomy import valid_interventions


def canonical_feature_value(value: Any) -> str:
    """Normalize logged feature values into stable one-hot suffixes."""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return f"{value:g}"
    return str(value)


def canonical_intervention_value(intervention_kind: str, value: Any) -> str:
    """Normalize user-supplied intervention values without rewriting string arms."""
    if intervention_kind == "temperature":
        numeric_value = _coerce_finite_number(value)
        if numeric_value is not None:
            return canonical_feature_value(numeric_value)
    return canonical_feature_value(value)


def intervention_feature_family(decision_type: str, intervention_kind: str) -> str:
    """Return the one-hot feature family used for a declared intervention kind."""
    if decision_type == "model_call" and intervention_kind == "temperature":
        return "model_call.temperature"
    return decision_type


def decision_feature_values(decision: Decision) -> list[tuple[str, str]]:
    """Return `(feature_family, value)` pairs contributed by one decision."""
    if not valid_interventions(decision.decision_type):
        return []

    values: list[tuple[str, str]] = []
    if decision.chosen_action is not None:
        values.append((decision.decision_type, canonical_feature_value(decision.chosen_action)))

    if decision.decision_type == "model_call":
        temperature = _model_temperature(decision.metadata)
        if temperature is not None:
            values.append(("model_call.temperature", canonical_feature_value(temperature)))

    return values


def _model_temperature(metadata: dict[str, Any]) -> Any | None:
    model_config = metadata.get("model_config")
    if isinstance(model_config, dict) and model_config.get("temperature") is not None:
        return model_config["temperature"]
    return metadata.get("temperature")


def _coerce_finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        if parsed.is_integer():
            return int(parsed)
        return parsed
    return None


__all__ = [
    "canonical_feature_value",
    "canonical_intervention_value",
    "decision_feature_values",
    "intervention_feature_family",
]
