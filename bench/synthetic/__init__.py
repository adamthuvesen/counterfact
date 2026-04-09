"""Synthetic SCM corpus — pure-Python simulator with known treatment effects."""

from bench.synthetic.scm import (
    HEADLINE_TRUE_EFFECT,
    MODEL_CHOICE_ARMS,
    RETRY_POLICY_ARMS,
    TOOL_CHOICE_ARMS,
    SyntheticSCM,
)
from bench.synthetic.generate import generate_corpus

__all__ = [
    "HEADLINE_TRUE_EFFECT",
    "MODEL_CHOICE_ARMS",
    "RETRY_POLICY_ARMS",
    "TOOL_CHOICE_ARMS",
    "SyntheticSCM",
    "generate_corpus",
]
