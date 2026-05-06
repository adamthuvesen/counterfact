"""Synthetic SCM corpus — pure-Python simulator with known treatment effects."""

from bench.synthetic.generate import generate_corpus, generate_traces
from bench.synthetic.scm import (
    CONFOUNDED_DO_HEADLINE,
    CONFOUNDED_NAIVE_HEADLINE,
    CONFOUNDED_NAIVE_VS_CAUSAL_GAP,
    HEADLINE_TRUE_EFFECT,
    MODEL_CHOICE_ARMS,
    RETRY_POLICY_ARMS,
    TOOL_CHOICE_ARMS,
    SyntheticSCM,
)

__all__ = [
    "CONFOUNDED_DO_HEADLINE",
    "CONFOUNDED_NAIVE_HEADLINE",
    "CONFOUNDED_NAIVE_VS_CAUSAL_GAP",
    "HEADLINE_TRUE_EFFECT",
    "MODEL_CHOICE_ARMS",
    "RETRY_POLICY_ARMS",
    "TOOL_CHOICE_ARMS",
    "SyntheticSCM",
    "generate_corpus",
    "generate_traces",
]
