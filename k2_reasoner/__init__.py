"""Utility package for the K2 Risk reasoning demo."""

from .portfolio import load_portfolio_dataframe, SAMPLE_COLUMNS
from .history import HISTORY_SCHEMA, HistoricalSeries, load_history, summarize_history
from .scenario import (
    ScenarioInsight,
    build_scenario_insights,
    breakdown_sentences,
    derive_scenarios,
    evaluate_custom_scenario,
    extract_pdf_text,
    interpret_shocks,
)
from .reasoner import (
    ReasoningArtifacts,
    answer_simple_question,
    build_reasoning_chain,
    generate_narrative_report,
    generate_reasoning_outputs,
    run_counterfactuals,
    summarize_exposures,
)
from .visuals import build_causal_graph

__all__ = [
    "SAMPLE_COLUMNS",
    "load_portfolio_dataframe",
    "load_history",
    "summarize_history",
    "HISTORY_SCHEMA",
    "summarize_exposures",
    "build_reasoning_chain",
    "generate_narrative_report",
    "generate_reasoning_outputs",
    "ReasoningArtifacts",
    "answer_simple_question",
    "run_counterfactuals",
    "build_causal_graph",
    "ScenarioInsight",
    "build_scenario_insights",
    "breakdown_sentences",
    "derive_scenarios",
    "evaluate_custom_scenario",
    "extract_pdf_text",
    "interpret_shocks",
]
