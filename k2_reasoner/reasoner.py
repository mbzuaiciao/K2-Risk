from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Dict, List, Tuple

import pandas as pd

from .portfolio import Portfolio
from .llm import (
    GeminiClientError,
    OpenRouterClientError,
    gemini_available,
    generate_openrouter_reasoning,
    generate_reasoning_response,
    openrouter_available,
    parse_reasoning_payload,
)


@dataclass
class ExposureSummary:
    totals: Dict[str, float]
    per_asset: pd.DataFrame


@dataclass
class ReasoningArtifacts:
    chain: List[str]
    narrative: str
    source: str
    note: str | None = None


RATES_COLUMNS = ["duration", "dv01", "convexity"]
CREDIT_COLUMNS = ["credit_spread_dv01"]
FX_COLUMNS = ["fx_delta"]
EQUITY_COLUMNS = ["beta", "notional"]


def summarize_exposures(portfolio: Portfolio) -> ExposureSummary:
    df = portfolio.df

    totals = {
        "notional": df["notional"].sum(),
        "duration": df["duration"].sum(),
        "dv01": df["dv01"].sum(),
        "convexity": df["convexity"].sum(),
        "credit_spread_dv01": df["credit_spread_dv01"].sum(),
        "fx_delta": df["fx_delta"].sum(),
        "beta_notional": (df["beta"] * df["notional"]).sum(),
    }

    per_asset = (
        df.groupby("asset_class")[
            ["notional", "duration", "dv01", "credit_spread_dv01", "fx_delta", "beta"]
        ]
        .sum()
        .reset_index()
    )

    return ExposureSummary(totals=totals, per_asset=per_asset)


def _describe_direction(value: float, positive_word: str, negative_word: str, *, tol: float = 1e-6) -> str:
    if value > tol:
        return positive_word
    if value < -tol:
        return negative_word
    return "balanced"


def _format_number(value: float, suffix: str = "") -> str:
    if abs(value) >= 1_000:
        base = value / 1_000
        return f"{base:,.1f}k{suffix}"
    return f"{value:,.0f}{suffix}"


def _theme_from_exposures(summary: ExposureSummary) -> List[str]:
    notes: List[str] = []
    duration_direction = _describe_direction(summary.totals["duration"], "long duration", "short duration")
    credit_direction = _describe_direction(
        summary.totals["credit_spread_dv01"], "long credit", "short credit"
    )
    fx_direction = _describe_direction(summary.totals["fx_delta"], "long USD", "short USD")
    equity_direction = _describe_direction(summary.totals["beta_notional"], "pro-cyclical beta", "defensive beta")

    if duration_direction != "balanced":
        notes.append(f"Portfolio is {duration_direction} via aggregated DV01 of {_format_number(summary.totals['dv01'])}.")

    if credit_direction != "balanced":
        notes.append(
            f"Credit exposure is {credit_direction}, suggesting spreads {'hurt' if credit_direction == 'long credit' else 'help'} during stress."
        )

    if fx_direction != "balanced":
        notes.append(f"FX delta skews {fx_direction}, linking rate moves to currency PnL.")

    if equity_direction != "balanced":
        notes.append(f"Equity book carries {equity_direction}, amplifying macro shocks.")

    if not notes:
        notes.append("Risk book is balanced across major factors; localized positions drive PnL.")

    return notes


def _build_rule_based_chain(summary: ExposureSummary, question: str) -> List[str]:
    themes = _theme_from_exposures(summary)

    chain = [
        f"Question received: \"{question or 'Explain current risk posture'}\"",
        "K2 Risk ingests aggregate risk factors across rates, credit, FX, and equity.",
    ]
    chain.extend(themes)

    # Linkages
    if summary.totals["duration"] > 0 and summary.totals["credit_spread_dv01"] > 0:
        chain.append(
            "Rates rally lowers yields, boosting the positive duration book; however the long-credit bias means recession-driven spread widening can offset gains."
        )
    if summary.totals["duration"] > 0 and summary.totals["fx_delta"] < 0:
        chain.append(
            "Long duration paired with short USD suggests reliance on lower U.S. yields to fund FX carry; a sharp USD squeeze would pressure both legs."
        )
    if summary.totals["beta_notional"] > 0 and summary.totals["credit_spread_dv01"] > 0:
        chain.append(
            "Positive equity beta co-moves with tight credit spreads; when growth fears appear, both legs deteriorate simultaneously."
        )
    if summary.totals["beta_notional"] < 0 and summary.totals["duration"] > 0:
        chain.append("Defensive equity hedges cushion rate-driven rallies, but basis risk remains against long duration.")

    chain.append("Result: K2 Risk maps quantitative sensitivities into a causal story connecting macro shocks to cross-asset PnL.")
    return chain


def _build_rule_based_narrative(summary: ExposureSummary, question: str) -> str:
    duration_pnl = summary.totals["dv01"] * 50  # 50bp rally
    credit_drag = summary.totals["credit_spread_dv01"] * 20
    fx_component = summary.totals["fx_delta"] * 0.01

    paragraphs = [
        f"K2 Risk interprets the submitted portfolio in response to: {question or 'Explain current risk posture.'}",
        f"Aggregate notional stands at {_format_number(summary.totals['notional'], 'mm')} with DV01 of {_format_number(summary.totals['dv01'])} and convexity {_format_number(summary.totals['convexity'])}.",
        f"A 50bp bull move in rates contributes roughly {_format_number(duration_pnl)} in gains, but contemporaneous 20bp spread widening would subtract {_format_number(credit_drag)}.",
        f"FX deltas translate U.S. rate changes into approximately {_format_number(fx_component)} of currency PnL for a 1% dollar move.",
    ]

    if summary.totals["beta_notional"] != 0:
        beta_effect = summary.totals["beta_notional"] * 0.015  # 1.5% equity shock
        direction = "loss" if beta_effect < 0 else "gain"
        paragraphs.append(f"Equity beta stacks to {_format_number(summary.totals['beta_notional'])}, implying a {direction} of {_format_number(abs(beta_effect))} for a 1.5% global equity move.")

    paragraphs.append(
        "Interpretation: duration hedges behave well when central banks ease, yet the credit and equity books import recession risk that can neutralize rate gains."
    )

    return "\n\n".join(paragraphs)


def run_counterfactuals(portfolio: Portfolio) -> List[Dict[str, str]]:
    summary = summarize_exposures(portfolio)
    scenarios: List[Tuple[str, Dict[str, float]]] = [
        ("Rates +75bp, USD +1.5%", {"rate_bps": 75, "credit_bps": 5, "fx_move": 0.015, "equity_move": -0.01}),
        ("Rates -50bp, spreads +25bp", {"rate_bps": -50, "credit_bps": 25, "fx_move": -0.01, "equity_move": -0.02}),
        ("Credit shock +60bp, equities -3%", {"rate_bps": 10, "credit_bps": 60, "fx_move": 0.005, "equity_move": -0.03}),
    ]

    outputs: List[Dict[str, str]] = []
    for label, s in scenarios:
        pnl_rates = -summary.totals["dv01"] * s["rate_bps"]
        pnl_credit = -summary.totals["credit_spread_dv01"] * s["credit_bps"]
        pnl_fx = summary.totals["fx_delta"] * s["fx_move"]
        pnl_equity = summary.totals["beta_notional"] * s["equity_move"]
        total = pnl_rates + pnl_credit + pnl_fx + pnl_equity

        explanation = (
            f"Rates move contributes {_format_number(pnl_rates)}; "
            f"credit adds {_format_number(pnl_credit)}; FX {_format_number(pnl_fx)}; "
            f"equities {_format_number(pnl_equity)}. Net impact {_format_number(total)}."
        )

        outputs.append({"scenario": label, "impact": explanation})

    return outputs


def answer_simple_question(summary: ExposureSummary, question: str | None) -> str | None:
    if not question:
        return None

    q = question.lower()
    totals = summary.totals

    if "dv01" in q and "credit" in q:
        return f"Total credit spread DV01 is {_format_number(totals['credit_spread_dv01'])}."
    if "dv01" in q:
        return f"Portfolio DV01 sums to {_format_number(totals['dv01'])}, meaning a 1bp parallel move shifts P&L by that amount."
    if "duration" in q:
        return f"Aggregate duration across the book is {totals['duration']:.2f} years."
    if "notional" in q or "size" in q:
        return f"Gross notional exposure is {_format_number(totals['notional'], 'mm')}."
    if "convexity" in q:
        return f"Total convexity measures {_format_number(totals['convexity'])}."
    if "fx" in q or "currency" in q:
        return f"Net FX delta sits at {_format_number(totals['fx_delta'])} in base currency terms."
    if "beta" in q or "equity" in q:
        return f"Equity beta-adjusted notional is {_format_number(totals['beta_notional'])}."

    return None


def _build_prompt(summary: ExposureSummary, question: str) -> str:
    def _pythonify_numbers(values: Dict[str, float]) -> Dict[str, float]:
        return {key: float(value) for key, value in values.items()}

    per_asset_rows = summary.per_asset.to_dict(orient="records")
    per_asset_lines = []
    for row in per_asset_rows:
        line = (
            f"- {row['asset_class']}: "
            f"notional {row['notional']:.1f}mm, duration {row['duration']:.1f}, "
            f"dv01 {row['dv01']:.1f}, credit {row['credit_spread_dv01']:.1f}, "
            f"fx {row['fx_delta']:.1f}, beta {row['beta']:.2f}"
        )
        per_asset_lines.append(line)

    totals_json = json.dumps(_pythonify_numbers(summary.totals), indent=2)
    per_asset_text = "\n".join(per_asset_lines)
    instructions = """
You are K2 Risk, an institutional risk reasoning agent built on Gemini.
Explain how multi-asset risk factors interact. Incorporate causal logic across rates, credit, FX, and equities.
Respond strictly in compact JSON with the following schema:
{
  "chain": ["Step 1 sentence", "Step 2 sentence", "..."],
  "narrative": "2-3 paragraph narrative explanation"
}
Ensure the chain has 4-6 concise steps covering cause-effect links and mitigation ideas.
"""

    prompt = f"""{instructions}
Question: {question}

Aggregate factor totals:
{totals_json}

Per-asset aggregates:
{per_asset_text}

Return only JSON."""
    return prompt


def _fallback_artifacts(summary: ExposureSummary, question: str, note: str | None = None) -> ReasoningArtifacts:
    return ReasoningArtifacts(
        chain=_build_rule_based_chain(summary, question),
        narrative=_build_rule_based_narrative(summary, question),
        source="rule-engine",
        note=note,
    )


def generate_reasoning_outputs(
    portfolio: Portfolio, question: str | None, llm_provider: str | None = None
) -> ReasoningArtifacts:
    """Create reasoning chain + narrative via selected LLM (or rule engine)."""

    summary = summarize_exposures(portfolio)
    query = question or "Explain current risk posture."

    if llm_provider == "gemini":
        if not gemini_available():
            note = "Gemini SDK or API key missing; fell back to deterministic reasoning."
            return _fallback_artifacts(summary, query, note=note)
        try:
            prompt = _build_prompt(summary, query)
            response_text = generate_reasoning_response(prompt)
            payload = parse_reasoning_payload(response_text, error_cls=GeminiClientError)
            chain = [step.strip() for step in payload["chain"] if step.strip()]
            narrative = payload["narrative"]
            if chain and narrative:
                return ReasoningArtifacts(chain=chain, narrative=narrative, source="gemini")
            note = "Gemini response lacked usable content; reverted to deterministic reasoning."
            return _fallback_artifacts(summary, query, note=note)
        except GeminiClientError as exc:
            note = f"Gemini request failed ({exc}); reverted to deterministic reasoning."
            return _fallback_artifacts(summary, query, note=note)

    if llm_provider == "openrouter":
        if not openrouter_available():
            note = "OpenRouter API key missing; reverted to deterministic reasoning."
            return _fallback_artifacts(summary, query, note=note)
        try:
            prompt = _build_prompt(summary, query)
            response_text = generate_openrouter_reasoning(prompt)
            payload = parse_reasoning_payload(response_text, error_cls=OpenRouterClientError)
            chain = [step.strip() for step in payload["chain"] if step.strip()]
            narrative = payload["narrative"]
            if chain and narrative:
                return ReasoningArtifacts(chain=chain, narrative=narrative, source="openrouter")
            note = "OpenRouter response lacked usable content; reverted to deterministic reasoning."
            return _fallback_artifacts(summary, query, note=note)
        except OpenRouterClientError as exc:
            note = f"OpenRouter request failed ({exc}); reverted to deterministic reasoning."
            return _fallback_artifacts(summary, query, note=note)

    return _fallback_artifacts(summary, query)


# Backwards-compatible helpers (rule-based only)
def build_reasoning_chain(portfolio: Portfolio, question: str) -> List[str]:
    summary = summarize_exposures(portfolio)
    return _build_rule_based_chain(summary, question)


def generate_narrative_report(portfolio: Portfolio, question: str) -> str:
    summary = summarize_exposures(portfolio)
    return _build_rule_based_narrative(summary, question)
