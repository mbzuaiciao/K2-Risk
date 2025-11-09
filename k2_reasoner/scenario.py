from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .llm import GeminiClientError, gemini_available, generate_scenario_shocks
from .reasoner import ExposureSummary, summarize_exposures


SCENARIO_KEYWORDS = {
    "rates": ["rate", "rates", "yield", "yields", "treasury", "curve", "tightening", "hike", "hikes"],
    "credit": ["credit", "spread", "spreads", "default", "ig", "hy"],
    "fx": ["fx", "currency", "currencies", "usd", "dollar", "eur", "yen"],
    "equity": ["equity", "equities", "stock", "stocks", "risk asset", "beta", "share"],
}


@dataclass
class ScenarioInsight:
    text: str
    shocks: Dict[str, float]
    impact: str
    note: str | None = None


def extract_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path.open("rb"))
    chunks: List[str] = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def derive_scenarios(raw_text: str) -> List[str]:
    lines = raw_text.splitlines()
    bullets: List[tuple[bool, str]] = []
    for line in lines:
        stripped = line.strip()
        if len(stripped.split()) <= 3:
            continue
        has_marker = bool(re.match(r"^\(?\d+\)", stripped) or stripped.startswith(("•", "○", "-", "–")))
        cleaned = stripped.lstrip("•○-– \t")
        bullets.append((has_marker, cleaned))
    results: List[str] = []
    current = []
    for has_marker, text in bullets:
        if has_marker:
            if current:
                results.append(" ".join(current).strip())
                current = []
        current.append(text)
    if current:
        results.append(" ".join(current).strip())
    return results[:8]  # limit noise


def _detect_sign(text: str, default: int = 1) -> int:
    lowered = text.lower()
    negative_triggers = ["fall", "drop", "decline", "tighten", "compress", "strengthen"]
    positive_triggers = ["rise", "increase", "widen", "selloff", "weaken"]
    for trigger in negative_triggers:
        if trigger in lowered:
            return -1
    for trigger in positive_triggers:
        if trigger in lowered:
            return 1
    return default


def _keyword_window_hits(
    source: str, start: int, end: int, keywords: List[str], window: int = 30
) -> tuple[bool, bool]:
    prefix = source[max(0, start - window) : start]
    suffix = source[end : min(len(source), end + window)]
    prefix_hit = any(keyword in prefix for keyword in keywords)
    suffix_hit = any(keyword in suffix for keyword in keywords)
    return prefix_hit, suffix_hit


def interpret_shocks(text: str) -> Dict[str, float]:
    lowered = text.lower()
    shocks = {"rate_bps": 0.0, "credit_bps": 0.0, "fx_move": 0.0, "equity_move": 0.0}

    bp_matches = list(re.finditer(r"([+-]?\d+)\s?bp", lowered))
    bp_values = [float(match.group(1)) for match in bp_matches]
    pct_matches = list(re.finditer(r"([+-]?\d+(?:\.\d+)?)\s?%", lowered))
    pct_values = [float(match.group(1)) / 100.0 for match in pct_matches]
    used_pct_indexes: set[int] = set()

    rate_indexes: List[int] = []
    credit_indexes: List[int] = []

    for idx, match in enumerate(bp_matches):
        value = bp_values[idx]
        start, end = match.span()
        rate_prefix, rate_suffix = _keyword_window_hits(lowered, start, end, SCENARIO_KEYWORDS["rates"])
        credit_prefix, credit_suffix = _keyword_window_hits(lowered, start, end, SCENARIO_KEYWORDS["credit"])

        bucket_assigned: str | None = None
        if rate_prefix and shocks["rate_bps"] == 0.0:
            bucket_assigned = "rate_bps"
        elif credit_prefix and shocks["credit_bps"] == 0.0:
            bucket_assigned = "credit_bps"
        elif rate_suffix and not credit_suffix and shocks["rate_bps"] == 0.0:
            bucket_assigned = "rate_bps"
        elif credit_suffix and not rate_suffix and shocks["credit_bps"] == 0.0:
            bucket_assigned = "credit_bps"

        if bucket_assigned:
            shocks[bucket_assigned] = value
            continue

        if rate_prefix or rate_suffix:
            rate_indexes.append(idx)
        if credit_prefix or credit_suffix:
            credit_indexes.append(idx)

    if not bp_matches:
        if shocks["rate_bps"] == 0.0 and any(k in lowered for k in SCENARIO_KEYWORDS["rates"]):
            shocks["rate_bps"] = _detect_sign(lowered) * 25.0
        if shocks["credit_bps"] == 0.0 and any(k in lowered for k in SCENARIO_KEYWORDS["credit"]):
            shocks["credit_bps"] = _detect_sign(lowered) * 15.0
    else:
        if shocks["rate_bps"] == 0.0:
            if rate_indexes:
                shocks["rate_bps"] = bp_values[rate_indexes[0]]
            else:
                shocks["rate_bps"] = bp_values[0]
        if shocks["credit_bps"] == 0.0 and (credit_indexes or any(k in lowered for k in SCENARIO_KEYWORDS["credit"])):
            if credit_indexes:
                shocks["credit_bps"] = bp_values[credit_indexes[-1]]
            else:
                shocks["credit_bps"] = bp_values[-1]

    equity_indexes: List[int] = []
    fx_indexes: List[int] = []

    for idx, match in enumerate(pct_matches):
        value = pct_values[idx]
        start, end = match.span()
        fx_prefix, fx_suffix = _keyword_window_hits(lowered, start, end, SCENARIO_KEYWORDS["fx"])
        eq_prefix, eq_suffix = _keyword_window_hits(lowered, start, end, SCENARIO_KEYWORDS["equity"])

        bucket_assigned: str | None = None
        if fx_prefix and shocks["fx_move"] == 0.0:
            bucket_assigned = "fx_move"
        elif eq_prefix and shocks["equity_move"] == 0.0:
            bucket_assigned = "equity_move"
        elif fx_suffix and not eq_suffix and shocks["fx_move"] == 0.0:
            bucket_assigned = "fx_move"
        elif eq_suffix and not fx_suffix and shocks["equity_move"] == 0.0:
            bucket_assigned = "equity_move"

        if bucket_assigned:
            shocks[bucket_assigned] = value
            used_pct_indexes.add(idx)
            continue

        if fx_prefix or fx_suffix:
            fx_indexes.append(idx)
        if eq_prefix or eq_suffix:
            equity_indexes.append(idx)

    def assign_next_available(bucket: str) -> bool:
        for idx, value in enumerate(pct_values):
            if idx in used_pct_indexes:
                continue
            shocks[bucket] = value
            used_pct_indexes.add(idx)
            return True
        return False

    if not pct_matches:
        if any(k in lowered for k in SCENARIO_KEYWORDS["equity"]):
            shocks["equity_move"] = _detect_sign(lowered) * 0.02
        if any(k in lowered for k in SCENARIO_KEYWORDS["fx"]):
            shocks["fx_move"] = _detect_sign(lowered) * 0.01
    else:
        if shocks["equity_move"] == 0.0 and any(k in lowered for k in SCENARIO_KEYWORDS["equity"]):
            if equity_indexes:
                shocks["equity_move"] = pct_values[equity_indexes[-1]]
                used_pct_indexes.add(equity_indexes[-1])
            else:
                assigned = assign_next_available("equity_move")
                if not assigned and pct_values:
                    shocks["equity_move"] = pct_values[-1]
        if shocks["fx_move"] == 0.0 and any(k in lowered for k in SCENARIO_KEYWORDS["fx"]):
            if fx_indexes:
                # Prefer the earliest FX reference to capture leading currency remarks.
                shocks["fx_move"] = pct_values[fx_indexes[0]]
                used_pct_indexes.add(fx_indexes[0])
            else:
                assigned = assign_next_available("fx_move")
                if not assigned and pct_values:
                    shocks["fx_move"] = pct_values[0]

    return shocks


def _shocks_are_zero(shocks: Dict[str, float], tol: float = 1e-6) -> bool:
    return all(abs(value) <= tol for value in shocks.values())


def _maybe_llm_shocks(text: str) -> tuple[Dict[str, float] | None, str | None]:
    if not gemini_available():
        return None, "Gemini scenario parser unavailable; used heuristic shocks."
    try:
        payload = generate_scenario_shocks(text)
        return payload, "Gemini-derived shock magnitudes."
    except GeminiClientError as exc:
        return None, f"Gemini scenario parse failed: {exc}"


def evaluate_custom_scenario(summary: ExposureSummary, shock: Dict[str, float]) -> str:
    pnl_rates = -summary.totals["dv01"] * shock.get("rate_bps", 0.0)
    pnl_credit = -summary.totals["credit_spread_dv01"] * shock.get("credit_bps", 0.0)
    pnl_fx = summary.totals["fx_delta"] * shock.get("fx_move", 0.0)
    pnl_equity = summary.totals["beta_notional"] * shock.get("equity_move", 0.0)
    total = pnl_rates + pnl_credit + pnl_fx + pnl_equity

    def fmt(value: float) -> str:
        if abs(value) >= 1_000:
            return f"{value/1000:,.1f}k"
        return f"{value:,.0f}"

    return (
        f"Rates {fmt(pnl_rates)}, credit {fmt(pnl_credit)}, "
        f"FX {fmt(pnl_fx)}, equities {fmt(pnl_equity)} ⇒ net {fmt(total)}"
    )


def breakdown_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip("-• \t") for sentence in sentences if sentence]


def build_scenario_insights(
    pdf_path: Path, summary: ExposureSummary, *, use_gemini: bool = False
) -> List[ScenarioInsight]:
    raw_text = extract_pdf_text(pdf_path)
    bullets = derive_scenarios(raw_text)
    insights: List[ScenarioInsight] = []
    for bullet in bullets:
        heuristics = interpret_shocks(bullet)
        shocks = heuristics
        note: str | None = None

        if use_gemini:
            llm_shocks, llm_note = _maybe_llm_shocks(bullet)
            note = llm_note
            if llm_shocks:
                shocks = llm_shocks
        elif _shocks_are_zero(heuristics):
            note = "Scenario text lacked explicit shocks; consider enabling Gemini for richer parsing."

        impact = evaluate_custom_scenario(summary, shocks)
        insights.append(ScenarioInsight(text=bullet, shocks=shocks, impact=impact, note=note))
    return insights
