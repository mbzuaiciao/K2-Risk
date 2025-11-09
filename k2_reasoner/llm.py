from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None


class GeminiClientError(RuntimeError):
    """Raised when the Gemini API cannot be used."""


def gemini_available() -> bool:
    """Return True when the SDK and API key are present."""
    return genai is not None and bool(os.getenv("GEMINI_API_KEY"))


def _configure_model():
    if not gemini_available():
        raise GeminiClientError("Gemini SDK not installed or GEMINI_API_KEY missing.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiClientError("GEMINI_API_KEY environment variable is not set.")

    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "models/gemini-1.0-pro")
    return genai.GenerativeModel(model_name)


@lru_cache(maxsize=1)
def _get_model():
    return _configure_model()


def generate_reasoning_response(prompt: str) -> str:
    """
    Call Gemini with the provided prompt and return the response text.
    """

    model = _get_model()
    try:
        response = model.generate_content(prompt)
    except Exception as exc:  # pragma: no cover - network path
        raise GeminiClientError(f"Gemini call failed: {exc}") from exc

    text = getattr(response, "text", None)
    if not text:
        raise GeminiClientError("Gemini returned an empty response.")
    return text


def _extract_json_fragment(payload: str) -> str:
    stripped = payload.strip()

    if stripped.startswith("```"):
        segments = stripped.split("```")
        for segment in segments:
            candidate = segment.strip()
            if candidate.startswith("{") and candidate.endswith("}"):
                return candidate

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]

    raise GeminiClientError("Unable to locate JSON object in Gemini response.")


def parse_reasoning_payload(payload: str) -> Dict[str, Any]:
    """
    Expect a JSON payload with fields:
      - chain: list[str]
      - narrative: str
    """

    try:
        fragment = _extract_json_fragment(payload)
        data = json.loads(fragment)
    except (json.JSONDecodeError, GeminiClientError) as exc:
        raise GeminiClientError(f"Unable to parse Gemini JSON: {exc}") from exc

    chain = data.get("chain")
    narrative = data.get("narrative")
    if not isinstance(chain, list) or not isinstance(narrative, str):
        raise GeminiClientError("Gemini response missing required fields.")
    return {"chain": [str(item) for item in chain], "narrative": narrative.strip()}


def _scenario_prompt(text: str) -> str:
    instructions = """
You are an institutional risk analyst. Convert the scenario text into numeric shocks.
Return strict JSON with floating point values:
{
  "rate_bps": float (positive for higher yields),
  "credit_bps": float (positive for wider spreads),
  "fx_move": float (as decimal, positive = USD stronger),
  "equity_move": float (as decimal, positive = equities rally)
}
Infer reasonable magnitudes when none are provided; use basis points / percent intuition.
"""
    return f"{instructions}\nScenario: {text}\nJSON:"


def parse_scenario_payload(payload: str) -> Dict[str, float]:
    try:
        fragment = _extract_json_fragment(payload)
        data = json.loads(fragment)
    except (json.JSONDecodeError, GeminiClientError) as exc:
        raise GeminiClientError(f"Unable to parse Gemini scenario JSON: {exc}") from exc

    required = ("rate_bps", "credit_bps", "fx_move", "equity_move")
    try:
        parsed = {key: float(data[key]) for key in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise GeminiClientError("Gemini scenario payload missing numeric fields.") from exc
    return parsed


def generate_scenario_shocks(text: str) -> Dict[str, float]:
    """Use Gemini to infer shock magnitudes from scenario prose."""

    if not gemini_available():
        raise GeminiClientError("Gemini SDK or API key missing; cannot generate scenario shocks.")

    model = _get_model()
    prompt = _scenario_prompt(text)
    try:
        response = model.generate_content(prompt)
    except Exception as exc:  # pragma: no cover - network path
        raise GeminiClientError(f"Gemini scenario call failed: {exc}") from exc

    text_response = getattr(response, "text", None)
    if not text_response:
        raise GeminiClientError("Gemini returned an empty scenario response.")
    return parse_scenario_payload(text_response)
