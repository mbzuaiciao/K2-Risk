from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, List

import requests

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None


class GeminiClientError(RuntimeError):
    """Raised when the Gemini API cannot be used."""


class OpenRouterClientError(RuntimeError):
    """Raised when the OpenRouter API cannot be used."""


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

    raise ValueError("Unable to locate JSON object within response.")


def parse_reasoning_payload(
    payload: str, *, error_cls: type[Exception] = GeminiClientError
) -> Dict[str, Any]:
    """
    Expect a JSON payload with fields:
      - chain: list[str]
      - narrative: str
    """

    try:
        fragment = _extract_json_fragment(payload)
        data = json.loads(fragment)
    except (json.JSONDecodeError, ValueError) as exc:
        raise error_cls(f"Unable to parse reasoning JSON: {exc}") from exc

    chain = data.get("chain")
    narrative = data.get("narrative")
    if not isinstance(chain, list) or not isinstance(narrative, str):
        raise error_cls("LLM response missing required fields.")
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


def parse_scenario_payload(
    payload: str, *, error_cls: type[Exception] = GeminiClientError
) -> Dict[str, float]:
    try:
        fragment = _extract_json_fragment(payload)
        data = json.loads(fragment)
    except (json.JSONDecodeError, ValueError) as exc:
        raise error_cls(f"Unable to parse scenario JSON: {exc}") from exc

    required = ("rate_bps", "credit_bps", "fx_move", "equity_move")
    try:
        parsed = {key: float(data[key]) for key in required}
    except (KeyError, TypeError, ValueError) as exc:
        raise error_cls("Scenario payload missing numeric fields.") from exc
    return parsed


def generate_gemini_scenario_shocks(text: str) -> Dict[str, float]:
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
    return parse_scenario_payload(text_response, error_cls=GeminiClientError)


def openrouter_available() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _openrouter_headers() -> Dict[str, str]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterClientError("OPENROUTER_API_KEY environment variable is not set.")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    referer = os.getenv("OPENROUTER_SITE_URL")
    if referer:
        headers["HTTP-Referer"] = referer
    app_title = os.getenv("OPENROUTER_APP_NAME", "K2 Risk Reasoner")
    headers["X-Title"] = app_title
    return headers


def _openrouter_model() -> str:
    return os.getenv("OPENROUTER_MODEL", "openrouter/llama-3.1-70b-instruct")


def _openrouter_endpoint() -> str:
    return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")


def _openrouter_call(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    if not openrouter_available():
        raise OpenRouterClientError("OpenRouter API key missing.")
    payload = {"model": _openrouter_model(), "messages": messages, "temperature": temperature}
    try:
        response = requests.post(
            _openrouter_endpoint(), headers=_openrouter_headers(), json=payload, timeout=60
        )
    except requests.RequestException as exc:  # pragma: no cover - network path
        raise OpenRouterClientError(f"OpenRouter call failed: {exc}") from exc

    if response.status_code >= 400:
        snippet = response.text[:200]
        raise OpenRouterClientError(f"OpenRouter error {response.status_code}: {snippet}")

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError) as exc:
        raise OpenRouterClientError("OpenRouter response missing usable content.") from exc

    if isinstance(content, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    else:
        text = str(content)
    if not text.strip():
        raise OpenRouterClientError("OpenRouter returned an empty response.")
    return text.strip()


def generate_openrouter_reasoning(prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are K2 Risk, an institutional risk analyst. Respond strictly in JSON per instructions.",
        },
        {"role": "user", "content": prompt},
    ]
    return _openrouter_call(messages)


def generate_openrouter_scenario_shocks(text: str) -> Dict[str, float]:
    prompt = _scenario_prompt(text)
    messages = [
        {
            "role": "system",
            "content": "You translate macro scenarios into numeric shocks. Respond strictly with JSON as instructed.",
        },
        {"role": "user", "content": prompt},
    ]
    response = _openrouter_call(messages)
    return parse_scenario_payload(response, error_cls=OpenRouterClientError)
