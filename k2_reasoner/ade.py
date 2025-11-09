from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests

ADE_DEFAULT_ENDPOINT = "https://api.va.landing.ai/v1/ade/parse"


class ADEClientError(RuntimeError):
    """Raised when ADE parsing fails."""


def ade_available() -> bool:
    return bool(os.getenv("ADE_API_KEY"))


def _ade_endpoint() -> str:
    return os.getenv("ADE_ENDPOINT", ADE_DEFAULT_ENDPOINT)


def _ade_headers() -> dict[str, str]:
    api_key = os.getenv("ADE_API_KEY")
    if not api_key:
        raise ADEClientError("ADE_API_KEY environment variable is not set.")
    return {"Authorization": f"Bearer {api_key}"}


def parse_document(
    *,
    file_path: Optional[Path] = None,
    document_url: Optional[str] = None,
    model: Optional[str] = None,
    split: Optional[str] = None,
) -> dict:
    if not ade_available():
        raise ADEClientError("ADE API key missing.")
    if file_path is None and not document_url:
        raise ADEClientError("Provide file_path or document_url to parse_document.")

    files = {}
    data: dict[str, str] = {}
    if file_path:
        files["document"] = (file_path.name, file_path.read_bytes())
    if document_url:
        data["document_url"] = document_url
    if model:
        data["model"] = model
    if split:
        data["split"] = split

    try:
        response = requests.post(
            _ade_endpoint(),
            headers=_ade_headers(),
            data=data or None,
            files=files or None,
            timeout=60,
        )
    except requests.RequestException as exc:  # pragma: no cover - network path
        raise ADEClientError(f"ADE request failed: {exc}") from exc

    if response.status_code >= 400:
        snippet = response.text[:200]
        raise ADEClientError(f"ADE error {response.status_code}: {snippet}")

    try:
        return response.json()
    except ValueError as exc:
        raise ADEClientError(f"ADE response not JSON: {exc}") from exc


def extract_markdown(payload: dict) -> str:
    primary = payload.get("markdown")
    if isinstance(primary, str) and primary.strip():
        return primary

    parts: list[str] = []
    for chunk in payload.get("chunks") or []:
        text = chunk.get("markdown")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    for split in payload.get("splits") or []:
        text = split.get("markdown")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(parts)
