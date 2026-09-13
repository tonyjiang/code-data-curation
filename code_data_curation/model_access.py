"""Guarded model resolution for mock, local, and endpoint-backed generation."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    source: str
    access: str
    server_path: str
    local_path_env: str
    endpoint_env: str


def load_model_specs(path: str | Path = "config/models.json") -> dict[str, ModelSpec]:
    data = json.loads(Path(path).read_text())
    return {
        name: ModelSpec(name=name, **values)
        for name, values in data["models"].items()
    }


def resolve_model(name: str, path: str | Path = "config/models.json", *, mock: bool = False) -> str:
    """Return ``mock``, a local path, or an endpoint; never downloads weights."""
    if mock:
        return "mock"
    spec = load_model_specs(path)[name]
    local_path = os.environ.get(spec.local_path_env)
    endpoint = os.environ.get(spec.endpoint_env)
    if endpoint:
        return f"endpoint:{endpoint}"
    if local_path and Path(local_path).exists():
        return f"local:{local_path}"
    raise RuntimeError(
        f"No access configured for {spec.model_id}. Set {spec.local_path_env} "
        f"to an existing model directory or {spec.endpoint_env} to an existing endpoint. "
        "Automatic weight downloads are disabled."
    )


def mock_generate(model: str, method: str, source: str) -> str:
    """Produce visibly mocked output for offline tests."""
    return (
        f"# MOCK_SYNTHETIC model={model} method={method}\n"
        "# This output is deterministic test data, not model inference.\n"
        f"{source}"
    )


def endpoint_generate(endpoint: str, model_id: str, prompt: str, *, max_tokens: int = 256) -> str:
    """Call an existing OpenAI-compatible endpoint; never starts a server."""
    url = endpoint.rstrip("/") + "/chat/completions"
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }).encode()
    request = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read())
    choice = body["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"Incomplete model response: {choice.get('finish_reason')}")
    content = choice["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Model returned empty content")
    return content
