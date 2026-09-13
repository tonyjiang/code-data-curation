"""Python-only CodeTrace pilot following CodeAlchemy's published procedure."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import ast
import uuid
from pathlib import Path


TRACE_IMAGE = os.environ.get(
    "CODE_DATA_TRACE_IMAGE", "docker.m.daocloud.io/library/python:3.12-slim"
)
TRACE_EVENT_RE = re.compile(r"TRACE:(IN|OUT|VAR|BRANCH|LOOP|ERR|TRANSFORM):")
TRACE_OUTPUT_RE = re.compile(r"^TRACE:(IN|OUT|VAR|BRANCH|LOOP|ERR|TRANSFORM):", re.MULTILINE)


def extract_fenced_block(text: str, language: str) -> str:
    match = re.search(rf"```(?:{language})?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


def parse_instrumentation(text: str) -> tuple[str, str]:
    blocks = re.findall(r"```([^\n`]*)\n(.*?)```", text, re.DOTALL)
    code = next((body.strip() for label, body in blocks if label.strip().lower() in {"python", "py"}), "")
    patterns = next((body.strip() for label, body in blocks if label.strip().lower() == "json"), "{}")
    if not code:
        raise ValueError("instrumentation response did not contain a Python code block")
    if not TRACE_EVENT_RE.search(code):
        raise ValueError("instrumented program contains no CodeAlchemy TRACE events")
    ast.parse(code)
    json.loads(patterns)
    return code, patterns


def parse_execution_script(text: str) -> str:
    script = extract_fenced_block(text, "bash")
    forbidden = re.compile(r"(?im)\b(curl|wget|ssh|scp|nc|sudo|docker|podman|pip|apt|brew)\b")
    if forbidden.search(script):
        raise ValueError("execution script contains a forbidden command")
    if "trace" not in script or "source.py" not in script:
        raise ValueError("execution script must run source.py and create trace files")
    return script


def _ensure_image_available(image: str) -> None:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, text=True, timeout=10
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Docker is required for real CodeTrace execution") from exc
    if result.returncode:
        raise RuntimeError(
            f"CodeTrace requires an existing {image} image; images are never pulled automatically"
        )


def execute_instrumented_python(
    source: str,
    execution_script: str,
    *,
    runs: int = 3,
    timeout_seconds: int = 30,
    image: str = TRACE_IMAGE,
) -> str:
    """Run identical inputs three times in a locked-down container."""
    if runs < 1 or timeout_seconds <= 0:
        raise ValueError("runs and timeout_seconds must be positive")
    _ensure_image_available(image)
    outputs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="codetrace-") as directory:
        root = Path(directory)
        (root / "source.py").write_text(source, encoding="utf-8")
        (root / "run.sh").write_text(execution_script, encoding="utf-8")
        collector = (
            "set -eu; mkdir -p /tmp/work; cp /work/input/* /tmp/work/; cd /tmp/work; "
            "bash run.sh >/dev/null; "
            "for f in $(find . -maxdepth 1 -name 'trace*.txt' -type f | sort); do "
            "printf '===STDERR:%s:START===\\n' \"${f#./}\"; cat \"$f\"; "
            "printf '===STDERR:%s:END===\\n' \"${f#./}\"; done"
        )
        command = [
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "30", "--memory", "512m", "--cpus", "1",
            "--tmpfs", "/tmp:rw,nosuid,size=64m",
            "--volume", f"{root}:/work/input:ro", image, "bash", "-lc", collector,
        ]
        for _ in range(runs):
            name = f"codetrace-{uuid.uuid4().hex}"
            try:
                result = subprocess.run(command[:2] + ["--name", name] + command[2:], capture_output=True, text=True, timeout=timeout_seconds)
            finally:
                # Killing the Docker client on timeout does not stop its container.
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=10)
            if result.returncode:
                raise RuntimeError(f"trace execution failed: {result.stderr.strip()[:500]}")
            if not TRACE_OUTPUT_RE.search(result.stdout):
                raise RuntimeError("execution produced an empty CodeAlchemy trace")
            outputs.append(result.stdout)
    if len(set(outputs)) != 1:
        raise RuntimeError("trace was nondeterministic across three identical runs")
    return outputs[0]


def trace_metrics(trace: str) -> dict:
    events = TRACE_OUTPUT_RE.findall(trace)
    files = re.findall(r"^===STDERR:[^:]+:START===$", trace, re.MULTILINE)
    counts = {event: events.count(event) for event in sorted(set(events))}
    return {
        "trace_files": len(files),
        "trace_events": len(events),
        "trace_characters": len(trace),
        "event_types": counts,
        "largest_event_type_fraction": max(counts.values(), default=0) / max(1, len(events)),
    }


def format_trace_document(instrumented_code: str, execution_script: str, trace: str) -> str:
    """Serialize as CodeAlchemy-style causal pretraining text."""
    return (
        "User:\nYou are provided an instrumented source file \"source.py\" and a bash execution "
        "script \"run.sh\". Trace through the execution and predict the generated trace files.\n\n"
        "INPUT FILES:\n\n---- FILENAME: source.py ----\n\n"
        + instrumented_code.rstrip()
        + "\n\n---- FILENAME: run.sh ----\n\n"
        + execution_script.rstrip()
        + "\n\nAssistant:\n"
        + trace.rstrip()
    )
