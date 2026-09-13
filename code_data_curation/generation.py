"""Generation adapters shared by mock and real endpoint execution."""

from __future__ import annotations

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .model_access import endpoint_generate, load_model_specs, resolve_model
from .code_trace import (
    execute_instrumented_python,
    format_trace_document,
    parse_execution_script,
    parse_instrumentation,
)


PROMPTS = {
    "swallowcode": "Rewrite this code into a self-contained, clear, idiomatic example while preserving behavior. Return code only.",
    "codedev": "Turn this source code into a realistic software-development task, such as a feature addition, bug fix, or behavior change. Include the task, relevant source context, and a complete solution as continuous pretraining text.",
    "codeqa": "Generate a technically precise question about this source code and an answer grounded only in the code. Focus on behavior, design, edge cases, or execution. Format as Question, Code, and Answer sections for pretraining; do not emit chat-role JSON.",
    "codetrace": "Instrument this Python program with deterministic TRACE:<TYPE>:<LOC>:<STATE> events on stderr, following CodeAlchemy Prompt C11. Use event types IN, OUT, VAR, BRANCH, LOOP, ERR, and TRANSFORM. Insert 15-25 selective trace points; trace conditions must depend on multiple state values and avoid randomness, timestamps, addresses, and system state. Preserve behavior, use only the standard library, read test data from stdin or CLI arguments, and return a complete ```python``` block followed by a ```json``` block describing the trace patterns.",
}

logger = logging.getLogger(__name__)


def build_prompt(document: dict, method: str, prompt_version: str = "v1") -> str:
    return (
        f"Prompt version: {prompt_version}\n{PROMPTS[method]}\n"
        f"Language: {document['language']}\n\nSource code:\n{document['content']}"
    )


def generate_document(document: dict, model_name: str, method: str, *, mock: bool = True, prompt_version: str = "v1") -> dict:
    if mock:
        from .pipeline import generate_mock
        return generate_mock(document, model_name, method, prompt_version)
    specs = load_model_specs()
    spec = specs[model_name]
    resolved = resolve_model(model_name, mock=False)
    prompt = build_prompt(document, method, prompt_version)
    if resolved.startswith("endpoint:"):
        endpoint = resolved.removeprefix("endpoint:")
        text = endpoint_generate(endpoint, spec.model_id, prompt, max_tokens=int(os.environ.get("CODE_DATA_MAX_TOKENS", "1024")))
        if method == "swallowcode":
            import re
            fenced = re.fullmatch(r"\s*```[^\n]*\n(.*?)\n```\s*", text, re.DOTALL)
            if fenced:
                text = fenced.group(1)
    else:
        raise RuntimeError(
            f"Local model path resolved for {spec.model_id}, but no local inference runtime is installed. "
            "Configure an OpenAI-compatible endpoint or install a compatible runtime explicitly."
        )
    execution_checked = False
    trace_deterministic = None
    if method == "codetrace":
        if document["language"].lower() != "python":
            raise ValueError("the CodeTrace pilot supports Python only")
        instrumented_code, patterns = parse_instrumentation(text)
        test_prompt = (
            "Following CodeAlchemy Prompt C12, generate a deterministic bash script with 3-5 "
            "structurally distinct tests for this instrumented Python file, saved as source.py. "
            "Use CLI arguments, stdin, or heredocs; redirect stderr to trace1.txt, trace2.txt, and "
            "so on. Use no packages, network, filesystem inputs, randomness, or interactive input. "
            "Return one ```bash``` block only.\n\nInstrumented code:\n"
            + instrumented_code
            + "\n\nTrace patterns:\n"
            + patterns
        )
        execution_script = parse_execution_script(
            endpoint_generate(endpoint, spec.model_id, test_prompt, max_tokens=1024)
        )
        trace = execute_instrumented_python(instrumented_code, execution_script)
        text = format_trace_document(instrumented_code, execution_script, trace)
        execution_checked = True
        trace_deterministic = True
    return {
        **document,
        "document_id": f"synthetic:{model_name}:{method}:{document['document_id']}",
        "parent_document_id": document["document_id"],
        "repo_id": document.get("repo_id"),
        "content_id_parent": document.get("content_id"),
        "mock": False,
        "content": text,
        "language": document["language"],
        "repo_path": document.get("repo_path"),
        "license_type": document.get("license_type"),
        "source_dataset": document.get("source_dataset"),
        "model": model_name,
        "model_version": spec.model_id,
        "method": method,
        "prompt_version": prompt_version,
        "synthetic": True,
        "token_count": len(text.split()),
        "execution_checked": execution_checked,
        "trace_deterministic": trace_deterministic,
    }


def generate_with_retries(document: dict, model_name: str, method: str, *, mock: bool, retries: int = 2, prompt_version: str = "v1") -> dict:
    last_error = None
    for attempt in range(retries + 1):
        try:
            result = generate_document(document, model_name, method, mock=mock, prompt_version=prompt_version)
            from .pipeline import validate_generated
            valid, invalid = validate_generated([result])
            if invalid:
                raise ValueError(invalid[0]["rejection_reason"])
            result = valid[0]
            result["generation_attempts"] = attempt + 1
            return result
        except Exception as exc:  # bounded and surfaced in the caller's shortfall report
            last_error = exc
    raise RuntimeError(f"generation failed after {retries + 1} attempts: {last_error}") from last_error


def generate_batch(documents: list[dict], model_name: str, method: str, *, mock: bool = True, retries: int = 2, concurrency: int = 2, prompt_version: str = "v1") -> tuple[list[dict], list[dict]]:
    if retries < 0 or concurrency < 1:
        raise ValueError("retries must be nonnegative and concurrency must be positive")
    accepted, failures = [], []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(generate_with_retries, d, model_name, method, mock=mock, retries=retries, prompt_version=prompt_version): d for d in documents}
        for future in as_completed(futures):
            parent = futures[future]
            try:
                accepted.append(future.result())
            except Exception as exc:
                failures.append({"parent_document_id": parent["document_id"], "model": model_name, "method": method, "error": str(exc)})
            logger.warning("%s/%s progress %d/%d: accepted=%d rejected=%d", model_name, method, len(accepted) + len(failures), len(documents), len(accepted), len(failures))
    accepted.sort(key=lambda d: d["document_id"])
    return accepted, failures
