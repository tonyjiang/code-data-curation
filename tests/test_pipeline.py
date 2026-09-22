import json
import pytest

from code_data_curation.pipeline import (
    allocate_branch_quotas,
    decontaminate,
    exact_deduplicate,
    generate_branches,
    quality_filter,
    validate_generated,
)
from code_data_curation import workflow
from code_data_curation.generation import build_prompt, generate_batch
from code_data_curation.code_trace import parse_execution_script, parse_instrumentation, trace_metrics


def doc(i, language="Python", repo=None, content=None):
    return {"document_id": str(i), "content_id": str(i), "content": content or f"print({i})", "language": language, "repo_id": repo or i}


def valid_python(i=1):
    return f'''def transform_{i}(input_{i}, factor_{i}):
    alpha_{i} = input_{i} + factor_{i}
    beta_{i} = alpha_{i} * factor_{i}
    gamma_{i} = beta_{i} - input_{i}
    delta_{i} = gamma_{i} / factor_{i}
    epsilon_{i} = delta_{i} + alpha_{i}
    zeta_{i} = epsilon_{i} * beta_{i}
    eta_{i} = zeta_{i} - gamma_{i}
    theta_{i} = eta_{i} + delta_{i}
    return theta_{i}
'''


def test_filters_secrets_vendor_and_duplicates():
    source = [doc(1, content=valid_python()), doc(1, content=valid_python()), doc(2, content="password = 'abcdefghijk'"), {**doc(3), "is_vendor": True}]
    unique = exact_deduplicate(source)
    accepted, rejected = quality_filter(unique)
    assert len(accepted) == 1
    assert {r["rejection_reason"] for r in rejected} == {"secret", "vendor"}


def test_decontamination():
    train, removed = decontaminate([doc(1), doc(2)], [doc("eval", content="print(1)")])
    assert len(train) == 1 and removed[0]["rejection_reason"] == "evaluation_contamination"


@pytest.mark.parametrize("audit", [False, True])
def test_export_retains_duplicates_and_optionally_decontaminates(tmp_path, audit):
    organic = [doc(1, content="print(1)")]
    evaluation = [doc("eval", content="eval_only()")]
    synthetic = [
        {**doc("copy"), "content": "print(1)", "model": "qwen_coder", "method": "swallowcode"},
        {**doc("leak"), "content": "eval_only()", "model": "qwen_coder", "method": "swallowcode"},
        {**doc("keep"), "content": "print(99)", "model": "qwen_coder", "method": "swallowcode"},
    ]
    kept = workflow.drop_evaluation_matches.task_function(organic + [doc(2, content="eval_only()")], evaluation)
    assert {d["document_id"] for d in kept} == {"1"}
    manifest = workflow.export.task_function(organic, synthetic, evaluation, str(tmp_path), 1000, post_generation_decontamination=audit)
    written = [json.loads(line) for line in (tmp_path / "train-00000.jsonl").read_text().splitlines()]
    ids = {d["document_id"] for d in written}
    assert ids == ({"1", "copy", "keep"} if audit else {"1", "copy", "leak", "keep"})
    assert manifest["post_generation_decontamination"] is audit
    assert manifest["post_generation_deduplication"] is False
    assert manifest["synthetic_contaminated"] == int(audit)


def test_eight_branches_with_paired_python_codetrace_pilot():
    organic = [doc(i, language="Python" if i % 2 else "Go") for i in range(20)]
    generated = generate_branches(organic)
    assert len({(d["model"], d["method"]) for d in generated}) == 8
    base = [d for d in generated if d["method"] != "codetrace"]
    traces = [d for d in generated if d["method"] == "codetrace"]
    assert len(base) == sum(allocate_branch_quotas(len(organic)))
    assert len(traces) == 20  # Ten available Python parents through both models.
    parents_by_model = {
        model: {d["parent_document_id"] for d in traces if d["model"] == model}
        for model in ("qwen_coder", "codegemma")
    }
    assert parents_by_model["qwen_coder"] == parents_by_model["codegemma"]


def test_real_branch_returns_empty_when_all_generations_fail(monkeypatch):
    parent = doc(1, language="Python")
    monkeypatch.setattr(
        workflow,
        "generate_batch",
        lambda *args, **kwargs: ([], [{"error": "instrumentation rejected"}]),
    )

    generated = workflow.synthesize_branch.task_function(
        [parent], "qwen_coder", "codetrace", "real", 2, 1, 1
    )

    assert generated == []
    accepted, rejected = validate_generated(generated)
    assert len(accepted) == len(generated) and not rejected


def test_generation_interface_is_bounded_and_grounded():
    source = [doc(i) for i in range(4)]
    prompt = build_prompt(source[0], "codeqa")
    assert "question" in prompt.lower() and source[0]["content"] in prompt
    outputs, failures = generate_batch(source, "qwen_coder", "codeqa", mock=True, concurrency=2)
    assert len(outputs) == 4 and failures == []
    assert all(d["generation_attempts"] == 1 for d in outputs)


def test_codealchemy_trace_format_parsing_and_metrics():
    response = '''```python
import sys
print("TRACE:VAR:main:x=1", file=sys.stderr)
```
```json
{"trace_patterns_used": [{"name": "state change"}]}
```'''
    code, patterns = parse_instrumentation(response)
    assert "TRACE:VAR" in code and "trace_patterns_used" in patterns
    script = parse_execution_script("```bash\npython source.py 2> trace1.txt\n```")
    assert "trace1.txt" in script
    metrics = trace_metrics("===STDERR:trace1.txt:START===\nTRACE:VAR:main:x=1\n===STDERR:trace1.txt:END===\n")
    assert metrics["trace_files"] == 1 and metrics["trace_events"] == 1
