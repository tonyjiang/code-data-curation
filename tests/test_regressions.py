import json
import subprocess

import pytest

from code_data_curation import code_trace, model_access, pipeline, workflow


def record(identifier, content="print(1)", **extra):
    return dict(document_id=str(identifier), content=content, language="Python", repo_id=str(identifier), **extra)


def test_exact_dedup_uses_content_not_untrusted_ids():
    rows = [record(1, content_id="a"), record(2, content_id="b"), record(3, "print(2)", content_id="a")]
    assert [d["document_id"] for d in pipeline.exact_deduplicate(rows)] == ["1", "3"]


def test_quality_checks_python_and_vendor_paths():
    accepted, rejected = pipeline.quality_filter([record(1), record(2, "def bad(:"), record(3, file_path="src/vendor/util.py"), record(4, None)])
    assert len(accepted) == 1
    assert {d["rejection_reason"] for d in rejected} == {"malformed_python", "vendor", "malformed"}


def test_embedded_python_and_secret_validation():
    bad = record(1, "Solution:\n```python\ndef bad(:\n```", method="codedev")
    secret = record(2, "password = 'abcdefghijk'", method="codeqa")
    assert not pipeline.validate_generated([bad, secret])[0]
    unchecked = record(3, "Question: Why? Answer: because.", method="codeqa")
    assert pipeline.validate_generated([unchecked])[0][0]["validation"]["syntax_valid"] is None


def test_quotas_preserve_skewed_language_proportions():
    rows = [record(i) for i in range(90)] + [dict(document_id=str(i), language="Go", content="package main") for i in range(90, 100)]
    selected = pipeline.select_branch_parents(rows, "qwen_coder", "swallowcode")
    assert len(selected) == 25
    assert sum(d["language"] == "Python" for d in selected) == 22  # Tie goes to Go alphabetically.
    assert pipeline.select_branch_parents(rows, "qwen_coder", "codetrace", codetrace_limit=0) == []


def test_export_refuses_overwrite_and_does_not_mutate(tmp_path):
    source = record(1)
    pipeline.export_dataset([source], [], tmp_path, codetrace_target_per_model=0)
    assert "token_count" not in source
    with pytest.raises(FileExistsError):
        pipeline.export_dataset([], [], tmp_path)


def test_model_truncation_is_rejected(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({"choices": [{"finish_reason": "length", "message": {"content": "def unfinished"}}]}).encode()
    monkeypatch.setattr(model_access.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="Incomplete"):
        model_access.endpoint_generate("http://localhost/v1", "test", "test")


def test_endpoint_wins_over_local_path(monkeypatch, tmp_path):
    monkeypatch.setenv("CODE_DATA_QWEN_ENDPOINT", "http://localhost/v1")
    monkeypatch.setenv("CODE_DATA_QWEN_MODEL_PATH", str(tmp_path))
    assert model_access.resolve_model("qwen_coder").startswith("endpoint:")


def test_trace_timeout_removes_container(monkeypatch):
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        if command[1] == "run":
            raise subprocess.TimeoutExpired(command, 1)
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(code_trace.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        code_trace.execute_instrumented_python("pass", "python source.py", timeout_seconds=1)
    assert commands[-1][:3] == ["docker", "rm", "-f"]
    assert commands[-1][-1] == commands[-2][commands[-2].index("--name") + 1]


def test_mock_flyte_exports_holdout_and_protects_directory(tmp_path, monkeypatch):
    from flytekit.core import local_cache
    monkeypatch.setattr(local_cache, "CACHE_LOCATION", str(tmp_path / "flyte-cache"))
    monkeypatch.setattr(local_cache.LocalTaskCache, "_initialized", False)
    source = tmp_path / "input.jsonl"
    source.write_text("\n".join(json.dumps(record(i, f"print({i})")) for i in range(20)))
    output = tmp_path / "new-run"
    result = workflow.code_data_curation_workflow(input_path=str(source), output_dir=str(output), codetrace_documents_per_model=2)
    assert result["mode"] == "mock"
    assert (output / "validation.jsonl").exists()
    assert (output / "quality-rejections.jsonl").exists()
    with pytest.raises(FileExistsError):
        workflow.ingest.task_function(str(source), str(output))
