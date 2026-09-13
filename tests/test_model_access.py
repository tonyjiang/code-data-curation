import json

import pytest

from code_data_curation.model_access import load_model_specs, mock_generate, resolve_model


def test_model_manifest_has_requested_models():
    specs = load_model_specs()
    assert specs["qwen_coder"].model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert specs["codegemma"].model_id == "google/codegemma-7b-it"


def test_resolution_requires_existing_access(monkeypatch, tmp_path):
    monkeypatch.delenv("CODE_DATA_QWEN_MODEL_PATH", raising=False)
    monkeypatch.delenv("CODE_DATA_QWEN_ENDPOINT", raising=False)
    with pytest.raises(RuntimeError, match="Automatic weight downloads are disabled"):
        resolve_model("qwen_coder")
    assert resolve_model("qwen_coder", mock=True) == "mock"


def test_mock_output_is_marked():
    output = mock_generate("qwen_coder", "swallowcode", "print('ok')")
    assert "MOCK_SYNTHETIC" in output
    assert "print('ok')" in output
