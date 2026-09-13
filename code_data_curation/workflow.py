"""Flyte 1 task graph for the small code-curation exercise."""

import logging
import json
from pathlib import Path

from flytekit import task, workflow

from .pipeline import (
    decontaminate,
    exact_deduplicate,
    export_dataset,
    load_jsonl,
    near_deduplicate,
    quality_filter,
    repository_split,
    select_branch_parents,
)
from .generation import generate_batch


@task(cache=False)
def ingest(path: str, output_dir: str = "", mode: str = "mock") -> list[dict]:
    if mode not in {"mock", "real"}:
        raise ValueError("mode must be mock or real")
    if mode == "real":
        from .model_access import resolve_model
        for model in ("qwen_coder", "codegemma"):
            if not resolve_model(model).startswith("endpoint:"):
                raise RuntimeError("Real generation requires endpoint configuration")
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=False)
    return load_jsonl(path)


@task(cache=False)
def load_evaluation(path: str) -> list[dict]:
    return load_jsonl(path) if path else []


@task(cache=False)
def clean(documents: list[dict], output_dir: str = "") -> list[dict]:
    accepted, rejected = quality_filter(documents)
    unique = near_deduplicate(exact_deduplicate(accepted))
    if output_dir:
        write_audit(output_dir, "quality-rejections", rejected)
    return unique


@task(cache=True, cache_version="review-v5")
def drop_evaluation_matches(documents: list[dict], evaluation: list[dict]) -> list[dict]:
    kept, _ = decontaminate(documents, evaluation)
    return kept


@task(cache=True, cache_version="v2")
def split(documents: list[dict]) -> tuple[list[dict], list[dict]]:
    return repository_split(documents)


logger = logging.getLogger(__name__)


def write_audit(output_dir: str, name: str, rows: list[dict]) -> None:
    # Do not duplicate potentially secret-bearing source text in audit logs.
    keys = ("document_id", "parent_document_id", "model", "method", "rejection_reason", "error")
    with (Path(output_dir) / f"{name}.jsonl").open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({key: row[key] for key in keys if key in row}) + "\n")


@task(cache=False)
def synthesize_branch(
    documents: list[dict], model: str, method: str, mode: str,
    generation_retries: int, generation_concurrency: int, codetrace_limit: int,
    output_dir: str = "",
) -> list[dict]:
    if mode not in {"mock", "real"}:
        raise ValueError("mode must be 'mock' or 'real'")
    parents = select_branch_parents(
        documents, model, method, codetrace_limit=codetrace_limit
    )
    generated, failures = generate_batch(
        parents, model, method, mock=mode == "mock",
        retries=generation_retries, concurrency=generation_concurrency,
    )
    if output_dir:
        write_audit(output_dir, f"failures-{model}-{method}", failures)
        with (Path(output_dir) / f"branch-{model}-{method}.jsonl").open("x", encoding="utf-8") as handle:
            for row in generated:
                handle.write(json.dumps(row) + "\n")
    if failures:
        logger.warning(
            "%s/%s accepted %d documents and rejected %d after bounded retries; "
            "the export manifest will report the resulting quota shortfall",
            model, method, len(generated), len(failures),
        )
    return generated


@task(cache=True, cache_version="v2")
def combine_branches(
    branch_1: list[dict], branch_2: list[dict], branch_3: list[dict],
    branch_4: list[dict], branch_5: list[dict], branch_6: list[dict],
    branch_7: list[dict], branch_8: list[dict],
) -> list[dict]:
    return [document for branch in (branch_1, branch_2, branch_3, branch_4, branch_5, branch_6, branch_7, branch_8) for document in branch]


@task(cache=False)
def export(
    organic: list[dict], synthetic: list[dict], evaluation: list[dict],
    output_dir: str, codetrace_target_per_model: int,
    validation: list[dict] = [], mode: str = "mock",
    post_generation_decontamination: bool = False,
) -> dict:
    clean_synthetic, contaminated = synthetic, []
    if post_generation_decontamination:
        clean_synthetic, contaminated = decontaminate(synthetic, evaluation + validation)
    manifest = export_dataset(
        organic, clean_synthetic, output_dir,
        codetrace_target_per_model=codetrace_target_per_model,
    )
    write_audit(output_dir, "export-rejections", contaminated)
    with (Path(output_dir) / "validation.jsonl").open("x", encoding="utf-8") as handle:
        for row in validation:
            handle.write(json.dumps(row) + "\n")
    manifest.update(mode=mode, validation_documents=len(validation), evaluation_documents=len(evaluation), evaluation_configured=bool(evaluation), status="completed_with_shortfalls" if manifest["synthetic_shortfall_documents"] else "completed")
    manifest.update(post_generation_deduplication=False, post_generation_decontamination=post_generation_decontamination, synthetic_contaminated=len(contaminated))
    (Path(output_dir) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@workflow
def code_data_curation_workflow(
    input_path: str,
    output_dir: str,
    mode: str = "mock",
    generation_retries: int = 2,
    generation_concurrency: int = 2,
    codetrace_documents_per_model: int = 1000,
    evaluation_path: str = "",
    post_generation_decontamination: bool = False,
) -> dict:
    evaluation = load_evaluation(evaluation_path)
    organic = drop_evaluation_matches(clean(ingest(input_path, output_dir, mode), output_dir), evaluation)
    train, _validation = split(organic)
    generated = combine_branches(
        synthesize_branch(train, "qwen_coder", "swallowcode", mode, generation_retries, generation_concurrency, codetrace_documents_per_model, output_dir),
        synthesize_branch(train, "qwen_coder", "codedev", mode, generation_retries, generation_concurrency, codetrace_documents_per_model, output_dir),
        synthesize_branch(train, "qwen_coder", "codeqa", mode, generation_retries, generation_concurrency, codetrace_documents_per_model, output_dir),
        synthesize_branch(train, "codegemma", "swallowcode", mode, generation_retries, generation_concurrency, codetrace_documents_per_model, output_dir),
        synthesize_branch(train, "codegemma", "codedev", mode, generation_retries, generation_concurrency, codetrace_documents_per_model, output_dir),
        synthesize_branch(train, "codegemma", "codeqa", mode, generation_retries, generation_concurrency, codetrace_documents_per_model, output_dir),
        synthesize_branch(train, "qwen_coder", "codetrace", mode, generation_retries, generation_concurrency, codetrace_documents_per_model, output_dir),
        synthesize_branch(train, "codegemma", "codetrace", mode, generation_retries, generation_concurrency, codetrace_documents_per_model, output_dir),
    )
    return export(train, generated, evaluation, output_dir, codetrace_documents_per_model, _validation, mode, post_generation_decontamination)
