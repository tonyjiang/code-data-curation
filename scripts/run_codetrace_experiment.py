#!/usr/bin/env python3
"""Run the paired, Python-only CodeTrace data-quality experiment."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_data_curation.code_trace import trace_metrics
from code_data_curation.generation import generate_batch
from code_data_curation.pipeline import (
    exact_deduplicate,
    load_jsonl,
    near_deduplicate,
    quality_filter,
    repository_split,
    select_branch_parents,
    validate_generated,
)


def wilson(successes: int, total: int) -> list[float]:
    if not total:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def mcnemar_exact(qwen_only: int, gemma_only: int) -> float:
    discordant = qwen_only + gemma_only
    if not discordant:
        return 1.0
    smaller = min(qwen_only, gemma_only)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/stack-v3-full-shard-python.jsonl")
    parser.add_argument("--output", default="data/codetrace-experiment")
    parser.add_argument("--documents", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--mode", choices=("mock", "real"), default="mock")
    args = parser.parse_args()

    unique = near_deduplicate(exact_deduplicate(load_jsonl(args.input)))
    clean, quality_rejected = quality_filter(unique)
    train, held_out = repository_split(clean)
    parents = select_branch_parents(
        train, "qwen_coder", "codetrace", codetrace_limit=args.documents
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    accepted_by_model: dict[str, list[dict]] = {}
    failures: list[dict] = []
    for model in ("qwen_coder", "codegemma"):
        generated, model_failures = generate_batch(
            parents, model, "codetrace", mock=args.mode == "mock",
            retries=args.retries, concurrency=args.concurrency,
        )
        accepted, invalid = validate_generated(generated)
        accepted_by_model[model] = accepted
        failures.extend(model_failures)
        failures.extend({
            "parent_document_id": row["parent_document_id"], "model": model,
            "method": "codetrace", "error": "post_generation_validation_failed",
        } for row in invalid)
        write_jsonl(output / f"{model}.jsonl", accepted)

    write_jsonl(output / "failures.jsonl", failures)
    success = {
        model: {row["parent_document_id"] for row in rows}
        for model, rows in accepted_by_model.items()
    }
    parent_ids = {row["document_id"] for row in parents}
    qwen_only = len(success["qwen_coder"] - success["codegemma"])
    gemma_only = len(success["codegemma"] - success["qwen_coder"])
    attempted = len(parents)
    models = {}
    for model, rows in accepted_by_model.items():
        metrics = [trace_metrics(row["content"].split("Assistant:\n", 1)[-1]) for row in rows] if args.mode == "real" else []
        models[model] = {
            "accepted": len(rows),
            "acceptance_rate": len(rows) / max(1, attempted),
            "acceptance_rate_95pct_wilson": wilson(len(rows), attempted),
            "median_trace_events": sorted(m["trace_events"] for m in metrics)[len(metrics) // 2] if metrics else None,
        }
    difference = models["qwen_coder"]["acceptance_rate"] - models["codegemma"]["acceptance_rate"]
    summary = {
        "mode": args.mode,
        "mock_results_are_not_evidence": args.mode == "mock",
        "requested_shared_parents": args.documents,
        "attempted_shared_parents": attempted,
        "unique_repositories": len({row.get("repo_id") or row.get("repo_path") for row in parents}),
        "quality_rejected_before_sampling": len(quality_rejected),
        "repository_held_out_documents": len(held_out),
        "models": models,
        "paired_outcomes": {
            "both_succeeded": len(success["qwen_coder"] & success["codegemma"]),
            "qwen_only": qwen_only,
            "codegemma_only": gemma_only,
            "neither": len(parent_ids - success["qwen_coder"] - success["codegemma"]),
            "acceptance_rate_difference_qwen_minus_codegemma": difference,
            "mcnemar_exact_p_value": mcnemar_exact(qwen_only, gemma_only),
        },
        "sampling_requirement_met": attempted >= 1000 and len({row.get("repo_id") or row.get("repo_path") for row in parents}) >= 200,
        "clear_model_difference": abs(difference) >= 0.05 and mcnemar_exact(qwen_only, gemma_only) < 0.05,
        "claim_scope": "generation reliability and trace-data characteristics only; pretraining benefit requires a controlled training ablation",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
