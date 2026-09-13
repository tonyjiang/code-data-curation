#!/usr/bin/env python3
"""Run the curation logic locally without a Flyte backend or model weights."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code_data_curation.pipeline import (
    cross_dataset_deduplicate,
    decontaminate,
    exact_deduplicate,
    export_dataset,
    generate_branches,
    load_jsonl,
    near_deduplicate,
    quality_filter,
    repository_split,
    validate_generated,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/stack-v3-sample.jsonl")
    parser.add_argument("--output", default="data/curated-mock")
    parser.add_argument("--evaluation", default=None)
    parser.add_argument("--codetrace-documents-per-model", type=int, default=1000)
    args = parser.parse_args()
    source = load_jsonl(args.input)
    unique = near_deduplicate(exact_deduplicate(source))
    clean, rejected = quality_filter(unique)
    eval_documents = load_jsonl(args.evaluation) if args.evaluation else []
    clean, contaminated_organic = decontaminate(clean, eval_documents)
    train, validation = repository_split(clean)
    synthetic = generate_branches(
        train, codetrace_limit=args.codetrace_documents_per_model
    )
    valid, invalid = validate_generated(synthetic)
    valid, contaminated = decontaminate(valid, eval_documents)
    valid, cross_branch_duplicates = cross_dataset_deduplicate(valid, train)
    manifest = export_dataset(
        train, valid, args.output,
        codetrace_target_per_model=args.codetrace_documents_per_model,
    )
    manifest.update({
        "input_documents": len(source),
        "quality_rejected": len(rejected),
        "organic_contaminated": len(contaminated_organic),
        "validation_documents": len(validation),
        "synthetic_invalid": len(invalid),
        "synthetic_contaminated": len(contaminated),
        "cross_branch_duplicates": len(cross_branch_duplicates),
        "mode": "mock",
    })
    Path(args.output, "manifest.json").write_text(__import__("json").dumps(manifest, indent=2) + "\n")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
