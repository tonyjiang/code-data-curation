#!/usr/bin/env python3
"""Download complete The Stack v3 shards and extract a language sample."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


DEFAULT_LANGUAGES = [
    "Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#", "Go",
    "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Dart", "Scala",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/stack-v3-full-shard-sample.jsonl"),
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--min-per-language", type=int, default=10)
    parser.add_argument("--dataset", default="HuggingFaceCode/stack-v3-train")
    parser.add_argument("--languages", nargs="+", choices=DEFAULT_LANGUAGES, default=DEFAULT_LANGUAGES)
    parser.add_argument("--max-per-repository", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("data/raw/stack-v3-shards"),
        help="Directory for complete downloaded Parquet shards.",
    )
    return parser.parse_args()


def iter_repositories(shard_path: Path) -> Iterator[dict[str, Any]]:
    parquet_file = pq.ParquetFile(shard_path)
    for batch in parquet_file.iter_batches(batch_size=32):
        yield from batch.to_pylist()


def main() -> int:
    args = parse_args()
    if args.max_shards < 1:
        raise ValueError("--max-shards must be at least 1")
    if args.start_shard < 0:
        raise ValueError("--start-shard cannot be negative")

    languages = args.languages
    wanted = set(languages)
    counts: Counter[str] = Counter()
    seen_content_ids: set[str] = set()
    documents: list[dict[str, Any]] = []
    repositories_scanned = 0

    api = HfApi()
    dataset_info = api.dataset_info(args.dataset, files_metadata=True)
    parquet_shards = sorted(
        (
            sibling
            for sibling in dataset_info.siblings
            if sibling.rfilename.endswith(".parquet")
        ),
        key=lambda sibling: sibling.rfilename,
    )
    selected_shards = parquet_shards[
        args.start_shard:args.start_shard + args.max_shards
    ]
    if not selected_shards:
        raise ValueError("No Parquet shards matched the requested shard range")

    processed_shards: list[dict[str, Any]] = []
    for shard in selected_shards:
        shard_path = Path(hf_hub_download(
            repo_id=args.dataset,
            filename=shard.rfilename,
            repo_type="dataset",
            revision=dataset_info.sha,
            local_dir=args.shard_dir,
        ))
        for repository in iter_repositories(shard_path):
            repositories_scanned += 1
            accepted_from_repository = 0
            for file_record in repository.get("files", []):
                language = file_record.get("language")
                content_id = file_record.get("content_id")
                content = file_record.get("content")
                if language not in wanted:
                    continue
                if args.max_per_repository and accepted_from_repository >= args.max_per_repository:
                    break
                if not content or not content_id or content_id in seen_content_ids:
                    continue
                seen_content_ids.add(content_id)
                document_id = ":".join(
                    str(part)
                    for part in (repository.get("repo_id"), file_record.get("file_path"), content_id)
                )
                documents.append({
                    "document_id": document_id,
                    "content_id": content_id,
                    "content": content,
                    "language": language,
                    "file_path": file_record.get("file_path"),
                    "repo_path": repository.get("repo_path"),
                    "repo_id": repository.get("repo_id"),
                    "commit_id": repository.get("commit_id"),
                    "license_type": file_record.get("license_type"),
                    "detected_licenses": file_record.get("detected_licenses", []),
                    "is_vendor": file_record.get("is_vendor", False),
                    "source_dataset": args.dataset,
                    "source_shard": shard.rfilename,
                })
                counts[language] += 1
                accepted_from_repository += 1

        processed_shards.append({
            "path": shard.rfilename,
            "size_bytes": shard.size,
            "local_path": str(shard_path),
        })
        if all(counts[language] >= args.min_per_language for language in languages):
            break

    documents.sort(key=lambda document: (document["language"], document["document_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")

    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "dataset": args.dataset,
        "dataset_revision": dataset_info.sha,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "requested_languages": languages,
        "min_per_language": args.min_per_language,
        "accepted_counts": dict(sorted(counts.items())),
        "shortfalls": {
            language: max(0, args.min_per_language - counts[language])
            for language in languages if counts[language] < args.min_per_language
        },
        "documents": len(documents),
        "repositories_scanned": repositories_scanned,
        "max_per_repository": args.max_per_repository or None,
        "start_shard": args.start_shard,
        "max_shards": args.max_shards,
        "processed_shards": processed_shards,
        "unique_content_ids": len(seen_content_ids),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "documents": len(documents),
        "counts": dict(sorted(counts.items())),
        "shards": len(processed_shards),
        "manifest": str(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
