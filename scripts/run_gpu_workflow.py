"""Run Flyte locally with endpoint defaults and durable exit status.

Launch with nohup; the caller redirects stdout/stderr to a run-specific log.
"""

import argparse
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/stack-v3-sample.jsonl")
    parser.add_argument("--output", default=f"data/curated-real-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}")
    parser.add_argument("--codetrace-documents", type=int, default=1000)
    parser.add_argument("--evaluation", default="")
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("Output already exists; choose a new directory")
    status_path = Path(args.output + ".status.json")
    status_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("CODE_DATA_QWEN_ENDPOINT", "http://127.0.0.1:8001/v1")
    env.setdefault("CODE_DATA_CODEGEMMA_ENDPOINT", "http://127.0.0.1:8002/v1")
    env["PYTHONUNBUFFERED"] = "1"
    command = [str(Path(sys.executable).parent / "pyflyte"), "run", "code_data_curation/workflow.py", "code_data_curation_workflow", "--input_path", args.input, "--output_dir", args.output, "--mode", "real", "--generation_retries", "2", "--generation_concurrency", "2", "--codetrace_documents_per_model", str(args.codetrace_documents), "--evaluation_path", args.evaluation]
    status = {"output": args.output, "started_at": datetime.now(UTC).isoformat(), "state": "running"}
    with status_path.open("x") as handle:
        json.dump(status, handle)
    try:
        result = subprocess.run(command, env=env)
        status.update(exit_code=result.returncode, state="finished" if result.returncode == 0 else "failed")
        return result.returncode
    finally:
        status["finished_at"] = datetime.now(UTC).isoformat()
        if status["state"] == "running":
            status["state"] = "interrupted"
        status_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
