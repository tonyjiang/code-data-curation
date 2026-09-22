# Code Data Curation

This project is a small, runnable Flyte pipeline for curating organic source code
and generating synthetic documents for LLM pretraining. It preserves source
and repository lineage, cleans the organic corpus, runs four generation
methods through two models, validates the results, and exports a versioned training
mixture. Mock mode is deterministic and requires no model inference.

The workflow has these stages:

1. Ingest source documents with repository, license, file, and content identifiers.
2. Remove exact and near duplicates.
3. Apply deterministic code-quality filters and remove malformed, vendor/generated, secret-bearing, or low-signal files.
4. Remove configured evaluation matches; retain all remaining organic documents for training.
5. Generate synthetic data through eight branches: two models × four methods.
6. Validate each branch, then merge accepted synthetic documents with organic training documents and export JSONL plus a manifest.

See the [workflow diagram](docs/workflow.html) for the full graph.

## Create the environment

```bash
uv sync --dev
source .venv/bin/activate
```

## Data

The checked-in [15-language sample](data/raw/stack-v3-sample.jsonl) contains 150
documents: 10 each for C, C#, C++, Dart, Go, Java, JavaScript, Kotlin, PHP,
Python, Ruby, Rust, Scala, Swift, and TypeScript.

The Stack v3 is stored on Hugging Face as 8,192 Parquet shards rather than as
individually downloadable source files. The fetcher downloads complete shards
with `huggingface_hub`, decodes them locally, and processes every repository in
each downloaded shard. `--min-per-language` is a lower bound checked only after
a complete shard, so counts can exceed the requested minimum. Each shard is
roughly 300–550 MB and remains under `data/raw/stack-v3-shards/`.

The real one-shard Python fetch used in this project is:

```bash
uv run python scripts/fetch_stack_sample.py \
  --languages Python \
  --min-per-language 1500 \
  --max-per-repository 5 \
  --max-shards 1 \
  --output data/raw/stack-v3-full-shard-python.jsonl
```

It produced 7,788 unique Python documents from 2,739 repositories. The repository
cap prevents a few large repositories from dominating the cohort; source files
themselves are never truncated. The adjacent manifest records the exact dataset
revision, shard path and size, retrieval time, counts, and any shortfalls. Increase
`--max-shards` if a shard does not meet the requested minimum.

## OpenCoder-inspired filtering

The filter adapts a focused subset of the MIT-licensed
[OpenCoder data-filtering rules](https://github.com/OpenCoder-llm/opc_data_filtering)
for this repository. It measures document size, line and token statistics,
repetition, character composition, comments, encoded data, long strings, and
Tree-sitter parse errors. Tree-sitter parsing covers all 15 languages in the
sample. Accepted records retain every `quality_signals` value; rejected records
also retain the failing signal, observed value, and threshold in the audit log.

This is a local adaptation rather than the complete upstream rule set. Generated
markers, vendor paths, and likely secrets remain explicit local checks. A neural
scoring stage should only be added after a separate matched-budget evaluation
shows that it improves downstream model quality enough to justify GPU cost.

## Synthetic generation methods

The active design uses four distinct methods. Every method runs once with Qwen
and once with CodeGemma, producing eight branches.

| Method | Generated pretraining document | What distinguishes it |
| --- | --- | --- |
| SwallowCode-style rewrite | A self-contained, idiomatic rewrite that preserves the source behavior | Improves the presentation and quality of an existing implementation without inventing a separate task |
| CodeDev | A realistic development request, relevant source context, and a complete solution | Converts source code into feature, repair, or behavior-change material |
| CodeQA | A technical question, the grounding code, and an answer about behavior, design, or edge cases | Converts code into explanatory question-answer text rather than a development task |
| CodeTrace | Instrumented Python, generated tests, and traces captured from sandbox execution | Adds observed runtime state and control-flow behavior |

SwallowCode follows the rewriting recipe in
[Rewriting Pre-Training Data Boosts LLM Performance in Math and Code](https://arxiv.org/abs/2505.02881).
CodeDev, CodeQA, and CodeTrace follow the method categories introduced by
[CodeAlchemy](https://arxiv.org/abs/2606.10087). CodeEnhance is not another
branch because its code-improvement role substantially overlaps the selected
SwallowCode rewriting branch.

CodeQA uses a source-grounded prompt implemented in
[`generation.py`](code_data_curation/generation.py). It does not depend on an
OSS-Instruct library. The output is serialized as continuous `Question`, `Code`,
and `Answer` text for pretraining rather than as chat-role JSON. The
[Magicoder OSS-Instruct paper](https://arxiv.org/abs/2312.02120) is relevant
background for source-grounded instruction synthesis, but OSS-Instruct is not a
separate ninth method here.

Let **N** be the number of cleaned organic training documents after filtering, deduplication, and
decontamination. The six SwallowCode, CodeDev, and CodeQA branches each
target `0.25N` accepted documents. Together they target `1.5N` synthetic
documents, producing approximately `2.5N` documents after merging with the
organic corpus. The two CodeTrace branches form an additional paired Python
cohort, capped by `codetrace_documents_per_model` and reported separately in the
manifest so they do not silently change the base mixture calculation.

All synthetic records retain the parent document ID, repository and license,
model and model version, method, prompt version, token estimate, and validation
results. Real endpoint generation also records the attempt count. Failed quotas
are reported rather than filled by lowering quality thresholds.

## Models

The two configured generators are recorded in [config/models.json](config/models.json):

- `Qwen/Qwen2.5-Coder-7B-Instruct` using the official Q4_K_M GGUF
- `google/codegemma-7b-it`

The workflow never downloads weights or launches paid inference. The guarded
resolver accepts an existing local model directory or OpenAI-compatible endpoint:

```bash
export CODE_DATA_QWEN_MODEL_PATH=/path/to/Qwen2.5-Coder-7B-Instruct-GGUF
export CODE_DATA_CODEGEMMA_MODEL_PATH=/path/to/codegemma-7b-it-GGUF

# Or use already-running model servers:
export CODE_DATA_QWEN_ENDPOINT=http://127.0.0.1:8001/v1
export CODE_DATA_CODEGEMMA_ENDPOINT=http://127.0.0.1:8002/v1
```

CodeGemma requires accepting Google’s Hugging Face license. Local paths confirm
model availability but do not provide an inference runtime; real generation
currently sends requests to the configured endpoints.

## Run locally in mock mode

The plain Python runner executes the complete curation logic without a Flyte
backend:

```bash
uv run python scripts/run_mock_workflow.py \
  --input data/raw/stack-v3-sample.jsonl \
  --output data/curated-mock
```

Pass `--evaluation` / `--evaluation_path` with a JSONL of held-out eval documents
to drop matching organic records before generation. Empty or omitted means no eval set.

After generation, accepted documents are merged without exact or near deduplication.
An optional second decontamination pass is off by default. Enable
`--post_generation_decontamination` in `pyflyte run`, or
`--post-generation-decontamination` in either Python runner, to reject synthetic
matches against the configured evaluation set. The manifest
records whether this pass ran and how many matches it removed.

The same graph can run through Flyte’s local runner:

```bash
uv run pyflyte run code_data_curation/workflow.py code_data_curation_workflow \
  --input_path data/raw/stack-v3-sample.jsonl \
  --output_dir data/flyte-mock
```

Mock records are labeled and are useful only for checking orchestration, quotas,
lineage, validation, and export. They are not evidence about model quality.

The output directory contains:

- `train-00000.jsonl`: organic and accepted synthetic documents.
- `manifest.json`: organic, synthetic, model-method, target, shortfall, and token counts.

## Evaluate the generation methods

All eight branches should first be compared on accepted yield, rejection reasons,
duplicate rates, token counts, language balance, and repository coverage. Those
measure generation and curation quality. Claims that one method improves a model
require matched-token continued-pretraining ablations from the same base checkpoint
and evaluation on both general code generation and method-relevant tasks.

CodeTrace additionally supports dynamic checks unavailable to the other methods.
For Python, the real adapter instruments code, generates 3–5 tests, and runs the
same inputs three times in a network-disabled, read-only Docker sandbox. Empty,
failed, or nondeterministic traces are rejected. The paired study gives Qwen and
CodeGemma the same 1,000 parent files and reports confidence intervals, trace
statistics, and an exact McNemar test. This evaluates CodeTrace generation
reliability; it does not establish a pretraining benefit.

```bash
# Offline structure check
uv run python scripts/run_codetrace_experiment.py \
  --input data/raw/stack-v3-full-shard-python.jsonl \
  --mode mock

# Real paired CodeTrace generation
uv run python scripts/run_codetrace_experiment.py \
  --input data/raw/stack-v3-full-shard-python.jsonl \
  --mode real --documents 1000 --concurrency 2
```

The real CodeTrace run requires the existing
`docker.m.daocloud.io/library/python:3.12-slim` Docker image; the project never
pulls images automatically. Set `CODE_DATA_TRACE_IMAGE` to use a different
pre-pulled image.

### CodeTrace experiment design

#### Registered comparison

The cohort contains 1,000 cleaned Python files from at least 200 repositories.
Qwen and CodeGemma receive exactly the same parents. Each model generates both
the instrumented program and its tests. Ground-truth traces come only from sandbox
execution.

Primary outcomes are accepted deterministic trace yield, failure categories,
trace length, event count, event-type coverage, and repository coverage. Report
95% Wilson intervals for each rate. Compare paired success/failure outcomes with
an exact McNemar test. Call a model difference clear only when the two-sided
`p < 0.05` and the absolute acceptance-rate difference is at least five percentage
points. This rule is registered before the real run.

At 1,000 samples, an individual proportion has at worst about a 3.1-point 95%
margin of error before repository clustering.

#### Claim boundary

This experiment can establish whether the two generators reliably produce usable
Python CodeTrace records and whether one has a meaningful yield advantage. A
pretraining-effect claim requires a separate, equal-token ablation from the same
base checkpoint, ideally with multiple seeds. Evaluate on TraceEval plus
HumanEval+, MBPP+, and CRUXEval so execution gains are not mistaken for broad code
improvement.

## Tests

```bash
uv run python -m pytest -q
```

## Run real generation on the GPU server

For a fresh monitored run, use the launcher from the project directory:

```bash
mkdir -p logs
nohup /root/.local/bin/uv run python scripts/run_gpu_workflow.py \
  > logs/generation.log 2>&1 < /dev/null &
tail -f logs/generation.log
```

The launcher supplies endpoint defaults and creates a unique timestamped output
directory, plus a sibling `.status.json` with the process exit code. Existing
datasets are never overwritten. Generation logs report each completed parent;
branch files and rejection audit files are saved before final export. The export
retains all cleaned organic documents, and the manifest explicitly labels quota shortfalls.
Responses cut off by the token limit are rejected; the default output budget is
1,024 tokens (`CODE_DATA_MAX_TOKENS`). Larger prompts can still exceed the running
servers' 2,048-token context. Syntax validity is `null` when no parser was used;
non-Python syntax and semantic correctness are not established by these checks.

The intended server layout is:

- project: `/root/code-data-curation`
- model files: `/root/models`
- Qwen API: `http://127.0.0.1:8001/v1`
- CodeGemma API: `http://127.0.0.1:8002/v1`

The RTX 5070 has 12 GB of VRAM. The selected Qwen and CodeGemma Q4_K_M files are
4.68 GB and 5.33 GB. They have been verified together with a 2,048-token context;
do not raise the context window until VRAM use has been measured.

### 1. Copy the project and data from the Mac

Run on the Mac:

```bash
rsync -az --exclude .venv --exclude '.pytest_cache' \
  --exclude 'data/raw/stack-v3-shards/' \
  ~/Workshop/code-data-curation/ \
  gaming-server:/root/code-data-curation/
```

The extracted JSONL inputs are copied; the 380 MB local Parquet source shard is
not needed for generation on the GPU server.

The server uses these quantized GGUF files:

```bash
ssh gaming-server 'mkdir -p /root/models'
rsync -ah --info=progress2 \
  ~/Models/huggingface/Qwen2.5-Coder-7B-Instruct-GGUF/ \
  gaming-server:/root/models/Qwen2.5-Coder-7B-Instruct-GGUF/
rsync -ah --info=progress2 \
  ~/Models/huggingface/codegemma-7b-it-GGUF/ \
  gaming-server:/root/models/codegemma-7b-it-GGUF/
```

### 2. Install the project environment

```bash
ssh gaming-server
cd /root/code-data-curation
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --dev
```

### 3. Start OpenAI-compatible model servers

The server has a Vulkan-enabled `llama-server`, which is required to use this
RTX 5070 without installing an obsolete CUDA toolkit. See the
[llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

```bash
llama-server \
  --model /root/models/Qwen2.5-Coder-7B-Instruct-GGUF/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --alias Qwen/Qwen2.5-Coder-7B-Instruct \
  --host 127.0.0.1 --port 8001 \
  --ctx-size 2048 --gpu-layers 999 --parallel 1
```

In a second terminal:

```bash
llama-server \
  --model /root/models/codegemma-7b-it-GGUF/codegemma-7b-it-Q4_K_M.gguf \
  --alias google/codegemma-7b-it \
  --host 127.0.0.1 --port 8002 \
  --ctx-size 2048 --gpu-layers 999 --parallel 1
```

Check both endpoints:

```bash
nvidia-smi  # shows the NVIDIA GPU’s status
curl -f http://127.0.0.1:8001/health
curl -f http://127.0.0.1:8002/health
```

Docker is installed for the CodeTrace sandbox. Its Python image is available as
`docker.m.daocloud.io/library/python:3.12-slim` because Docker Hub times out from
this server.

### 4. Run the real Flyte workflow

```bash
cd /root/code-data-curation
export CODE_DATA_QWEN_ENDPOINT=http://127.0.0.1:8001/v1
export CODE_DATA_CODEGEMMA_ENDPOINT=http://127.0.0.1:8002/v1

uv run pyflyte run code_data_curation/workflow.py code_data_curation_workflow \
  --input_path data/raw/stack-v3-sample.jsonl \
  --output_dir data/curated-real \
  --mode real \
  --generation_retries 2 \
  --generation_concurrency 2 \
  --codetrace_documents_per_model 1000
```

For a disconnect-resistant server run, pass both endpoints directly to `nohup`
and use the absolute `uv` path (non-interactive shells do not load its PATH):

```bash
mkdir -p logs
nohup env \
  CODE_DATA_QWEN_ENDPOINT=http://127.0.0.1:8001/v1 \
  CODE_DATA_CODEGEMMA_ENDPOINT=http://127.0.0.1:8002/v1 \
  /root/.local/bin/uv run pyflyte run code_data_curation/workflow.py code_data_curation_workflow \
  --input_path data/raw/stack-v3-sample.jsonl \
  --output_dir data/curated-real \
  --mode real --generation_retries 2 --generation_concurrency 2 \
  --codetrace_documents_per_model 1000 \
  > logs/real-workflow.log 2>&1 < /dev/null &
```

Follow it with `tail -f logs/real-workflow.log`.

Flyte orchestrates the eight branches while generation requests go to the two
configured endpoints. Results are written to `data/curated-real/train-00000.jsonl`
and `data/curated-real/manifest.json`.
