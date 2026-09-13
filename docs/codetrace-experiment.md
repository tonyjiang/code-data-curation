# CodeTrace experiment design

## What prior work establishes

- [CodeAlchemy](https://arxiv.org/abs/2606.10087) instruments real files with
  structured trace events, generates 3–5 test inputs, executes each file three
  times in isolated sandboxes, and removes empty or inconsistent traces. It
  retained about 1.3 million code/trace pairs from 4 million instrumented files.
- Its CodeTrace-only ablation annealed a 3B base model on 10B tokens. CodeTrace
  produced the strongest TraceEval result among individual CodeAlchemy components,
  while mixed data preserved broader coding performance.
- [CodeExecutor](https://arxiv.org/abs/2305.05383) treats trace prediction as a
  pretraining objective and uses curriculum learning.
- [NExT](https://arxiv.org/abs/2404.14662) conditions reasoning on observed runtime
  states and filters examples by whether they lead to correct repairs.
- [SemCoder](https://arxiv.org/abs/2406.01006) combines execution behavior with
  higher-level semantic explanations and evaluates execution reasoning alongside
  ordinary code generation.

## Registered comparison

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
margin of error before repository clustering. The final analysis should also use
repository-level bootstrap intervals.

## Claim boundary

This experiment can establish whether the two generators reliably produce usable
Python CodeTrace records and whether one has a meaningful yield advantage. A
pretraining-effect claim requires a separate, equal-token ablation from the same
base checkpoint, ideally with multiple seeds. Evaluate on TraceEval plus
HumanEval+, MBPP+, and CRUXEval so execution gains are not mistaken for broad code
improvement.
