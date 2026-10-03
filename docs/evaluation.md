# Evaluation

All benchmarks are run against the SGLang server ([serving.md](serving.md)) in non-thinking mode, using the
model's bundled chat template and one decode preset per run (`nfe32`, `nfe8` or `nfe4`).

## lm-evaluation-harness

```bash
bash scripts/serve_sglang.sh nfe8 &          # wait until the server is ready
bash scripts/run_lm_eval.sh                  # IFEval, GPQA Diamond, MMLU-Pro
```

## Benchmarks in the paper

| Benchmark | Harness |
|---|---|
| IFEval | lm-eval `ifeval` |
| GPQA Diamond | lm-eval `gpqa_diamond_cot_zeroshot` |
| MMLU-Pro | lm-eval `mmlu_pro` (the paper uses a 0-shot prompt) |
| Multi-IF | [facebook/Multi-IF](https://github.com/facebookresearch/Multi-IF) |
| IFBench | [allenai/IFBench](https://github.com/allenai/IFBench) |
| BFCL v3 / v4 | [gorilla](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard), OpenAI-compatible endpoint |
| τ²-Bench | [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) |
| CaseReportBench | [dataset](https://huggingface.co/datasets/cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark) |

The paper's numbers were produced with Liquid AI's internal evaluation harness on the same served model. Public
harnesses differ in prompts and scoring, so expect small differences.
