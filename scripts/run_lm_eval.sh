#!/usr/bin/env bash
# Run lm-evaluation-harness against a running SGLang server. Usage: bash scripts/run_lm_eval.sh [tasks]
set -euo pipefail
TASKS="${1:-ifeval,gpqa_diamond_cot_zeroshot,mmlu_pro}"
MODEL="${MODEL:-LiquidAI/lfm2.5-350m-diffusion-exp}"
BASE_URL="${BASE_URL:-http://localhost:30000/v1/chat/completions}"

uv run --extra eval lm_eval --model local-chat-completions \
  --model_args "model=${MODEL},base_url=${BASE_URL},num_concurrent=16,max_retries=3,tokenized_requests=False" \
  --tasks "$TASKS" --apply_chat_template --output_path results/ "${@:2}"
