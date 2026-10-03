#!/usr/bin/env bash
# Launch an SGLang server for LFM2.5-350M-Diffusion. Usage: [MODE=latency] bash scripts/serve_sglang.sh [nfe32|nfe8|nfe4]
set -euo pipefail
PRESET="${1:-nfe8}"
MODEL="${MODEL:-LiquidAI/lfm2.5-350m-diffusion-exp}"
PORT="${PORT:-30000}"
# The SGLang ROCm images enable AITER, which slows DuoBlock by ~30%; no effect on NVIDIA.
export SGLANG_USE_AITER="${SGLANG_USE_AITER:-0}"
CONFIG="$(cd "$(dirname "$0")/.." && pwd)/lfm_diffusion/decode_configs/${PRESET}.yaml"

if [[ "${MODE:-throughput}" == "latency" ]]; then
  FUSED="$(mktemp --suffix .yaml)"
  cat "$CONFIG" > "$FUSED" && echo "commit_fusion: true" >> "$FUSED"
  CONFIG="$FUSED"; MAX_RUNNING=1; EXTRA=()
else
  MAX_RUNNING=32; EXTRA=(--tokenizer-worker-num 4)
  export SGLANG_DLLM_PREFILL_BATCH=4 SGLANG_DLLM_PREFILL_MAX_WAIT_MS=10
fi

exec python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
  --dllm-algorithm DuoBlock --dllm-algorithm-config "$CONFIG" \
  --dllm-prefix-attention causal --attention-backend triton --no-dllm-fdfo \
  --dllm-cuda-graph --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32 \
  --max-running-requests "$MAX_RUNNING" --disable-radix-cache --dtype bfloat16 \
  --mem-fraction-static 0.85 --tool-call-parser lfm2 --port "$PORT" "${EXTRA[@]}"
