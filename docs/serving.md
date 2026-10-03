# Serving with SGLang

Block diffusion is served by Liquid AI's SGLang extension ([PR #3](https://github.com/Liquid4All/sglang-diffusion/pull/3)),
which adds the `DuoBlock` algorithm with fused kernels, CUDA graphs and continuous batching.

## Install

```bash
git clone -b release https://github.com/Liquid4All/sglang-diffusion && cd sglang-diffusion
pip install -e "python"                                                         # NVIDIA
mv python/pyproject_other.toml python/pyproject.toml && pip install -e "python[all_hip]"   # AMD
```

## Launch

```bash
bash scripts/serve_sglang.sh nfe8              # throughput: batching up to 32 requests
MODE=latency bash scripts/serve_sglang.sh nfe8 # single request, commit fusion
```

Equivalent command (throughput mode):

```bash
python -m sglang.launch_server --model-path LiquidAI/lfm2.5-350m-diffusion-exp --trust-remote-code \
  --dllm-algorithm DuoBlock --dllm-algorithm-config lfm_diffusion/decode_configs/nfe8.yaml \
  --dllm-prefix-attention causal --attention-backend triton --no-dllm-fdfo \
  --dllm-cuda-graph --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32 \
  --max-running-requests 32 --disable-radix-cache --dtype bfloat16 \
  --tool-call-parser lfm2 --port 30000
```

Decoding parameters (rho schedule, temperature anneal, sigma floor) are set by the server config. Per request,
`dllm_steps_per_block` overrides the steps per block, and `top_p` / `top_k` are accepted.

## Query

```bash
curl http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "LiquidAI/lfm2.5-350m-diffusion-exp",
  "messages": [{"role": "user", "content": "What is C. elegans?"}],
  "max_tokens": 256
}'
```

## Speed

Decode speedup over autoregressive LFM2.5-350M at batch size 1 (1024-token prompt, 1024 new tokens):

| GPU | NFE 32 | NFE 16 | NFE 8 | NFE 4 |
|---|---|---|---|---|
| H100 | 0.89× | 1.65× | 2.93× | 4.74× |
| B200 | 0.75× | 1.38× | 2.47× | 4.07× |
| MI325X | 0.78× | 1.50× | 2.81× | 4.98× |

The advantage is a low-batch effect: every denoising step evaluates the whole block, so throughput falls below
autoregressive decoding as concurrency grows.
