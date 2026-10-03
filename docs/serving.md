# Serving with SGLang

Block diffusion is served by Liquid AI's SGLang extension, [sglang-diffusion-lfm](https://github.com/Liquid4All/sglang-diffusion-lfm),
which adds the `DuoBlock` algorithm with fused kernels, CUDA graphs and continuous batching.

## Install

```bash
git clone https://github.com/Liquid4All/sglang-diffusion-lfm && cd sglang-diffusion-lfm
SGLANG_BUILD_RUST_EXTS=none pip install -e "python"
```

`SGLANG_BUILD_RUST_EXTS=none` skips SGLang's optional Rust extensions, which otherwise need a Rust toolchain.

**AMD (ROCm).** Start from SGLang's ROCm image (`mi30x` for MI300X/MI325X, `mi35x` for MI350/MI355) and install
`sglang-diffusion-lfm` on top of it:

```bash
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 16g \
  -v ~/.cache/huggingface:/root/.cache/huggingface -p 30000:30000 lmsysorg/sglang:v0.5.17-rocm720-mi30x bash
git clone https://github.com/Liquid4All/sglang-diffusion-lfm && cd sglang-diffusion-lfm
SGLANG_BUILD_RUST_EXTS=none pip install -e "python" --no-deps
```

`scripts/serve_sglang.sh` sets `SGLANG_USE_AITER=0`: the ROCm images enable AITER, which slows DuoBlock by about 30%.

## Launch

Run from the `lfm-diffusion` checkout, which holds the launch script and the decode configs:

```bash
cd /path/to/lfm-diffusion   # not the sglang-diffusion-lfm checkout
bash scripts/serve_sglang.sh nfe8              # throughput: batching up to 32 requests
MODE=latency bash scripts/serve_sglang.sh nfe8 # single request, commit fusion
```

Equivalent command (throughput mode):

```bash
SGLANG_DLLM_PREFILL_BATCH=4 SGLANG_DLLM_PREFILL_MAX_WAIT_MS=10 \
python -m sglang.launch_server --model-path LiquidAI/lfm2.5-350m-diffusion-exp --trust-remote-code \
  --dllm-algorithm DuoBlock --dllm-algorithm-config lfm_diffusion/decode_configs/nfe8.yaml \
  --dllm-prefix-attention causal --attention-backend triton --no-dllm-fdfo \
  --dllm-cuda-graph --cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32 \
  --max-running-requests 32 --disable-radix-cache --dtype bfloat16 --mem-fraction-static 0.85 \
  --tokenizer-worker-num 4 --tool-call-parser lfm2 --port 30000
```

Decoding parameters (rho schedule, temperature anneal, sigma floor) are set by the server config. Per request,
`dllm_steps_per_block` overrides the steps per block, and `top_p` / `top_k` are accepted.

## Query

```bash
curl http://localhost:30000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "LiquidAI/lfm2.5-350m-diffusion-exp",
  "messages": [{"role": "user", "content": "Give three tips for getting better sleep."}],
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

Median of 5 machines per GPU, with the autoregressive model and the diffusion model measured back to back on the
same machine. The autoregressive baseline uses its fastest SGLang configuration (CUDA graphs and continuous decode
steps; `SGLANG_USE_AITER=1` on MI325X). B200 and H100: 32 pinned CPU cores; MI325X: 14 cores. Autoregressive
decoding at batch size 1 is bound by the host CPU, so on slower CPUs the speedup is larger.
