<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/2b08LKpev0DNEk6DlnWkY.png" alt="Liquid AI" width="100%"/>
  <p>
    <a href="https://huggingface.co/LiquidAI/lfm2.5-350m-diffusion-exp"><strong>Model</strong></a> •
    <a href="https://github.com/Liquid4All/sglang-diffusion-lfm"><strong>SGLang</strong></a> •
    <a href="https://docs.liquid.ai/lfm/getting-started/welcome"><strong>Docs</strong></a> •
    <a href="https://discord.com/invite/liquid-ai"><strong>Discord</strong></a>
  </p>
</div>

# LFM Diffusion

Block-diffusion language models from Liquid AI. **LFM2.5-350M-Diffusion** is LFM2.5-350M converted into a
uniform-state block-diffusion model: it denoises 32-token blocks in parallel and decodes several times faster
than its autoregressive parent at low batch size.

> [!NOTE]
> `lfm2.5-350m-diffusion-exp` is an experimental release. Training, fine-tuning, TRL and native
> Transformers `generate()` support will be added to this repository.

## Install

```bash
git clone https://github.com/Liquid4All/lfm-diffusion && cd lfm-diffusion
uv sync --extra cuda   # NVIDIA
uv sync --extra rocm   # AMD
```

## Inference

**PyTorch** (reference implementation, any GPU):

```bash
uv run scripts/generate.py --prompt "Give three tips for getting better sleep." --nfe 8
```

```python
from lfm_diffusion import generate, load_model

model, tokenizer = load_model("LiquidAI/lfm2.5-350m-diffusion-exp")
messages = [{"role": "user", "content": "Give three tips for getting better sleep."}]
print(generate(model, tokenizer, messages, config="nfe8", max_new_tokens=256))
```

**SGLang** (fast serving, OpenAI-compatible): see [docs/serving.md](docs/serving.md).

## Decoding

`--nfe` sets the denoising steps per 32-token block. Fewer steps decode faster at some cost in quality.

| Preset | Steps per block | Use |
|---|---|---|
| `nfe32` | 32 | Highest quality |
| `nfe8` | 8 | Recommended balance |
| `nfe4` | 4 | Fastest |

All presets use ancestral sampling with a temperature anneal; do not decode greedily. The YAML files in
[`lfm_diffusion/decode_configs`](lfm_diffusion/decode_configs) list every parameter.

## Evaluation

See [docs/evaluation.md](docs/evaluation.md) to evaluate the model through the SGLang endpoint with
`lm-evaluation-harness` and the official harnesses of the remaining benchmarks.

## Citation

```bibtex
@inproceedings{tafreshi2026blockdiffusion,
  title     = {From Autoregression to Block Diffusion: Adapting Language Models for Efficient Parallel Decoding},
  author    = {Amin Tafreshi, Rouzbeh and Fan, Jack and Mosca, Edoardo and Lechner, Mathias and Amini, Alexander},
  booktitle = {NeurIPS 2026 Workshop},
  year      = {2026}
}
```

## License

Code: Apache 2.0. Model weights: [LFM Open License v1.0](https://huggingface.co/LiquidAI/lfm2.5-350m-diffusion-exp/blob/main/LICENSE).
