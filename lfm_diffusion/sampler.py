"""DuoBlock sampler for LFM2.5 block-diffusion models (plain PyTorch, any GPU or CPU).

Usage: `from lfm_diffusion import load_model, generate` -- see scripts/generate.py.

Decoding runs block by block on an answer-anchored grid: the prompt is the committed,
token-causal context, and each new block of `block_size` tokens starts from uniform
random tokens and is denoised with the uniform-state (DUO) ancestral posterior for
`steps_per_block` steps, followed by one readout step. The block is then committed and
decoding moves on until an end-of-sequence token is produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
import yaml

CONFIG_DIR = Path(__file__).parent / "decode_configs"


@dataclass
class DecodeConfig:
    """Decoding parameters. The shipped presets live in `decode_configs/`."""

    steps_per_block: int = 8          # denoising steps (NFE) per block, readout excluded
    block_size: int = 32              # must match the checkpoint
    schedule: str = "rho"             # "rho" or "linear" time grid
    rho: float = 0.3                  # grid warp t = u**rho; < 1 spends more steps at low noise
    eps: float = 1e-3                 # smallest noise level of the grid
    temp_anneal: bool = True          # anneal temperature linearly within each block
    temp_start: float = 0.8
    temp_end: float = 0.4             # also the readout temperature
    temperature: float = 1.0          # used only when temp_anneal is False
    terminal_sigma_floor: float = 0.7  # noise level the readout step is conditioned on
    kappa: float = 1.0                # 1.0 = exact ancestral; < 1 mixes in the forward marginal
    top_p: float = 1.0                # nucleus on the denoiser's x0 belief
    top_k: int = 0                    # top-k on the denoiser's x0 belief (0 = off)
    self_cond: bool = True            # feed the previous step's belief back as input
    use_float64: bool = True          # posterior and sampling in float64

    def temperature_at(self, step: int) -> float:
        if not self.temp_anneal:
            return self.temperature
        frac = step / max(self.steps_per_block - 1, 1)
        return self.temp_start + frac * (self.temp_end - self.temp_start)

    @property
    def readout_temperature(self) -> float:
        return self.temp_end if self.temp_anneal else self.temperature


# Keys that may appear in a preset but are fixed by this sampler. A preset that asks for
# something else is rejected rather than silently decoded with different settings.
_FIXED = {
    "greedy": False,
    "noise_removal": "ancestral",
    "adaptive_stop": False,
    "top_p_site": "belief",
    "temp_schedule": "linear",
}
_IGNORED = {"max_steps_per_block", "stop_entropy", "top_k_decode"}


def load_config(config: Union[str, Path, DecodeConfig, dict] = "nfe8") -> DecodeConfig:
    """Load a preset by name ("nfe32", "nfe8", "nfe4"), from a YAML path, or a dict."""
    if isinstance(config, DecodeConfig):
        return config
    if isinstance(config, dict):
        raw = dict(config)
    else:
        path = Path(config)
        if not path.suffix:
            path = CONFIG_DIR / f"{config}.yaml"
        raw = yaml.safe_load(path.read_text())
    for key, value in _FIXED.items():
        if key in raw and raw.pop(key) != value:
            raise ValueError(f"{key}={raw.get(key)!r} is not supported; this sampler uses {key}={value!r}")
    if "top_k_decode" in raw and "top_k" not in raw:
        raw["top_k"] = raw["top_k_decode"]
    known = {f.name for f in fields(DecodeConfig)}
    unknown = set(raw) - known - _IGNORED
    if unknown:
        raise ValueError(f"unknown decode config keys: {sorted(unknown)}")
    cfg = DecodeConfig(**{k: v for k, v in raw.items() if k in known})
    if cfg.schedule not in ("rho", "linear"):
        raise ValueError(f"schedule={cfg.schedule!r} is not supported (use 'rho' or 'linear')")
    return cfg


def _time_grid(cfg: DecodeConfig, device) -> torch.Tensor:
    """steps+1 times from 1 (pure noise) to eps (clean)."""
    u = torch.linspace(1.0, 0.0, cfg.steps_per_block + 1, device=device)
    warped = u ** cfg.rho if cfg.schedule == "rho" else u
    return cfg.eps + (1.0 - cfg.eps) * warped


def _alpha(t: torch.Tensor, eps: float) -> torch.Tensor:
    """Log-linear signal level alpha(t) = 1 - (1 - eps) t, as a [1, 1, 1] tensor."""
    return (1.0 - (1.0 - eps) * t).reshape(1, 1, 1)


def _filter_top_p_top_k(logits: torch.Tensor, top_p: float, top_k: int) -> torch.Tensor:
    logits = logits.clone()
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(-1)
        remove = probs.cumsum(-1) - probs > top_p
        remove = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


def _duo_posterior(x0_probs, xt, alpha_s, alpha_t, vocab_size):
    """Uniform-state reverse posterior q(z_s | z_t, x0 belief), over the vocabulary."""
    alpha_ts = alpha_t / alpha_s
    d_alpha = alpha_s - alpha_t
    xt_one_hot = F.one_hot(xt, vocab_size).to(x0_probs.dtype)
    numer = (
        alpha_t * vocab_size * x0_probs * xt_one_hot
        + (alpha_ts - alpha_t) * xt_one_hot
        + d_alpha * x0_probs
        + (1 - alpha_ts) * (1 - alpha_s) / vocab_size
    )
    denom = alpha_t * vocab_size * x0_probs.gather(-1, xt[..., None]) + (1 - alpha_t)
    return numer / denom


def _reverse_step(logits, x_blk, alpha_t, alpha_s, temperature, cfg, vocab_size):
    """One ancestral step on the active block. Returns (new block tokens, log belief)."""
    scaled = logits / temperature
    log_x0 = F.log_softmax(scaled, dim=-1)
    if cfg.top_p < 1.0 or cfg.top_k > 0:
        x0_probs = F.log_softmax(_filter_top_p_top_k(scaled, cfg.top_p, cfg.top_k), dim=-1).exp()
    else:
        x0_probs = log_x0.exp()
    if cfg.use_float64:
        x0_probs, alpha_t, alpha_s = x0_probs.double(), alpha_t.double(), alpha_s.double()
    q = _duo_posterior(x0_probs, x_blk, alpha_s, alpha_t, vocab_size)
    if cfg.kappa < 1.0:
        q = cfg.kappa * q + (1.0 - cfg.kappa) * (alpha_s * x0_probs + (1.0 - alpha_s) / vocab_size)
    q = q.clamp_min(1e-30)
    gumbel = 1e-10 - (torch.rand_like(q) + 1e-10).log()
    return (q / gumbel).argmax(dim=-1), log_x0


class _BlockDenoiser:
    """Runs the model on [committed prefix | active block] and returns block logits."""

    def __init__(self, model, vocab_size: int):
        self.model = model
        self.vocab_size = vocab_size
        self.embed = model.get_input_embeddings().weight

    def logits(self, x, bstart, sigma_t, selfcond):
        b, length = x.shape
        sigma = torch.zeros((b, length), device=x.device, dtype=torch.float32)
        sigma[:, bstart:] = sigma_t.reshape(-1, 1).float()
        kw = {}
        if selfcond is not None:
            sc = torch.zeros((b, length, selfcond.shape[-1]), device=x.device, dtype=selfcond.dtype)
            sc[:, bstart:] = selfcond
            mask = torch.zeros((b, length), device=x.device, dtype=torch.bool)
            mask[:, bstart:] = True
            kw = dict(selfcond_input=sc, selfcond_pos_mask=mask)
        out = self.model(
            input_ids=x,
            sigma=sigma,
            block_start=bstart,
            position_ids=torch.arange(length, device=x.device).unsqueeze(0).expand(b, -1),
            logits_index=torch.arange(bstart, length, device=x.device).unsqueeze(0).expand(b, -1),
            use_cache=False,
            **kw,
        )
        return out.logits[..., : self.vocab_size].float()

    def selfcond(self, log_x0):
        """Expected input embedding under the belief: softmax(log_x0) @ E."""
        probs = log_x0[..., : self.vocab_size].float().softmax(dim=-1)
        emb = self.embed[: self.vocab_size]
        return probs.to(emb.dtype) @ emb


@torch.no_grad()
def generate_ids(
    model,
    prompt_ids: list[int],
    config: Union[str, Path, DecodeConfig, dict] = "nfe8",
    max_new_tokens: int = 256,
    eos_token_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> list[int]:
    """Generate token ids after `prompt_ids` (a chat-formatted prompt). Stops at EOS."""
    cfg = load_config(config)
    device = next(model.parameters()).device
    vocab_size = int(model.config.real_vocab_size)
    eos = int(model.config.eos_token_id if eos_token_id is None else eos_token_id)
    bsz = cfg.block_size
    if bsz != int(getattr(model.config, "diffusion_block_size", bsz)):
        raise ValueError(f"block_size={bsz} does not match the checkpoint's {model.config.diffusion_block_size}")
    if seed is not None:
        torch.manual_seed(seed)

    plen = len(prompt_ids)
    n_blocks = max(1, math.ceil(max_new_tokens / bsz))
    canvas = torch.full((1, plen + n_blocks * bsz), eos, dtype=torch.long, device=device)
    canvas[0, :plen] = torch.as_tensor(prompt_ids, dtype=torch.long, device=device)
    denoiser = _BlockDenoiser(model, vocab_size)
    ts = _time_grid(cfg, device)
    readout_alpha = _alpha(ts[-1], cfg.eps)
    if cfg.terminal_sigma_floor > 0.0:
        readout_alpha = readout_alpha.clamp_max(math.exp(-cfg.terminal_sigma_floor))

    for blk in range(n_blocks):
        bstart = plen + blk * bsz
        bend = bstart + bsz
        x = canvas[:, :bend].clone()
        x[:, bstart:] = torch.randint(0, vocab_size, (1, bsz), device=device)
        sc = None
        for i in range(cfg.steps_per_block):
            a_t, a_s = _alpha(ts[i], cfg.eps), _alpha(ts[i + 1], cfg.eps)
            logits = denoiser.logits(x, bstart, -a_t.clamp_min(1e-30).log(), sc)
            x[:, bstart:], log_x0 = _reverse_step(
                logits, x[:, bstart:], a_t, a_s, cfg.temperature_at(i), cfg, vocab_size)
            if cfg.self_cond:
                sc = denoiser.selfcond(log_x0)
        logits = denoiser.logits(x, bstart, -readout_alpha.clamp_min(1e-30).log(), sc)
        x[:, bstart:], _ = _reverse_step(
            logits, x[:, bstart:], readout_alpha, torch.ones_like(readout_alpha),
            cfg.readout_temperature, cfg, vocab_size)
        canvas[:, bstart:bend] = x[:, bstart:]
        if bool((x[:, bstart:] == eos).any()):
            break

    out = canvas[0, plen:].tolist()
    return out[: out.index(eos)] if eos in out else out


def generate(
    model,
    tokenizer,
    messages: list[dict],
    config: Union[str, Path, DecodeConfig, dict] = "nfe8",
    max_new_tokens: int = 256,
    seed: Optional[int] = None,
    **chat_template_kwargs,
) -> str:
    """Generate an assistant reply for a list of chat messages."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **chat_template_kwargs)
    prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids = generate_ids(model, prompt_ids, config, max_new_tokens,
                       eos_token_id=tokenizer.eos_token_id, seed=seed)
    return tokenizer.decode(ids, skip_special_tokens=True)


def load_model(model_id: str, device: Optional[str] = None, dtype=torch.bfloat16):
    """Load the model (with its bundled modeling code) and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, dtype=dtype)
    return model.to(device).eval(), tokenizer
