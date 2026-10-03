"""Generate a reply with PyTorch. Run: uv run scripts/generate.py --prompt "Hello" --nfe 8"""

import argparse
import time

from lfm_diffusion import generate, load_config, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="LiquidAI/lfm2.5-350m-diffusion-exp")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--nfe", type=int, choices=[32, 8, 4], default=8,
                       help="denoising steps per 32-token block (shipped presets)")
    group.add_argument("--config", help="path to a decode config YAML")
    p.add_argument("--prompt", default="What is C. elegans?")
    p.add_argument("--system", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    model, tokenizer = load_model(args.model, device=args.device)
    config = load_config(args.config or f"nfe{args.nfe}")
    messages = ([{"role": "system", "content": args.system}] if args.system else [])
    messages.append({"role": "user", "content": args.prompt})

    start = time.perf_counter()
    reply = generate(model, tokenizer, messages, config, args.max_new_tokens, seed=args.seed)
    elapsed = time.perf_counter() - start
    print(reply)
    n = len(tokenizer(reply, add_special_tokens=False)["input_ids"])
    print(f"\n[{n} tokens in {elapsed:.2f}s, {config.steps_per_block} steps per block]")


if __name__ == "__main__":
    main()
