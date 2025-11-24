#!/usr/bin/env python3
"""
Simple smoke test for Kimi Linear checkpoints.

Usage:
    python examples/kimi_linear_test.py -m /path/to/Kimi-Linear-48B-A3B-Instruct -p "Hello!"
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from exllamav3 import Cache, Config, Generator, Model, Tokenizer  # noqa: E402


def collect_stop_conditions(tokenizer: Tokenizer) -> list[int]:
    """Return a deduplicated list of useful stop tokens for chat-style prompts."""
    candidates = ["<|im_end|>", "[EOT]", "<|eot_id|>"]
    stops: list[int] = []
    for token in candidates:
        token_id = tokenizer.single_id(token)
        if token_id is not None:
            stops.append(token_id)
    if tokenizer.eos_token_id is not None:
        stops.append(tokenizer.eos_token_id)
    # Deduplicate while preserving order.
    deduped = []
    for token_id in stops:
        if token_id not in deduped:
            deduped.append(token_id)
    return deduped


def build_prompt(model: Model, user_prompt: str, system_prompt: str, raw: bool) -> str:
    if raw:
        return user_prompt
    try:
        return model.default_chat_prompt(user_prompt, system_prompt)
    except NotImplementedError:
        return user_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick test runner for Kimi Linear checkpoints.")
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Path to the model directory (converted EXL3 weights + config).",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default="Hello! Can you introduce yourself?",
        help="User prompt to send to the model.",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant provided by Moonshot-AI.",
        help="System prompt injected when --raw-prompt is not set.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to sample.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=131072,
        help="Total tokens to allocate for the KV cache (must be a multiple of 256).",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=4096,
        help="Prefill chunk size passed to the generator.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional single-device override, e.g. cuda:0.",
    )
    parser.add_argument(
        "--tensor-parallel",
        action="store_true",
        help="Enable tensor parallel loading (required when specifying per-device VRAM budgets).",
    )
    parser.add_argument(
        "--use-per-device",
        type=float,
        nargs="+",
        default=None,
        help="Maximum GB of VRAM to use per device (one value per GPU).",
    )
    parser.add_argument(
        "--reserve-per-device",
        type=float,
        nargs="+",
        default=None,
        help="VRAM to reserve (GB) per device before loading (one value per GPU).",
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Send the prompt exactly as provided (skip default chat template).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the load progress bar.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.cache_size % 256 != 0:
        raise ValueError("--cache-size must be a multiple of 256 tokens.")

    config = Config.from_directory(args.model)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.cache_size)
    load_kwargs = {
        "device": args.device,
        "progressbar": not args.quiet,
    }
    if args.tensor_parallel or args.use_per_device or args.reserve_per_device:
        load_kwargs["tensor_p"] = True
        if args.use_per_device:
            load_kwargs["use_per_device"] = args.use_per_device
        if args.reserve_per_device:
            load_kwargs["reserve_per_device"] = args.reserve_per_device

    model.load(**load_kwargs)

    generator = Generator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        max_batch_size=1,
        max_chunk_size=args.max_chunk_size,
    )

    prompt = build_prompt(model, args.prompt, args.system, args.raw_prompt)
    stop_conditions = collect_stop_conditions(tokenizer)

    response = generator.generate(
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        stop_conditions=stop_conditions if stop_conditions else None,
        add_bos=True,
        completion_only=not args.raw_prompt,
    )

    if isinstance(response, list):
        response = response[0]

    print("\n=== Response ===\n")
    print(response.strip())
    print("\n===============")

    model.unload()


if __name__ == "__main__":
    main()
