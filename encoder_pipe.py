#!/usr/bin/env python3
"""Train or sample from a bidirectional masked-LM diffusion model."""
import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--model", default="bert-base-uncased")
    train.add_argument("--dataset", required=True, help="JSONL/JSON file or Hugging Face dataset ID")
    train.add_argument("--output-dir", required=True)
    train.add_argument("--input-column", default="instruction", help="Use an empty string for unconditional text training")
    train.add_argument("--output-column", default="output")
    train.add_argument("--max-length", type=int, default=256)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--max-updates", type=int, default=1000)
    train.add_argument("--max-samples", type=int, help="Stream at most this many usable rows (avoids downloading the whole dataset)")
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--ngram-repetition-weight", type=float, default=1.0,
                       help="Boost loss when an incorrect prediction repeats a visible answer n-gram; 1 disables it")
    train.add_argument("--ngram-size", type=int, default=3, help="Size of repeated answer n-grams (at least 2)")
    train.add_argument("--validation-fraction", type=float, default=0.05)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", help="Defaults to CUDA if available, otherwise CPU")
    sample = commands.add_parser("sample")
    sample.add_argument("--model", required=True, help="Directory saved by the train command")
    sample.add_argument("--prompt", required=True)
    sample.add_argument("--gen-length", type=int, default=64)
    sample.add_argument("--steps", type=int, default=64)
    sample.add_argument("--temperature", type=float, default=0.0)
    sample.add_argument("--repetition-penalty", type=float, default=1.0, help="Penalize previously revealed answer tokens; 1 disables it")
    sample.add_argument("--eos-penalty", type=float, default=1.0, help="Penalize the tokenizer's SEP stop token; 1 disables it")
    sample.add_argument("--suppress-eos", action="store_true", help="Disallow SEP; generate up to the full token budget")
    sample.add_argument("--gif-output", help="Optionally save a GIF of answer slots filling over time")
    sample.add_argument("--device")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parent / ".env")
    os.environ["USE_TF"] = "0"
    os.environ["USE_TORCH"] = "1"
    # Poetry --no-root installs dependencies but not this src-layout package.
    # Run the script from any directory without relying on an editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from diffusion_lm.encoder import generate_encoder, load_encoder, train_encoder

    if args.command == "train":
        train_encoder(args.model, args.dataset, Path(args.output_dir),
                      input_column=args.input_column or None, output_column=args.output_column,
                      max_length=args.max_length, batch_size=args.batch_size,
                      max_updates=args.max_updates, learning_rate=args.learning_rate,
                      validation_fraction=args.validation_fraction, seed=args.seed, device=args.device,
                      max_samples=args.max_samples,
                      ngram_repetition_weight=args.ngram_repetition_weight,
                      ngram_size=args.ngram_size)
    else:
        model, tokenizer = load_encoder(args.model, args.device)
        print(generate_encoder(model, tokenizer, args.prompt, gen_length=args.gen_length,
                               steps=args.steps, temperature=args.temperature,
                               repetition_penalty=args.repetition_penalty,
                               eos_penalty=args.eos_penalty, suppress_eos=args.suppress_eos,
                               gif_output=args.gif_output))


if __name__ == "__main__":
    main()
