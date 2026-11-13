# BYOD: Build Your Own DLM

Convert an instruction model into a bidirectional masked-diffusion language
model on one GPU.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RuurdKuiper/BYOD/blob/main/notebooks/train_diffusion.ipynb)

Start with the notebook to select a model, train, and watch the loss curves.
Step 6 opens an interactive inference page with generation settings and live
denoising updates, including from a saved interrupted run. Setup reuses Colab’s
PyTorch/CUDA packages and installs Gradio only when you run step 6. For a local
inference web app, install `pip install -e ".[demo]"`.
Try the trained models in the hosted demos:
[Llama](https://huggingface.co/spaces/Dovelove/byod-llama-3.1-8b),
[Gemma](https://huggingface.co/spaces/Dovelove/byod-gemma-2-9b),
[Qwen](https://huggingface.co/spaces/Dovelove/byod-qwen2.5-7b),
[Ministral](https://huggingface.co/spaces/Dovelove/byod-ministral-8b).

## Train locally

```bash
pip install -e .
cp .env.example .env  # add your Hugging Face read token
python train.py --model llama --mode quick
```

Accept the parent model's Hugging Face terms before downloading gated models.
Training requires an NVIDIA GPU. Use `--max-updates 2` for a short smoke run.

### CUDA 12.4 PyTorch (Poetry)

Poetry pins `torch==2.6.0+cu124` from the explicit `pytorch-cu124` source
for Windows/Linux NVIDIA training; standard `pip install -e .` retains the
generic `torch>=2.5` requirement. CUDA 12.4 works with the locally observed
NVIDIA driver 552.31. To install the Poetry dependencies:

```bash
poetry install --no-root
poetry run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`--no-root` is needed because this setuptools project is not configured for
Poetry package installation. CUDA should print `2.6.0+cu124`, `12.4`, `True`.
The Poetry pin is platform-specific and will not install on macOS or CPU-only
machines; use the generic pip installation instead. On UNC network paths,
`poetry run` may hang in `cmd.exe`; run the virtualenv's `Scripts/python.exe`
directly or work from a mapped drive.

| Model | `quick` | `paper` |
| --- | --- | --- |
| `llama` | Llama 3.2 1B Instruct | Llama 3.1 8B Instruct |
| `gemma` | Gemma 3 1B IT | Gemma 2 9B IT |
| `qwen` | Qwen2.5 1.5B Instruct | Qwen2.5 7B Instruct |
| `ministral` | — | Ministral 8B Instruct 2410 |

Both modes default to 25,000 updates. Quick mode uses smaller models and
rank-128 LoRA; paper mode uses rank-1024 LoRA and needs a high-memory GPU
(the original runs used 96 GB). Quick mode does not reproduce the paper scores.
This version uses FP16/BF16 training without FP8. Quick Gemma uses BF16 on
supported GPUs and FP32 otherwise to avoid FP16 overflow; the FP32 fallback
needs more GPU memory. Training stops with an explicit error if a training or
validation loss is NaN or infinite, preserving any previously saved best adapter.

Training downloads the matching [pretokenized dataset](https://huggingface.co/datasets/Ruurd/BYOD-tokenized-256)
automatically. Each profile covers the default 400,000-example training budget.
Random masking is sampled online, with answer/padding losses weighted 0.9/0.1.
See [reproducibility notes](REPRODUCIBILITY.md) for provenance and validation limits.

Runs evaluate at each checkpoint and keep only `best/` under `outputs/`.
Stopping training manually also saves `interrupted/`.
Merge an adapter into its base model with:

```bash
python merge_adapter.py outputs/llama-3.1-8b-mask/best merged/llama
```

## Change settings or data

Edit [`configs/paper.yaml`](configs/paper.yaml) for shared hyperparameters,
[`configs/quick.yaml`](configs/quick.yaml) for quick-mode overrides, and
[`configs/models.yaml`](configs/models.yaml) for model choices. To override a
preset without editing it, copy and edit [`configs/custom.example.yaml`](configs/custom.example.yaml):

```bash
python train.py --model llama --mode quick --config configs/my_run.yaml
```

In Colab, enter the file path in **Config YAML**. Overrides apply after the
selected preset; CLI flags take precedence. Set `dataset_name` to your Hugging
Face dataset; it needs text columns `instruction` and `output` (`input` is optional).
Use `dataset_columns: {instruction: question, output: answer}` for other names.
Missing validation/test splits are reserved from train using seed 42. Custom
data is tokenized once using the selected model and cached, including on Drive
when enabled. Random training masking is still sampled online.

## Experimental encoder masked diffusion

An encoder such as BERT already attends bidirectionally and has a same-position
masked-language-model head. It cannot use the decoder-specific LoRA/chat-template
pipeline: use the separate full-model encoder pipe instead. Given a JSONL file
with `instruction` and `output` strings (or a Hub dataset with these columns):

```bash
python encoder_pipe.py train --model bert-base-uncased --dataset examples.jsonl --output-dir outputs/bert-diffusion --max-updates 1000
python encoder_pipe.py sample --model outputs/bert-diffusion --prompt "Explain diffusion" --gen-length 64 --steps 64
```

For a text-only corpus with a `text` column, use `--input-column ''
--output-column text`; sampling then uses an empty prompt. Training randomly masks
answer tokens plus the terminating `[SEP]` at variable noise levels; loss is
computed only at masked answer positions. For optional training-time correction,
pass `--ngram-repetition-weight 2 --ngram-size 3`: when the model's top
prediction is **incorrect** and would complete a trigram already visible
earlier in the same answer, its class gets a higher weight inside the masked
cross-entropy. Only fully visible answer n-grams count; masked tokens, prompt,
and padding cannot supply evidence. Correct repetitions are never penalized.
Try `--ngram-size 2` for a stronger bigram signal; `--ngram-repetition-weight 1`
keeps the original loss. Validation uses the same weighted objective. This
replaces the former `--neighbor-repetition-weight` option and is separate from
the inference-time (single-token) `--repetition-penalty`. With high mask ratios,
complete visible n-grams are rare, so this correction may affect few positions.

Sampling starts from a fixed number
of `[MASK]` slots, repeatedly reveals high-confidence predictions, and stops
at the first predicted `[SEP]`. `--gen-length` is a maximum *token* budget,
not an exact length. Sampling defaults to no logit penalties; use
`--repetition-penalty 1.2` to discourage already revealed answer tokens,
`--eos-penalty 2` to delay the first SEP (`</s>` for RoBERTa), or
`--suppress-eos` to forbid SEP and fill the entire token budget. These affect
inference only; suppressing SEP can produce trailing or less coherent text.
BERT WordPiece generation is experimental: it is not an
instruction-tuned model, and fixed-budget parallel sampling can be less fluent
than autoregressive decoding. This route saves full model weights (not a LoRA
adapter), does not use the decoder evaluation/demo/merge commands, and can be
run on CPU for small tests but benefits from a GPU for real training.

### Dutch clinical model smoke test

`UMCU/CardioBERTa.nl_clinical` is a RoBERTa masked LM (`<mask>` and
`</s>`); `UMCU/MIRIAD4.4.nl` provides `question` and `answer`. The Hub
training split has **4,487,542** rows, so use `--max-samples 100000` to
stream a seed-shuffled bounded subset instead of downloading all rows. This
is a buffer shuffle, not a uniform sample of all 4.49 million records.
The model is GPL-3.0 licensed; check its and the dataset's terms before
redistribution. Training saves full weights and may need several GB of GPU
memory and disk space.

```bash
python encoder_pipe.py train --model UMCU/CardioBERTa.nl_clinical --dataset UMCU/MIRIAD4.4.nl --input-column question --output-column answer --max-samples 100000 --output-dir outputs/cardio-diffusion --max-updates 1000
python encoder_pipe.py sample --model outputs/cardio-diffusion --prompt "Wat zijn de symptomen van hartfalen?" --gen-length 64 --steps 64
```

For a short real-model **train → save → reload → sample** smoke test on CUDA,
install `pip install -e '.[test]'` and run:

```bash
BYOD_RUN_CARDIOBERTA_SMOKE=1 pytest -q -s tests/test_encoder_cardio_smoke.py
```

On PowerShell, set `$env:BYOD_RUN_CARDIOBERTA_SMOKE="1"` before running
`pytest -q -s tests/test_encoder_cardio_smoke.py`. The opt-in test downloads
the checkpoint and streams only 32 usable examples, trains for one update,
and checks inference plumbing; it does **not** establish answer quality.

## Evaluate

Set your adapter paths in the evaluation configs, then run:

```bash
pip install -e '.[evaluation]'
python evaluate_benchmarks.py --config configs/evaluation.yaml
python evaluate_benchmarks.py --config configs/open_ended.yaml
```

The open-ended profile's 4-bit Phi-4 scorer also needs `pip install -e '.[cuda]'`.
Code benchmarks execute generated Python; use a disposable environment.

## Development

Core code is in [`src/diffusion_lm/`](src/diffusion_lm/): `training.py` runs
optimization, `data.py` and `corruption.py` handle batches and masking, `loss.py`
defines the objective, and `inference.py` implements denoising generation.

```bash
pip install -e '.[test]'
pytest -q
```

[MIT license](LICENSE) for the software; upstream model and dataset terms apply.
Citation metadata is in [CITATION.cff](CITATION.cff).
