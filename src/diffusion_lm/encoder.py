"""Masked-diffusion training and fixed-budget generation for encoder MLMs.

Unlike the decoder adapters, encoders already have bidirectional attention and a
same-position masked-token head. An answer is represented by tokens followed by
its native SEP token; SEP is the learned stop marker at generation time.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer


def prompt_text(instruction: str) -> str:
    return f"Instruction: {instruction}\nResponse:"


def _special_ids(tokenizer):
    if tokenizer.mask_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("Encoder diffusion requires a tokenizer with mask_token_id and sep_token_id")
    return tokenizer.mask_token_id, tokenizer.sep_token_id


class EncoderCollator:
    """Pad clean examples, then independently mask answer tokens at a random t."""

    def __init__(self, tokenizer, max_length: int, t_min: float = 0.05, seed: int | None = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.t_min = t_min
        self.seed = seed
        _special_ids(tokenizer)
        if tokenizer.pad_token_id is None:
            raise ValueError("Encoder diffusion requires a pad token")
        if max_length < 4 or not 0 < t_min <= 1:
            raise ValueError("max_length must be >= 4 and t_min must be in (0, 1]")

    def __call__(self, examples):
        rows = []
        for example in examples:
            prefix = self.tokenizer.encode(prompt_text(str(example.get("instruction", ""))), add_special_tokens=True)
            if len(prefix) >= self.max_length - 1:
                # Preserve the initial BOS/CLS and the response cue when the
                # question is longer than the training context.
                prefix = prefix[:1] + prefix[-(self.max_length - 3):]
            answer = self.tokenizer.encode(str(example["output"]), add_special_tokens=False)
            if not answer:
                raise ValueError("Empty output is not a trainable example")
            answer = answer[: self.max_length - len(prefix) - 1]
            clean = prefix + answer + [self.tokenizer.sep_token_id]
            rows.append((clean, len(prefix), int(example.get("_index", 0))))
        width = max(len(clean) for clean, _, _ in rows)
        ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        answer_mask = torch.zeros_like(ids, dtype=torch.bool)
        ts = []
        for row, (clean, start, index) in enumerate(rows):
            end = len(clean)
            ids[row, :end] = torch.tensor(clean)
            attention[row, :end] = 1
            answer_mask[row, start:end - 1] = True
            generator = torch.Generator().manual_seed(self.seed + index) if self.seed is not None else None
            t = float(torch.empty(()).uniform_(self.t_min, 1.0, generator=generator))
            draw = torch.rand(end - start, generator=generator) < t
            if not draw.any():
                draw[torch.randint(end - start, (1,), generator=generator)] = True
            positions = torch.arange(start, end)[draw]
            labels[row, positions] = ids[row, positions]
            ids[row, positions] = self.tokenizer.mask_token_id
            ts.append(t)
        return {"input_ids": ids, "attention_mask": attention, "labels": labels,
                "answer_mask": answer_mask, "sampled_t": torch.tensor(ts),
                "answer_lengths": torch.tensor([len(c) - s for c, s, _ in rows])}


def _wrong_repeated_ngrams(batch, example_ids, token_ids, predictions, labels,
                           mask_token_id: int, ngram_size: int):
    """Find wrong predictions completing an earlier, fully visible answer n-gram."""
    inputs = batch["input_ids"]
    if ngram_size > inputs.shape[1]:
        return torch.zeros_like(predictions, dtype=torch.bool)
    visible = batch["answer_mask"] & (inputs != mask_token_id)
    offsets = torch.arange(ngram_size - 1, device=inputs.device)
    context_positions = (token_ids[:, None] - (ngram_size - 1) + offsets).clamp_min(0)
    context = inputs[example_ids[:, None], context_positions]
    context_visible = visible[example_ids[:, None], context_positions].all(dim=1)
    previous = inputs.unfold(1, ngram_size, 1)[example_ids]
    previous_visible = visible.unfold(1, ngram_size, 1).all(dim=-1)[example_ids]
    previous_starts = torch.arange(previous.shape[1], device=inputs.device)
    earlier = previous_starts[None, :] + ngram_size <= token_ids[:, None]
    repeated = (
        previous_visible & earlier
        & (previous[:, :, :-1] == context[:, None, :]).all(dim=-1)
        & (previous[:, :, -1] == predictions[:, None])
    ).any(dim=1)
    return (predictions != labels) & context_visible & repeated


def encoder_loss(logits, batch, *, ngram_repetition_weight: float = 1.0,
                 ngram_size: int = 3, mask_token_id: int | None = None):
    """Inverse-t masked CE with optional conditional wrong n-gram class weight.

    Boost the predicted class inside CE only when an incorrect prediction
    completes an n-gram already visible earlier in the same answer. Masked
    context and legitimate repeats never trigger the extra loss.
    """
    if not math.isfinite(ngram_repetition_weight) or ngram_repetition_weight < 1:
        raise ValueError("ngram_repetition_weight must be finite and at least 1")
    if ngram_size < 2:
        raise ValueError("ngram_size must be at least 2")
    selected = batch["labels"] != -100
    example_ids, token_ids = selected.nonzero(as_tuple=True)
    selected_logits = logits[example_ids, token_ids].float()
    selected_labels = batch["labels"][selected]
    if ngram_repetition_weight > 1:
        if mask_token_id is None or "answer_mask" not in batch:
            raise ValueError("N-gram repetition weighting requires mask_token_id and answer_mask")
        predicted = selected_logits.detach().argmax(dim=-1)
        copy = _wrong_repeated_ngrams(batch, example_ids, token_ids, predicted, selected_labels,
                                       mask_token_id, ngram_size)
        if copy.any():
            selected_logits = selected_logits.clone()
            rows = copy.nonzero(as_tuple=True)[0]
            selected_logits[rows, predicted[rows]] += math.log(ngram_repetition_weight)
    ce = F.cross_entropy(selected_logits, selected_labels, reduction="none")
    totals = torch.zeros(logits.shape[0], device=logits.device, dtype=ce.dtype).scatter_add(0, example_ids, ce)
    return (totals / batch["sampled_t"].to(logits.device) / batch["answer_lengths"].to(logits.device)).mean()


def _rows(dataset, input_column, output_column):
    for index, item in enumerate(dataset):
        yield {"instruction": item[input_column] if input_column else "", "output": item[output_column], "_index": index}


def train_encoder(
    model_name: str, dataset_name: str, output_dir: str | Path, *,
    input_column: str | None = "instruction", output_column: str = "output",
    max_length: int = 256, batch_size: int = 8, max_updates: int = 1000,
    learning_rate: float = 5e-5, validation_fraction: float = 0.05,
    seed: int = 42, device: str | None = None, max_samples: int | None = None,
    ngram_repetition_weight: float = 1.0, ngram_size: int = 3,
):
    """Fine-tune a full masked-LM and save the best validation checkpoint.

    dataset_name may be a local JSON/JSONL file or a Hugging Face dataset ID.
    max_samples streams a bounded, seed-shuffled subset instead of downloading
    a multi-million-row dataset. This does not load decoder token caches/adapters.
    """
    from datasets import load_dataset

    if max_updates < 1 or batch_size < 1 or learning_rate <= 0 or not 0 < validation_fraction < 1:
        raise ValueError("max_updates, batch_size and learning_rate must be positive; validation_fraction must be in (0, 1)")
    if max_samples is not None and max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    if not math.isfinite(ngram_repetition_weight) or ngram_repetition_weight < 1:
        raise ValueError("ngram_repetition_weight must be finite and at least 1")
    if ngram_size < 2:
        raise ValueError("ngram_size must be at least 2")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Encoder output directory is not empty: {output}")
    torch.manual_seed(seed)
    source = Path(dataset_name)
    source_kwargs = {"data_files": str(source)} if source.is_file() else {}
    source_name = "json" if source.is_file() else dataset_name
    if max_samples is not None:
        # Shuffling an iterable uses a bounded buffer, not a uniform sample of
        # the entire Hub dataset. It avoids fetching all 4.49M MIRIAD rows.
        stream = load_dataset(source_name, split="train", streaming=True, **source_kwargs)
        stream = stream.filter(lambda row: isinstance(row.get(output_column), str) and bool(row[output_column].strip()))
        dataset = list(stream.shuffle(seed=seed, buffer_size=min(max_samples, 10_000)).take(max_samples))
        columns = dataset[0].keys() if dataset else ()
    else:
        dataset = load_dataset(source_name, split="train", **source_kwargs)
        columns = dataset.column_names
    needed = [output_column] + ([input_column] if input_column else [])
    if any(column not in columns for column in needed):
        raise ValueError(f"Dataset must contain columns {needed}")
    if len(dataset) < 2:
        raise ValueError("At least two examples are needed for training and validation")
    if max_samples is not None:
        import random
        random.Random(seed).shuffle(dataset)
        n_validation = max(1, round(len(dataset) * validation_fraction))
        train_data, val_data = dataset[n_validation:], dataset[:n_validation]
    else:
        split = dataset.train_test_split(test_size=max(1, round(len(dataset) * validation_fraction)), seed=seed)
        train_data, val_data = split["train"], split["test"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    _special_ids(tokenizer)
    limit = getattr(model.config, "max_position_embeddings", None)
    if limit is not None and max_length > limit:
        raise ValueError(f"max_length={max_length} exceeds encoder position limit {limit}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    train_rows = list(_rows(train_data, input_column, output_column))
    val_rows = list(_rows(val_data, input_column, output_column))
    collate = EncoderCollator(tokenizer, max_length)
    validation_collate = EncoderCollator(tokenizer, max_length, seed=seed)
    train_loader = DataLoader(train_rows, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_rows, batch_size=batch_size, collate_fn=validation_collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    best = math.inf
    iterator = iter(train_loader)
    for step in range(1, max_updates + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        model.train()
        optimizer.zero_grad()
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss = encoder_loss(logits, batch, ngram_repetition_weight=ngram_repetition_weight,
                            ngram_size=ngram_size, mask_token_id=tokenizer.mask_token_id)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite encoder training loss at step {step}")
        loss.backward()
        optimizer.step()
        if step == max_updates or step % max(1, max_updates // 20) == 0:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for val in val_loader:
                    val = {key: value.to(device) for key, value in val.items()}
                    logits = model(input_ids=val["input_ids"], attention_mask=val["attention_mask"]).logits
                    total += encoder_loss(logits, val, ngram_repetition_weight=ngram_repetition_weight,
                                          ngram_size=ngram_size, mask_token_id=tokenizer.mask_token_id).item() * len(val["input_ids"])
            metric = total / len(val_rows)
            if not math.isfinite(metric):
                raise FloatingPointError(f"Non-finite encoder validation loss at step {step}")
            print(f"step {step}/{max_updates}: train={loss.item():.4f} validation={metric:.4f}", flush=True)
            if metric < best:
                best = metric
                model.save_pretrained(output, safe_serialization=True)
                tokenizer.save_pretrained(output)
    return best


@torch.no_grad()
def generate_encoder(model, tokenizer, instruction: str, *, gen_length: int = 64,
                     steps: int = 64, temperature: float = 0.0, device: str | torch.device | None = None,
                     repetition_penalty: float = 1.0, eos_penalty: float = 1.0,
                     suppress_eos: bool = False) -> str:
    """Progressively reveal the most confident masked answer positions.

    Fixed-length slots are all visible to the encoder; the first predicted SEP
    ends the response unless suppressed. Penalties divide the unnormalized
    probability by their value per already-revealed occurrence (or for SEP).
    Unresolved positions are recomputed at every step.
    """
    mask_id, sep_id = _special_ids(tokenizer)
    if gen_length < 1 or steps < 1 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("gen_length and steps must be positive; temperature must be finite and non-negative")
    if any(not math.isfinite(value) or value < 1 for value in (repetition_penalty, eos_penalty)):
        raise ValueError("repetition_penalty and eos_penalty must be finite and at least 1")
    prefix = tokenizer.encode(prompt_text(instruction), add_special_tokens=True)
    limit = getattr(model.config, "max_position_embeddings", None)
    if limit is not None and len(prefix) + gen_length > limit:
        raise ValueError(f"Prompt plus gen_length exceeds encoder position limit {limit}")
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    x = torch.tensor([prefix + [mask_id] * gen_length], device=device)
    model.eval()
    blocked = set(tokenizer.all_special_ids) - {sep_id}
    blocked.add(mask_id)
    if suppress_eos:
        blocked.add(sep_id)
    for step in range(steps):
        remaining = torch.where(x[0, len(prefix):] == mask_id)[0]
        if not len(remaining):
            break
        logits = model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0, len(prefix):][remaining].float()
        if repetition_penalty > 1:
            revealed = x[0, len(prefix):]
            revealed = revealed[(revealed != mask_id) & (revealed != sep_id)]
            if len(revealed):
                token_ids, counts = torch.unique(revealed, return_counts=True)
                logits[:, token_ids] -= counts.to(logits.dtype) * math.log(repetition_penalty)
        if eos_penalty > 1:
            logits[:, sep_id] -= math.log(eos_penalty)
        logits[:, list(blocked)] = -torch.inf
        if temperature:
            probs = F.softmax(logits / temperature, dim=-1)
            predictions = torch.multinomial(probs, 1).squeeze(-1)
            confidence = probs.gather(1, predictions[:, None]).squeeze(-1)
        else:
            confidence, predictions = F.softmax(logits, dim=-1).max(dim=-1)
        count = math.ceil(len(remaining) / (steps - step))
        chosen = torch.topk(confidence, count).indices
        x[0, len(prefix) + remaining[chosen]] = predictions[chosen]
    answer = x[0, len(prefix):].tolist()
    if sep_id in answer:
        answer = answer[:answer.index(sep_id)]
    return tokenizer.decode([token for token in answer if token != mask_id], skip_special_tokens=True).strip()


def load_encoder(path: str | Path, device: str | None = None):
    """Load a saved full encoder checkpoint for generation."""
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForMaskedLM.from_pretrained(path)
    model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model, tokenizer
