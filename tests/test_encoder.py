import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from diffusion_lm.encoder import EncoderCollator, encoder_loss, generate_encoder, load_encoder


class TinyTokenizer:
    mask_token_id = 3
    sep_token_id = 2
    pad_token_id = 0
    all_special_ids = [0, 1, 2, 3]

    def encode(self, text, add_special_tokens=True):
        ids = [4 + (ord(c) % 4) for c in text]
        return [1] + ids + [2] if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if not skip_special_tokens or i not in self.all_special_ids)


class FixedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))
        self.config = type("Config", (), {"max_position_embeddings": 100})()
        self.calls = []

    def forward(self, input_ids, attention_mask):
        self.calls.append((input_ids.clone(), attention_mask.clone()))
        logits = torch.full((*input_ids.shape, 8), -100.0)
        logits[..., 4] = 10.0
        logits[..., 2] = 9.0
        # Prefer SEP at the first answer position; keep it visible after selection.
        prefix_length = self.prefix_length
        logits[:, prefix_length, 2] = 20.0
        return type("Output", (), {"logits": logits})()


def test_encoder_corruption_only_masks_answer_and_sep_not_prompt_or_padding():
    tokenizer = TinyTokenizer()
    collator = EncoderCollator(tokenizer, max_length=100, t_min=1, seed=123)
    batch = collator([{"instruction": "a", "output": "xy", "_index": 0},
                      {"instruction": "longer", "output": "z", "_index": 1}])
    first_start = len(tokenizer.encode("Instruction: a\nResponse:"))
    assert torch.equal(batch["input_ids"][0, :first_start],
                       torch.tensor(tokenizer.encode("Instruction: a\nResponse:")))
    assert (batch["labels"][0, :first_start] == -100).all()
    assert not batch["answer_mask"][0, :first_start].any()
    assert batch["answer_mask"][0, first_start:first_start + 2].all()
    assert not batch["answer_mask"][0, first_start + 2:].any()
    assert batch["labels"][0, first_start:first_start + 3].tolist() == tokenizer.encode("xy", False) + [2]
    assert (batch["input_ids"][0, first_start:first_start + 3] == 3).all()
    assert (batch["labels"][0, first_start + 3:] == -100).all()
    assert (batch["attention_mask"][0, first_start + 3:] == 0).all()
    assert torch.equal(batch["sampled_t"], collator([
        {"instruction": "a", "output": "xy", "_index": 0},
        {"instruction": "longer", "output": "z", "_index": 1}])["sampled_t"])
    logits = torch.randn(*batch["labels"].shape, 8, requires_grad=True)
    loss = encoder_loss(logits, batch)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum() > 0
    assert (logits.grad[batch["labels"] == -100] == 0).all()


def test_neighbor_weight_increases_loss_and_gradient_only_for_wrong_visible_answer_copy():
    batch = {
        "input_ids": torch.tensor([[4, 5, 3, 7, 2]]),
        "answer_mask": torch.tensor([[False, True, True, True, False]]),
        "labels": torch.tensor([[-100, -100, 6, -100, -100]]),
        "sampled_t": torch.ones(1), "answer_lengths": torch.tensor([4]),
    }
    logits = torch.zeros(1, 5, 8, requires_grad=True)
    with torch.no_grad():
        logits[0, 2, 5] = 4  # Wrong prediction copies the visible left answer token.
        logits[0, 2, 6] = 3  # Correct target.
    base = encoder_loss(logits, batch)
    boosted = encoder_loss(logits, batch, neighbor_repetition_weight=4, mask_token_id=3)
    assert boosted > base
    base_gradient = torch.autograd.grad(base, logits, retain_graph=True)[0]
    boosted_gradient = torch.autograd.grad(boosted, logits)[0]
    assert boosted_gradient[0, 2, 5] > base_gradient[0, 2, 5]

    # The true answer may legitimately repeat the preceding answer token.
    batch["labels"][0, 2] = 5
    assert torch.allclose(encoder_loss(logits, batch),
                          encoder_loss(logits, batch, neighbor_repetition_weight=4, mask_token_id=3))
    batch["labels"][0, 2] = 6
    # Do not punish copying a prompt token or a masked answer neighbor.
    batch["answer_mask"][0, 1] = False
    assert torch.allclose(encoder_loss(logits, batch),
                          encoder_loss(logits, batch, neighbor_repetition_weight=4, mask_token_id=3))
    batch["answer_mask"][0, 1] = True
    batch["input_ids"][0, 1] = 3
    assert torch.allclose(encoder_loss(logits, batch),
                          encoder_loss(logits, batch, neighbor_repetition_weight=4, mask_token_id=3))

    # The right answer neighbor also counts, but only if visible.
    batch["input_ids"][0, 3] = 5
    assert encoder_loss(logits, batch, neighbor_repetition_weight=4, mask_token_id=3) > encoder_loss(logits, batch)
    with pytest.raises(ValueError, match="at least 1"):
        encoder_loss(logits, batch, neighbor_repetition_weight=0)
    with pytest.raises(ValueError, match="requires mask_token_id"):
        encoder_loss(logits, batch, neighbor_repetition_weight=2)


def test_encoder_sampler_keeps_prompt_and_uses_sep_to_stop():
    tokenizer = TinyTokenizer()
    model = FixedModel()
    model.prefix_length = len(tokenizer.encode("Instruction: hello\nResponse:"))
    result = generate_encoder(model, tokenizer, "hello", gen_length=3, steps=3)
    assert result == ""
    assert len(model.calls) >= 1
    first_ids, first_attention = model.calls[0]
    assert first_ids[0, model.prefix_length:].tolist() == [3, 3, 3]
    assert (first_attention == 1).all()
    assert all(torch.equal(ids[0, :model.prefix_length], first_ids[0, :model.prefix_length])
               for ids, _ in model.calls)


def test_encoder_sampler_can_penalize_or_suppress_stop_token():
    tokenizer = TinyTokenizer()
    model = FixedModel()
    model.prefix_length = len(tokenizer.encode("Instruction: hello\nResponse:"))
    assert generate_encoder(model, tokenizer, "hello", gen_length=1, steps=1) == ""
    assert generate_encoder(model, tokenizer, "hello", gen_length=1, steps=1,
                            eos_penalty=1e6) == "4"
    assert generate_encoder(model, tokenizer, "hello", gen_length=1, steps=1,
                            suppress_eos=True) == "4"


class RepeatingModel(FixedModel):
    def forward(self, input_ids, attention_mask):
        logits = torch.full((*input_ids.shape, 8), -100.0)
        logits[..., 4] = 10.0
        logits[..., 5] = 9.0
        return type("Output", (), {"logits": logits})()


def test_encoder_sampler_penalizes_only_revealed_answer_tokens():
    tokenizer = TinyTokenizer()
    model = RepeatingModel()
    original = generate_encoder(model, tokenizer, "hello", gen_length=2, steps=2)
    penalized = generate_encoder(model, tokenizer, "hello", gen_length=2, steps=2,
                                 repetition_penalty=4)
    assert original == "4 4"
    assert sorted(penalized.split()) == ["4", "5"]
    with pytest.raises(ValueError, match="at least 1"):
        generate_encoder(model, tokenizer, "hello", repetition_penalty=0.5)
    with pytest.raises(ValueError, match="at least 1"):
        generate_encoder(model, tokenizer, "hello", eos_penalty=0)


def test_tiny_local_mlm_train_save_reload_and_sample(tmp_path):
    """Exercise the same streaming, update and checkpoint path without Hub/CUDA."""
    from transformers import BertConfig, BertForMaskedLM, BertTokenizer

    base = tmp_path / "base"
    base.mkdir()
    vocab = base / "vocab.txt"
    vocab.write_text("\n".join([
        "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "instruction", "response", ":", "question", "antwoord", "hart",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ]) + "\n")
    tokenizer = BertTokenizer(vocab_file=str(vocab))
    tokenizer.save_pretrained(base)
    BertForMaskedLM(BertConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=64,
    )).save_pretrained(base)
    data = tmp_path / "qa.jsonl"
    data.write_text("".join(json.dumps({"question": f"question {i}", "answer": "antwoord hart"}) + "\n"
                            for i in range(10)))
    output = tmp_path / "trained"
    script = Path(__file__).resolve().parents[1] / "encoder_pipe.py"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    train = subprocess.run([
        sys.executable, str(script), "train", "--model", str(base),
        "--dataset", str(data), "--input-column", "question",
        "--output-column", "answer", "--max-samples", "6",
        "--output-dir", str(output), "--max-length", "48",
        "--batch-size", "2", "--max-updates", "1",
        "--neighbor-repetition-weight", "2",
        "--validation-fraction", "0.25", "--device", "cpu",
    ], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=90, check=True)
    assert "step 1/1" in train.stdout
    model, loaded_tokenizer = load_encoder(output, device="cpu")
    answer = generate_encoder(model, loaded_tokenizer, "question 0", gen_length=4, steps=2)
    assert isinstance(answer, str)
    assert loaded_tokenizer.mask_token not in answer
    sample = subprocess.run([
        sys.executable, str(script), "sample", "--model", str(output),
        "--prompt", "question 0", "--gen-length", "4", "--steps", "2",
        "--device", "cpu",
    ], cwd=tmp_path, env=env, text=True, capture_output=True, timeout=90, check=True)
    assert sample.stdout.strip() == answer


def test_encoder_checks_context_length_and_mask_tokens():
    tokenizer = TinyTokenizer()
    model = FixedModel()
    model.config.max_position_embeddings = 4
    with pytest.raises(ValueError, match="position limit"):
        generate_encoder(model, tokenizer, "hello", gen_length=3, steps=2)
    batch = EncoderCollator(tokenizer, max_length=4, t_min=1)([
        {"instruction": "hello", "output": "yes"}])
    assert batch["input_ids"].shape[1] == 4
    assert batch["labels"][0, -1] == tokenizer.sep_token_id
