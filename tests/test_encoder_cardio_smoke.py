"""Opt-in Hub/GPU end-to-end smoke test for the Dutch clinical encoder.

Run with BYOD_RUN_CARDIOBERTA_SMOKE=1 pytest -q -s tests/test_encoder_cardio_smoke.py
Downloads the public checkpoint and a bounded streamed dataset sample.
"""
import math
import os

import pytest
import torch

from diffusion_lm.encoder import generate_encoder, load_encoder, train_encoder


@pytest.mark.skipif(os.getenv("BYOD_RUN_CARDIOBERTA_SMOKE") != "1", reason="opt-in Hub download and full-model GPU training")
def test_cardio_bert_miriad_train_save_reload_and_sample(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("The real 166M-parameter model smoke test needs a CUDA GPU")

    checkpoint = tmp_path / "cardio-diffusion"
    best_loss = train_encoder(
        "UMCU/CardioBERTa.nl_clinical", "UMCU/MIRIAD4.4.nl", checkpoint,
        input_column="question", output_column="answer", max_samples=32,
        max_length=128, batch_size=2, max_updates=1,
        validation_fraction=0.125, device="cuda",
    )
    assert math.isfinite(best_loss)
    assert (checkpoint / "config.json").is_file()
    model, tokenizer = load_encoder(checkpoint, device="cuda")
    answer = generate_encoder(
        model, tokenizer, "Wat zijn de symptomen van hartfalen?",
        gen_length=8, steps=4,
    )
    print(f"CardioBERTa smoke sample: {answer!r}")
    assert isinstance(answer, str)
    assert tokenizer.mask_token not in answer
