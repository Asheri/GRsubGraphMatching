"""随机初始化的小型 BART 区域路径生成器。"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from transformers import BartConfig, BartForConditionalGeneration


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(vocab_size: int, *, max_input_length: int = 4096, max_target_length: int = 128) -> BartForConditionalGeneration:
    config = BartConfig(
        vocab_size=vocab_size,
        d_model=128,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=4,
        decoder_attention_heads=4,
        encoder_ffn_dim=512,
        decoder_ffn_dim=512,
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        max_position_embeddings=max_input_length + 2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=2,
        forced_eos_token_id=2,
        use_cache=True,
    )
    model = BartForConditionalGeneration(config)
    model.generation_config.max_length = max_target_length
    model.generation_config.num_beams = 10
    model.generation_config.num_return_sequences = 5
    return model


def generate_paths(
    model: BartForConditionalGeneration,
    input_ids: Sequence[int],
    *,
    device: torch.device,
    max_input_length: int = 4096,
    max_target_length: int = 128,
) -> List[List[int]]:
    if len(input_ids) > max_input_length:
        raise ValueError(f"模型输入长度 {len(input_ids)} 超过上限 {max_input_length}。")
    tensor = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    mask = torch.ones_like(tensor)
    model.eval()
    with torch.no_grad():
        output = model.generate(
            input_ids=tensor,
            attention_mask=mask,
            num_beams=10,
            num_return_sequences=5,
            max_length=max_target_length,
            early_stopping=True,
            do_sample=False,
        )
    return [row.tolist() for row in output]


__all__ = ["generate_paths", "make_model", "set_seed"]
