"""Model loading utilities."""

import torch
from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)


def load_model_and_processor(model_id: str, model_dir: str, model_type: str):
    """
    Load a multimodal LLM and its processor.

    Args:
        model_id: HuggingFace model identifier.
        model_dir: Local directory containing model weights.
        model_type: "Llava" or "Lingshu" (Qwen2.5-VL).

    Returns:
        (model, processor) tuple.
    """
    if model_type == "Llava":
        print(f"Loading LLaVA model from {model_dir}")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        try:
            processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"

    elif model_type == "Lingshu":
        print(f"Loading Qwen2.5-VL model from {model_dir}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        try:
            processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        except Exception:
            processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model, processor
