"""
Fine-tune a Qwen2.5-VL based multimodal LLM on medical VQA data using LoRA.

This script trains the model on parquet-formatted datasets containing medical
images and question-answer pairs. It supports:
  - LoRA with configurable target modules (language-only or language+merger)
  - Gradient accumulation and gradient checkpointing
  - Automatic checkpoint resume
  - FlashAttention-2 with SDPA fallback
"""

import os
import json
import argparse
import sys

import torch
import pandas as pd
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    get_scheduler,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from accelerate import Accelerator

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_process.data_preprocess import train_collate_fn_lingshu


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
CHECKPOINT_DIRNAME = "checkpoints"
LATEST_CHECKPOINT_FILE = "latest_checkpoint.json"


def _checkpoint_root(save_dir: str) -> str:
    return os.path.join(save_dir, CHECKPOINT_DIRNAME)


def _latest_meta_path(save_dir: str) -> str:
    return os.path.join(_checkpoint_root(save_dir), LATEST_CHECKPOINT_FILE)


def _save_checkpoint_meta(save_dir: str, ckpt_dir: str, epoch: int) -> None:
    os.makedirs(_checkpoint_root(save_dir), exist_ok=True)
    with open(_latest_meta_path(save_dir), "w") as f:
        json.dump({"checkpoint_dir": ckpt_dir, "completed_epoch": epoch}, f, indent=2)


def _load_checkpoint_meta(save_dir: str):
    path = _latest_meta_path(save_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            meta = json.load(f)
        if "checkpoint_dir" in meta and "completed_epoch" in meta:
            return meta
    except Exception:
        pass
    return None


def _safe_rmtree(path: str) -> None:
    """Remove a directory tree inside the checkpoint root."""
    if not path or not os.path.isdir(path):
        return
    try:
        for root, dirs, files in os.walk(path, topdown=False):
            for fname in files:
                os.remove(os.path.join(root, fname))
            for dname in dirs:
                os.rmdir(os.path.join(root, dname))
        os.rmdir(path)
    except Exception as e:
        print(f"Warning: failed to cleanup old checkpoint '{path}': {e}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MedicalVQADataset(Dataset):
    """Loads medical VQA data from a parquet file.

    Each row contains an ``image`` column (dict with ``bytes`` key) and a
    ``metadata`` column (JSON string or dict/list of QA pairs).  The dataset
    is flattened so that every QA pair becomes a separate sample.
    """

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.samples = self._flatten(df)

    def _flatten(self, df: pd.DataFrame):
        data = []
        for idx, row in df.iterrows():
            image_bytes = row["image"].get("bytes")
            try:
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                print(f"Skipping image at index {idx}: {e}")
                continue

            try:
                metadata = row["metadata"]
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                if not isinstance(metadata, list):
                    metadata = [metadata]
            except json.JSONDecodeError as e:
                print(f"Skipping metadata at index {idx}: {e}")
                continue

            for qa in metadata:
                q, a = qa.get("Question", ""), qa.get("Answer", "")
                if q and a:
                    data.append({
                        "image": image,
                        "question": q,
                        "answer": a,
                        "task_type": "generation",
                    })

        print(f"Loaded {len(data)} QA pairs from {len(df)} images")
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# LoRA target module discovery
# ---------------------------------------------------------------------------
def find_linear_modules(model, include_merger: bool = False):
    """Return names of linear layers suitable for LoRA.

    Args:
        model: The base model.
        include_merger: If True, include the vision-language merger/projector
            layers in LoRA training.  If False (default), only language layers
            are trained.
    """
    if include_merger:
        excluded = ["vision_model", "visual"]
    else:
        excluded = ["multi_modal_projector", "vision_model", "visual"]

    names = set()
    for name, module in model.named_modules():
        if any(kw in name for kw in excluded):
            continue
        if isinstance(module, torch.nn.Linear):
            short = name.split(".")[-1] if "." in name else name
            if not short.isdigit():
                names.add(short)

    names.discard("lm_head")
    print(f"LoRA target modules ({len(names)}): {sorted(names)}")
    return list(names)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    print("=" * 70)
    print("Fine-tuning Multimodal LLM on Medical VQA")
    print("=" * 70)
    print(f"  Base model : {args.model_id}")
    print(f"  Data       : {args.data_dir}")
    print(f"  Output     : {args.save_dir}")
    print(f"  Epochs     : {args.num_epochs}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  LR         : {args.lr}")
    print("=" * 70)

    # -- Load model ----------------------------------------------------------
    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id, attn_implementation="flash_attention_2", **load_kwargs
        )
        print("Loaded with FlashAttention-2")
    except Exception:
        print("FlashAttention-2 unavailable, falling back to SDPA")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id, attn_implementation="sdpa", **load_kwargs
        )

    # -- Processor & tokenizer -----------------------------------------------
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "right"

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("Resizing embedding matrix to match tokenizer")
        model.resize_token_embeddings(len(tokenizer))

    # -- LoRA ----------------------------------------------------------------
    target_modules = find_linear_modules(model, include_merger=args.include_merger)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # Ensure bf16
    for _, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    model.print_trainable_parameters()

    # -- Gradient checkpointing ----------------------------------------------
    if not args.no_gradient_checkpointing:
        try:
            model.enable_input_require_grads()
        except AttributeError:
            pass
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

    # -- Dataset & dataloader ------------------------------------------------
    df = pd.read_parquet(args.data_dir)
    dataset = MedicalVQADataset(df)
    print(f"Dataset: {len(dataset)} samples")
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Check data format.")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_fn_lingshu(x, processor, args),
    )

    # -- Optimizer & scheduler -----------------------------------------------
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(train_loader) * args.num_epochs
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=0, num_training_steps=num_training_steps,
    )

    # -- Accelerator ---------------------------------------------------------
    accelerator = Accelerator()
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    # -- Resume from checkpoint ----------------------------------------------
    accum_steps = args.gradient_accumulation_steps
    start_epoch = 0

    meta = _load_checkpoint_meta(args.save_dir)
    if meta and os.path.isdir(meta["checkpoint_dir"]):
        print(f"Resuming from checkpoint: {meta['checkpoint_dir']}")
        accelerator.load_state(meta["checkpoint_dir"])
        start_epoch = int(meta["completed_epoch"])

    # -- Training loop -------------------------------------------------------
    print(f"\nTraining for epochs {start_epoch + 1}..{args.num_epochs}"
          f"  (accum_steps={accum_steps})")

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        if accum_steps > 1:
            optimizer.zero_grad()
            for step, batch in enumerate(pbar):
                input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
                with accelerator.accumulate(model):
                    outputs = model(
                        input_ids=input_ids, attention_mask=attention_mask,
                        pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                        labels=labels,
                    )
                    loss = outputs.loss / accum_steps
                    accelerator.backward(loss)
                    if (step + 1) % accum_steps == 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()
                total_loss += outputs.loss.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")
        else:
            for step, batch in enumerate(pbar):
                input_ids, attention_mask, pixel_values, image_grid_thw, labels = batch
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                    labels=labels,
                )
                accelerator.backward(outputs.loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                total_loss += outputs.loss.item()
                n_batches += 1
                pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1} - avg loss: {avg_loss:.4f}")

        # Save checkpoint (keep only the latest to save disk)
        old_meta = _load_checkpoint_meta(args.save_dir)
        old_ckpt = old_meta["checkpoint_dir"] if old_meta else None

        ckpt_dir = os.path.join(_checkpoint_root(args.save_dir), f"epoch_{epoch + 1}")
        accelerator.save_state(ckpt_dir)
        _save_checkpoint_meta(args.save_dir, ckpt_dir, epoch + 1)

        if accelerator.is_main_process and old_ckpt and old_ckpt != ckpt_dir:
            ckpt_root = os.path.abspath(_checkpoint_root(args.save_dir))
            if os.path.abspath(old_ckpt).startswith(ckpt_root):
                _safe_rmtree(old_ckpt)
        accelerator.wait_for_everyone()

    # -- Save final model ----------------------------------------------------
    print("\nSaving final merged model...")
    os.makedirs(args.save_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    merged = accelerator.unwrap_model(model).merge_and_unload()
    merged.save_pretrained(args.save_dir)
    processor.save_pretrained(args.save_dir)

    config = {
        "model_id": args.model_id,
        "data_dir": args.data_dir,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.lr,
        "gradient_accumulation_steps": accum_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "include_merger": args.include_merger,
    }
    with open(os.path.join(args.save_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Model saved to: {args.save_dir}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-VL on medical VQA with LoRA"
    )

    # Model
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to training parquet file")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Output directory for the merged model")

    # Training
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps (1 = no accumulation)")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing (uses more VRAM)")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--include_merger", action="store_true",
                        help="Include VL merger/projector in LoRA (default: language only)")

    args = parser.parse_args()
    main(args)
