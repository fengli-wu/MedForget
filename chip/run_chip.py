"""
Entry point for running CHIP unlearning.

Usage:
    python -m chip.run_chip \
        --model_id lingshu-medical-mllm/Lingshu-7B \
        --vanilla_dir /path/to/finetuned_model \
        --forget_file /path/to/forget_set.parquet \
        --retain_file /path/to/retain_set.parquet \
        --target_level patient \
        --save_dir /path/to/output
"""

import argparse

import pandas as pd

from chip.chip import CHIP, build_hierarchy_from_parquet
from utils.model_utils import load_model_and_processor


def main():
    parser = argparse.ArgumentParser(
        description="CHIP: Cross-modal Hierarchy-Informed Projection for Unlearning"
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace model ID")
    parser.add_argument("--vanilla_dir", type=str, required=True,
                        help="Path to fine-tuned (pre-unlearning) model checkpoint")
    parser.add_argument("--model_type", type=str, default="Lingshu",
                        choices=["Llava", "Lingshu"])
    parser.add_argument("--forget_file", type=str, required=True)
    parser.add_argument("--retain_file", type=str, required=True)
    parser.add_argument("--target_level", type=str, default="institution",
                        choices=["institution", "patient", "study", "section"])
    parser.add_argument("--top_k_percent", type=float, default=10.0,
                        help="Percentage of neurons to select (k)")
    parser.add_argument("--variance_threshold", type=float, default=0.95,
                        help="SVD variance threshold (tau)")
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Vision weight in language layer activations (Eq. 1)")
    parser.add_argument("--vision_text_separation", action="store_true", default=True)
    parser.add_argument("--no_vision_text_separation", action="store_false",
                        dest="vision_text_separation")
    parser.add_argument("--lang_layers", type=int, nargs="+",
                        default=[22, 23, 24, 25, 26, 27])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples_per_node", type=int, default=500)
    parser.add_argument("--max_targets", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="./chip_output")
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, processor = load_model_and_processor(
        args.model_id, args.vanilla_dir, args.model_type)

    # Build hierarchy graphs
    print("Building hierarchy graphs...")
    forget_df = pd.read_parquet(args.forget_file)
    retain_df = pd.read_parquet(args.retain_file)
    print(f"  Forget: {len(forget_df)} samples, Retain: {len(retain_df)} samples")

    fg = build_hierarchy_from_parquet(forget_df)
    rg = build_hierarchy_from_parquet(retain_df)
    for lv in ["institution", "patient", "study", "section"]:
        print(f"  {lv}: forget={len(fg.level_to_nodes.get(lv, []))}, "
              f"retain={len(rg.level_to_nodes.get(lv, []))}")

    # Run CHIP
    chip = CHIP(model, processor, args.model_type,
                top_k_percent=args.top_k_percent,
                variance_threshold=args.variance_threshold,
                alpha=args.alpha,
                vision_text_separation=args.vision_text_separation,
                lang_layers=args.lang_layers)

    chip.run(fg, rg, level=args.target_level, batch_size=args.batch_size,
             max_samples_per_node=args.max_samples_per_node,
             max_targets=args.max_targets)

    chip.save(args.save_dir)


if __name__ == "__main__":
    main()
