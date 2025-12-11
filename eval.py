"""
Evaluation script for machine unlearning models
Computes ROUGE-L and LLM-based factuality scores

This script evaluates multimodal language models by:
1. Generating answers for VQA samples using batch inference
2. Computing ROUGE-L scores against ground truth (enables accuracy calculation for 
   classification/cloze tasks where rouge_l == 1.0 indicates exact match)
3. Evaluating factuality using an LLM-as-a-judge approach

For classification and cloze tasks, accuracy is computed as the proportion of 
samples with rouge_l == 1.0. For generation tasks, ROUGE-L mean and factuality 
mean are the primary metrics.
"""

import gc
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from io import BytesIO
import pandas as pd
from tqdm import tqdm
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Metrics
from rouge_score import rouge_scorer

# Model loading
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel

# LLM API for factuality evaluation
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def clear_cuda_cache():
    """Clear CUDA cache and run garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class LLMEvaluator:
    """Evaluator using LLM API for factuality scoring
    
    Currently configured for DeepSeek API, but can be adapted for other LLM providers.
    """

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 5):
        """Initialize LLM evaluator

        Args:
            api_key: API key for LLM service (reads from DEEPSEEK_API_KEY env var if None)
            max_retries: Maximum number of retry attempts for API calls
        """
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not found in environment variables")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.max_retries = max_retries

    def build_factuality_prompt(self, question: str, generated: str, ground_truth: str) -> str:
        """Build factuality evaluation prompt optimized for medical VQA
        
        Args:
            question: The input question
            generated: Model-generated answer
            ground_truth: Reference answer
            
        Returns:
            Evaluation prompt string
        """
        return f"""You will evaluate the **factuality** of the "generated_answer" against the "ground_truth" for medical image analysis questions.

Your task: Assess how well the generated response captures the KEY MEDICAL INFORMATION from the ground truth and assign a **factuality score (1-10)**.

IMPORTANT EVALUATION PRINCIPLES:
1. **Medical Terminology Accuracy**: Prioritize correct medical terms
2. **Core Clinical Content**: Focus on anatomical structures, findings, and diagnostic information
3. **Partial Credit**: Give credit for capturing key medical concepts even if incomplete
4. **Context Matters**: Consider that some details may be less critical than core findings

Scoring rubric:
- 10-9: Fully correct; all key medical terms and findings present
- 8-7: Core medical information correct; minor omissions or slight incompleteness
- 6-5: Key medical terms present but missing important context OR partially correct
- 4-3: Contains some relevant medical terminology but misses critical findings
- 2-1: Mostly incorrect, irrelevant, or only trivial overlap

Task Context:
- Question: {question}
- Generated Answer: {generated}
- Ground Truth: {ground_truth}

Response format (JSON):
{{"score": <1-10>, "reasoning": "<brief explanation>"}}"""

    def evaluate_factuality(self, question: str, generated: str, ground_truth: str) -> Dict:
        """Evaluate factuality of generated answer using LLM API

        Args:
            question: The input question
            generated: Model-generated answer
            ground_truth: Reference answer

        Returns:
            Dictionary with 'score', 'reasoning', and 'success' keys
        """
        prompt = self.build_factuality_prompt(question, generated, ground_truth)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256,
                )

                result_text = response.choices[0].message.content.strip()

                # Parse JSON response
                try:
                    result = json.loads(result_text)
                    return {
                        "score": float(result.get("score", 0)),
                        "reasoning": result.get("reasoning", ""),
                        "success": True
                    }
                except json.JSONDecodeError:
                    # Fallback: extract score using regex
                    import re
                    score_match = re.search(r'"score":\s*(\d+(?:\.\d+)?)', result_text)
                    if score_match:
                        return {
                            "score": float(score_match.group(1)),
                            "reasoning": result_text,
                            "success": True
                        }
                    else:
                        print(f"Failed to parse LLM response: {result_text[:200]}")

            except Exception as e:
                error_msg = str(e)
                print(f"LLM API error (attempt {attempt + 1}/{self.max_retries}): {error_msg}")

                # Handle rate limiting
                if "rate" in error_msg.lower() or "429" in error_msg:
                    wait_time = min(60, 5 * (2 ** attempt))
                    print(f"Rate limited. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                # Handle connection errors
                elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                    wait_time = 2 ** attempt
                    print(f"Connection issue. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                # Other errors
                elif attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)

        return {"score": 0, "reasoning": "Failed to evaluate after retries", "success": False}


def load_model_and_processor(model_path: str, base_model_path: Optional[str] = None):
    """Load model and processor, handling both full models and LoRA adapters
    
    Args:
        model_path: Path to model or adapter checkpoint
        base_model_path: Path to base model (required for adapter checkpoints)
    
    Returns:
        Tuple of (model, processor)
    """
    print(f"\nLoading model from: {model_path}")
    
    # Check if this is a LoRA adapter checkpoint
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_adapter_checkpoint = os.path.exists(adapter_config_path)
    
    if is_adapter_checkpoint:
        print("Detected LoRA adapter checkpoint")
        
        if base_model_path is None:
            raise ValueError(
                "This is a LoRA adapter checkpoint but no base_model_path provided. "
                "Please specify --base_model_path."
            )
        
        print(f"Loading base model from: {base_model_path}")
        
        # Load base model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        
        # Load processor
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True,
            )
            print("Loaded processor from adapter checkpoint")
        except Exception as e:
            print(f"Warning: Could not load processor from checkpoint: {e}")
            print("Loading processor from base model...")
            processor = AutoProcessor.from_pretrained(
                base_model_path,
                local_files_only=True,
            )
        
        # Resize embeddings to match processor vocab size
        print(f"Resizing token embeddings to {len(processor.tokenizer)} tokens...")
        model.resize_token_embeddings(len(processor.tokenizer))
        
        # Load LoRA adapters
        print(f"Loading LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        print("LoRA adapters loaded")
        
        # Merge LoRA weights for faster inference
        print("Merging LoRA weights into base model...")
        model = model.merge_and_unload()
        print("LoRA weights merged - inference speed optimized")
        
    else:
        print("Loading complete model")
        
        # Load complete model
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        
        # Load processor
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                local_files_only=True,
            )
        except (OSError, Exception) as e:
            print(f"Warning: Could not load processor from {model_path}")
            print("Loading processor from original model...")
            processor = AutoProcessor.from_pretrained("lingshu-medical-mllm/Lingshu-7B")
    
    processor.tokenizer.padding_side = "left"
    print("Model and processor loaded successfully")
    
    return model, processor


def generate_answer_batch(model, processor, images, questions, max_new_tokens=128, batch_size=4):
    """Generate answers for a batch of image-question pairs
    
    Args:
        model: Vision-language model
        processor: Model processor
        images: List of PIL images
        questions: List of question strings
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for inference
        
    Returns:
        List of generated answer strings
    """
    all_outputs = []
    
    for i in tqdm(range(0, len(images), batch_size), desc="Batch inference"):
        batch_images = images[i:i+batch_size]
        batch_questions = questions[i:i+batch_size]
        
        # Prepare batch messages
        batch_messages = []
        for img, q in zip(batch_images, batch_questions):
            batch_messages.append([
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": q},
                    ],
                }
            ])
        
        # Process batch
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in batch_messages
        ]
        
        # Collect image inputs
        image_inputs_list = []
        for msg in batch_messages:
            img_inputs, _ = process_vision_info(msg)
            image_inputs_list.extend(img_inputs if img_inputs else [])
        
        # Batch tokenization
        inputs = processor(
            text=texts,
            images=image_inputs_list if image_inputs_list else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,  # Explicitly set to None for greedy decoding
                top_p=None,
                top_k=None,
            )
        
        # Decode outputs
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        all_outputs.extend(outputs)
        
        # Periodic cache cleanup
        if i % 10 == 0:
            del inputs, generated_ids, generated_ids_trimmed
            clear_cuda_cache()
    
    return all_outputs


def compute_rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between prediction and reference"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores['rougeL'].fmeasure


def load_data_samples(data_path: str, max_samples: int = 1000) -> List[Dict]:
    """Load VQA samples from parquet file
    
    Args:
        data_path: Path to parquet file
        max_samples: Maximum number of samples to load
        
    Returns:
        List of sample dictionaries
    """
    print(f"\nLoading samples from: {data_path}")

    df = pd.read_parquet(data_path)
    samples = []

    for idx, row in df.iterrows():
        if len(samples) >= max_samples:
            break

        image_data = row['image']['bytes']
        image = Image.open(BytesIO(image_data)).convert("RGB")
        metadata = json.loads(row['metadata'])

        for qa_idx, qa in enumerate(metadata):
            if len(samples) >= max_samples:
                break

            # Handle Answer field (may be list or string)
            answer = qa.get('Answer', '')
            if isinstance(answer, list):
                answer = ' '.join(str(a) for a in answer if a)
            elif not isinstance(answer, str):
                answer = str(answer) if answer else ''

            samples.append({
                'sample_id': f"{idx}_{qa_idx}",
                'image': image,
                'question': qa.get('Question', ''),
                'ground_truth': answer,
            })

    print(f"Loaded {len(samples)} samples")
    return samples


def evaluate_sample_batch(
    samples: List[Dict],
    model,
    processor,
    evaluator: LLMEvaluator,
    max_new_tokens: int = 128,
    max_workers: int = 10,
    inference_batch_size: int = 4
) -> List[Dict]:
    """Evaluate a batch of samples with batch inference and parallel LLM scoring

    Args:
        samples: List of sample dictionaries
        model: Vision-language model
        processor: Model processor
        evaluator: LLM evaluator for factuality scoring
        max_new_tokens: Maximum tokens to generate
        max_workers: Maximum parallel workers for LLM evaluation
        inference_batch_size: Batch size for model inference

    Returns:
        List of evaluation results
    """
    results = []

    # Step 1: Batch generation
    print("\nGenerating answers with batch inference...")
    images = [s['image'] for s in samples]
    questions = [s['question'] for s in samples]
    
    generated_answers = generate_answer_batch(
        model, processor, images, questions, 
        max_new_tokens, inference_batch_size
    )
    
    # Step 2: Compute ROUGE-L
    print("\nComputing ROUGE-L scores...")
    for sample, generated in tqdm(zip(samples, generated_answers), 
                                  total=len(samples), 
                                  desc="Computing ROUGE"):
        sample['generated'] = generated
        sample['rouge_l'] = compute_rouge_l(generated, sample['ground_truth'])

    # Step 3: Parallel factuality evaluation
    print("\nEvaluating factuality with LLM...")

    def evaluate_single(sample):
        """Evaluate factuality for a single sample"""
        result = evaluator.evaluate_factuality(
            sample['question'],
            sample['generated'],
            sample['ground_truth']
        )
        return {
            'sample_id': sample['sample_id'],
            'question': sample['question'],
            'ground_truth': sample['ground_truth'],
            'generated': sample['generated'],
            'rouge_l': sample['rouge_l'],
            'factuality_score': result['score'],
            'factuality_reasoning': result['reasoning'],
            'evaluation_success': result['success']
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_single, sample): sample
                   for sample in samples}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Factuality"):
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                sample = futures[future]
                print(f"Error evaluating sample {sample['sample_id']}: {e}")
                results.append({
                    'sample_id': sample['sample_id'],
                    'question': sample['question'],
                    'ground_truth': sample['ground_truth'],
                    'generated': sample['generated'],
                    'rouge_l': sample['rouge_l'],
                    'factuality_score': 0,
                    'factuality_reasoning': f"Evaluation failed: {str(e)}",
                    'evaluation_success': False
                })

    return results


def save_results(results: List[Dict], output_dir: str):
    """Save evaluation results and compute summary statistics
    
    The total score is computed as: 0.25 * ROUGE-L + 0.75 * (factuality/10)
    
    Args:
        results: List of evaluation result dictionaries
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate total score for each sample
    for r in results:
        factuality_norm = r['factuality_score'] / 10.0 if r['evaluation_success'] else 0.0
        r['total_score'] = 0.25 * r['rouge_l'] + 0.75 * factuality_norm

    # Save detailed results
    detailed_path = output_path / "detailed_results.json"
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {detailed_path}")

    # Compute summary statistics
    successful_results = [r for r in results if r['evaluation_success']]
    all_total_scores = [r['total_score'] for r in results]

    summary = {
        'total_samples': len(results),
        'successful_evaluations': len(successful_results),
        'failed_evaluations': len(results) - len(successful_results),
        'avg_rouge_l': float(np.mean([r['rouge_l'] for r in results])),
        'std_rouge_l': float(np.std([r['rouge_l'] for r in results])),
        'avg_factuality': float(np.mean([r['factuality_score'] for r in successful_results])) if successful_results else 0.0,
        'std_factuality': float(np.std([r['factuality_score'] for r in successful_results])) if successful_results else 0.0,
        'avg_total_score': float(np.mean(all_total_scores)),
        'std_total_score': float(np.std(all_total_scores)),
        'timestamp': datetime.now().isoformat(),
    }

    # Compute percentiles
    rouge_scores = [r['rouge_l'] for r in results]
    summary['rouge_percentiles'] = {
        '25th': float(np.percentile(rouge_scores, 25)),
        '50th': float(np.percentile(rouge_scores, 50)),
        '75th': float(np.percentile(rouge_scores, 75)),
        '90th': float(np.percentile(rouge_scores, 90)),
    }

    if successful_results:
        factuality_scores = [r['factuality_score'] for r in successful_results]
        summary['factuality_percentiles'] = {
            '25th': float(np.percentile(factuality_scores, 25)),
            '50th': float(np.percentile(factuality_scores, 50)),
            '75th': float(np.percentile(factuality_scores, 75)),
            '90th': float(np.percentile(factuality_scores, 90)),
        }

    summary['total_score_percentiles'] = {
        '25th': float(np.percentile(all_total_scores, 25)),
        '50th': float(np.percentile(all_total_scores, 50)),
        '75th': float(np.percentile(all_total_scores, 75)),
        '90th': float(np.percentile(all_total_scores, 90)),
    }

    # Save summary
    summary_path = output_path / "evaluation_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_path}")

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Total Samples          : {summary['total_samples']}")
    print(f"Successful Evaluations : {summary['successful_evaluations']}")
    print(f"\nROUGE-L Scores:")
    print(f"  Mean ± Std : {summary['avg_rouge_l']:.4f} ± {summary['std_rouge_l']:.4f}")
    print(f"  Median     : {summary['rouge_percentiles']['50th']:.4f}")
    print(f"\nFactuality Scores (1-10):")
    if successful_results:
        print(f"  Mean ± Std : {summary['avg_factuality']:.2f} ± {summary['std_factuality']:.2f}")
        print(f"  Median     : {summary['factuality_percentiles']['50th']:.2f}")
    print(f"\nTotal Score (25% ROUGE-L + 75% Factuality/10):")
    print(f"  Mean ± Std : {summary['avg_total_score']:.4f} ± {summary['std_total_score']:.4f}")
    print(f"  Median     : {summary['total_score_percentiles']['50th']:.4f}")
    print(f"  25th-75th  : {summary['total_score_percentiles']['25th']:.4f} - {summary['total_score_percentiles']['75th']:.4f}")
    print("="*70)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate machine unlearning models with ROUGE-L and LLM-based factuality'
    )
    parser.add_argument('--model_path', required=True, 
                        help='Path to model or adapter checkpoint')
    parser.add_argument('--base_model_path', default=None,
                        help='Path to base model (required for adapter checkpoints)')
    parser.add_argument('--data_path', required=True, 
                        help='Path to evaluation data (parquet format)')
    parser.add_argument('--output_dir', default='eval_results', 
                        help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=500, 
                        help='Maximum number of samples to evaluate')
    parser.add_argument('--max_new_tokens', type=int, default=128, 
                        help='Maximum tokens for generation')
    parser.add_argument('--max_workers', type=int, default=10, 
                        help='Maximum parallel workers for LLM evaluation')
    parser.add_argument('--inference_batch_size', type=int, default=4, 
                        help='Batch size for model inference')
    parser.add_argument('--deepseek_api_key', 
                        help='LLM API key (optional, reads from environment if not provided)')

    args = parser.parse_args()

    # Initialize evaluator
    print("\nInitializing LLM evaluator...")
    evaluator = LLMEvaluator(api_key=args.deepseek_api_key)

    # Load model
    model, processor = load_model_and_processor(args.model_path, args.base_model_path)

    # Load data
    samples = load_data_samples(args.data_path, args.max_samples)

    # Evaluate
    print(f"\nStarting evaluation of {len(samples)} samples...")
    print(f"Inference batch size: {args.inference_batch_size}")
    start_time = time.time()

    results = evaluate_sample_batch(
        samples, model, processor, evaluator,
        max_new_tokens=args.max_new_tokens,
        max_workers=args.max_workers,
        inference_batch_size=args.inference_batch_size
    )

    # Save results
    summary = save_results(results, args.output_dir)

    elapsed_time = time.time() - start_time
    print(f"\nEvaluation completed in {elapsed_time:.2f} seconds")
    print(f"Average time per sample: {elapsed_time/len(samples):.2f} seconds")

    return summary


if __name__ == "__main__":
    main()