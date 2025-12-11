# MedForget

This repository includes code for the paper: [*MedForget: Hierarchy-Aware Multimodal Unlearning Testbed for Medical AI*](https://arxiv.org/) by [Fengli Wu](https://fengliwu.com/), [Vaidehi Patil](https://vaidehi99.github.io/), [Jaehong Yoon](https://jaehong31.github.io/), [Yue Zhang](https://zhangyuejoslin.github.io/), and [Mohit Bansal](https://www.cs.unc.edu/~mbansal/)

![image](./asset/teaser.png)

## Abstract
Pretrained Multimodal Large Language Models (MLLMs) are increasingly deployed in medical AI systems for clinical reasoning, diagnosis support, and report generation. However, their training on sensitive patient data raises critical privacy and compliance challenges under regulations such as HIPAA and GDPR, which enforce the "right to be forgotten." Unlearning, the process of tuning models to selectively remove the influence of specific training data points, offers a potential solution, yet its effectiveness in complex medical settings remains underexplored. To systematically study this, we introduce MedForget, a Hierarchy-Aware Multimodal Unlearning Testbed with explicit retain and forget splits and evaluation sets containing rephrased variants. MedForget models hospital data as a nested hierarchy (Institution → Patient → Study → Section), enabling fine-grained assessment across eight organizational levels. The benchmark contains 3840 multimodal (image, question, answer) instances, each hierarchy level having a dedicated unlearning target, reflecting distinct unlearning challenges. Experiments with four SOTA unlearning methods on three tasks (generation, classification, cloze) show that existing methods struggle to achieve complete, hierarchy-aware forgetting without reducing diagnostic performance. To test whether unlearning truly deletes hierarchical pathways, we introduce a reconstruction attack that progressively adds hierarchical level context to prompts. Models unlearned at a coarse granularity show strong resistance, while fine-grained unlearning leaves models vulnerable to such reconstruction. MedForget provides a practical, HIPAA-aligned testbed for building compliant medical AI systems.

## Requirements

```bash
pip install torch transformers rouge-score openai peft pandas pillow python-dotenv qwen-vl-utils
```

Set LLM API key:
```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```bash
# Evaluate both forget and retain sets
./run_eval.sh --model /path/to/model --level patient_level --dataset both
```

### With LoRA Adapter

```bash
./run_eval.sh \
  --model /path/to/adapter \
  --base-model /path/to/base/model \
  --level study_level \
  --batch-size 32
```

### Direct Python Script

```bash
python eval.py \
  --model_path /path/to/model \
  --data_path /path/to/data.parquet \
  --output_dir results \
  --inference_batch_size 8
```

## Key Arguments

- `--model`: Path to model or adapter checkpoint
- `--base-model`: Path to base model (required for adapters)
- `--level`: Hierarchy level (patient_level, study_level, etc.)
- `--dataset`: Dataset type (forget, retain, both)
- `--batch-size`: Inference batch size (default: 64)
- `--samples`: Number of samples to evaluate (default: 1200)

## Output

Results are saved in `eval_results_LEVEL_TIMESTAMP/`:
- `detailed_results.json`: Per-sample scores (ROUGE-L, factuality, total)
- `evaluation_summary.json`: Statistical summary with mean, std, and percentiles

## Metrics

- **ROUGE-L**: Lexical overlap with ground truth
- **Factuality**: LLM-based score (1-10) assessing medical accuracy
- **Total Score**: 0.25 × ROUGE-L + 0.75 × (Factuality/10)

<!-- ## Citation

```bibtex
@inproceedings{your-paper-2026,
  title={Your Paper Title},
  author={Your Name},
  booktitle={Conference Name},
  year={2026}
}
``` -->