"""
Task Type Analysis for Evaluation Results

Analyzes evaluation results (from eval.py) by task type:
- Classification: Accuracy computed from ROUGE-L (rouge_l == 1.0 means correct answer)
- Cloze: Accuracy computed from ROUGE-L (rouge_l == 1.0 means correct answer)
- Generation: ROUGE-L mean, Factuality mean, and Total Score mean

The script takes detailed_results.json files produced by eval.py and computes
task-specific metrics by classifying questions based on their format.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from collections import defaultdict


class TaskTypeClassifier:
    """Classifier to identify task types from questions"""

    @staticmethod
    def is_classification(question: str) -> bool:
        """Check if question is a classification task (contains A), B), C))"""
        has_a = "A)" in question
        has_b = "B)" in question
        has_c = "C)" in question
        return has_a and has_b and has_c

    @staticmethod
    def is_cloze(question: str) -> bool:
        """Check if question is a cloze task (contains [blank])"""
        return "[blank]" in question

    @staticmethod
    def classify(question: str) -> str:
        """Classify question into task type"""
        if TaskTypeClassifier.is_cloze(question):
            return "cloze"
        elif TaskTypeClassifier.is_classification(question):
            return "classification"
        else:
            return "generation"


class EvaluationAnalyzer:
    """Analyzer for evaluation results"""

    def __init__(self, results_dir: str):
        """Initialize analyzer with results directory

        Args:
            results_dir: Path to directory containing forget/ and retain/ subdirectories
        """
        self.results_dir = Path(results_dir)
        self.forget_data = None
        self.retain_data = None

    def load_data(self):
        """Load detailed_results.json from forget and retain directories"""
        forget_path = self.results_dir / "forget" / "detailed_results.json"
        retain_path = self.results_dir / "retain" / "detailed_results.json"

        if not forget_path.exists():
            raise FileNotFoundError(f"Forget results not found: {forget_path}")

        with open(forget_path, 'r') as f:
            self.forget_data = json.load(f)

        if retain_path.exists():
            with open(retain_path, 'r') as f:
                self.retain_data = json.load(f)
            print(f"Loaded {len(self.forget_data)} forget samples")
            print(f"Loaded {len(self.retain_data)} retain samples")
        else:
            self.retain_data = None
            print(f"Loaded {len(self.forget_data)} forget samples")
            print("Warning: Retain data not found, skipping comparison")

    def classify_samples(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Classify samples by task type

        Returns:
            Dict mapping task type to list of samples
        """
        classified = defaultdict(list)

        for sample in samples:
            task_type = TaskTypeClassifier.classify(sample['question'])
            classified[task_type].append(sample)

        return dict(classified)

    def compute_accuracy_metrics(self, samples: List[Dict]) -> Dict:
        """Compute accuracy metrics for classification/cloze tasks

        Args:
            samples: List of samples with rouge_l scores

        Returns:
            Dict with accuracy metrics
        """
        total = len(samples)
        if total == 0:
            return {
                "total_samples": 0,
                "correct": 0,
                "incorrect": 0,
                "accuracy": 0.0,
                "failed_evaluations": 0
            }

        correct = sum(1 for s in samples if s.get('rouge_l', 0) == 1.0)
        failed = sum(1 for s in samples if not s.get('evaluation_success', True))

        return {
            "total_samples": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": correct / total if total > 0 else 0.0,
            "failed_evaluations": failed
        }

    def compute_generation_metrics(self, samples: List[Dict]) -> Dict:
        """Compute ROUGE-L and Factuality metrics for generation tasks

        Args:
            samples: List of samples with rouge_l and factuality_score

        Returns:
            Dict with generation metrics
        """
        total = len(samples)
        if total == 0:
            return {
                "total_samples": 0,
                "rouge_l_mean": 0.0,
                "factuality_mean": 0.0,
                "total_score_mean": 0.0,
                "successful_samples": 0,
                "failed_evaluations": 0
            }

        rouge_scores = [s.get('rouge_l', 0) for s in samples]
        factuality_scores = [s.get('factuality_score', 0) for s in samples
                            if s.get('evaluation_success', True)]
        failed = sum(1 for s in samples if not s.get('evaluation_success', True))

        rouge_mean = float(np.mean(rouge_scores))

        if factuality_scores:
            factuality_mean = float(np.mean(factuality_scores))
            successful_samples = len(factuality_scores)
        else:
            factuality_mean = 0.0
            successful_samples = 0

        # Compute total_score = 0.25 * rouge_l + 0.75 * (factuality / 10)
        total_scores = []
        for s in samples:
            if s.get('evaluation_success', True):
                rouge = s.get('rouge_l', 0)
                fact = s.get('factuality_score', 0)
                total_score = 0.25 * rouge + 0.75 * (fact / 10.0)
                total_scores.append(total_score)

        total_score_mean = float(np.mean(total_scores)) if total_scores else 0.0

        return {
            "total_samples": total,
            "rouge_l_mean": rouge_mean,
            "factuality_mean": factuality_mean,
            "total_score_mean": total_score_mean,
            "successful_samples": successful_samples,
            "failed_evaluations": failed
        }

    def analyze_dataset(self, samples: List[Dict], dataset_name: str) -> Dict:
        """Analyze a dataset (forget or retain)

        Args:
            samples: List of evaluation samples
            dataset_name: Name of dataset (forget/retain)

        Returns:
            Dict with analysis results ordered by task type
        """
        classified = self.classify_samples(samples)

        results = {
            "dataset": dataset_name,
            "total_samples": len(samples),
            "task_types": {}
        }

        # Order: generation, classification, cloze
        for task_type in ["generation", "classification", "cloze"]:
            task_samples = classified.get(task_type, [])
            task_count = len(task_samples)
            task_percent = (task_count / len(samples) * 100) if len(samples) > 0 else 0.0

            if task_type in ["classification", "cloze"]:
                metrics = self.compute_accuracy_metrics(task_samples)
            else:
                metrics = self.compute_generation_metrics(task_samples)

            results["task_types"][task_type] = {
                "count": task_count,
                "percentage": task_percent,
                "metrics": metrics
            }

        return results

    def compare_datasets(self, forget_results: Dict, retain_results: Dict) -> Dict:
        """Compare forget and retain datasets

        Args:
            forget_results: Analysis results for forget set
            retain_results: Analysis results for retain set

        Returns:
            Dict with comparative analysis
        """
        comparison = {
            "task_type_distribution": {},
            "performance_comparison": {}
        }

        for task_type in ["generation", "classification", "cloze"]:
            forget_count = forget_results["task_types"][task_type]["count"]
            retain_count = retain_results["task_types"][task_type]["count"]

            comparison["task_type_distribution"][task_type] = {
                "forget_count": forget_count,
                "retain_count": retain_count,
                "forget_percentage": forget_results["task_types"][task_type]["percentage"],
                "retain_percentage": retain_results["task_types"][task_type]["percentage"]
            }

        for task_type in ["generation", "classification", "cloze"]:
            forget_metrics = forget_results["task_types"][task_type]["metrics"]
            retain_metrics = retain_results["task_types"][task_type]["metrics"]

            if task_type in ["classification", "cloze"]:
                forget_acc = forget_metrics.get("accuracy", 0.0)
                retain_acc = retain_metrics.get("accuracy", 0.0)

                comparison["performance_comparison"][task_type] = {
                    "forget_accuracy": forget_acc,
                    "retain_accuracy": retain_acc,
                    "accuracy_difference": retain_acc - forget_acc,
                    "forget_correct": forget_metrics.get("correct", 0),
                    "retain_correct": retain_metrics.get("correct", 0)
                }
            else:  # generation
                forget_rouge = forget_metrics.get("rouge_l_mean", 0.0)
                retain_rouge = retain_metrics.get("rouge_l_mean", 0.0)
                forget_fact = forget_metrics.get("factuality_mean", 0.0)
                retain_fact = retain_metrics.get("factuality_mean", 0.0)
                forget_total = forget_metrics.get("total_score_mean", 0.0)
                retain_total = retain_metrics.get("total_score_mean", 0.0)

                comparison["performance_comparison"][task_type] = {
                    "rouge_l_mean": {
                        "forget": forget_rouge,
                        "retain": retain_rouge,
                        "difference": retain_rouge - forget_rouge
                    },
                    "factuality_mean": {
                        "forget": forget_fact,
                        "retain": retain_fact,
                        "difference": retain_fact - forget_fact
                    },
                    "total_score_mean": {
                        "forget": forget_total,
                        "retain": retain_total,
                        "difference": retain_total - forget_total
                    }
                }

        return comparison

    def print_results(self, forget_results: Dict, retain_results: Dict, comparison: Dict):
        """Print analysis results"""
        print("\n" + "="*80)
        print("TASK TYPE ANALYSIS")
        print("="*80)

        if comparison:
            print("\nCOMPARATIVE ANALYSIS (Retain vs Forget)")
            print("-"*80)

            perf_comp = comparison["performance_comparison"]

            for task_type in ["generation", "classification", "cloze"]:
                print(f"\n{task_type.upper()}:")
                if task_type in ["classification", "cloze"]:
                    comp = perf_comp[task_type]
                    forget_acc = comp["forget_accuracy"]
                    retain_acc = comp["retain_accuracy"]
                    diff = comp["accuracy_difference"]

                    print(f"  Forget Accuracy: {forget_acc:.4f}")
                    print(f"  Retain Accuracy: {retain_acc:.4f}")
                    print(f"  Δ Accuracy: {diff:+.4f}", end="")
                    if diff > 0:
                        print(" (Better retention)")
                    elif diff < 0:
                        print(" (Better forgetting)")
                    else:
                        print()
                else:  # generation
                    comp = perf_comp[task_type]

                    total_comp = comp["total_score_mean"]
                    print(f"  Total Score:")
                    print(f"    Forget: {total_comp['forget']:.4f}")
                    print(f"    Retain: {total_comp['retain']:.4f}")
                    print(f"    Δ: {total_comp['difference']:+.4f}")

                    rouge_comp = comp["rouge_l_mean"]
                    print(f"  ROUGE-L Mean:")
                    print(f"    Forget: {rouge_comp['forget']:.4f}")
                    print(f"    Retain: {rouge_comp['retain']:.4f}")
                    print(f"    Δ: {rouge_comp['difference']:+.4f}")

                    fact_comp = comp["factuality_mean"]
                    if fact_comp['forget'] > 0 or fact_comp['retain'] > 0:
                        print(f"  Factuality Mean:")
                        print(f"    Forget: {fact_comp['forget']:.2f}")
                        print(f"    Retain: {fact_comp['retain']:.2f}")
                        print(f"    Δ: {fact_comp['difference']:+.2f}")

        # Print detailed results for each dataset
        for results in [forget_results, retain_results]:
            if results is None:
                continue
                
            dataset_name = results["dataset"].upper()
            print(f"\n{dataset_name} SET BREAKDOWN ({results['total_samples']} samples)")
            print("-"*80)

            for task_type in ["generation", "classification", "cloze"]:
                task_data = results["task_types"][task_type]
                count = task_data["count"]
                pct = task_data["percentage"]
                metrics = task_data["metrics"]

                print(f"\n{task_type.upper()}: {count} samples ({pct:.1f}%)")

                if task_type in ["classification", "cloze"]:
                    acc = metrics["accuracy"]
                    correct = metrics["correct"]
                    print(f"  Accuracy: {acc:.4f} ({correct}/{count} correct)")
                else:  # generation
                    total_score = metrics.get("total_score_mean", 0.0)
                    rouge_mean = metrics.get("rouge_l_mean", 0.0)
                    fact_mean = metrics.get("factuality_mean", 0.0)
                    
                    print(f"  Total Score: {total_score:.4f}")
                    print(f"  ROUGE-L Mean: {rouge_mean:.4f}")
                    if metrics.get("successful_samples", 0) > 0:
                        print(f"  Factuality Mean: {fact_mean:.2f}")

        print("\n" + "="*80 + "\n")

    def save_results(self, forget_results: Dict, retain_results: Dict, comparison: Dict):
        """Save analysis results to JSON file"""
        output = {
            "comparison": comparison,
            "forget_set": forget_results,
            "retain_set": retain_results
        }

        output_path = self.results_dir / "task_type_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Analysis saved to: {output_path}")

    def run_analysis(self):
        """Run complete analysis"""
        print("\nStarting task type analysis...")

        self.load_data()

        print("\nAnalyzing forget set...")
        forget_results = self.analyze_dataset(self.forget_data, "forget")

        if self.retain_data:
            print("Analyzing retain set...")
            retain_results = self.analyze_dataset(self.retain_data, "retain")
            
            print("Computing comparative analysis...")
            comparison = self.compare_datasets(forget_results, retain_results)
        else:
            retain_results = None
            comparison = None

        self.print_results(forget_results, retain_results, comparison)
        self.save_results(forget_results, retain_results, comparison)

        print("Analysis completed\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze evaluation results by task type'
    )

    parser.add_argument(
        '--results_dir', '-r',
        required=True,
        help='Path to results directory containing forget/ and retain/ subdirectories'
    )

    args = parser.parse_args()

    analyzer = EvaluationAnalyzer(args.results_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
