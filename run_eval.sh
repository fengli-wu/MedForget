#!/bin/bash

# ==============================================================================
# Evaluation Script for Machine Unlearning Models
# Computes ROUGE-L and LLM-based factuality scores
# ==============================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "================================================"
echo "  Machine Unlearning Model Evaluation"
echo "  ROUGE-L + LLM-based Factuality Assessment"
echo "================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

################################################################################
# Configuration
################################################################################

# Default data configuration
UNLEARNING_BASE="${SCRIPT_DIR}/../unlearning/generated_vqa_dataset_20251112_124819"
FILTER="all"
MODEL_PATH="/root/autodl-tmp/vqa_mimic_format_20251128_153105"
BASE_MODEL_PATH="/root/autodl-tmp/vqa_mimic_format_20251128_153105"

# Evaluation parameters
MAX_SAMPLES=1200
MAX_NEW_TOKENS=128
MAX_WORKERS=100
INFERENCE_BATCH_SIZE=64

# Model and data search paths
MODEL_SEARCH_PATHS=(
    "./pruned_output"
    "../pruned_output"
    "./saved_model"
    "../saved_model"
)

DATA_SEARCH_PATHS=(
    "$UNLEARNING_BASE"
    "../unlearning_finetune_20251108_190328"
    "./unlearning_finetune_20251108_190328"
    "/root/autodl-tmp/unlearning_finetune_20251108_190328"
)

################################################################################
# Parse arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            MODEL_PATH="$2"
            shift 2
            ;;
        --base-model|-b)
            BASE_MODEL_PATH="$2"
            shift 2
            ;;
        --level|-l)
            LEVEL="$2"
            shift 2
            ;;
        --filter|-f)
            FILTER="$2"
            shift 2
            ;;
        --dataset|-d)
            DATASET_TYPE="$2"
            shift 2
            ;;
        --samples|-n)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --batch-size|--bs)
            INFERENCE_BATCH_SIZE="$2"
            shift 2
            ;;
        --api-key|-k)
            LLM_API_KEY="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model, -m PATH          Path to model or adapter checkpoint"
            echo "  --base-model, -b PATH     Path to base model (for adapter checkpoints)"
            echo "  --level, -l LEVEL         Hierarchy level (patient_level, study_level, etc.)"
            echo "  --filter, -f FILTER       Filter type (default: all)"
            echo "  --dataset, -d TYPE        Dataset type (forget, retain, both)"
            echo "  --samples, -n NUM         Number of samples (default: 1200)"
            echo "  --batch-size, --bs NUM    Inference batch size (default: 64)"
            echo "  --api-key, -k KEY         LLM API key for factuality evaluation"
            echo "  --help, -h                Show this help"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            exit 1
            ;;
    esac
done

# Set defaults
DATASET_TYPE="${DATASET_TYPE:-both}"

################################################################################
# Helper functions
################################################################################

check_if_adapter_checkpoint() {
    local model_path="$1"
    [ -f "$model_path/adapter_config.json" ]
}

find_pruned_model() {
    for dir in "${MODEL_SEARCH_PATHS[@]}"; do
        if [ -d "$dir" ]; then
            local model=$(ls -td "$dir"/lingshu_pruned_* 2>/dev/null | head -1)
            if [ -n "$model" ] && [ -d "$model" ]; then
                local iter=$(ls -d "$model"/lingshu-*_iteration_* 2>/dev/null | sort -V | tail -1)
                if [ -n "$iter" ]; then
                    echo "$iter"
                else
                    echo "$model"
                fi
                return 0
            fi
        fi
    done
    return 1
}

find_data_base() {
    for dir in "${DATA_SEARCH_PATHS[@]}"; do
        if [ -d "$dir" ]; then
            echo "$dir"
            return 0
        fi
    done
    return 1
}

run_evaluation() {
    local data_file="$1"
    local dataset_name="$2"
    local output_dir="$3"

    echo ""
    echo "================================================"
    echo "Evaluating ${dataset_name} set"
    echo "================================================"
    echo "Data: $data_file"
    echo "Output: $output_dir"
    echo "Batch size: $INFERENCE_BATCH_SIZE"
    echo ""

    # Set API key if provided
    if [ -n "$LLM_API_KEY" ]; then
        export DEEPSEEK_API_KEY="$LLM_API_KEY"
    fi

    # Build evaluation command
    local eval_cmd="python eval.py \
        --model_path \"$MODEL_PATH\" \
        --data_path \"$data_file\" \
        --output_dir \"$output_dir\" \
        --max_samples $MAX_SAMPLES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --max_workers $MAX_WORKERS \
        --inference_batch_size $INFERENCE_BATCH_SIZE"
    
    if [ -n "$BASE_MODEL_PATH" ]; then
        eval_cmd="$eval_cmd --base_model_path \"$BASE_MODEL_PATH\""
    fi

    # Run evaluation
    eval $eval_cmd

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${dataset_name} evaluation completed${NC}"

        # Display summary
        if [ -f "$output_dir/evaluation_summary.json" ]; then
            echo ""
            echo "Summary for ${dataset_name}:"
            python -c "
import json
with open('$output_dir/evaluation_summary.json', 'r') as f:
    s = json.load(f)
    print(f'  ROUGE-L: {s[\"avg_rouge_l\"]:.4f} ± {s[\"std_rouge_l\"]:.4f}')
    if s.get('avg_factuality', 0) > 0:
        print(f'  Factuality: {s[\"avg_factuality\"]:.2f} ± {s[\"std_factuality\"]:.2f}')
    if s.get('avg_total_score'):
        print(f'  Total Score: {s[\"avg_total_score\"]:.4f} ± {s[\"std_total_score\"]:.4f}')
    print(f'  Total samples: {s[\"total_samples\"]}')
"
        fi
        return 0
    else
        echo -e "${RED}✗ ${dataset_name} evaluation failed${NC}"
        return 1
    fi
}

################################################################################
# Main execution
################################################################################

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}WARNING: GPU not detected${NC}"
else
    echo -e "${GREEN}✓ GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
fi

# Verify API key
if [ -z "$LLM_API_KEY" ] && [ -z "$(printenv DEEPSEEK_API_KEY)" ]; then
    echo -e "${RED}ERROR: LLM API key not set${NC}"
    echo "Please set via --api-key or DEEPSEEK_API_KEY environment variable"
    exit 1
fi

# Find or validate model path
if [ -z "$MODEL_PATH" ]; then
    echo "Searching for model..."
    MODEL_PATH=$(find_pruned_model)
    if [ -z "$MODEL_PATH" ]; then
        echo -e "${RED}ERROR: No model found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Found model: $MODEL_PATH${NC}"
else
    if [ ! -d "$MODEL_PATH" ]; then
        echo -e "${RED}ERROR: Model not found: $MODEL_PATH${NC}"
        exit 1
    fi
fi

# Check for adapter checkpoint and validate base model
if check_if_adapter_checkpoint "$MODEL_PATH"; then
    echo "Detected LoRA adapter checkpoint"
    if [ -z "$BASE_MODEL_PATH" ]; then
        echo -e "${RED}ERROR: Adapter checkpoint requires base model${NC}"
        echo "Please provide --base-model PATH"
        exit 1
    fi
    if [ ! -d "$BASE_MODEL_PATH" ]; then
        echo -e "${RED}ERROR: Base model not found: $BASE_MODEL_PATH${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Base model: $BASE_MODEL_PATH${NC}"
else
    echo "Using complete model (not an adapter)"
    if [ -n "$BASE_MODEL_PATH" ]; then
        BASE_MODEL_PATH=""
    fi
fi

# Find data directory
if [ ! -d "$UNLEARNING_BASE" ]; then
    UNLEARNING_BASE=$(find_data_base)
    if [ -z "$UNLEARNING_BASE" ]; then
        echo -e "${RED}ERROR: Data directory not found${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Found data directory: $UNLEARNING_BASE${NC}"
fi

# Construct data file paths
FORGET_FILE="${UNLEARNING_BASE}/${LEVEL}/forget_set_${FILTER}.parquet"
RETAIN_FILE="${UNLEARNING_BASE}/${LEVEL}/retain_set_${FILTER}.parquet"

# Verify data files exist
if [[ "$DATASET_TYPE" == "forget" || "$DATASET_TYPE" == "both" ]]; then
    if [ ! -f "$FORGET_FILE" ]; then
        echo -e "${RED}ERROR: Forget file not found: $FORGET_FILE${NC}"
        exit 1
    fi
fi

if [[ "$DATASET_TYPE" == "retain" || "$DATASET_TYPE" == "both" ]]; then
    if [ ! -f "$RETAIN_FILE" ]; then
        echo -e "${RED}ERROR: Retain file not found: $RETAIN_FILE${NC}"
        exit 1
    fi
fi

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="eval_results_${LEVEL}_${TIMESTAMP}"
mkdir -p "$OUTPUT_BASE"

# Setup logging
LOG_DIR="eval_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/eval_${LEVEL}_${TIMESTAMP}.log"
exec > >(tee "$LOG_FILE") 2>&1

echo "================================================"
echo "  Evaluation Configuration"
echo "================================================"
echo "Model: $MODEL_PATH"
[ -n "$BASE_MODEL_PATH" ] && echo "Base Model: $BASE_MODEL_PATH"
echo "Level: $LEVEL, Filter: $FILTER"
echo "Dataset: $DATASET_TYPE"
echo "Samples: $MAX_SAMPLES"
echo "Batch size: $INFERENCE_BATCH_SIZE"
echo ""

# Setup Python environment
echo "Setting up Python environment..."
if command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate base
    echo "Python: $(which python)"
    python --version
else
    echo -e "${YELLOW}WARNING: Conda not found, using system Python${NC}"
fi

# Verify required packages
python -c "
import sys
required = ['torch', 'transformers', 'rouge_score', 'openai', 'peft']
missing = []
for pkg in required:
    try:
        __import__(pkg.replace('-', '_'))
        print(f'✓ {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'✗ {pkg}')
if missing:
    print(f'\nERROR: Missing packages: {missing}')
    sys.exit(1)
"
if [ $? -ne 0 ]; then
    exit 1
fi

################################################################################
# Run evaluations
################################################################################

echo ""
echo "================================================"
echo "  Starting Evaluation"
echo "================================================"

OVERALL_SUCCESS=true

# Evaluate forget set
if [[ "$DATASET_TYPE" == "forget" || "$DATASET_TYPE" == "both" ]]; then
    run_evaluation "$FORGET_FILE" "FORGET" "$OUTPUT_BASE/forget"
    [ $? -ne 0 ] && OVERALL_SUCCESS=false
fi

# Evaluate retain set
if [[ "$DATASET_TYPE" == "retain" || "$DATASET_TYPE" == "both" ]]; then
    run_evaluation "$RETAIN_FILE" "RETAIN" "$OUTPUT_BASE/retain"
    [ $? -ne 0 ] && OVERALL_SUCCESS=false
fi

################################################################################
# Generate comparison report
################################################################################

if [[ "$DATASET_TYPE" == "both" ]] && [ "$OVERALL_SUCCESS" = true ]; then
    echo ""
    echo "================================================"
    echo "  Comparative Results"
    echo "================================================"

    python -c "
import json

def load_summary(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None

forget = load_summary('$OUTPUT_BASE/forget/evaluation_summary.json')
retain = load_summary('$OUTPUT_BASE/retain/evaluation_summary.json')

if forget and retain:
    print('\nUnlearning Performance:')
    print('-' * 50)
    print(f'Forget Set:')
    print(f'  ROUGE-L: {forget[\"avg_rouge_l\"]:.4f}')
    print(f'  Factuality: {forget.get(\"avg_factuality\", 0):.2f}/10')
    if forget.get('avg_total_score'):
        print(f'  Total Score: {forget[\"avg_total_score\"]:.4f}')
    print(f'\nRetain Set:')
    print(f'  ROUGE-L: {retain[\"avg_rouge_l\"]:.4f}')
    print(f'  Factuality: {retain.get(\"avg_factuality\", 0):.2f}/10')
    if retain.get('avg_total_score'):
        print(f'  Total Score: {retain[\"avg_total_score\"]:.4f}')
    print(f'\nDifferences (Retain - Forget):')
    print(f'  ΔROUGE-L: {retain[\"avg_rouge_l\"] - forget[\"avg_rouge_l\"]:+.4f}')
    if forget.get('avg_factuality', 0) > 0 and retain.get('avg_factuality', 0) > 0:
        print(f'  ΔFactuality: {retain[\"avg_factuality\"] - forget[\"avg_factuality\"]:+.2f}')
    if forget.get('avg_total_score') and retain.get('avg_total_score'):
        print(f'  ΔTotal Score: {retain[\"avg_total_score\"] - forget[\"avg_total_score\"]:+.4f}')
    print('\nInterpretation:')
    print('  • Lower forget scores indicate better forgetting')
    print('  • Higher retain scores indicate better retention')
    print('  • Larger positive Δ suggests better unlearning')
"
fi

################################################################################
# Task type analysis
################################################################################

if [ "$OVERALL_SUCCESS" = true ]; then
    echo ""
    echo "================================================"
    echo "  Task Type Analysis"
    echo "================================================"
    echo ""
    
    if [ -f "scripts/analyze_eval_results.py" ]; then
        python scripts/analyze_eval_results.py --results_dir "$OUTPUT_BASE"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ Task type analysis completed${NC}"
            echo ""
            echo "Analysis output:"
            echo "  • task_type_analysis.json - Task type breakdown and comparison"
        else
            echo -e "${YELLOW}⚠ Task analysis failed but evaluation results are valid${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ Analysis script not found: scripts/analyze_eval_results.py${NC}"
        echo "You can run it manually later:"
        echo "  python scripts/analyze_eval_results.py -r $OUTPUT_BASE"
    fi
fi

# Final summary
echo ""
echo "================================================"
if [ "$OVERALL_SUCCESS" = true ]; then
    echo -e "${GREEN}Evaluation completed successfully${NC}"
    echo ""
    echo "Results: $OUTPUT_BASE/"
    echo "Log: $LOG_FILE"
else
    echo -e "${RED}Some evaluations failed${NC}"
    echo "Check log: $LOG_FILE"
fi
echo "Finished at: $(date)"
echo "================================================"