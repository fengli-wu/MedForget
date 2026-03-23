#!/bin/bash

# ==============================================================================
# CHIP: Cross-modal Hierarchy-Informed Projection for Multimodal Unlearning
# Training-free unlearning via orthogonal weight projection
# ==============================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo "================================================"
echo "  CHIP: Hierarchy-Aware Multimodal Unlearning"
echo "  Training-Free Orthogonal Weight Projection"
echo "================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

################################################################################
# Configuration
################################################################################

# Default model configuration
MODEL_ID="lingshu-medical-mllm/Lingshu-7B"
MODEL_TYPE="Lingshu"
MODEL_PATH=""

# Default data configuration
DATA_BASE=""
LEVEL="institution_level"
FILTER="all"

# CHIP hyperparameters
TOP_K=10
VARIANCE=0.95
ALPHA=0.3
VISION_TEXT_SEP="--vision_text_separation"
LANG_LAYERS="22 23 24 25 26 27"

# Runtime parameters
BATCH_SIZE=4
MAX_SAMPLES_PER_NODE=500
MAX_TARGETS=""

# Output
SAVE_DIR=""

################################################################################
# Parse arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-id)
            MODEL_ID="$2"
            shift 2
            ;;
        --model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --data|-d)
            DATA_BASE="$2"
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
        --top-k|-k)
            TOP_K="$2"
            shift 2
            ;;
        --variance|-t)
            VARIANCE="$2"
            shift 2
            ;;
        --alpha|-a)
            ALPHA="$2"
            shift 2
            ;;
        --no-vision-sep)
            VISION_TEXT_SEP="--no_vision_text_separation"
            shift
            ;;
        --lang-layers)
            LANG_LAYERS="$2"
            shift 2
            ;;
        --batch-size|--bs)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES_PER_NODE="$2"
            shift 2
            ;;
        --max-targets)
            MAX_TARGETS="$2"
            shift 2
            ;;
        --output|-o)
            SAVE_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Model Options:"
            echo "  --model, -m PATH          Path to fine-tuned (pre-unlearning) model"
            echo "  --model-id ID             HuggingFace model ID (default: lingshu-medical-mllm/Lingshu-7B)"
            echo "  --model-type TYPE         Model type: Lingshu or Llava (default: Lingshu)"
            echo ""
            echo "Data Options:"
            echo "  --data, -d PATH           Base directory for MedForget data"
            echo "  --level, -l LEVEL         Hierarchy level (default: patient_level)"
            echo "                            Options: institution_level, patient_level, study_level, section_level"
            echo "  --filter, -f FILTER       Filter type (default: all)"
            echo ""
            echo "CHIP Hyperparameters:"
            echo "  --top-k, -k NUM           Percentage of neurons to select (default: 10)"
            echo "  --variance, -t NUM        SVD variance threshold tau (default: 0.95)"
            echo "  --alpha, -a NUM           Vision weight in lang layer activations (default: 0.3)"
            echo "  --no-vision-sep           Disable vision-text separation"
            echo "  --lang-layers LAYERS      Language layer indices (default: \"22 23 24 25 26 27\")"
            echo ""
            echo "Runtime Options:"
            echo "  --batch-size, --bs NUM    Batch size for activation collection (default: 4)"
            echo "  --max-samples NUM         Max QA samples per hierarchy node (default: 500)"
            echo "  --max-targets NUM         Max number of forget targets (default: all)"
            echo "  --output, -o DIR          Output directory for unlearned model"
            echo "  --help, -h                Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --model /path/to/model --data /path/to/data --level patient_level"
            echo "  $0 -m /path/to/model -d /path/to/data -l institution_level -k 15 -t 0.95"
            echo ""
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Validate inputs
################################################################################

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}WARNING: GPU not detected${NC}"
else
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    echo ""
fi

# Validate model path
if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}ERROR: Model path required${NC}"
    echo "Please provide --model PATH"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}ERROR: Model not found: $MODEL_PATH${NC}"
    exit 1
fi

if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo -e "${RED}ERROR: Invalid model directory (missing config.json): $MODEL_PATH${NC}"
    exit 1
fi
echo -e "${GREEN}Model: $MODEL_PATH${NC}"

# Validate data
if [ -z "$DATA_BASE" ]; then
    echo -e "${RED}ERROR: Data directory required${NC}"
    echo "Please provide --data PATH"
    exit 1
fi

# Derive target level (strip _level suffix)
TARGET_LEVEL="${LEVEL%_level}"

FORGET_FILE="${DATA_BASE}/${LEVEL}/forget_set_${FILTER}.parquet"
RETAIN_FILE="${DATA_BASE}/${LEVEL}/retain_set_${FILTER}.parquet"

if [ ! -f "$FORGET_FILE" ]; then
    echo -e "${RED}ERROR: Forget file not found: $FORGET_FILE${NC}"
    exit 1
fi
if [ ! -f "$RETAIN_FILE" ]; then
    echo -e "${RED}ERROR: Retain file not found: $RETAIN_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}Forget: $FORGET_FILE${NC}"
echo -e "${GREEN}Retain: $RETAIN_FILE${NC}"

# Set output directory
if [ -z "$SAVE_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    SAVE_DIR="chip_output_${TARGET_LEVEL}_${TIMESTAMP}"
fi

echo ""

################################################################################
# Verify Python packages
################################################################################

echo "Checking Python packages..."
python -c "
required = ['torch', 'transformers', 'pandas', 'PIL']
missing = []
for pkg in required:
    try:
        __import__(pkg)
        print(f'  {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'  {pkg} (MISSING)')
if missing:
    import sys
    print(f'\nERROR: Missing packages: {missing}')
    sys.exit(1)
"
if [ $? -ne 0 ]; then
    exit 1
fi
echo ""

################################################################################
# Run CHIP
################################################################################

echo "================================================"
echo "  Configuration"
echo "================================================"
echo "  Model:         $MODEL_PATH"
echo "  Model ID:      $MODEL_ID"
echo "  Model Type:    $MODEL_TYPE"
echo "  Level:         $TARGET_LEVEL"
echo "  Top-k:         ${TOP_K}%"
echo "  Variance (τ):  $VARIANCE"
echo "  Alpha (α):     $ALPHA"
echo "  Lang layers:   $LANG_LAYERS"
echo "  Batch size:    $BATCH_SIZE"
echo "  Max samples:   $MAX_SAMPLES_PER_NODE"
echo "  Output:        $SAVE_DIR"
echo ""

# Build command
CMD="python -m chip.run_chip \
    --model_id \"$MODEL_ID\" \
    --vanilla_dir \"$MODEL_PATH\" \
    --model_type \"$MODEL_TYPE\" \
    --forget_file \"$FORGET_FILE\" \
    --retain_file \"$RETAIN_FILE\" \
    --target_level \"$TARGET_LEVEL\" \
    --top_k_percent $TOP_K \
    --variance_threshold $VARIANCE \
    --alpha $ALPHA \
    $VISION_TEXT_SEP \
    --lang_layers $LANG_LAYERS \
    --batch_size $BATCH_SIZE \
    --max_samples_per_node $MAX_SAMPLES_PER_NODE \
    --save_dir \"$SAVE_DIR\""

# Add optional max_targets
if [ -n "$MAX_TARGETS" ]; then
    CMD="$CMD --max_targets $MAX_TARGETS"
fi

echo "================================================"
echo "  Starting CHIP Unlearning"
echo "  Time: $(date)"
echo "================================================"
echo ""

eval $CMD
EXIT_CODE=$?

echo ""
echo "================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}CHIP unlearning completed successfully${NC}"
    echo ""
    echo "Unlearned model saved to: $SAVE_DIR"
    echo ""
    if [ -f "$SAVE_DIR/chip_config.json" ]; then
        echo "Surgery config:"
        cat "$SAVE_DIR/chip_config.json"
        echo ""
    fi
    echo ""
    echo "To evaluate the unlearned model:"
    echo "  ./run_eval.sh --model $SAVE_DIR --level $LEVEL --dataset both"
else
    echo -e "${RED}CHIP unlearning failed (exit code: $EXIT_CODE)${NC}"
fi
echo "Finished at: $(date)"
echo "================================================"

exit $EXIT_CODE
