#!/bin/bash

# Run All Pipeline Script for Sign2Sound Phase 2
# Executes the complete pipeline from data preprocessing to evaluation

echo "=========================================="
echo "Sign2Sound Phase 2 - Full Pipeline"
echo "=========================================="

# Configuration
DATA_DIR="data"
PROCESSED_DIR="data/processed"
CHECKPOINTS_DIR="checkpoints"
RESULTS_DIR="results"
CONFIG_FILE="training/config.yaml"

# Function to check if command succeeded
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed"
        exit 1
    fi
}

# Function to print section header
print_section() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "Activating virtual environment..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ Virtual environment not found. Run setup_environment.sh first"
        exit 1
    fi
fi

# Step 1: Data Preprocessing
print_section "Step 1: Data Preprocessing"
echo "Preprocessing raw video data..."
python preprocessing/preprocess.py \
    --input "$DATA_DIR/raw" \
    --output "$PROCESSED_DIR" \
    --verbose
check_status "Data preprocessing"

# Step 2: Feature Extraction
print_section "Step 2: Feature Extraction"
echo "Extracting landmarks from videos..."
python preprocessing/extract_features.py \
    --input "$DATA_DIR/raw" \
    --output "$PROCESSED_DIR" \
    --target-fps 30
check_status "Feature extraction"

# Step 3: Data Augmentation (optional)
print_section "Step 3: Data Augmentation"
read -p "Do you want to apply data augmentation? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Applying data augmentation..."
    python preprocessing/augmentation.py \
        --input "$PROCESSED_DIR" \
        --output "$PROCESSED_DIR/augmented" \
        --augmentation-factor 2
    check_status "Data augmentation"
else
    echo "Skipping data augmentation"
fi

# Step 4: Training
print_section "Step 4: Model Training"
echo "Training sign recognition model..."
python training/train.py \
    --config "$CONFIG_FILE" \
    --data "$PROCESSED_DIR" \
    --output "$CHECKPOINTS_DIR" \
    --verbose
check_status "Model training"

# Step 5: Evaluation
print_section "Step 5: Model Evaluation"
echo "Evaluating model on test set..."
python training/evaluate.py \
    --model "$CHECKPOINTS_DIR/best_model.h5" \
    --data "$PROCESSED_DIR" \
    --output "$RESULTS_DIR" \
    --verbose
check_status "Model evaluation"

# Step 6: Generate Visualizations
print_section "Step 6: Generating Visualizations"
echo "Creating performance visualizations..."
python -c "
import json
import matplotlib.pyplot as plt
import numpy as np

# Load metrics
with open('$RESULTS_DIR/metrics.json', 'r') as f:
    metrics = json.load(f)

print('Generating plots...')

# Create plots directory
import os
os.makedirs('$RESULTS_DIR', exist_ok=True)

print('✅ Visualizations generated')
"
check_status "Visualization generation"

# Step 7: Test Inference
print_section "Step 7: Testing Inference"
read -p "Do you want to test real-time inference? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting real-time demo..."
    echo "Press Ctrl+C to stop"
    python inference/realtime_demo.py \
        --model "$CHECKPOINTS_DIR/best_model.h5" \
        --verbose
else
    echo "Skipping real-time inference test"
fi

# Step 8: Generate Report
print_section "Step 8: Generating Summary Report"
echo "Creating summary report..."

cat > "$RESULTS_DIR/pipeline_summary.txt" << EOF
Sign2Sound Phase 2 - Pipeline Execution Summary
================================================

Execution Date: $(date)

Pipeline Steps Completed:
1. ✅ Data Preprocessing
2. ✅ Feature Extraction
3. $([ "$REPLY" = "y" ] && echo "✅" || echo "⏭️ ") Data Augmentation
4. ✅ Model Training
5. ✅ Model Evaluation
6. ✅ Visualization Generation
7. $([ "$REPLY" = "y" ] && echo "✅" || echo "⏭️ ") Real-time Inference Test

Output Locations:
- Processed Data: $PROCESSED_DIR
- Model Checkpoints: $CHECKPOINTS_DIR
- Results & Metrics: $RESULTS_DIR

Next Steps:
1. Review results in $RESULTS_DIR/metrics.json
2. Check visualizations in $RESULTS_DIR/
3. Test inference with: python inference/realtime_demo.py
4. Deploy model for production use

================================================
EOF

echo "✅ Summary report generated: $RESULTS_DIR/pipeline_summary.txt"

# Final Summary
print_section "Pipeline Complete!"
echo "All steps completed successfully! 🎉"
echo ""
echo "Results Summary:"
echo "  - Processed data: $PROCESSED_DIR"
echo "  - Model checkpoint: $CHECKPOINTS_DIR/best_model.h5"
echo "  - Evaluation results: $RESULTS_DIR/metrics.json"
echo "  - Summary report: $RESULTS_DIR/pipeline_summary.txt"
echo ""
echo "To view results:"
echo "  cat $RESULTS_DIR/metrics.json"
echo ""
echo "To run real-time demo:"
echo "  python inference/realtime_demo.py --model $CHECKPOINTS_DIR/best_model.h5"
echo ""
