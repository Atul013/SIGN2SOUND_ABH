#!/bin/bash
# Dataset Download Script for ASL Alphabet Recognition
# This script downloads and prepares the ASL alphabet dataset

echo "======================================================================"
echo "ASL Alphabet Dataset Download Script"
echo "======================================================================"

# Configuration
DATA_DIR="data"
RAW_DIR="$DATA_DIR/raw"
PROCESSED_DIR="$DATA_DIR/processed"
SPLITS_DIR="$DATA_DIR/splits"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p "$RAW_DIR"
mkdir -p "$PROCESSED_DIR/train"
mkdir -p "$PROCESSED_DIR/val"
mkdir -p "$PROCESSED_DIR/test"
mkdir -p "$SPLITS_DIR"

echo "✅ Directories created"

# Dataset options
echo ""
echo "Available ASL Alphabet Datasets:"
echo "======================================================================"
echo "1. Kaggle ASL Alphabet Dataset"
echo "   - URL: https://www.kaggle.com/datasets/grassknoted/asl-alphabet"
echo "   - Size: ~1.1 GB"
echo "   - Samples: ~87,000 images (3,000+ per letter)"
echo "   - Format: JPG images"
echo ""
echo "2. ASL Fingerspelling Recognition Dataset"
echo "   - URL: https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out"
echo "   - Size: ~5 GB"
echo "   - Samples: Video sequences"
echo "   - Format: Video files"
echo ""
echo "3. Custom Dataset"
echo "   - Record your own samples"
echo "   - Use webcam or video files"
echo "======================================================================"

echo ""
echo "Download Instructions:"
echo "----------------------------------------------------------------------"
echo "Option 1: Kaggle CLI (Recommended)"
echo "  1. Install Kaggle CLI: pip install kaggle"
echo "  2. Setup API credentials: https://www.kaggle.com/docs/api"
echo "  3. Run: kaggle datasets download -d grassknoted/asl-alphabet"
echo "  4. Unzip to $RAW_DIR"
echo ""
echo "Option 2: Manual Download"
echo "  1. Visit: https://www.kaggle.com/datasets/grassknoted/asl-alphabet"
echo "  2. Click 'Download' button"
echo "  3. Extract ZIP to $RAW_DIR"
echo ""
echo "Option 3: Use wget (if direct link available)"
echo "  wget <dataset_url> -O $RAW_DIR/asl_alphabet.zip"
echo "  unzip $RAW_DIR/asl_alphabet.zip -d $RAW_DIR"
echo "======================================================================"

# Check if Kaggle CLI is available
if command -v kaggle &> /dev/null; then
    echo ""
    echo "✅ Kaggle CLI detected!"
    echo ""
    read -p "Download ASL Alphabet dataset using Kaggle CLI? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Downloading dataset..."
        kaggle datasets download -d grassknoted/asl-alphabet -p "$RAW_DIR"
        
        echo ""
        echo "Extracting dataset..."
        unzip "$RAW_DIR/asl-alphabet.zip" -d "$RAW_DIR"
        
        echo "✅ Dataset downloaded and extracted!"
    fi
else
    echo ""
    echo "⚠️  Kaggle CLI not found"
    echo "   Install with: pip install kaggle"
    echo "   Then setup API credentials: https://www.kaggle.com/docs/api"
fi

# Dataset organization
echo ""
echo "======================================================================"
echo "Dataset Organization"
echo "======================================================================"
echo ""
echo "Expected directory structure:"
echo "$DATA_DIR/"
echo "├── raw/                    # Original downloaded data"
echo "│   ├── A/                  # Letter A images"
echo "│   ├── B/                  # Letter B images"
echo "│   └── ...                 # Other letters"
echo "├── processed/              # Preprocessed landmarks"
echo "│   ├── train/"
echo "│   │   ├── A/"
echo "│   │   ├── B/"
echo "│   │   └── ..."
echo "│   ├── val/"
echo "│   └── test/"
echo "└── splits/                 # Train/val/test split files"
echo "    ├── train.txt"
echo "    ├── val.txt"
echo "    └── test.txt"
echo ""

# Next steps
echo "======================================================================"
echo "Next Steps"
echo "======================================================================"
echo ""
echo "After downloading the dataset:"
echo ""
echo "1. Preprocess with Developer A's pipeline:"
echo "   python preprocessing/preprocess.py --input data/raw --output data/processed"
echo ""
echo "2. Create train/val/test splits:"
echo "   python scripts/create_splits.py --data-dir data/processed --split-ratio 0.7 0.15 0.15"
echo ""
echo "3. Verify dataset statistics:"
echo "   python scripts/verify_dataset.py --data-dir data"
echo ""
echo "4. Start training:"
echo "   python training/train.py"
echo ""
echo "======================================================================"

echo ""
echo "✅ Dataset download script complete!"
echo ""
