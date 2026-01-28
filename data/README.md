# Sign2Sound Phase 2 - Dataset & Vocabulary

## 📚 Phase-2 Vocabulary (15 Signs)

This vocabulary focuses on **functional communication** - signs that enable basic conversation and needs expression.

### Vocabulary List

| Sign ID | Sign Name | Category | Description | Usage Context |
|---------|-----------|----------|-------------|---------------|
| 0 | **HELLO** | Greeting | Standard greeting gesture | Starting conversations |
| 1 | **GOODBYE** | Greeting | Farewell gesture | Ending conversations |
| 2 | **THANK_YOU** | Courtesy | Expression of gratitude | Acknowledging help |
| 3 | **PLEASE** | Courtesy | Polite request | Making requests |
| 4 | **YES** | Response | Affirmative response | Confirming/agreeing |
| 5 | **NO** | Response | Negative response | Declining/disagreeing |
| 6 | **HELP** | Need | Request for assistance | Emergency/assistance |
| 7 | **WATER** | Need | Request for water | Basic needs |
| 8 | **FOOD** | Need | Request for food | Basic needs |
| 9 | **BATHROOM** | Need | Request for restroom | Basic needs |
| 10 | **SORRY** | Courtesy | Apology | Expressing regret |
| 11 | **UNDERSTAND** | Question | "Do you understand?" | Checking comprehension |
| 12 | **WHAT** | Question | Question word | Asking questions |
| 13 | **WHERE** | Question | Question word | Location queries |
| 14 | **HOW** | Question | Question word | Process queries |

### Label Mapping

```python
SIGN_TO_ID = {
    "HELLO": 0,
    "GOODBYE": 1,
    "THANK_YOU": 2,
    "PLEASE": 3,
    "YES": 4,
    "NO": 5,
    "HELP": 6,
    "WATER": 7,
    "FOOD": 8,
    "BATHROOM": 9,
    "SORRY": 10,
    "UNDERSTAND": 11,
    "WHAT": 12,
    "WHERE": 13,
    "HOW": 14
}

ID_TO_SIGN = {v: k for k, v in SIGN_TO_ID.items()}
NUM_CLASSES = 15
```

## 📊 Dataset Information

### Source Datasets

We will use a combination of:
1. **WLASL (Word-Level American Sign Language)** - Primary source
2. **MS-ASL (Microsoft American Sign Language)** - Supplementary
3. **Custom recordings** (if needed for specific signs)

### Dataset Structure

```
data/
├── raw/                    # Original video files
│   ├── HELLO/
│   ├── GOODBYE/
│   └── ...
├── processed/              # Preprocessed landmark sequences (from Developer A)
│   ├── train/
│   ├── val/
│   └── test/
├── splits/                 # Train/val/test split definitions
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
└── statistics.txt          # Dataset statistics
```

### Data Splits

- **Training**: 70% of data
- **Validation**: 15% of data
- **Testing**: 15% of data

**Minimum samples per sign**: 50 videos (target: 100+ per sign)

### Expected Statistics

```
Total Signs: 15
Target Videos per Sign: 100
Total Target Videos: 1,500

Split Distribution:
- Train: ~1,050 videos (70 per sign)
- Val: ~225 videos (15 per sign)
- Test: ~225 videos (15 per sign)
```

## 🎯 Data Requirements

### Video Requirements
- **Format**: MP4, AVI, or MOV
- **Frame Rate**: 30 FPS (minimum)
- **Resolution**: 640x480 minimum
- **Duration**: 1-5 seconds per sign
- **Quality**: Clear hand visibility, minimal occlusion

### Landmark Requirements (from Developer A)
- **Hand Landmarks**: 21 points per hand (42 total)
- **Pose Landmarks**: 8 points (shoulders, elbows, wrists, hips)
- **Format**: JSON with normalized coordinates
- **Temporal**: Variable-length sequences (handled by model)

## 📝 Usage Notes

### For Training
```python
from data.vocabulary import SIGN_TO_ID, ID_TO_SIGN, NUM_CLASSES

# Get class ID for a sign
class_id = SIGN_TO_ID["HELLO"]  # Returns 0

# Get sign name from ID
sign_name = ID_TO_SIGN[0]  # Returns "HELLO"
```

### Vocabulary Expansion (Future Phases)
Phase 3 can expand to:
- Numbers (0-10)
- Family members
- Common verbs (eat, drink, sleep)
- Emotions (happy, sad, angry)

## 🔗 Integration with Developer A

Developer A's preprocessing pipeline outputs:
- **Input**: Raw video files
- **Output**: JSON files with segmented landmark sequences
- **Location**: `data/processed/`

Developer B will:
1. Use preprocessed landmark sequences for training
2. Load sequences using `training/data_loader.py`
3. Train model on these sequences

## 📚 References

- WLASL Dataset: https://dxli94.github.io/WLASL/
- MS-ASL Dataset: https://www.microsoft.com/en-us/research/project/ms-asl/
- ASL Sign Bank: https://aslsignbank.haskins.yale.edu/
