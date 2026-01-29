# Sign2Sound Phase 2 - Dataset & Vocabulary

## 📚 Phase-2 Vocabulary: ASL Fingerspelling Alphabet

This vocabulary focuses on the **26 letters of the American Sign Language (ASL) fingerspelling alphabet** (A-Z).

### Why Alphabet Instead of Words?

✅ **Universal**: Users can spell ANY word they want  
✅ **Complete**: 26 letters cover all possible words  
✅ **Foundational**: Alphabet is the basis for all ASL communication  
✅ **Scalable**: No need to add more vocabulary - it's already complete  
✅ **Educational**: Easier to learn and demonstrate  

### Alphabet Overview

| Letter | ID | Difficulty | Hand Shape Description |
|--------|----|-----------|-----------------------|
| A | 0 | Easy | Closed fist, thumb on side |
| B | 1 | Easy | Flat hand, fingers together, thumb across palm |
| C | 2 | Easy | Curved hand forming "C" shape |
| D | 3 | Medium | Index finger up, other fingers touch thumb |
| E | 4 | Hard | Fingers curled, thumb across fingertips |
| F | 5 | Medium | Index and thumb touch, other fingers up |
| G | 6 | Medium | Index and thumb parallel, pointing sideways |
| H | 7 | Medium | Index and middle fingers extended sideways |
| I | 8 | Medium | Pinky finger up, others closed |
| J | 9 | Very Hard | Pinky up with "J" motion |
| K | 10 | Medium | Index and middle up in "V", thumb between |
| L | 11 | Medium | "L" shape with index and thumb |
| M | 12 | Hard | Thumb under first three fingers |
| N | 13 | Hard | Thumb under first two fingers |
| O | 14 | Easy | Fingers form circle with thumb |
| P | 15 | Medium | Like "K" but pointing down |
| Q | 16 | Medium | Like "G" but pointing down |
| R | 17 | Hard | Index and middle crossed |
| S | 18 | Easy | Closed fist, thumb over fingers |
| T | 19 | Hard | Thumb between index and middle |
| U | 20 | Medium | Index and middle up together |
| V | 21 | Medium | Index and middle up in "V" |
| W | 22 | Medium | Index, middle, ring up |
| X | 23 | Medium | Index finger crooked |
| Y | 24 | Medium | Pinky and thumb out |
| Z | 25 | Hard | "Z" motion with index finger |

### Letter Groups (for Learning)

```
A-E: A, B, C, D, E
F-J: F, G, H, I, J
K-O: K, L, M, N, O
P-T: P, Q, R, S, T
U-Z: U, V, W, X, Y, Z
```

### Difficulty Levels

- **Easy** (5 letters): A, B, C, O, S - Simple closed hand shapes
- **Medium** (14 letters): D, F, G, H, I, K, L, P, Q, U, V, W, X, Y - Standard finger positions
- **Hard** (6 letters): E, M, N, R, T, Z - Complex finger arrangements
- **Very Hard** (1 letter): J - Requires motion (draws "J" in air)

### Label Mapping

```python
# Letter to ID (A=0, B=1, ..., Z=25)
LETTER_TO_ID = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7,
    "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15,
    "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20, "V": 21, "W": 22,
    "X": 23, "Y": 24, "Z": 25
}

NUM_CLASSES = 26
```

## 📊 Dataset Information

### Source Datasets

We will use:
1. **ASL Alphabet Dataset** (Kaggle) - Primary source for static letters
2. **Custom recordings** - For motion-based letter (J) and validation
3. **Synthetic augmentation** - To increase dataset size

### Dataset Structure

```
data/
├── raw/                    # Original video/image files
│   ├── A/
│   ├── B/
│   ├── ...
│   └── Z/
├── processed/              # Preprocessed landmark sequences (from Developer A)
│   ├── train/
│   │   ├── A/
│   │   ├── B/
│   │   └── ...
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

**Minimum samples per letter**: 200 images/videos (target: 500+ per letter)

### Expected Statistics

```
Total Letters: 26
Target Samples per Letter: 500
Total Target Samples: 13,000

Split Distribution:
- Train: ~9,100 samples (350 per letter)
- Val: ~1,950 samples (75 per letter)
- Test: ~1,950 samples (75 per letter)
```

## 🎯 Data Requirements

### Image/Video Requirements
- **Format**: JPG, PNG for images; MP4, AVI for videos
- **Frame Rate**: 30 FPS for videos
- **Resolution**: 640x480 minimum
- **Duration**: 0.5-2 seconds per letter (static letters can be single frames)
- **Quality**: Clear hand visibility, minimal occlusion
- **Background**: Varied backgrounds for robustness

### Landmark Requirements (from Developer A)
- **Hand Landmarks**: 21 points per hand
- **Pose Landmarks**: 8 points (shoulders, elbows, wrists, hips)
- **Format**: JSON with normalized coordinates
- **Temporal**: Single frame for static letters, short sequence for J

## 📝 Usage Notes

### For Training
```python
from data.vocabulary import LETTER_TO_ID, ID_TO_LETTER, NUM_CLASSES, word_to_ids, ids_to_word

# Get class ID for a letter
class_id = LETTER_TO_ID["A"]  # Returns 0

# Get letter from ID
letter = ID_TO_LETTER[0]  # Returns "A"

# Convert word to sequence of IDs
word = "HELLO"
ids = word_to_ids(word)  # Returns [7, 4, 11, 11, 14]

# Convert IDs back to word
reconstructed = ids_to_word(ids)  # Returns "HELLO"
```

### Fingerspelling Recognition Flow

1. **Input**: Continuous video of user fingerspelling a word
2. **Segmentation**: Developer A's pipeline segments into individual letters
3. **Recognition**: Model predicts each letter (A-Z)
4. **Word Formation**: Combine predicted letters into words
5. **Output**: Text-to-speech speaks the spelled word

### Example Use Cases

- User spells "H-E-L-L-O" → System outputs "HELLO" (spoken)
- User spells "H-E-L-P" → System outputs "HELP" (spoken)
- User spells their name → System speaks their name

## 🔗 Integration with Developer A

Developer A's preprocessing pipeline outputs:
- **Input**: Raw video/images of fingerspelling
- **Output**: JSON files with hand landmark sequences
- **Location**: `data/processed/`

Developer B will:
1. Use preprocessed landmark sequences for training
2. Load sequences using `training/data_loader.py`
3. Train model to recognize 26 letters (A-Z)
4. Combine letter predictions into words

## 📚 Dataset Sources

### Recommended Datasets

1. **ASL Alphabet Dataset (Kaggle)**
   - URL: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
   - ~87,000 images (3,000+ per letter)
   - Static hand positions
   - Varied backgrounds and lighting

2. **ASL Fingerspelling Recognition Dataset**
   - URL: https://www.kaggle.com/datasets/mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out
   - Video sequences
   - Real-world fingerspelling

3. **Custom Collection**
   - Record additional samples for validation
   - Ensure diversity in hand sizes, skin tones, lighting

## 🎓 Training Strategy

### Phase 1: Static Letters (A-Z except J)
- Train on 25 static letters first
- Use single-frame images
- Achieve high accuracy (>95%)

### Phase 2: Motion Letter (J)
- Add temporal component for "J"
- Use short video sequences
- Fine-tune model

### Phase 3: Word Recognition
- Combine letter predictions
- Add temporal smoothing
- Handle repeated letters

## 📈 Success Metrics

- **Per-Letter Accuracy**: >95% for each letter
- **Overall Accuracy**: >90% across all 26 letters
- **Inference Speed**: <50ms per letter
- **Word Accuracy**: >85% for common words (3-7 letters)

## 🚀 Future Enhancements

- Add numbers (0-9)
- Add common ASL words (non-fingerspelled)
- Support for double letters (e.g., "LL" in "HELLO")
- Auto-correction using dictionary
- Real-time word suggestions
