# Quick Start Guide: Sign-to-Speech Pipeline

This guide will help you quickly set up and test the Sign-to-Speech pipeline with SLM text normalization and Kokoro-TTS.

## Installation (5 minutes)

### Step 1: Install Dependencies

```bash
# Navigate to project directory
cd c:\Users\ZAYED\s2s

# Install all dependencies
pip install -r requirements.txt
```

**Note**: The first installation may take 5-10 minutes as it downloads PyTorch, Transformers, and other large packages.

### Step 2: Verify Installation

```bash
# Test text normalizer
python inference/text_normalizer.py

# Test Kokoro-TTS
python inference/kokoro_tts.py

# Test complete pipeline
python inference/sign2speech_pipeline.py --demo
```

## Quick Test (2 minutes)

### Test 1: Command Line

```bash
# Process a single text
python inference/sign2speech_pipeline.py --text "who eat now"

# Expected output:
# 📥 Input: 'who eat now'
# 🤖 Normalized: 'Who is eating now?' (150ms)
# 🔊 Speech synthesized (75ms)
# ⏱️  Total latency: 225ms
```

### Test 2: Python Script

Create a file `test_pipeline.py`:

```python
from inference.sign2speech_pipeline import Sign2SpeechPipeline

# Initialize pipeline
pipeline = Sign2SpeechPipeline(
    use_normalizer=True,
    use_kokoro=True,
    verbose=True
)

# Test cases
test_texts = [
    "who eat now",
    "i want water",
    "hello how you"
]

# Process each text
for text in test_texts:
    print(f"\nProcessing: {text}")
    normalized = pipeline.process(
        text,
        return_normalized=True
    )
    print(f"Result: {normalized}")
```

Run it:
```bash
python test_pipeline.py
```

## Integration with Existing Code

### Option 1: Add to Real-time Demo

Edit `inference/realtime_demo.py`:

```python
# Add at the top
from inference.sign2speech_pipeline import Sign2SpeechPipeline

# In __init__ method
class ASLRealtimeDemo:
    def __init__(self, model_path, use_cuda=True):
        # ... existing code ...
        
        # Add pipeline
        self.speech_pipeline = Sign2SpeechPipeline(
            use_normalizer=True,
            use_kokoro=True,
            verbose=False
        )
    
    # In run method, after prediction
    def run(self, camera_id=0):
        # ... existing code ...
        
        # After getting predicted_letter
        if predicted_letter and predicted_letter != self.last_spoken:
            # Speak the letter
            self.speech_pipeline.process(predicted_letter)
            self.last_spoken = predicted_letter
```

### Option 2: Standalone Usage

```python
from inference.sign2speech_pipeline import Sign2SpeechPipeline

# Initialize once (takes ~10 seconds on first run)
pipeline = Sign2SpeechPipeline()

# Use in your code
def on_sign_detected(sign_text):
    """Called when a sign is detected."""
    pipeline.process(sign_text)

# Example
on_sign_detected("who eat now")  # Speaks: "Who is eating now?"
```

## Configuration Options

### Minimal Configuration (Fastest)

```python
pipeline = Sign2SpeechPipeline(
    use_normalizer=False,  # Skip normalization
    use_kokoro=False,      # Use basic TTS
    verbose=False
)
```

### Full Configuration (Best Quality)

```python
pipeline = Sign2SpeechPipeline(
    use_normalizer=True,
    use_kokoro=True,
    normalizer_config={
        'temperature': 0.1,  # Conservative
        'max_tokens': 50     # Short output
    },
    tts_config={
        'voice': 'af_bella',
        'speed': 1.0
    },
    verbose=True
)
```

## Troubleshooting

### Issue: "Model download failed"

**Solution**: Models download automatically on first use. If download fails:

```bash
# Manually download Qwen2.5
huggingface-cli login  # Optional, for faster downloads
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

### Issue: "CUDA out of memory"

**Solution**: Force CPU usage:

```python
pipeline = Sign2SpeechPipeline(
    normalizer_config={'device': 'cpu'},
    tts_config={'device': 'cpu'}
)
```

### Issue: "No audio output"

**Solution**: Install audio backends:

```bash
pip install sounddevice soundfile pygame
```

Or use fallback TTS:

```python
pipeline = Sign2SpeechPipeline(use_kokoro=False)
```

### Issue: "Import errors"

**Solution**: Reinstall dependencies:

```bash
pip install --upgrade transformers accelerate bitsandbytes
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Performance Tips

### 1. Reduce Latency

```python
# Use smaller max_tokens
pipeline = Sign2SpeechPipeline(
    normalizer_config={'max_tokens': 30}  # Faster
)
```

### 2. Reduce Memory Usage

```python
# Ensure INT4 quantization is enabled
pipeline = Sign2SpeechPipeline(
    normalizer_config={'load_in_4bit': True}  # ~500MB instead of ~2GB
)
```

### 3. Batch Processing

```python
# Process multiple texts at once
texts = ["who eat now", "i want water", "hello"]
results = pipeline.process_batch(texts)
```

## Next Steps

1. **Read Full Documentation**: See `docs/SIGN2SPEECH_PIPELINE.md`
2. **Run Tests**: `python tests/test_pipeline.py`
3. **Integrate with Your Code**: Add to `realtime_demo.py` or `sentence_builder_demo.py`
4. **Customize Configuration**: Adjust temperature, max_tokens, voice, etc.
5. **Optimize for Your Hardware**: Test different configurations

## Example Output

```
🤖 Initializing Text Normalizer (Qwen2.5-0.5B INT4)...
  Loading model: Qwen/Qwen2.5-0.5B-Instruct
  Device: cpu
  Quantization: INT4
✅ Model loaded successfully!

🔊 Initializing Kokoro-TTS...
  Voice: af_bella
  Speed: 1.0x
  Sample Rate: 24000 Hz
✅ Kokoro-TTS loaded successfully!

======================================================================
Pipeline Ready!
======================================================================

----------------------------------------------------------------------
📥 Input: 'who eat now'
📝 Normalized: 'who eat now' → 'Who is eating now?'
🔊 Synthesizing: 'Who is eating now?'
🔊 Speech synthesized (78ms)
⏱️  Total latency: 198ms
----------------------------------------------------------------------
```

## Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review `docs/SIGN2SPEECH_PIPELINE.md`
3. Run tests: `python tests/test_pipeline.py`
4. Check requirements: `pip list | grep -E "transformers|torch|kokoro"`

## Summary

✅ **Installation**: `pip install -r requirements.txt`  
✅ **Quick Test**: `python inference/sign2speech_pipeline.py --demo`  
✅ **Integration**: Import `Sign2SpeechPipeline` and call `process(text)`  
✅ **Performance**: <280ms total latency on CPU  
✅ **Memory**: ~600MB with INT4 quantization  

You're ready to go! 🚀
