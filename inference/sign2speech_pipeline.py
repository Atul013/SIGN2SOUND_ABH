"""
Integrated Sign-to-Speech Pipeline with SLM Normalization

This module implements the complete pipeline:
Sign-to-text output → SLM text normalization (Qwen2.5-0.5B INT4) → Kokoro-TTS → Audio output

Key Features:
- On-device text normalization using Qwen2.5-0.5B (INT4)
- High-fidelity speech synthesis using Kokoro-TTS
- Edge-compatible (low memory, CPU-only)
- No cloud dependencies
- Conservative decoding for reliable output
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.text_normalizer import TextNormalizer
from inference.kokoro_tts import KokoroTTS


class Sign2SpeechPipeline:
    """
    Complete sign-to-speech pipeline with SLM normalization.
    
    Pipeline stages:
    1. Sign-to-text output (from sign recognition model)
    2. SLM text normalization (Qwen2.5-0.5B INT4)
    3. Kokoro-TTS speech synthesis
    4. Audio output
    """
    
    def __init__(
        self,
        use_normalizer: bool = True,
        use_kokoro: bool = True,
        normalizer_config: Optional[Dict[str, Any]] = None,
        tts_config: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ):
        """
        Initialize the sign-to-speech pipeline.
        
        Args:
            use_normalizer: Whether to use SLM text normalization
            use_kokoro: Whether to use Kokoro-TTS (fallback to pyttsx3 if False)
            normalizer_config: Configuration for text normalizer
            tts_config: Configuration for TTS engine
            verbose: Whether to print status messages
        """
        self.use_normalizer = use_normalizer
        self.use_kokoro = use_kokoro
        self.verbose = verbose
        
        # Default configurations
        self.normalizer_config = normalizer_config or {
            'model_name': 'Qwen/Qwen2.5-0.5B-Instruct',
            'device': 'cpu',
            'load_in_4bit': True,
            'max_tokens': 50,
            'temperature': 0.1,
            'verbose': verbose
        }
        
        self.tts_config = tts_config or {
            'voice': 'af_bella',
            'speed': 1.0,
            'sample_rate': 24000,
            'device': 'cpu',
            'verbose': verbose
        }
        
        # Initialize components
        self.normalizer = None
        self.tts = None
        
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initialize pipeline components."""
        if self.verbose:
            print("=" * 70)
            print("Initializing Sign-to-Speech Pipeline")
            print("=" * 70)
            print()
        
        # Initialize text normalizer
        if self.use_normalizer:
            try:
                if self.verbose:
                    print("Stage 1: Loading Text Normalizer (Qwen2.5-0.5B INT4)...")
                self.normalizer = TextNormalizer(**self.normalizer_config)
            except Exception as e:
                print(f"⚠️  Could not load text normalizer: {e}")
                print("  Continuing without normalization...")
                self.use_normalizer = False
        
        # Initialize TTS
        if self.use_kokoro:
            try:
                if self.verbose:
                    print("Stage 2: Loading TTS Engine (Kokoro-TTS)...")
                self.tts = KokoroTTS(**self.tts_config)
            except Exception as e:
                print(f"⚠️  Could not load Kokoro-TTS: {e}")
                print("  Using fallback TTS...")
                # Fallback to basic TTS
                from inference.tts import TextToSpeech
                self.tts = TextToSpeech(engine='pyttsx3', rate=150, volume=1.0)
        else:
            # Use basic TTS
            from inference.tts import TextToSpeech
            self.tts = TextToSpeech(engine='pyttsx3', rate=150, volume=1.0)
        
        if self.verbose:
            print()
            print("=" * 70)
            print("Pipeline Ready!")
            print("=" * 70)
            print()
    
    def process(
        self,
        sign_text: str,
        save_audio: bool = False,
        output_path: Optional[str] = None,
        return_normalized: bool = False
    ) -> Optional[str]:
        """
        Process sign-to-text output through the complete pipeline.
        
        Args:
            sign_text: Raw text from sign recognition
            save_audio: Whether to save audio file
            output_path: Path to save audio (if save_audio=True)
            return_normalized: Whether to return normalized text
            
        Returns:
            Normalized text (if return_normalized=True), otherwise None
        """
        if not sign_text or not sign_text.strip():
            if self.verbose:
                print("⚠️  Empty input text")
            return None
        
        start_time = time.time()
        
        if self.verbose:
            print("-" * 70)
            print(f"📥 Input: '{sign_text}'")
        
        # Stage 1: Text Normalization
        if self.use_normalizer and self.normalizer:
            try:
                norm_start = time.time()
                normalized_text = self.normalizer.normalize(sign_text)
                norm_time = (time.time() - norm_start) * 1000
                
                if self.verbose:
                    print(f"🤖 Normalized: '{normalized_text}' ({norm_time:.0f}ms)")
            except Exception as e:
                print(f"⚠️  Normalization failed: {e}")
                normalized_text = sign_text
        else:
            normalized_text = sign_text
            if self.verbose:
                print(f"⏭️  Skipping normalization")
        
        # Stage 2: Text-to-Speech
        try:
            tts_start = time.time()
            
            if isinstance(self.tts, KokoroTTS):
                # Use Kokoro-TTS
                self.tts.synthesize(
                    normalized_text,
                    output_path=output_path if save_audio else None,
                    play=True
                )
            else:
                # Use fallback TTS
                self.tts.speak(normalized_text)
            
            tts_time = (time.time() - tts_start) * 1000
            
            if self.verbose:
                print(f"🔊 Speech synthesized ({tts_time:.0f}ms)")
        
        except Exception as e:
            print(f"⚠️  TTS failed: {e}")
        
        # Total time
        total_time = (time.time() - start_time) * 1000
        
        if self.verbose:
            print(f"⏱️  Total latency: {total_time:.0f}ms")
            print("-" * 70)
            print()
        
        if return_normalized:
            return normalized_text
        return None
    
    def process_batch(
        self,
        sign_texts: list,
        save_audio: bool = False,
        output_dir: Optional[str] = None
    ) -> list:
        """
        Process multiple sign-to-text outputs.
        
        Args:
            sign_texts: List of raw texts from sign recognition
            save_audio: Whether to save audio files
            output_dir: Directory to save audio files
            
        Returns:
            List of normalized texts
        """
        results = []
        
        if save_audio and output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, text in enumerate(sign_texts):
            output_path = None
            if save_audio and output_dir:
                output_path = str(Path(output_dir) / f"output_{i:03d}.wav")
            
            normalized = self.process(
                text,
                save_audio=save_audio,
                output_path=output_path,
                return_normalized=True
            )
            results.append(normalized)
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline configuration.
        
        Returns:
            Dictionary with pipeline info
        """
        return {
            'normalizer_enabled': self.use_normalizer,
            'normalizer_model': self.normalizer_config.get('model_name') if self.use_normalizer else None,
            'normalizer_quantization': 'INT4' if self.normalizer_config.get('load_in_4bit') else 'None',
            'tts_engine': 'Kokoro-TTS' if self.use_kokoro else 'pyttsx3',
            'tts_voice': self.tts_config.get('voice'),
            'device': 'CPU (edge-compatible)',
            'cloud_dependency': False
        }


def demo_pipeline():
    """Demo the complete sign-to-speech pipeline."""
    print("=" * 70)
    print("Sign-to-Speech Pipeline Demo")
    print("=" * 70)
    print()
    
    # Test cases (simulating sign-to-text output)
    test_cases = [
        "who eat now",
        "i want water",
        "hello how you",
        "thank you very much",
        "what time is",
        "where bathroom",
        "help me please"
    ]
    
    try:
        # Initialize pipeline
        pipeline = Sign2SpeechPipeline(
            use_normalizer=True,
            use_kokoro=True,
            verbose=True
        )
        
        # Show pipeline info
        info = pipeline.get_pipeline_info()
        print("Pipeline Configuration:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()
        
        # Process test cases
        print("Processing test cases:")
        print("=" * 70)
        
        for text in test_cases:
            pipeline.process(text)
            time.sleep(1)  # Pause between utterances
        
        print("=" * 70)
        print("Demo complete!")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point with CLI arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Sign-to-Speech Pipeline with SLM Normalization'
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Text to process (sign-to-text output)'
    )
    parser.add_argument(
        '--no-normalizer',
        action='store_true',
        help='Disable SLM text normalization'
    )
    parser.add_argument(
        '--no-kokoro',
        action='store_true',
        help='Use fallback TTS instead of Kokoro'
    )
    parser.add_argument(
        '--save-audio',
        action='store_true',
        help='Save audio output'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.wav',
        help='Output audio file path'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo with test cases'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        demo_pipeline()
    elif args.text:
        pipeline = Sign2SpeechPipeline(
            use_normalizer=not args.no_normalizer,
            use_kokoro=not args.no_kokoro,
            verbose=True
        )
        
        pipeline.process(
            args.text,
            save_audio=args.save_audio,
            output_path=args.output if args.save_audio else None
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
