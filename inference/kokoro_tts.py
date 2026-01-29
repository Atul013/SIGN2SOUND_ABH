"""
Kokoro-TTS Module for High-Fidelity Speech Synthesis

This module provides a wrapper for Kokoro-TTS, a lightweight neural TTS engine
with high-fidelity output and low latency (<80ms).

Key Features:
- High-quality, expressive voices
- Low latency synthesis
- Fully offline execution
- Multiple voice profiles
- Emotional intonation support
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Optional, Union
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class KokoroTTS:
    """
    Kokoro-TTS wrapper for high-fidelity speech synthesis.
    
    Provides natural, human-like voice synthesis with low latency
    for real-time applications.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        voice: str = "af_bella",
        speed: float = 1.0,
        sample_rate: int = 24000,
        device: str = "cpu",
        verbose: bool = True
    ):
        """
        Initialize Kokoro-TTS.
        
        Args:
            model_path: Path to Kokoro model file (optional, will download if needed)
            voice: Voice profile to use
            speed: Speech speed multiplier (1.0 = normal)
            sample_rate: Audio sample rate
            device: Device to run on ('cpu' for edge deployment)
            verbose: Whether to print initialization messages
        """
        self.model_path = model_path
        self.voice = voice
        self.speed = speed
        self.sample_rate = sample_rate
        self.device = device
        self.verbose = verbose
        
        self.model = None
        self.voices = None
        
        if self.verbose:
            print("🔊 Initializing Kokoro-TTS...")
            print(f"  Voice: {self.voice}")
            print(f"  Speed: {self.speed}x")
            print(f"  Sample Rate: {self.sample_rate} Hz")
        
        self._load_model()
    
    def _load_model(self):
        """Load Kokoro-TTS model."""
        try:
            # Try to import kokoro
            try:
                import kokoro
                self.kokoro = kokoro
            except ImportError:
                print("⚠️  Kokoro not installed. Attempting to install...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "kokoro-onnx"])
                import kokoro
                self.kokoro = kokoro
            
            # Initialize model
            if self.model_path and Path(self.model_path).exists():
                self.model = self.kokoro.KokoroTTS(model_path=self.model_path)
            else:
                # Use default model (will download if needed)
                self.model = self.kokoro.KokoroTTS()
            
            # Get available voices
            self.voices = self.model.get_voices() if hasattr(self.model, 'get_voices') else []
            
            if self.verbose:
                print("✅ Kokoro-TTS loaded successfully!")
                if self.voices:
                    print(f"  Available voices: {', '.join(self.voices)}")
                print()
            
        except Exception as e:
            print(f"⚠️  Could not load Kokoro-TTS: {e}")
            print("  Falling back to pyttsx3...")
            self._load_fallback_tts()
    
    def _load_fallback_tts(self):
        """Load fallback TTS (pyttsx3) if Kokoro is not available."""
        try:
            import pyttsx3
            self.model = pyttsx3.init()
            self.model.setProperty('rate', 150)
            self.model.setProperty('volume', 1.0)
            self.is_fallback = True
            
            if self.verbose:
                print("✅ Fallback TTS (pyttsx3) loaded")
                print()
        except Exception as e:
            print(f"❌ Error loading fallback TTS: {e}")
            self.model = None
            self.is_fallback = True
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        play: bool = True
    ) -> Optional[np.ndarray]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize (plain text, no quotes or metadata)
            output_path: Optional path to save audio file
            play: Whether to play the audio
            
        Returns:
            Audio waveform as numpy array (if not using fallback)
        """
        if not text or not text.strip():
            return None
        
        # Clean text (remove quotes, newlines, etc.)
        text = self._clean_text(text)
        
        if self.verbose:
            print(f"🔊 Synthesizing: '{text}'")
        
        try:
            if hasattr(self, 'is_fallback') and self.is_fallback:
                # Use fallback TTS
                return self._synthesize_fallback(text, play)
            else:
                # Use Kokoro-TTS
                return self._synthesize_kokoro(text, output_path, play)
                
        except Exception as e:
            print(f"⚠️  Synthesis error: {e}")
            # Try fallback
            return self._synthesize_fallback(text, play)
    
    def _synthesize_kokoro(
        self,
        text: str,
        output_path: Optional[str],
        play: bool
    ) -> Optional[np.ndarray]:
        """Synthesize using Kokoro-TTS."""
        try:
            # Generate audio
            audio = self.model.synthesize(
                text,
                voice=self.voice,
                speed=self.speed
            )
            
            # Save if requested
            if output_path:
                self._save_audio(audio, output_path)
                if self.verbose:
                    print(f"  Audio saved to {output_path}")
            
            # Play if requested
            if play:
                self._play_audio(audio)
            
            return audio
            
        except Exception as e:
            print(f"⚠️  Kokoro synthesis error: {e}")
            raise e
    
    def _synthesize_fallback(self, text: str, play: bool) -> None:
        """Synthesize using fallback TTS (pyttsx3)."""
        if self.model is None:
            print("⚠️  No TTS engine available")
            return None
        
        try:
            self.model.say(text)
            if play:
                self.model.runAndWait()
        except Exception as e:
            print(f"⚠️  Fallback TTS error: {e}")
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for TTS input.
        
        - Remove quotes
        - Remove newlines
        - Trim whitespace
        - Ensure single sentence
        """
        # Remove quotes
        text = text.replace('"', '').replace("'", '')
        
        # Remove newlines
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Collapse multiple spaces
        text = ' '.join(text.split())
        
        # Trim
        text = text.strip()
        
        return text
    
    def _save_audio(self, audio: np.ndarray, output_path: str):
        """Save audio to file."""
        try:
            import soundfile as sf
            sf.write(output_path, audio, self.sample_rate)
        except ImportError:
            print("⚠️  soundfile not installed. Install with: pip install soundfile")
            # Try scipy as fallback
            try:
                from scipy.io import wavfile
                wavfile.write(output_path, self.sample_rate, audio)
            except ImportError:
                print("⚠️  scipy not installed either. Cannot save audio.")
    
    def _play_audio(self, audio: np.ndarray):
        """Play audio using sounddevice."""
        try:
            import sounddevice as sd
            sd.play(audio, self.sample_rate)
            sd.wait()
        except ImportError:
            print("⚠️  sounddevice not installed. Install with: pip install sounddevice")
            # Try pygame as fallback
            try:
                import pygame
                import tempfile
                from scipy.io import wavfile
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    temp_path = f.name
                    wavfile.write(temp_path, self.sample_rate, audio)
                
                # Play with pygame
                pygame.mixer.init()
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                # Clean up
                os.remove(temp_path)
                
            except Exception as e:
                print(f"⚠️  Could not play audio: {e}")
    
    def speak(self, text: str):
        """
        Convenience method to synthesize and play text.
        
        Args:
            text: Text to speak
        """
        self.synthesize(text, play=True)
    
    def set_voice(self, voice: str):
        """
        Change the voice profile.
        
        Args:
            voice: Voice name
        """
        if self.voices and voice in self.voices:
            self.voice = voice
            if self.verbose:
                print(f"Voice changed to: {voice}")
        else:
            print(f"⚠️  Voice '{voice}' not available")
            if self.voices:
                print(f"  Available voices: {', '.join(self.voices)}")
    
    def get_available_voices(self) -> list:
        """
        Get list of available voices.
        
        Returns:
            List of voice names
        """
        return self.voices if self.voices else []


def demo_kokoro():
    """Demo Kokoro-TTS functionality."""
    print("=" * 70)
    print("Kokoro-TTS Demo")
    print("=" * 70)
    print()
    
    # Test sentences
    test_sentences = [
        "Hello, how are you?",
        "I want water.",
        "Thank you very much!",
        "Where is the bathroom?",
        "Can you help me please?"
    ]
    
    try:
        tts = KokoroTTS(verbose=True)
        
        print("Testing speech synthesis:")
        print("-" * 70)
        
        for sentence in test_sentences:
            print(f"\nSpeaking: '{sentence}'")
            tts.speak(sentence)
            
            # Small pause between sentences
            import time
            time.sleep(0.5)
        
        print("\n" + "=" * 70)
        print("Demo complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_kokoro()
