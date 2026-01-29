"""
Text-to-Speech Module for ASL Alphabet Recognition

This module provides text-to-speech functionality to convert predicted
ASL letters/words into spoken audio output.
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TextToSpeech:
    """
    Text-to-Speech engine for converting predicted text to speech.
    
    Supports multiple TTS backends:
    - pyttsx3 (offline, cross-platform)
    - gTTS (online, Google TTS)
    - System TTS (platform-specific)
    """
    
    def __init__(
        self,
        engine: str = 'pyttsx3',
        rate: int = 150,
        volume: float = 1.0,
        voice: Optional[str] = None,
        save_audio: bool = False,
        audio_dir: str = 'results/audio'
    ):
        """
        Initialize TTS engine.
        
        Args:
            engine: TTS engine to use ('pyttsx3', 'gtts', or 'system')
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
            voice: Voice ID (engine-specific)
            save_audio: Whether to save audio files
            audio_dir: Directory to save audio files
        """
        self.engine_name = engine
        self.rate = rate
        self.volume = volume
        self.voice = voice
        self.save_audio = save_audio
        self.audio_dir = Path(audio_dir)
        
        if self.save_audio:
            self.audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize engine
        self.engine = self._initialize_engine()
        
        print(f"TTS Engine initialized: {engine}")
        print(f"  Rate: {rate} WPM")
        print(f"  Volume: {volume}")
        print(f"  Save audio: {save_audio}")
    
    def _initialize_engine(self):
        """Initialize the selected TTS engine."""
        if self.engine_name == 'pyttsx3':
            return self._initialize_pyttsx3()
        elif self.engine_name == 'gtts':
            return self._initialize_gtts()
        elif self.engine_name == 'system':
            return self._initialize_system_tts()
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine_name}")
    
    def _initialize_pyttsx3(self):
        """Initialize pyttsx3 engine."""
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
            
            if self.voice:
                engine.setProperty('voice', self.voice)
            
            return engine
            
        except ImportError:
            print("⚠️  pyttsx3 not installed. Install with: pip install pyttsx3")
            return None
        except Exception as e:
            print(f"⚠️  Error initializing pyttsx3: {e}")
            return None
    
    def _initialize_gtts(self):
        """Initialize gTTS engine."""
        try:
            from gtts import gTTS
            return gTTS  # gTTS is used per-call, not as a persistent engine
            
        except ImportError:
            print("⚠️  gTTS not installed. Install with: pip install gtts")
            return None
    
    def _initialize_system_tts(self):
        """Initialize system TTS."""
        # Platform-specific TTS initialization
        import platform
        
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            return 'say'
        elif system == 'Windows':
            return 'powershell'
        elif system == 'Linux':
            return 'espeak'
        else:
            print(f"⚠️  System TTS not supported on {system}")
            return None
    
    def speak(self, text: str, blocking: bool = True):
        """
        Convert text to speech.
        
        Args:
            text: Text to speak
            blocking: Whether to wait for speech to complete
        """
        if not text or not text.strip():
            return
        
        print(f"🔊 Speaking: '{text}'")
        
        if self.engine_name == 'pyttsx3':
            self._speak_pyttsx3(text, blocking)
        elif self.engine_name == 'gtts':
            self._speak_gtts(text)
        elif self.engine_name == 'system':
            self._speak_system(text)
    
    def _speak_pyttsx3(self, text: str, blocking: bool):
        """Speak using pyttsx3."""
        if self.engine is None:
            print("⚠️  TTS engine not available")
            return
        
        try:
            self.engine.say(text)
            
            if blocking:
                self.engine.runAndWait()
            
            # Save audio if requested
            if self.save_audio:
                audio_file = self.audio_dir / f"{text.replace(' ', '_')}.mp3"
                self.engine.save_to_file(text, str(audio_file))
                self.engine.runAndWait()
                print(f"  Audio saved to {audio_file}")
                
        except Exception as e:
            print(f"⚠️  Error speaking: {e}")
    
    def _speak_gtts(self, text: str):
        """Speak using gTTS."""
        if self.engine is None:
            print("⚠️  gTTS not available")
            return
        
        try:
            from gtts import gTTS
            import pygame
            
            # Create TTS object
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to temporary file
            audio_file = self.audio_dir / f"{text.replace(' ', '_')}.mp3" if self.save_audio else "temp_tts.mp3"
            tts.save(str(audio_file))
            
            # Play audio
            pygame.mixer.init()
            pygame.mixer.music.load(str(audio_file))
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            if not self.save_audio:
                os.remove(audio_file)
            else:
                print(f"  Audio saved to {audio_file}")
                
        except ImportError:
            print("⚠️  gTTS or pygame not installed")
        except Exception as e:
            print(f"⚠️  Error with gTTS: {e}")
    
    def _speak_system(self, text: str):
        """Speak using system TTS."""
        if self.engine is None:
            print("⚠️  System TTS not available")
            return
        
        try:
            import subprocess
            import platform
            
            system = platform.system()
            
            if system == 'Darwin':  # macOS
                subprocess.run(['say', text])
            elif system == 'Windows':
                command = f'Add-Type -AssemblyName System.Speech; $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; $speak.Speak("{text}")'
                subprocess.run(['powershell', '-Command', command])
            elif system == 'Linux':
                subprocess.run(['espeak', text])
                
        except Exception as e:
            print(f"⚠️  Error with system TTS: {e}")
    
    def speak_letter(self, letter: str):
        """
        Speak a single letter.
        
        Args:
            letter: Letter to speak (A-Z)
        """
        self.speak(f"Letter {letter}")
    
    def speak_word(self, word: str):
        """
        Speak a word.
        
        Args:
            word: Word to speak
        """
        self.speak(word)
    
    def spell_word(self, word: str, pause_between_letters: float = 0.5):
        """
        Spell out a word letter by letter.
        
        Args:
            word: Word to spell
            pause_between_letters: Pause duration between letters (seconds)
        """
        import time
        
        for letter in word:
            if letter.isalpha():
                self.speak_letter(letter)
                time.sleep(pause_between_letters)
    
    def get_available_voices(self) -> list:
        """
        Get list of available voices.
        
        Returns:
            List of voice IDs
        """
        if self.engine_name == 'pyttsx3' and self.engine:
            try:
                voices = self.engine.getProperty('voices')
                return [(v.id, v.name) for v in voices]
            except:
                return []
        return []
    
    def set_voice(self, voice_id: str):
        """
        Set the voice to use.
        
        Args:
            voice_id: Voice ID
        """
        if self.engine_name == 'pyttsx3' and self.engine:
            try:
                self.engine.setProperty('voice', voice_id)
                self.voice = voice_id
                print(f"Voice set to: {voice_id}")
            except Exception as e:
                print(f"⚠️  Error setting voice: {e}")


def demo_tts():
    """Demo TTS functionality."""
    print("Text-to-Speech Demo")
    print("=" * 60)
    
    # Try different engines
    engines = ['pyttsx3', 'system']
    
    for engine_name in engines:
        print(f"\nTesting {engine_name}:")
        print("-" * 60)
        
        try:
            tts = TextToSpeech(
                engine=engine_name,
                rate=150,
                volume=1.0,
                save_audio=False
            )
            
            # Test single letter
            print("\n1. Speaking single letter:")
            tts.speak_letter('A')
            
            # Test word
            print("\n2. Speaking word:")
            tts.speak_word('HELLO')
            
            # Test spelling
            print("\n3. Spelling word:")
            tts.spell_word('HI', pause_between_letters=0.3)
            
            print(f"\n✅ {engine_name} test complete!")
            break  # Use first working engine
            
        except Exception as e:
            print(f"⚠️  {engine_name} not available: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("TTS Demo Complete!")
    print("\nNote: Install TTS engines with:")
    print("  pip install pyttsx3")
    print("  pip install gtts pygame")


if __name__ == "__main__":
    demo_tts()
