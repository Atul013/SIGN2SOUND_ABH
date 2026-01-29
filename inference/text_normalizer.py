"""
Text Normalization Module using Qwen2.5-0.5B SLM

This module provides lightweight text cleanup for sign-to-text output
before TTS synthesis. It fixes spelling, grammar, and punctuation without
changing meaning or adding/removing information.

Key Features:
- Qwen2.5-0.5B model quantized to INT4
- CPU-only execution for edge deployment
- Conservative decoding (low temperature, small max tokens)
- Single sentence output
- No cloud dependencies
"""

import os
import sys
import warnings
from typing import Optional
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TextNormalizer:
    """
    Text normalization using Qwen2.5-0.5B SLM (INT4 quantized).
    
    This class provides lightweight text cleanup for sign-to-text output:
    - Fixes spelling errors
    - Corrects grammar
    - Adds proper punctuation
    - Does NOT change meaning
    - Does NOT add or remove information
    - Outputs exactly one sentence
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cpu",
        load_in_4bit: bool = True,
        max_tokens: int = 50,
        temperature: float = 0.1,
        verbose: bool = True
    ):
        """
        Initialize the text normalizer.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ('cpu' only for edge deployment)
            load_in_4bit: Whether to use INT4 quantization
            max_tokens: Maximum tokens to generate (conservative)
            temperature: Sampling temperature (low for conservative output)
            verbose: Whether to print initialization messages
        """
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose
        
        self.model = None
        self.tokenizer = None
        
        # System prompt for text normalization
        self.system_prompt = """You are a text cleanup assistant. Your ONLY job is to fix spelling, grammar, and punctuation in the given text. 

Rules:
1. Fix spelling errors
2. Correct grammar mistakes
3. Add proper punctuation
4. Do NOT change the meaning
5. Do NOT add new information
6. Do NOT remove information
7. Output EXACTLY ONE sentence
8. Keep it concise and natural

Output only the corrected sentence, nothing else."""
        
        if self.verbose:
            print("🤖 Initializing Text Normalizer (Qwen2.5-0.5B INT4)...")
        
        self._load_model()
    
    def _load_model(self):
        """Load the Qwen2.5-0.5B model with INT4 quantization."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            import torch
            
            if self.verbose:
                print(f"  Loading model: {self.model_name}")
                print(f"  Device: {self.device}")
                print(f"  Quantization: INT4" if self.load_in_4bit else "  Quantization: None")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Configure quantization for edge deployment
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map=self.device,
                    trust_remote_code=True
                )
            
            self.model.eval()
            
            if self.verbose:
                print("✅ Model loaded successfully!")
                print(f"  Max tokens: {self.max_tokens}")
                print(f"  Temperature: {self.temperature}")
                print()
            
        except ImportError as e:
            print("❌ Error: Required packages not installed.")
            print("Install with: pip install transformers accelerate bitsandbytes")
            raise e
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e
    
    def normalize(self, text: str) -> str:
        """
        Normalize text using the SLM.
        
        Args:
            text: Raw text from sign-to-text output
            
        Returns:
            Cleaned, normalized text (single sentence)
        """
        if not text or not text.strip():
            return ""
        
        # Skip normalization if text is already clean and short
        if len(text.split()) <= 2 and text[0].isupper():
            return text.strip()
        
        try:
            # Create prompt
            user_message = f"Fix this text: {text.strip()}"
            
            # Format as chat
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate with conservative settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=True if self.temperature > 0 else False,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Clean output
            normalized = self._clean_output(generated_text)
            
            if self.verbose:
                print(f"📝 Normalized: '{text}' → '{normalized}'")
            
            return normalized
            
        except Exception as e:
            print(f"⚠️  Normalization error: {e}")
            # Fallback: return original text with basic cleanup
            return self._basic_cleanup(text)
    
    def _clean_output(self, text: str) -> str:
        """
        Clean the model output.
        
        - Trim at first newline
        - Trim at sentence end
        - Remove quotes
        - Strip whitespace
        """
        # Remove quotes
        text = text.replace('"', '').replace("'", '')
        
        # Trim at first newline
        if '\n' in text:
            text = text.split('\n')[0]
        
        # Trim at second sentence (keep only first)
        sentences = text.split('. ')
        if len(sentences) > 1:
            text = sentences[0] + '.'
        
        # Ensure proper capitalization
        text = text.strip()
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Ensure ends with punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def _basic_cleanup(self, text: str) -> str:
        """
        Basic text cleanup as fallback.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        text = text.strip()
        
        # Capitalize first letter
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        
        # Add period if missing
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def batch_normalize(self, texts: list) -> list:
        """
        Normalize multiple texts.
        
        Args:
            texts: List of texts to normalize
            
        Returns:
            List of normalized texts
        """
        return [self.normalize(text) for text in texts]


def demo_normalizer():
    """Demo text normalization functionality."""
    print("=" * 70)
    print("Text Normalizer Demo (Qwen2.5-0.5B INT4)")
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
        "help me please",
        "A B C",
        "HELLO"
    ]
    
    try:
        normalizer = TextNormalizer(verbose=True)
        
        print("Testing normalization:")
        print("-" * 70)
        
        for text in test_cases:
            normalized = normalizer.normalize(text)
            print(f"Input:  '{text}'")
            print(f"Output: '{normalized}'")
            print()
        
        print("=" * 70)
        print("Demo complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_normalizer()
