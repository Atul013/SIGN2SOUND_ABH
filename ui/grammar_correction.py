"""
Small Language Model (SLM) Integration for Grammar Correction
Uses Hugging Face Transformers for text correction and improvement
"""

from flask import jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Global variables for model
grammar_model = None
grammar_tokenizer = None

def load_grammar_model():
    """Load a small language model for grammar correction"""
    global grammar_model, grammar_tokenizer
    
    try:
        print("Loading grammar correction model...")
        
        # Use DistilGPT-2 for lightweight grammar correction
        model_name = "distilgpt2"
        
        # Load tokenizer and model
        grammar_tokenizer = AutoTokenizer.from_pretrained(model_name)
        grammar_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad token
        if grammar_tokenizer.pad_token is None:
            grammar_tokenizer.pad_token = grammar_tokenizer.eos_token
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        grammar_model = grammar_model.to(device)
        grammar_model.eval()
        
        print(f"Grammar model loaded successfully on {device}")
        return True
        
    except Exception as e:
        print(f"Error loading grammar model: {e}")
        return False

def correct_grammar(text):
    """
    Correct grammar in ASL gloss text
    
    Args:
        text: Raw ASL gloss (e.g., "WHO EAT NOW")
        
    Returns:
        Corrected natural English (e.g., "Who is eating now?")
    """
    global grammar_model, grammar_tokenizer
    
    if not text or len(text.strip()) == 0:
        return text
    
    # If model not loaded, return simple corrections
    if grammar_model is None or grammar_tokenizer is None:
        return simple_grammar_correction(text)
    
    try:
        # Prepare prompt for grammar correction
        prompt = f"Correct this sentence to proper English: {text}\nCorrected:"
        
        # Tokenize
        inputs = grammar_tokenizer(prompt, return_tensors="pt", padding=True)
        device = next(grammar_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = grammar_model.generate(
                **inputs,
                max_new_tokens=50,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=grammar_tokenizer.pad_token_id
            )
        
        # Decode
        corrected = grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the corrected part (after "Corrected:")
        if "Corrected:" in corrected:
            corrected = corrected.split("Corrected:")[-1].strip()
        
        # Clean up
        corrected = corrected.split('\n')[0].strip()  # Take first line
        
        return corrected if corrected else simple_grammar_correction(text)
        
    except Exception as e:
        print(f"Error in grammar correction: {e}")
        return simple_grammar_correction(text)

def simple_grammar_correction(text):
    """
    Simple rule-based grammar correction as fallback
    """
    if not text:
        return ""
    
    # Convert to lowercase
    corrected = text.lower().strip()
    
    # Common ASL gloss to English patterns
    patterns = {
        # Question words
        'who': 'Who',
        'what': 'What',
        'where': 'Where',
        'when': 'When',
        'why': 'Why',
        'how': 'How',
        
        # Common verbs - add "is/are" for present continuous
        'eat': 'eating',
        'drink': 'drinking',
        'walk': 'walking',
        'run': 'running',
        'sleep': 'sleeping',
        'work': 'working',
        'play': 'playing',
        'read': 'reading',
        'write': 'writing',
        'learn': 'learning',
        
        # Time markers
        'now': 'now',
        'today': 'today',
        'tomorrow': 'tomorrow',
        'yesterday': 'yesterday',
    }
    
    words = corrected.split()
    result = []
    
    for i, word in enumerate(words):
        # Capitalize first word
        if i == 0 and word in patterns:
            result.append(patterns[word])
        elif word in patterns:
            result.append(patterns[word])
        else:
            result.append(word)
    
    # Join words
    corrected = ' '.join(result)
    
    # Capitalize first letter
    if corrected:
        corrected = corrected[0].upper() + corrected[1:]
    
    # Capitalize 'I'
    corrected = corrected.replace(' i ', ' I ')
    if corrected.startswith('i '):
        corrected = 'I' + corrected[1:]
    
    # Add period if missing
    if corrected and not corrected[-1] in '.!?':
        corrected += '.'
    
    return corrected

def add_grammar_routes(app):
    """Add grammar correction routes to Flask app"""
    
    @app.route('/api/correct_grammar', methods=['POST'])
    def api_correct_grammar():
        """API endpoint for grammar correction"""
        from flask import request
        
        try:
            data = request.get_json()
            text = data.get('text', '')
            
            if not text:
                return jsonify({
                    'success': False,
                    'error': 'No text provided'
                }), 400
            
            # Correct grammar
            corrected = correct_grammar(text)
            
            return jsonify({
                'success': True,
                'original': text,
                'corrected': corrected
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/grammar_status', methods=['GET'])
    def api_grammar_status():
        """Check if grammar model is loaded"""
        return jsonify({
            'loaded': grammar_model is not None,
            'model': 'distilgpt2' if grammar_model is not None else None
        })
