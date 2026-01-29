"""
Sign2Sound UI - Enhanced Flask Backend with Model Integration
Provides real-time ASL recognition through webcam
"""

from flask import Flask, render_template, jsonify, request, send_from_directory, Response
import os
import sys
import json
import time
import base64
import io
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.asl_model import create_model
from features.hand_landmarks import HandLandmarks
from data.vocabulary import ID_TO_LETTER, NUM_CLASSES

app = Flask(__name__, 
            static_folder='static',
            template_folder='.')

# Global model and extractor
model = None
extractor = None
device = None

def load_model():
    """Load the trained model"""
    global model, extractor, device
    
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load model checkpoint
        model_path = os.path.join('..', 'checkpoints', 'best_model.pth')
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}")
            return False
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Create model
        model = create_model(
            model_type='gru',
            input_dim=63,
            num_classes=NUM_CLASSES,
            hidden_size=128,
            num_layers=2,
            dropout=0.3
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully (Val Acc: {checkpoint['val_accuracy']:.4f})")
        
        # Initialize MediaPipe hand landmarker
        landmarker_path = os.path.join('..', 'models', 'hand_landmarker.task')
        if not os.path.exists(landmarker_path):
            print(f"Warning: Hand landmarker not found at {landmarker_path}")
            return False
        
        extractor = HandLandmarks(
            model_path=landmarker_path,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        print("MediaPipe Hand Landmarker initialized")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def preprocess_landmarks(landmarks, handedness='Right'):
    """Preprocess landmarks for model input - matches training preprocessing exactly"""
    from features.feature_utils import FeatureNormalizer
    
    # Create hand_data structure matching training format
    hand_data = {
        'type': handedness,
        'landmarks': landmarks
    }
    
    # Use the SAME normalization as training
    normalized_hand = FeatureNormalizer.normalize_hand_to_wrist(hand_data)
    normalized_landmarks = normalized_hand['landmarks']
    
    # Convert to numpy array (21 landmarks * 3 coordinates = 63 features)
    features = np.array([[lm['x'], lm['y'], lm['z']] for lm in normalized_landmarks])
    features = features.flatten()
    
    # Convert to tensor
    features = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0)
    
    return features

def predict(features):
    """Make prediction"""
    global model, device
    
    if model is None:
        return None, 0.0
    
    with torch.no_grad():
        features = features.to(device)
        outputs = model(features)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        return predicted.item(), confidence.item()

@app.route('/')
def index():
    """Serve the main UI"""
    return send_from_directory('.', 'index.html')

@app.route('/api/model/status')
def model_status():
    """Get model status"""
    global model, extractor
    
    return jsonify({
        'model_loaded': model is not None,
        'extractor_loaded': extractor is not None,
        'device': str(device) if device else 'unknown',
        'num_classes': NUM_CLASSES
    })

@app.route('/api/inference', methods=['POST'])
def inference():
    """Perform inference on webcam frame"""
    global model, extractor
    
    if model is None or extractor is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        # Get image from request
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided'
            }), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract landmarks
        detection_result, result = extractor.extract_landmarks(rgb_frame)
        
        if result and len(result) > 0:
            hand_data = result[0]
            landmarks = hand_data['landmarks']
            handedness = hand_data.get('handedness', 'Unknown')
            
            # Preprocess and predict (pass handedness for correct normalization)
            features = preprocess_landmarks(landmarks, handedness)
            class_id, confidence = predict(features)
            
            if class_id is not None:
                predicted_letter = ID_TO_LETTER[class_id]
                
                # Convert landmarks to list for JSON
                landmarks_list = [
                    {'x': lm['x'], 'y': lm['y'], 'z': lm['z']}
                    for lm in landmarks
                ]
                
                return jsonify({
                    'success': True,
                    'prediction': predicted_letter,
                    'confidence': float(confidence),
                    'handedness': handedness,
                    'landmarks': landmarks_list,
                    'hand_detected': True
                })
        
        # No hand detected
        return jsonify({
            'success': True,
            'hand_detected': False,
            'prediction': None,
            'confidence': 0.0
        })
        
    except Exception as e:
        print(f"Inference error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/metrics')
def get_metrics():
    """Get training metrics"""
    history_path = os.path.join('..', 'results', 'training_history.json')
    eval_path = os.path.join('..', 'results', 'evaluation_metrics.json')
    
    metrics = {}
    
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
            metrics['history'] = history
            if history.get('val_accuracy'):
                metrics['best_val_accuracy'] = max(history['val_accuracy'])
                metrics['final_train_accuracy'] = history['train_accuracy'][-1]
                metrics['epochs'] = len(history['train_accuracy'])
    
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            eval_metrics = json.load(f)
            metrics['evaluation'] = eval_metrics
    
    return jsonify(metrics)

@app.route('/api/preprocessing/status')
def get_preprocessing_status():
    """Get preprocessing status"""
    stats_path = os.path.join('..', 'data', 'processed', 'train', 'preprocessing_stats.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            return jsonify({
                'status': 'complete',
                'total_images': stats.get('total_images', 0),
                'successful': stats.get('successful', 0),
                'failed': stats.get('failed', 0),
                'success_rate': stats.get('successful', 0) / stats.get('total_images', 1) * 100
            })
    
    return jsonify({
        'status': 'not_started',
        'total_images': 0,
        'successful': 0,
        'failed': 0,
        'success_rate': 0
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    global model, extractor
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'extractor_loaded': extractor is not None
    })

if __name__ == '__main__':
    print("=" * 70)
    print("Sign2Sound UI Server with Model Integration")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    if load_model():
        print("[OK] Model loaded successfully!")
    else:
        print("[ERROR] Failed to load model. Inference will not work.")
    
    print("\nStarting server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
