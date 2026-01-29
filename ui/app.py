"""
Sign2Sound UI - Flask Backend
Simple web server to serve the UI and provide API endpoints
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import json
import time
from datetime import datetime

app = Flask(__name__, 
            static_folder='static',
            template_folder='.')

# Global training status
training_status = {
    'status': 'idle',
    'phase': 'Not Started',
    'progress': 0,
    'epoch': 0,
    'total_epochs': 100,
    'train_accuracy': 0,
    'val_accuracy': 0,
    'train_loss': 0,
    'val_loss': 0,
    'time_elapsed': '0m 0s',
    'eta': '-',
    'best_accuracy': 0,
    'start_time': None
}

@app.route('/')
def index():
    """Serve the main UI"""
    return send_from_directory('.', 'index.html')

@app.route('/api/training/status')
def get_training_status():
    """Get current training status"""
    return jsonify(training_status)

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start training (placeholder)"""
    training_status['status'] = 'running'
    training_status['phase'] = 'Preprocessing'
    training_status['start_time'] = time.time()
    return jsonify({'success': True, 'message': 'Training started'})

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop training (placeholder)"""
    training_status['status'] = 'stopped'
    return jsonify({'success': True, 'message': 'Training stopped'})

@app.route('/api/inference', methods=['POST'])
def inference():
    """Perform inference on image (placeholder)"""
    # This would be replaced with actual model inference
    return jsonify({
        'success': True,
        'prediction': 'A',
        'confidence': 0.95,
        'landmarks': []
    })

@app.route('/api/metrics')
def get_metrics():
    """Get training metrics"""
    # Try to read from results/metrics.json if available
    metrics_path = os.path.join('..', 'results', 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return jsonify(json.load(f))
    
    return jsonify({
        'accuracy': 0,
        'loss': 0,
        'val_accuracy': 0,
        'val_loss': 0
    })

@app.route('/api/preprocessing/status')
def get_preprocessing_status():
    """Get preprocessing status"""
    # Try to read preprocessing stats
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
        'status': 'running',
        'total_images': 0,
        'successful': 0,
        'failed': 0,
        'success_rate': 0
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 70)
    print("Sign2Sound UI Server")
    print("=" * 70)
    print("\nStarting server...")
    print("Open your browser and navigate to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
