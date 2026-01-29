"""
PyTorch Model Evaluation Script for ASL Alphabet Recognition
Generates comprehensive metrics, confusion matrix, and visualizations
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.asl_model import create_model
from data.vocabulary import LETTER_TO_ID, ID_TO_LETTER, NUM_CLASSES


class ASLDataset(Dataset):
    """PyTorch Dataset for ASL landmarks"""
    
    def __init__(self, data_dir, split='val'):
        self.data_dir = Path(data_dir) / 'processed' / split
        self.samples = []
        self.labels = []
        
        # Load all .npy files
        for class_dir in sorted(self.data_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Get class ID
            if class_name.upper() in LETTER_TO_ID:
                class_id = LETTER_TO_ID[class_name.upper()]
            elif class_name.lower() in LETTER_TO_ID:
                class_id = LETTER_TO_ID[class_name.lower()]
            else:
                continue
            
            # Load all samples for this class
            for npy_file in class_dir.glob('*.npy'):
                self.samples.append(str(npy_file))
                self.labels.append(class_id)
        
        print(f"Loaded {len(self.samples)} samples from {split} set")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features = np.load(self.samples[idx])
        label = self.labels[idx]
        
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])[0]
        features = features.unsqueeze(0)
        
        return features, label


def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions and labels"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('ASL Alphabet Recognition - Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_training_history(history_path, save_dir):
    """Plot training history curves"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc * 100 for acc in history['train_accuracy']], 
             'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, [acc * 100 for acc in history['val_accuracy']], 
             'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(save_dir) / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {save_path}")
    plt.close()


def save_per_class_metrics(y_true, y_pred, class_names, save_path):
    """Save per-class performance metrics"""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0
    )
    
    metrics = []
    for i, class_name in enumerate(class_names):
        metrics.append({
            'class': class_name,
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        })
    
    # Save as JSON
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nPer-class metrics saved to {save_path}")
    
    # Print summary
    print("\nPer-Class Performance:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for m in metrics:
        print(f"{m['class']:<10} {m['precision']:<12.4f} {m['recall']:<12.4f} "
              f"{m['f1_score']:<12.4f} {m['support']:<10}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate ASL Recognition Model')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Data directory')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'],
                        help='Dataset split to evaluate')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use CUDA if available')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"ASL Alphabet Recognition - Model Evaluation")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"{'='*70}\n")
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(args.model, map_location=device)
    
    # Create model architecture
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
    print(f"Model loaded successfully (Epoch {checkpoint['epoch']}, "
          f"Val Acc: {checkpoint['val_accuracy']:.4f})")
    
    # Create dataloader
    print(f"\nLoading {args.split} dataset...")
    dataset = ASLDataset(args.data_dir, split=args.split)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4)
    
    # Evaluate
    print("\nEvaluating model...")
    predictions, labels = evaluate_model(model, dataloader, device)
    
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"{'='*70}\n")
    
    # Get class names
    class_names = [ID_TO_LETTER[i] for i in range(NUM_CLASSES)]
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Generate confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(labels, predictions)
    plot_confusion_matrix(cm, class_names, results_dir / 'confusion_matrix.png')
    
    # Plot training history
    history_path = Path('results/training_history.json')
    if history_path.exists():
        print("\nGenerating training curves...")
        plot_training_history(history_path, results_dir)
    
    # Save per-class metrics
    print("\nComputing per-class metrics...")
    save_per_class_metrics(labels, predictions, class_names, 
                          results_dir / 'per_class_metrics.json')
    
    # Save overall metrics
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'total_samples': len(labels),
        'model_path': args.model,
        'split': args.split
    }
    
    metrics_path = results_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nOverall metrics saved to {metrics_path}")
    
    print(f"\n{'='*70}")
    print("Evaluation Complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
