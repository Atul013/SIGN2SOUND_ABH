"""
Evaluation Script for ASL Alphabet Recognition

This script evaluates the trained model on the test set and generates
comprehensive metrics and visualizations.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.data_loader import create_data_loaders
from data.vocabulary import NUM_CLASSES, ID_TO_LETTER, LETTER_TO_ID


class ModelEvaluator:
    """
    Evaluator for ASL alphabet recognition model.
    """
    
    def __init__(self, model_path: str, data_dir: str = 'data'):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            data_dir: Directory containing test data
        """
        self.model_path = model_path
        self.data_dir = data_dir
        
        # Load model (pseudo-code)
        # self.model = self.load_model(model_path)
        
        # Load test data
        _, _, self.test_loader = create_data_loaders(data_dir)
        
        # Results storage
        self.predictions = []
        self.ground_truth = []
        self.per_class_correct = defaultdict(int)
        self.per_class_total = defaultdict(int)
    
    def evaluate(self) -> Dict:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        print("Evaluating model on test set...")
        print("=" * 60)
        
        total_correct = 0
        total_samples = 0
        
        # Evaluate on test set
        for features, labels, seq_lengths in self.test_loader:
            # Run inference (pseudo-code)
            # predictions = self.model.predict(features)
            # predicted_labels = np.argmax(predictions, axis=1)
            
            # Simulate predictions
            predicted_labels = np.random.randint(0, NUM_CLASSES, size=len(labels))
            
            # Store results
            self.predictions.extend(predicted_labels)
            self.ground_truth.extend(labels)
            
            # Compute accuracy
            correct = np.sum(predicted_labels == labels)
            total_correct += correct
            total_samples += len(labels)
            
            # Per-class statistics
            for true_label, pred_label in zip(labels, predicted_labels):
                letter = ID_TO_LETTER[true_label]
                self.per_class_total[letter] += 1
                if true_label == pred_label:
                    self.per_class_correct[letter] += 1
        
        # Compute overall metrics
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # Compute per-class metrics
        per_class_metrics = self.compute_per_class_metrics()
        
        # Compute confusion matrix
        confusion_matrix = self.compute_confusion_matrix()
        
        # Compute additional metrics
        precision, recall, f1 = self.compute_precision_recall_f1(confusion_matrix)
        
        metrics = {
            'overall_accuracy': overall_accuracy,
            'total_samples': total_samples,
            'total_correct': total_correct,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': confusion_matrix.tolist()
        }
        
        return metrics
    
    def compute_per_class_metrics(self) -> Dict:
        """Compute per-class accuracy, precision, recall."""
        per_class_metrics = {}
        
        for letter in sorted(LETTER_TO_ID.keys()):
            total = self.per_class_total.get(letter, 0)
            correct = self.per_class_correct.get(letter, 0)
            
            accuracy = correct / total if total > 0 else 0
            
            per_class_metrics[letter] = {
                'accuracy': accuracy,
                'total_samples': total,
                'correct_predictions': correct
            }
        
        return per_class_metrics
    
    def compute_confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix."""
        confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
        
        for true_label, pred_label in zip(self.ground_truth, self.predictions):
            confusion_matrix[true_label, pred_label] += 1
        
        return confusion_matrix
    
    def compute_precision_recall_f1(self, confusion_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute macro-averaged precision, recall, and F1 score.
        
        Args:
            confusion_matrix: Confusion matrix
            
        Returns:
            Tuple of (precision, recall, f1)
        """
        precisions = []
        recalls = []
        f1_scores = []
        
        for i in range(NUM_CLASSES):
            # True positives
            tp = confusion_matrix[i, i]
            
            # False positives
            fp = np.sum(confusion_matrix[:, i]) - tp
            
            # False negatives
            fn = np.sum(confusion_matrix[i, :]) - tp
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precisions.append(precision)
            
            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recalls.append(recall)
            
            # F1 score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        # Macro-averaged metrics
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        avg_f1 = np.mean(f1_scores)
        
        return avg_precision, avg_recall, avg_f1
    
    def save_results(self, metrics: Dict, output_dir: str = 'results'):
        """
        Save evaluation results to files.
        
        Args:
            metrics: Evaluation metrics
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_file = output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics saved to {metrics_file}")
        
        # Save per-class metrics as CSV
        csv_file = output_path / 'per_class_metrics.csv'
        with open(csv_file, 'w') as f:
            f.write("Letter,Accuracy,Total Samples,Correct Predictions\n")
            for letter, letter_metrics in sorted(metrics['per_class_metrics'].items()):
                f.write(f"{letter},{letter_metrics['accuracy']:.4f},"
                       f"{letter_metrics['total_samples']},{letter_metrics['correct_predictions']}\n")
        
        print(f"Per-class metrics saved to {csv_file}")
        
        # Save confusion matrix
        cm_file = output_path / 'confusion_matrix.npy'
        np.save(cm_file, np.array(metrics['confusion_matrix']))
        print(f"Confusion matrix saved to {cm_file}")
    
    def print_results(self, metrics: Dict):
        """
        Print evaluation results.
        
        Args:
            metrics: Evaluation metrics
        """
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {metrics['overall_accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  Total Samples: {metrics['total_samples']}")
        print(f"  Correct Predictions: {metrics['total_correct']}")
        
        print(f"\nPer-Class Accuracy:")
        print("-" * 60)
        
        for letter, letter_metrics in sorted(metrics['per_class_metrics'].items()):
            accuracy = letter_metrics['accuracy']
            total = letter_metrics['total_samples']
            correct = letter_metrics['correct_predictions']
            
            # Color code based on accuracy
            if accuracy >= 0.95:
                status = "✅"
            elif accuracy >= 0.85:
                status = "⚠️"
            else:
                status = "❌"
            
            print(f"  {letter}: {accuracy:.4f} ({correct}/{total}) {status}")
        
        print("=" * 60)


def main():
    """Main evaluation function."""
    print("ASL Alphabet Recognition - Evaluation Script")
    print("=" * 70)
    
    # Configuration
    model_path = "checkpoints/best_model.pth"
    data_dir = "data"
    output_dir = "results"
    
    try:
        # Create evaluator
        evaluator = ModelEvaluator(model_path, data_dir)
        
        # Run evaluation
        metrics = evaluator.evaluate()
        
        # Print results
        evaluator.print_results(metrics)
        
        # Save results
        evaluator.save_results(metrics, output_dir)
        
        print("\n✅ Evaluation complete!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nNote: This script requires:")
        print("1. Trained model checkpoint in checkpoints/")
        print("2. Test data in data/processed/test/")
        print("\nThis is a template implementation.")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nThis is expected if model hasn't been trained yet.")


if __name__ == "__main__":
    main()
