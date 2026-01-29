"""
PyTorch Training Script for ASL Alphabet Recognition
With CUDA support and comprehensive training pipeline
"""

import os
import sys
import yaml
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.asl_model import create_model
from data.vocabulary import LETTER_TO_ID, ID_TO_LETTER, NUM_CLASSES


class ASLDataset(Dataset):
    """PyTorch Dataset for ASL landmarks"""
    
    def __init__(self, data_dir, split='train'):
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
                print(f"Warning: Unknown class {class_name}, skipping...")
                continue
            
            # Load all samples for this class
            for npy_file in class_dir.glob('*.npy'):
                self.samples.append(str(npy_file))
                self.labels.append(class_id)
        
        print(f"Loaded {len(self.samples)} samples from {split} set")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load landmark features
        features = np.load(self.samples[idx])
        label = self.labels[idx]
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])[0]
        
        # Add sequence dimension (batch, seq_len=1, features)
        features = features.unsqueeze(0)
        
        return features, label


def create_dataloaders(data_dir, batch_size=32, num_workers=4):
    """Create train, validation, and test dataloaders"""
    
    train_dataset = ASLDataset(data_dir, split='train')
    val_dataset = ASLDataset(data_dir, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


class Trainer:
    """Trainer for ASL Recognition Model"""
    
    def __init__(self, config_path='training/config.yaml', use_cuda=True):
        self.config = self.load_config(config_path)
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        print(f"\n{'='*70}")
        print(f"Device: {self.device}")
        if self.use_cuda:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"{'='*70}\n")
        
        self.setup_directories()
        self.setup_logging()
        
        # Create model
        self.model = self.build_model()
        self.model = self.model.to(self.device)
        
        # Create dataloaders
        self.train_loader, self.val_loader = self.create_dataloaders()
        
        # Setup training
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def load_config(self, config_path):
        """Load configuration from YAML"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config['training']['checkpoint']['checkpoint_dir'],
            self.config['logging']['log_dir'],
            'results'
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = Path(self.config['logging']['log_dir']) / f'training_{timestamp}.log'
        self.log(f"Training started at {datetime.now()}")
    
    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def build_model(self):
        """Build model from config"""
        model_config = self.config['model']
        model_type = model_config['type']
        model_params = model_config.get(model_type, {})
        
        self.log(f"Building {model_type.upper()} model...")
        model = create_model(
            model_type=model_type,
            input_dim=model_config['input_dim'],
            num_classes=model_config['num_classes'],
            **model_params
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log(f"Total parameters: {total_params:,}")
        self.log(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def create_dataloaders(self):
        """Create data loaders"""
        data_config = self.config['data']
        self.log("Creating data loaders...")
        
        train_loader, val_loader = create_dataloaders(
            data_dir=data_config['data_dir'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers']
        )
        
        self.log(f"Train batches: {len(train_loader)}")
        self.log(f"Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def create_optimizer(self):
        """Create optimizer"""
        opt_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        
        if opt_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr)
        elif opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        elif opt_name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def create_scheduler(self):
        """Create learning rate scheduler"""
        sched_config = self.config['training']['lr_schedule']
        sched_type = sched_config['type']
        
        if sched_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=sched_config['factor'],
                patience=sched_config['patience'],
                min_lr=sched_config['min_lr']
            )
        else:
            return None
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['training']['epochs']}")
        
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clipping']['enabled']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clipping']['max_norm']
                )
            
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc="Validating"):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        val_loss = running_loss / len(self.val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        # Compute precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        self.log(f"\nValidation Results:")
        self.log(f"  Loss: {val_loss:.4f}")
        self.log(f"  Accuracy: {val_acc:.4f}")
        self.log(f"  Precision: {precision:.4f}")
        self.log(f"  Recall: {recall:.4f}")
        self.log(f"  F1-Score: {f1:.4f}")
        
        return val_loss, val_acc
    
    def save_checkpoint(self, epoch, val_accuracy):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['training']['checkpoint']['checkpoint_dir'])
        
        # Save best model
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'config': self.config
            }, checkpoint_path)
            self.log(f"Saved best model (accuracy: {val_accuracy:.4f}) to {checkpoint_path}")
        
        # Save periodic checkpoint
        if epoch % self.config['training']['checkpoint']['save_frequency'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_accuracy': val_accuracy
            }, checkpoint_path)
            self.log(f"Saved checkpoint to {checkpoint_path}")
    
    def should_stop_early(self, val_accuracy):
        """Check early stopping"""
        if not self.config['training']['early_stopping']['enabled']:
            return False
        
        patience = self.config['training']['early_stopping']['patience']
        min_delta = self.config['training']['early_stopping']['min_delta']
        
        if val_accuracy < self.best_val_accuracy + min_delta:
            self.patience_counter += 1
        else:
            self.patience_counter = 0
        
        if self.patience_counter >= patience:
            self.log(f"Early stopping triggered after {patience} epochs without improvement")
            return True
        
        return False
    
    def train(self):
        """Main training loop"""
        self.log("\n" + "="*70)
        self.log("Starting Training")
        self.log("="*70)
        
        num_epochs = self.config['training']['epochs']
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_accuracy'].append(train_acc)
            
            self.log(f"\nEpoch {epoch}/{num_epochs}")
            self.log(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_accuracy'].append(val_acc)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_acc)
                current_lr = self.optimizer.param_groups[0]['lr']
                self.log(f"  Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_acc)
            
            # Early stopping
            if self.should_stop_early(val_acc):
                break
            
            # Clear CUDA cache
            if self.use_cuda:
                torch.cuda.empty_cache()
        
        # Save training history
        self.save_training_history()
        
        self.log("\n" + "="*70)
        self.log("Training Complete!")
        self.log(f"Best validation accuracy: {self.best_val_accuracy:.4f}")
        self.log("="*70)
    
    def save_training_history(self):
        """Save training history"""
        history_file = Path(self.config['logging']['log_dir']) / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        self.log(f"Training history saved to {history_file}")


def main():
    parser = argparse.ArgumentParser(description='Train ASL Recognition Model')
    parser.add_argument('--config', type=str, default='training/config.yaml', help='Config file path')
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use CUDA if available')
    parser.add_argument('--no-cuda', dest='use_cuda', action='store_false', help='Disable CUDA')
    args = parser.parse_args()
    
    print("="*70)
    print("ASL Alphabet Recognition - PyTorch Training")
    print("="*70)
    
    try:
        trainer = Trainer(config_path=args.config, use_cuda=args.use_cuda)
        trainer.train()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
