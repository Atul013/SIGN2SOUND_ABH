"""
ASL Alphabet Recognition - PyTorch Training Script with CUDA Support
Optimized for 4GB VRAM (RTX 3050)
"""

import os
import sys
import yaml
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vocabulary import NUM_CLASSES, ID_TO_LETTER, LETTER_TO_ID


class ASLDataset(Dataset):
    """Dataset for ASL landmarks"""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir) / 'processed' / split
        self.samples = []
        self.labels = []
        
        # Load all .npy files
        for class_dir in self.data_dir.iterdir():
            if class_dir.is_dir():
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
        # Load landmarks
        landmarks = np.load(self.samples[idx])
        landmarks = torch.FloatTensor(landmarks)
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return landmarks, label


class ASLModel(nn.Module):
    """Simple GRU model for ASL recognition"""
    
    def __init__(self, input_dim=63, hidden_size=128, num_layers=2, num_classes=29, dropout=0.3):
        super(ASLModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Reshape input to sequence
        self.fc_input = nn.Linear(input_dim, hidden_size)
        
        # GRU layers
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classifier
        self.dropout = nn.Dropout(dropout)
        self.fc_output = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # x shape: (batch, 63)
        batch_size = x.size(0)
        
        # Transform to sequence
        x = self.fc_input(x)  # (batch, hidden_size)
        x = x.unsqueeze(1)  # (batch, 1, hidden_size)
        
        # GRU
        out, _ = self.gru(x)  # (batch, 1, hidden_size)
        out = out[:, -1, :]  # (batch, hidden_size)
        
        # Classifier
        out = self.dropout(out)
        out = self.fc_output(out)  # (batch, num_classes)
        
        return out


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for features, labels in pbar:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(features)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for features, labels in pbar:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    return total_loss / len(val_loader), 100. * correct / total


def main():
    print("="*70)
    print("ASL Alphabet Recognition - PyTorch Training with CUDA")
    print("="*70)
    
    # Load config
    with open('training/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create directories
    Path('checkpoints').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = ASLDataset('data', split='train')
    val_dataset = ASLDataset('data', split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    print("\nCreating model...")
    model_config = config['model']['gru']
    model = ASLModel(
        input_dim=config['model']['input_dim'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_classes=config['model']['num_classes'],
        dropout=model_config['dropout']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['training']['lr_schedule']['factor'],
        patience=config['training']['lr_schedule']['patience'],
        min_lr=config['training']['lr_schedule']['min_lr']
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config['hardware']['mixed_precision'] else None
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(1, config['training']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['training']['epochs']}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, 'checkpoints/best_model.pth')
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if config['training']['early_stopping']['enabled']:
            if patience_counter >= config['training']['early_stopping']['patience']:
                print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
                break
        
        # Save periodic checkpoint
        if epoch % config['training']['checkpoint']['save_frequency'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, f'checkpoints/checkpoint_epoch_{epoch}.pth')
    
    # Save training history
    with open('results/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*70)


if __name__ == "__main__":
    main()
