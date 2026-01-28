"""
Training Callbacks for ASL Alphabet Recognition

This module provides callback functions for training monitoring and control.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class Callback:
    """Base callback class."""
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of an epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of an epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Called at the beginning of a batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of a batch."""
        pass
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training."""
        pass


class EarlyStopping(Callback):
    """
    Stop training when a monitored metric has stopped improving.
    """
    
    def __init__(
        self,
        monitor: str = 'val_accuracy',
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',
        verbose: bool = True
    ):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement after which training will be stopped
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' or 'max'
            verbose: Whether to print messages
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Check if training should stop."""
        if logs is None:
            return
        
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
        
        # Check if improved
        if self.mode == 'min':
            improved = current_value < (self.best_value - self.min_delta)
        else:
            improved = current_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = current_value
            self.wait = 0
            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} improved to {current_value:.4f}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} did not improve ({self.wait}/{self.patience})")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                if self.verbose:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    print(f"Best {self.monitor}: {self.best_value:.4f}")


class LearningRateScheduler(Callback):
    """
    Learning rate scheduler callback.
    """
    
    def __init__(
        self,
        schedule_type: str = 'reduce_on_plateau',
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-6,
        monitor: str = 'val_loss',
        verbose: bool = True
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            schedule_type: Type of schedule ('step', 'exponential', 'reduce_on_plateau')
            factor: Factor by which to reduce learning rate
            patience: Number of epochs with no improvement after which LR will be reduced
            min_lr: Minimum learning rate
            monitor: Metric to monitor
            verbose: Whether to print messages
        """
        super().__init__()
        self.schedule_type = schedule_type
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.monitor = monitor
        self.verbose = verbose
        
        self.wait = 0
        self.best_value = float('inf')
        self.current_lr = None
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Update learning rate if needed."""
        if logs is None:
            return
        
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
        
        if self.schedule_type == 'reduce_on_plateau':
            if current_value < self.best_value:
                self.best_value = current_value
                self.wait = 0
            else:
                self.wait += 1
                
                if self.wait >= self.patience:
                    # Reduce learning rate
                    new_lr = max(self.current_lr * self.factor, self.min_lr)
                    
                    if new_lr != self.current_lr:
                        self.current_lr = new_lr
                        self.wait = 0
                        
                        if self.verbose:
                            print(f"\nReducing learning rate to {new_lr:.6f}")


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_accuracy',
        save_best_only: bool = True,
        mode: str = 'max',
        verbose: bool = True
    ):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path to save checkpoints
            monitor: Metric to monitor
            save_best_only: Only save when metric improves
            mode: 'min' or 'max'
            verbose: Whether to print messages
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Save checkpoint if needed."""
        if logs is None:
            return
        
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
        
        # Check if should save
        should_save = False
        
        if self.save_best_only:
            if self.mode == 'min':
                should_save = current_value < self.best_value
            else:
                should_save = current_value > self.best_value
            
            if should_save:
                self.best_value = current_value
        else:
            should_save = True
        
        if should_save:
            filepath = self.filepath.format(epoch=epoch, **logs)
            
            if self.verbose:
                print(f"\nSaving checkpoint to {filepath}")
                print(f"{self.monitor}: {current_value:.4f}")
            
            # Save model (pseudo-code)
            # model.save(filepath)


class MetricsLogger(Callback):
    """
    Log training metrics to file.
    """
    
    def __init__(self, log_dir: str = 'results', filename: str = 'training_log.txt'):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory to save logs
            filename: Log filename
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename
        
        self.metrics_history = []
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Log metrics for this epoch."""
        if logs is None:
            return
        
        # Add epoch number
        log_entry = {'epoch': epoch, **logs}
        self.metrics_history.append(log_entry)
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Save complete metrics history."""
        history_file = self.log_dir / 'metrics_history.json'
        
        with open(history_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)


class ProgressBar(Callback):
    """
    Display progress bar during training.
    """
    
    def __init__(self, total_epochs: int):
        """
        Initialize progress bar.
        
        Args:
            total_epochs: Total number of epochs
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.start_time = None
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Start timer."""
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Update progress bar."""
        if logs is None:
            return
        
        # Calculate progress
        progress = epoch / self.total_epochs
        elapsed_time = time.time() - self.start_time
        eta = elapsed_time / progress - elapsed_time if progress > 0 else 0
        
        # Format metrics
        metrics_str = ' - '.join([f"{k}: {v:.4f}" for k, v in logs.items()])
        
        # Print progress
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        
        print(f"\rEpoch {epoch}/{self.total_epochs} [{bar}] - "
              f"ETA: {eta:.0f}s - {metrics_str}", end='')
        
        if epoch == self.total_epochs:
            print()  # New line at end


class CallbackList:
    """
    Container for managing multiple callbacks.
    """
    
    def __init__(self, callbacks: list):
        """
        Initialize callback list.
        
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
    
    def on_train_begin(self, logs: Optional[Dict] = None):
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict] = None):
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)


if __name__ == "__main__":
    print("Training Callbacks Demo")
    print("=" * 60)
    
    # Create callbacks
    callbacks = CallbackList([
        EarlyStopping(monitor='val_accuracy', patience=5),
        LearningRateScheduler(schedule_type='reduce_on_plateau', patience=3),
        ModelCheckpoint(filepath='checkpoints/model_epoch_{epoch}.pth'),
        MetricsLogger(log_dir='results'),
        ProgressBar(total_epochs=10)
    ])
    
    # Simulate training
    callbacks.on_train_begin()
    
    for epoch in range(1, 11):
        callbacks.on_epoch_begin(epoch)
        
        # Simulate epoch metrics
        logs = {
            'loss': np.random.uniform(0.5, 2.0),
            'accuracy': np.random.uniform(0.7, 0.95),
            'val_loss': np.random.uniform(0.3, 1.5),
            'val_accuracy': np.random.uniform(0.75, 0.98)
        }
        
        callbacks.on_epoch_end(epoch, logs)
    
    callbacks.on_train_end()
    
    print("\n\nCallbacks demonstration complete!")
