import torch

checkpoint = torch.load('checkpoints/best_model.pth')
print(f"Best Model Information:")
print(f"  Epoch: {checkpoint['epoch']}")
print(f"  Validation Accuracy: {checkpoint['val_acc']:.2f}%")
print(f"  Model saved successfully!")
