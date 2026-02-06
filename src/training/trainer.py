#!/usr/bin/env python3
"""
Training module for Dog Emotion Recognition
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EmotionTrainer:
    """Trainer class for emotion recognition models"""
    
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, train_loader, val_loader, epochs, lr):
        """Train the model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data) if not isinstance(self.model(data), tuple) else self.model(data)[0]
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            history['train_loss'].append(train_loss / len(train_loader))
        
        return history
    
    def evaluate(self, test_loader, class_names):
        """Evaluate model on test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data) if not isinstance(self.model(data), tuple) else self.model(data)[0]
                
                if isinstance(output, tuple):
                    output = output[0]
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = correct / max(total, 1)
        return {'accuracy': accuracy, 'correct': correct, 'total': total}
    
    def save_checkpoint(self, epoch, val_acc, filename):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
        }, filename)
        print(f"Checkpoint saved: {filename}")
    
    def plot_training_history(self):
        """Plot training history"""
        pass
