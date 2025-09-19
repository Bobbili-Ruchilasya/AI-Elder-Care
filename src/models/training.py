"""
Model Training and Optimization Module
Implements training loops, loss functions, and hyperparameter optimization
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional
import optuna
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import logging
from tqdm import tqdm
import json
import os

from .ensemble_model import LonelinessDetectionModel

class LonelinessDataset(Dataset):
    """
    Dataset class for loneliness detection
    """
    
    def __init__(self, 
                 speech_features: List[np.ndarray],
                 texts: List[str],
                 labels: List[float],
                 tokenizer,
                 max_length: int = 512):
        self.speech_features = speech_features
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get speech features
        speech_feat = torch.FloatTensor(self.speech_features[idx])
        
        # Tokenize text
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'speech_features': speech_feat,
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.FloatTensor([self.labels[idx]])
        }

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining classification and regression objectives
    """
    
    def __init__(self, classification_weight: float = 0.7, regression_weight: float = 0.3):
        super(MultiTaskLoss, self).__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Classification loss (binary)
        binary_targets = (targets > 0.5).float()
        classification_loss = self.bce_loss(predictions, binary_targets)
        
        # Regression loss (continuous)
        regression_loss = self.mse_loss(predictions, targets)
        
        # Combined loss
        total_loss = (self.classification_weight * classification_loss + 
                     self.regression_weight * regression_loss)
        
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'regression_loss': regression_loss
        }

class ModelTrainer:
    """
    Training pipeline for loneliness detection model
    """
    
    def __init__(self, 
                 model: LonelinessDetectionModel,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 2e-5,
                 weight_decay: float = 0.01):
        
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.criterion = MultiTaskLoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'val_auc': []
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            speech_features = batch['speech_features'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            output = self.model(speech_features, input_ids, attention_mask)
            
            # Calculate loss
            loss_dict = self.criterion(output['predictions'], labels)
            loss = loss_dict['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
            
            predictions.extend(output['predictions'].detach().cpu().numpy())
            targets.extend(labels.detach().cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        # Convert to binary for accuracy calculation
        binary_preds = (predictions > 0.5).astype(int)
        binary_targets = (targets > 0.5).astype(int)
        accuracy = accuracy_score(binary_targets, binary_preds)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                speech_features = batch['speech_features'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                output = self.model(speech_features, input_ids, attention_mask)
                
                # Calculate loss
                loss_dict = self.criterion(output['predictions'], labels)
                loss = loss_dict['total_loss']
                
                # Update metrics
                total_loss += loss.item() * len(labels)
                total_samples += len(labels)
                
                predictions.extend(output['predictions'].detach().cpu().numpy())
                targets.extend(labels.detach().cpu().numpy())
        
        # Calculate epoch metrics
        avg_loss = total_loss / total_samples
        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        
        # Binary classification metrics
        binary_preds = (predictions > 0.5).astype(int)
        binary_targets = (targets > 0.5).astype(int)
        accuracy = accuracy_score(binary_targets, binary_preds)
        
        # AUC score
        try:
            auc = roc_auc_score(binary_targets, predictions)
        except ValueError:
            auc = 0.5  # Random performance if only one class present
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            binary_targets, binary_preds, average='binary', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 50,
              early_stopping_patience: int = 10,
              save_path: str = 'best_model.pth') -> Dict[str, List[float]]:
        """
        Complete training loop with early stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            scheduler.step(val_metrics['loss'])
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['val_auc'].append(val_metrics['auc'])
            
            # Log metrics
            self.logger.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_metrics': val_metrics
                }, save_path)
                
                self.logger.info(f"New best model saved with val_loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping after {epoch + 1} epochs")
                    break
        
        # Load best model
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self.training_history

class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna
    """
    
    def __init__(self, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 n_trials: int = 50):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_trials = n_trials
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def objective(self, trial):
        """Objective function for hyperparameter optimization"""
        
        # Suggest hyperparameters
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
        hidden_dim = trial.suggest_categorical('hidden_dim', [128, 256, 512])
        dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
        
        # Create model with suggested hyperparameters
        model = LonelinessDetectionModel(
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Create trainer
        trainer = ModelTrainer(
            model=model,
            device=self.device,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        # Train for a few epochs (quick evaluation)
        try:
            trainer.train(
                self.train_loader,
                self.val_loader,
                num_epochs=10,
                early_stopping_patience=5
            )
            
            # Evaluate final performance
            val_metrics = trainer.validate_epoch(self.val_loader)
            
            # Return metric to optimize (minimize validation loss)
            return val_metrics['loss']
            
        except Exception as e:
            print(f"Trial failed: {e}")
            return float('inf')
    
    def optimize(self) -> Dict:
        """Run hyperparameter optimization"""
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=self.n_trials)
        
        print(f"Best trial: {study.best_trial.value}")
        print(f"Best params: {study.best_trial.params}")
        
        return {
            'best_params': study.best_trial.params,
            'best_value': study.best_trial.value,
            'study': study
        }
