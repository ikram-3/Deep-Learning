"""
Professional Transformer Model for Text Classification/Generation
==================================================================
This module provides a complete pipeline for training transformer models on text data.
Includes data preprocessing, model architecture, training with callbacks, and evaluation.

Author: Professional ML Engineer
Version: 1.0.0
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd
from tqdm import tqdm
import yaml


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transformer_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Custom PyTorch Dataset for text data with tokenization."""
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: AutoTokenizer = None,
        max_length: int = 512,
        padding: str = 'max_length',
        truncation: bool = True
    ):
        """
        Initialize the text dataset.
        
        Args:
            texts: List of text samples
            labels: Optional list of labels for classification
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length' or 'longest')
            truncation: Whether to truncate sequences longer than max_length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        
        if self.labels is not None:
            encoding['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return encoding


class TransformerConfig:
    """Configuration class for transformer model training."""
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        task_type: str = 'classification',  # 'classification' or 'generation'
        num_labels: int = 2,
        max_length: int = 512,
        batch_size: int = 32,
        learning_rate: float = 2e-5,
        num_epochs: int = 10,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        output_dir: str = './outputs',
        evaluation_strategy: str = 'epoch',
        save_strategy: str = 'epoch',
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = 'accuracy',
        early_stopping_patience: int = 3,
        use_mixed_precision: bool = False,
        device: str = 'auto'
    ):
        self.model_name = model_name
        self.task_type = task_type
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.output_dir = output_dir
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.load_best_model_at_end = load_best_model_at_end
        self.metric_for_best_model = metric_for_best_model
        self.early_stopping_patience = early_stopping_patience
        self.use_mixed_precision = use_mixed_precision
        self.device = device
        
    def save_config(self, path: str):
        """Save configuration to YAML file."""
        config_dict = self.__dict__.copy()
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        logger.info(f"Configuration saved to {path}")
        
    @classmethod
    def load_config(cls, path: str) -> 'TransformerConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class TransformerTrainer:
    """Professional trainer for transformer models with comprehensive features."""
    
    def __init__(self, config: TransformerConfig):
        """
        Initialize the transformer trainer.
        
        Args:
            config: TransformerConfig object with training parameters
        """
        self.config = config
        self.set_seed(config.seed)
        self.device = self._get_device()
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Using device: {self.device}")
        
    def _get_device(self) -> torch.device:
        """Get the appropriate device for training."""
        if self.config.device == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info("Using GPU for training")
            elif torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("Using MPS (Apple Silicon) for training")
            else:
                device = torch.device('cpu')
                logger.info("Using CPU for training")
        else:
            device = torch.device(self.config.device)
            
        return device
    
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed set to {seed}")
    
    def load_tokenizer(self, tokenizer_name: Optional[str] = None):
        """Load the tokenizer."""
        tokenizer_name = tokenizer_name or self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logger.info(f"Loaded tokenizer: {tokenizer_name}")
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
    
    def load_model(self, model_name: Optional[str] = None):
        """Load the pre-trained transformer model."""
        model_name = model_name or self.config.model_name
        
        if self.config.task_type == 'classification':
            config = AutoConfig.from_pretrained(
                model_name,
                num_labels=self.config.num_labels
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                config=config
            )
        elif self.config.task_type == 'generation':
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported task type: {self.config.task_type}")
        
        self.model.to(self.device)
        logger.info(f"Loaded model: {model_name}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_data(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare and split data into train, validation, and test sets.
        
        Args:
            texts: List of text samples
            labels: Optional list of labels
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            random_state: Random seed for splitting
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.tokenizer is None:
            self.load_tokenizer()
        
        # Split data
        if labels is not None:
            # First split: train+val vs test
            texts_train_val, texts_test, labels_train_val, labels_test = train_test_split(
                texts, labels, test_size=test_size, random_state=random_state, stratify=labels
            )
            
            # Second split: train vs val
            val_ratio = val_size / (1 - test_size)
            texts_train, texts_val, labels_train, labels_val = train_test_split(
                texts_train_val, labels_train_val, 
                test_size=val_ratio, random_state=random_state, stratify=labels_train_val
            )
        else:
            texts_train, texts_test = train_test_split(
                texts, test_size=test_size, random_state=random_state
            )
            texts_train, texts_val = train_test_split(
                texts_train, test_size=val_size/(1-test_size), random_state=random_state
            )
            labels_train = labels_val = labels_test = None
        
        logger.info(f"Train size: {len(texts_train)}, Val size: {len(texts_val)}, Test size: {len(texts_test)}")
        
        # Create datasets
        self.train_dataset = TextDataset(
            texts_train, labels_train, self.tokenizer, 
            self.config.max_length
        )
        self.val_dataset = TextDataset(
            texts_val, labels_val, self.tokenizer, 
            self.config.max_length
        )
        self.test_dataset = TextDataset(
            texts_test, labels_test, self.tokenizer, 
            self.config.max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type != 'cpu' else False
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type != 'cpu' else False
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type != 'cpu' else False
        )
        
        return train_loader, val_loader, test_loader
    
    def compute_metrics(self, predictions, labels) -> Dict[str, float]:
        """Compute evaluation metrics."""
        preds = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        Train the model with comprehensive monitoring and callbacks.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        if self.model is None:
            self.load_model()
        
        # Setup optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate
        )
        
        # Setup scheduler
        num_training_steps = len(train_loader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Resume from checkpoint if provided
        start_epoch = 0
        best_metric = 0.0
        patience_counter = 0
        
        if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            checkpoint = torch.load(resume_from_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_metric = checkpoint.get('best_metric', 0.0)
            self.training_history = checkpoint.get('training_history', self.training_history)
        
        # Training loop
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Training steps per epoch: {len(train_loader)}")
        
        for epoch in range(start_epoch, self.config.num_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            logger.info(f"{'='*60}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} - Training")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass with gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                # Gradient clipping
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * self.config.gradient_accumulation_steps
                
                # Calculate accuracy
                if self.config.task_type == 'classification':
                    predictions = outputs.logits.detach().cpu().numpy()
                    labels = batch['labels'].detach().cpu().numpy()
                    preds = np.argmax(predictions, axis=1)
                    train_correct += (preds == labels).sum()
                    train_total += len(labels)
                
                progress_bar.set_postfix({'loss': f'{train_loss / (batch_idx + 1):.4f}'})
            
            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = train_correct / train_total if train_total > 0 else 0
            
            # Validation phase
            val_loss, val_metrics = self.evaluate(val_loader)
            
            # Update training history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_accuracy'].append(avg_train_acc)
            if 'accuracy' in val_metrics:
                self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Log results
            logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            if val_metrics:
                logger.info(f"Val Metrics: {val_metrics}")
            
            # Save checkpoint
            current_metric = val_metrics.get(self.config.metric_for_best_model, val_loss)
            if self.config.metric_for_best_model == 'loss':
                current_metric = -current_metric  # Lower loss is better
            
            is_best = current_metric > best_metric
            if is_best:
                best_metric = current_metric
                patience_counter = 0
                self.save_checkpoint(
                    epoch, optimizer, scheduler, 
                    f"{self.config.output_dir}/best_model.pth",
                    best_metric=best_metric
                )
                logger.info(f"✓ New best model saved! Metric: {current_metric:.4f}")
            else:
                patience_counter += 1
                logger.info(f"Patience: {patience_counter}/{self.config.early_stopping_patience}")
            
            # Regular checkpoint
            self.save_checkpoint(
                epoch, optimizer, scheduler,
                f"{self.config.output_dir}/checkpoint_epoch_{epoch}.pth",
                training_history=self.training_history
            )
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
        
        # Load best model
        if self.config.load_best_model_at_end:
            best_model_path = f"{self.config.output_dir}/best_model.pth"
            if os.path.exists(best_model_path):
                logger.info(f"Loading best model from {best_model_path}")
                checkpoint = torch.load(best_model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info("\nTraining completed!")
        return self.training_history
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            outputs = self.model(**batch)
            loss = outputs.loss
            
            total_loss += loss.item()
            
            if self.config.task_type == 'classification':
                predictions = outputs.logits.detach().cpu().numpy()
                labels = batch['labels'].detach().cpu().numpy()
                all_predictions.append(predictions)
                all_labels.append(labels)
        
        avg_loss = total_loss / len(data_loader)
        
        metrics = {}
        if all_predictions and self.config.task_type == 'classification':
            all_predictions = np.concatenate(all_predictions)
            all_labels = np.concatenate(all_labels)
            metrics = self.compute_metrics(all_predictions, all_labels)
        
        return avg_loss, metrics
    
    def predict(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Make predictions on new text data.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Predictions (probabilities or logits)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        self.model.eval()
        
        dataset = TextDataset(
            texts, None, self.tokenizer,
            self.config.max_length, padding='longest'
        )
        
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                if self.config.task_type == 'classification':
                    probs = torch.softmax(outputs.logits, dim=-1)
                    all_predictions.extend(probs.cpu().numpy())
                else:
                    all_predictions.extend(outputs.logits.cpu().numpy())
        
        return all_predictions if len(all_predictions) > 1 else all_predictions[0]
    
    def save_checkpoint(
        self,
        epoch: int,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        path: str,
        best_metric: Optional[float] = None,
        training_history: Optional[Dict] = None
    ):
        """Save a training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': self.config.__dict__,
            'training_history': training_history or self.training_history
        }
        
        if best_metric is not None:
            checkpoint['best_metric'] = best_metric
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def save_model(self, path: str):
        """Save the final model and tokenizer."""
        os.makedirs(path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(path)
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)
        
        # Save config
        self.config.save_config(os.path.join(path, 'training_config.yaml'))
        
        logger.info(f"Model and tokenizer saved to {path}")
    
    def generate_report(self, test_loader: DataLoader, output_path: str):
        """Generate a detailed evaluation report."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Generating predictions"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                
                predictions = outputs.logits.detach().cpu().numpy()
                labels = batch['labels'].detach().cpu().numpy()
                
                all_predictions.extend(np.argmax(predictions, axis=1))
                all_labels.extend(labels)
        
        # Generate classification report
        report = classification_report(all_labels, all_predictions, output_dict=True)
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Evaluation report saved to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
        print("="*60)
        
        return report


def load_data_from_file(file_path: str, text_column: str = 'text', label_column: str = 'label') -> Tuple[List[str], List[int]]:
    """
    Load text data from CSV, JSON, or TXT file.
    
    Args:
        file_path: Path to the data file
        text_column: Name of the text column (for CSV/JSON)
        label_column: Name of the label column (for CSV/JSON)
        
    Returns:
        Tuple of (texts, labels)
    """
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(file_path)
        texts = df[text_column].astype(str).tolist()
        labels = df[label_column].tolist() if label_column in df.columns else None
    elif file_ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        texts = [item[text_column] for item in data]
        labels = [item[label_column] for item in data if label_column in item]
        if not labels:
            labels = None
    elif file_ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f.readlines()]
        labels = None
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    return texts, labels


def main():
    """Example usage of the transformer trainer."""
    
    # Configuration
    config = TransformerConfig(
        model_name='bert-base-uncased',
        task_type='classification',
        num_labels=2,
        max_length=256,
        batch_size=16,
        learning_rate=2e-5,
        num_epochs=10,
        weight_decay=0.01,
        warmup_ratio=0.1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        seed=42,
        output_dir='./transformer_outputs',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        early_stopping_patience=3,
        use_mixed_precision=False
    )
    
    # Initialize trainer
    trainer = TransformerTrainer(config)
    
    # Example: Load data from file
    # texts, labels = load_data_from_file('data/train.csv', text_column='text', label_column='label')
    
    # Example: Use sample data
    texts = [
        "This is a positive example.",
        "This is another positive sentence.",
        "This is a negative example.",
        "This is another negative sentence.",
    ] * 100  # Repeat for demonstration
    labels = [1, 1, 0, 0] * 100  # Binary classification
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(
        texts, labels, test_size=0.2, val_size=0.1
    )
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Evaluate on test set
    test_loss, test_metrics = trainer.evaluate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Metrics: {test_metrics}")
    
    # Generate detailed report
    trainer.generate_report(test_loader, f"{config.output_dir}/evaluation_report.json")
    
    # Save final model
    trainer.save_model(f"{config.output_dir}/final_model")
    
    # Make predictions
    sample_texts = ["This is a great product!", "This is terrible."]
    predictions = trainer.predict(sample_texts)
    logger.info(f"Sample predictions: {predictions}")


if __name__ == '__main__':
    main()
