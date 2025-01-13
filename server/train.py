import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.audio_model import AudioEmotionModel
# from models.video_model import VideoEmotionModel  # Commented out for now
from utils.data_preprocessor import DataPreprocessor
from config import Config
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json

# Confusion matrix and classification report functions remain unchanged
def plot_confusion_matrix(y_true, y_pred, labels, model_type):
    """Plot confusion matrix and save to file."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {model_type}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    metrics_dir = os.path.join(os.path.dirname(__file__), 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    plt.savefig(os.path.join(metrics_dir, f'{model_type.lower()}_confusion_matrix.png'))
    plt.close()

def save_classification_report(y_true, y_pred, labels, model_type):
    """Save classification report to JSON file."""
    report = classification_report(y_true, y_pred, 
                                 target_names=labels, 
                                 output_dict=True)
    
    metrics_dir = os.path.join(os.path.dirname(__file__), 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    report_path = os.path.join(metrics_dir, f'{model_type.lower()}_classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

# Model evaluation and training functions remain unchanged
def evaluate_model(model, test_loader, labels, model_type, device):
    """Evaluate model and generate metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predictions = outputs.max(1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(targets.numpy())
    
    # Generate and save metrics
    plot_confusion_matrix(all_labels, all_preds, labels, model_type)
    save_classification_report(all_labels, all_preds, labels, model_type)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    labels: list,
    model_type: str,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> nn.Module:
    """Train the model and generate metrics."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    best_model = None
    
    metrics_dir = os.path.join(os.path.dirname(__file__), 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    training_metrics = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        accuracy = 100.*correct/total
        
        # Store metrics
        training_metrics['epochs'].append(epoch + 1)
        training_metrics['train_loss'].append(avg_train_loss)
        training_metrics['val_loss'].append(avg_val_loss)
        training_metrics['val_accuracy'].append(accuracy)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {accuracy:.2f}%')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
    
    # Save training metrics
    metrics_path = os.path.join(metrics_dir, f'{model_type.lower()}_training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=4)
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(training_metrics['epochs'], training_metrics['train_loss'], label='Train Loss')
    plt.plot(training_metrics['epochs'], training_metrics['val_loss'], label='Val Loss')
    plt.title(f'{model_type} Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(metrics_dir, f'{model_type.lower()}_training_curves.png'))
    plt.close()
    
    # Load best model and evaluate
    model.load_state_dict(best_model)
    evaluate_model(model, test_loader, labels, model_type, device)
    
    return model

def main():
    # Initialize data preprocessor
    preprocessor = DataPreprocessor()
    
    # Emotion labels
    emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    
    # Prepare audio dataset
    print("Preparing audio dataset...")
    audio_features, audio_labels = preprocessor.prepare_audio_dataset()
    
    # Split data (70% train, 15% val, 15% test)
    def split_data(features, labels):
        num_samples = len(labels)
        indices = torch.randperm(num_samples)
        
        train_size = int(0.7 * num_samples)
        val_size = int(0.15 * num_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:]
        
        return (
            (features[train_indices], labels[train_indices]),
            (features[val_indices], labels[val_indices]),
            (features[test_indices], labels[test_indices])
        )
    
    # Split audio dataset
    (audio_train, audio_train_labels), (audio_val, audio_val_labels), (audio_test, audio_test_labels) = \
        split_data(audio_features, audio_labels)
    
    # Create audio data loaders
    batch_size = 32
    
    audio_train_loader = DataLoader(
        TensorDataset(audio_train, audio_train_labels),
        batch_size=batch_size,
        shuffle=True
    )
    
    audio_val_loader = DataLoader(
        TensorDataset(audio_val, audio_val_labels),
        batch_size=batch_size
    )
    
    audio_test_loader = DataLoader(
        TensorDataset(audio_test, audio_test_labels),
        batch_size=batch_size
    )
    
    # Train audio model
    print("Training audio model...")
    audio_model = AudioEmotionModel()
    trained_audio_model = train_model(
        audio_model,
        audio_train_loader,
        audio_val_loader,
        audio_test_loader,
        emotion_labels,
        'Audio'
    )
    
    # Save audio model
    os.makedirs(os.path.dirname(Config.AUDIO_MODEL_PATH), exist_ok=True)
    torch.save(trained_audio_model.state_dict(), Config.AUDIO_MODEL_PATH)
    
    print("Audio model training completed!")

if __name__ == '__main__':
    main()
