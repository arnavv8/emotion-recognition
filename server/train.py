import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.audio_model import AudioEmotionModel
from utils.data_preprocessor import DataPreprocessor
from config import Config
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json

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

def convert_tensors_to_python(obj):
    """Convert tensors to Python native types for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, dict):
        return {str(key): convert_tensors_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_tensors_to_python(item) for item in obj]
    return obj

def save_classification_report(y_true, y_pred, labels, model_type):
    """Save classification report to JSON file."""

    print(f"Labels provided to classification_report: {labels}")
    print(f"Number of unique classes in dataset: {len(set(y_true))}")

    # Convert tensors to numpy arrays if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Convert labels to list of strings if they're tensors
    if isinstance(labels[0], torch.Tensor):
        labels = [str(label.item()) for label in labels]

    # Ensure that the labels are passed correctly as integers (use the 7 emotion labels)
    labels = [int(label) for label in labels]  

    # Generate classification report
    report = classification_report(y_true, y_pred, 
                                 labels=labels,  # Ensure this matches the number of classes (7)
                                 target_names=['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'],
                                 output_dict=True,
                                 zero_division=0)

    # Convert any remaining tensors to Python native types
    report = convert_tensors_to_python(report)

    print("True Labels: ", y_true)
    print("Predicted Labels: ", y_pred)

    metrics_dir = os.path.join(os.path.dirname(__file__), 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    report_path = os.path.join(metrics_dir, f'{model_type.lower()}_classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

    print(f"Classification report saved to {report_path}")

def evaluate_model(model, test_loader, labels, model_type, device):
    """Evaluate model and generate metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predictions = outputs.max(1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # Debugging: Check unique labels in dataset and predictions
    print(f"Unique labels in test set: {set(all_labels)}")
    print(f"Unique labels in predictions: {set(all_preds)}")
    print(f"Expected labels: {labels}")
    
    plot_confusion_matrix(all_labels, all_preds, labels, model_type)
    save_classification_report(all_labels, all_preds, labels, model_type)

def train_model(model, train_loader, val_loader, test_loader, labels, model_type, num_epochs=Config.NUM_EPOCHS, learning_rate=Config.LEARNING_RATE, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train the model and generate metrics."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    best_model = None
    
    metrics_dir = os.path.join(os.getcwd(), 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    training_metrics = {'epochs': [], 'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        training_metrics['epochs'].append(epoch + 1)
        training_metrics['train_loss'].append(avg_train_loss)
        training_metrics['val_loss'].append(avg_val_loss)
        training_metrics['val_accuracy'].append(accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
    
    metrics_path = os.path.join(metrics_dir, f'{model_type.lower()}_training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=4)
    
    model.load_state_dict(best_model)
    evaluate_model(model, test_loader, labels, model_type, device)
    return model

def main():
    preprocessor = DataPreprocessor()
    emotion_labels = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    
    print("Preparing audio dataset...")
    audio_features, audio_labels = preprocessor.prepare_audio_dataset()
    
    def split_data(features, labels):
        num_samples = len(labels)
        indices = torch.randperm(num_samples)
        train_size, val_size = int(0.7 * num_samples), int(0.15 * num_samples)
        return (
            (features[indices[:train_size]], labels[indices[:train_size]]),
            (features[indices[train_size:train_size+val_size]], labels[indices[train_size:train_size+val_size]]),
            (features[indices[train_size+val_size:]], labels[indices[train_size+val_size:]]),
        )
    
    (audio_train, audio_train_labels), (audio_val, audio_val_labels), (audio_test, audio_test_labels) = split_data(audio_features, audio_labels)
    batch_size = Config.BATCH_SIZE
    
    audio_train_loader = DataLoader(TensorDataset(audio_train, audio_train_labels), batch_size=batch_size, shuffle=True)
    audio_val_loader = DataLoader(TensorDataset(audio_val, audio_val_labels), batch_size=batch_size)
    audio_test_loader = DataLoader(TensorDataset(audio_test, audio_test_labels), batch_size=batch_size)
    
    print("Training audio model...")
    audio_model = AudioEmotionModel()
    trained_audio_model = train_model(audio_model, audio_train_loader, audio_val_loader, audio_test_loader, emotion_labels, 'Audio')
    
    os.makedirs(os.path.dirname(Config.AUDIO_MODEL_PATH), exist_ok=True)
    torch.save(trained_audio_model.state_dict(), Config.AUDIO_MODEL_PATH)
    print("Audio model training completed!")

if __name__ == '__main__':
    main()
