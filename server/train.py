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
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels) 
    plt.title(f'Confusion Matrix - {model_type}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Define emotion labels explicitly within the function
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    # Set x and y tick labels to emotion labels
    plt.xticks(range(len(emotion_labels)), emotion_labels, rotation=45)
    plt.yticks(range(len(emotion_labels)), emotion_labels)

    metrics_dir = os.path.join(os.getcwd(), 'metrics')
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
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    labels = [int(label) for label in labels]
    unique_classes = sorted(set(y_true) | set(y_pred))
    
    class_names = ['angry', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

    report = classification_report(
        y_true, y_pred, labels=unique_classes, 
        target_names=[class_names[i] for i in unique_classes],
        output_dict=True, zero_division=0
    )

    report = convert_tensors_to_python(report)

    metrics_dir = os.path.join(os.getcwd(), 'metrics')
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

    plot_confusion_matrix(all_labels, all_preds, labels, model_type)
    save_classification_report(all_labels, all_preds, labels, model_type)

def train_model(model, train_loader, val_loader, test_loader, labels, model_type, num_epochs=Config.NUM_EPOCHS, learning_rate=Config.LEARNING_RATE, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train the model and save the best model for production."""

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_model_path = os.path.join(os.getcwd(), f'models\{model_type.lower()}_model.pth')

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
            torch.save(model.state_dict(), best_model_path)

    metrics_path = os.path.join(metrics_dir, f'{model_type.lower()}_training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=4)

    # Plot validation loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(training_metrics['epochs'], training_metrics['val_loss'], label='Validation Loss', marker='o', color='r')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    val_loss_path = os.path.join(metrics_dir, f'{model_type.lower()}_training_curves.png')
    plt.savefig(val_loss_path)
    plt.close()
    print(f"Validation loss curve saved to {val_loss_path}")

    # Plot accuracy curve
    plt.figure(figsize=(10, 5))
    plt.plot(training_metrics['epochs'], training_metrics['val_accuracy'], label='Validation Accuracy', marker='o', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Curve')
    plt.legend()
    plt.grid(True)
    accuracy_path = os.path.join(metrics_dir, f'{model_type.lower()}_training_accuracy.png')
    plt.savefig(accuracy_path)
    plt.close()
    print(f"Validation accuracy curve saved to {accuracy_path}")

    # Load the best model before evaluating and using it for production
    model.load_state_dict(torch.load(best_model_path, map_location=device,weights_only=True))
    print(f"Loaded best model from {best_model_path} for evaluation and production.")

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
    
    print("Audio model training completed!")

if __name__ == '__main__':
    main()
