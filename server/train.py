import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
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
    """Plot an improved confusion matrix with better visualization."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, square=True, annot_kws={'size': 10}, cbar_kws={'shrink': .8})
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
    plt.ylabel('True Label', fontsize=12, labelpad=10)
    plt.title(f'Confusion Matrix - {model_type}', pad=20, fontsize=14)
    plt.tight_layout()

    metrics_dir = os.path.join(os.path.dirname(__file__),'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    plt.savefig(os.path.join(metrics_dir, f'{model_type.lower()}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_classification_report(y_true, y_pred, labels, model_type):
    """Save classification report to JSON file."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)

    metrics_dir = os.path.join(os.path.dirname(__file__),'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    report_path = os.path.join(metrics_dir, f'{model_type.lower()}_classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)

def evaluate_model(model, test_loader, labels, model_type, device):
    """Evaluate the model and generate metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    plot_confusion_matrix(all_labels, all_preds, labels, model_type)
    save_classification_report(all_labels, all_preds, labels, model_type)

def train_model(model, train_loader, val_loader, test_loader, labels, model_type, device, 
                num_epochs=Config.NUM_EPOCHS, learning_rate=Config.LEARNING_RATE):
    """Train the model with validation monitoring."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch + 1}/{num_epochs} | Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()

    # Save the best model
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(best_model_state, os.path.join(model_dir, 'audio_model.pth'))
    
    # Save loss and accuracy plots
    metrics_dir = os.path.join(os.path.dirname(__file__),'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_type} - Training & Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(metrics_dir, 'audio_training_curves.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_type} - Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(metrics_dir, 'audio_training_accuracy.png'), dpi=300)
    plt.close()

    # Save training metrics
    training_metrics = {
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": max(val_accuracies)
    }
    with open(os.path.join(metrics_dir, 'audio_training_metrics.json'), 'w') as f:
        json.dump(training_metrics, f, indent=4)

    model.load_state_dict(best_model_state)
    evaluate_model(model, test_loader, labels, model_type, device)

    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    preprocessor = DataPreprocessor()
    dataset = 'ravdess'  # Change dataset here: 'ravdess', 'cremad', 'both'

    if dataset == 'ravdess':
        emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    elif dataset == 'cremad':
        emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad']

    print(f"Preparing audio dataset from {dataset.upper()}...")
    audio_features, audio_labels = preprocessor.prepare_audio_dataset(dataset)

    dataset = TensorDataset(audio_features, audio_labels)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE)

    print("Training audio model...")
    audio_model = AudioEmotionModel().to(device)

    trained_audio_model=train_model(audio_model, train_loader, val_loader, test_loader, emotion_labels, 'audio', device)

    # Save the trained model
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(
        trained_audio_model.state_dict(),
        os.path.join(model_dir, 'audio_model.pth')
    )
    
    print("Training complete! Model saved and metrics generated.")

if __name__ == '__main__':
    main()
