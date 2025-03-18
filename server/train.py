import os
import json
from sklearn.utils import compute_class_weight
import torch
import numpy as np
import argparse
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from config import Config
from torchvision import models
from torchaudio.transforms import MelSpectrogram
from utils.data_preprocessor import DataPreprocessor
from models.audio_model import AudioEmotionModel
from models.video_model import VideoEmotionModel

def get_class_weights(labels):
    """ Compute class weights for balancing dataset """
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return torch.tensor(class_weights, dtype=torch.float)

def train_model(model, train_data, val_data, test_data, labels, model_type, dataset, device, k_folds=5):
    """ Train model with k-fold cross-validation and balanced dataset """

    # Get model path from Config
    if model_type == "Audio":
        model_save_path = Config.AUDIO_MODEL_RAVDESS_PATH if dataset == "ravdess" else Config.AUDIO_MODEL_CREMAD_PATH
    elif model_type == "Video":
        model_save_path = Config.VIDEO_MODEL_CREMAD_PATH
    else:
        raise ValueError("Invalid model type. Choose 'Audio' or 'Video'.")

    X_train = train_data["features"].clone().detach()
    y_train = train_data["labels"].clone().detach().cpu().squeeze()

    X_val = val_data["features"].clone().detach()
    y_val = val_data["labels"].clone().detach().cpu().squeeze()

    X_test = test_data["features"].clone().detach()
    y_test = test_data["labels"].clone().detach().cpu().squeeze()

    print(f"Train Features Shape: {X_train.shape}, Train Labels Shape: {y_train.shape}")
    print(f"Val Features Shape: {X_val.shape}, Val Labels Shape: {y_val.shape}")
    print(f"Test Features Shape: {X_test.shape}, Test Labels Shape: {y_test.shape}")

    best_val_accuracy = 0.0 
    best_model_state = None  

    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train.numpy(), y_train.numpy())):
        print(f"\n📌 Training Fold {fold+1}/{k_folds}")

        train_features, val_features = X_train[train_idx], X_train[val_idx]
        train_labels, val_labels = y_train[train_idx], y_train[val_idx]

        train_labels = train_labels.to(torch.long)
        val_labels = val_labels.to(torch.long)

        train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=Config.BATCH_SIZE, shuffle=True, pin_memory=True)
        val_loader = DataLoader(TensorDataset(val_features, val_labels), batch_size=Config.BATCH_SIZE, shuffle=False, pin_memory=True)


        # Reinitialize optimizer and scheduler for each fold
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=Config.LR_SCHEDULER_FACTOR,
                                                        patience=Config.LR_SCHEDULER_PATIENCE, min_lr=Config.LR_SCHEDULER_MIN_LR)
        criterion = nn.CrossEntropyLoss()

        # Reset model weights after each fold
        model.apply(model._init_weights)

        # Initialize mixed precision scaler
        scaler = torch.amp.GradScaler(enabled=(device.type == 'cuda'))
        torch.backends.cudnn.benchmark = True

        # Reset training history for this fold
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        best_fold_val_accuracy = 0.0  
        patience_counter = 0  

        for epoch in range(Config.NUM_EPOCHS):  
            model.train()
            total_loss, correct, total = 0, 0, 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                total_loss += loss.item()

            train_accuracy = 100 * correct / total
            val_accuracy, val_loss = validate_model(model, val_loader, criterion, device)

            train_losses.append(total_loss / len(train_loader))
            val_losses.append(val_loss)
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)

            print(f"Epoch [{epoch+1}] - Train Loss: {total_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

            if val_accuracy > best_fold_val_accuracy:
                best_fold_val_accuracy = val_accuracy
                best_model_state = model.state_dict()
                patience_counter = 0  
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(f"🔥 Early stopping at epoch {epoch + 1}")
                    break  

            scheduler.step(val_loss)

    if best_model_state:
        torch.save(best_model_state, model_save_path)
        print(f"✅ Model saved to {model_save_path}")

    model.load_state_dict(best_model_state)

    test_loader = DataLoader(TensorDataset(test_data["features"], test_data["labels"]), batch_size=Config.BATCH_SIZE, shuffle=False)

    # ✅ Pass training curves to evaluate_model
    test_accuracy = evaluate_model(model, test_loader, labels, model_type, dataset, device,
                                   train_losses=train_losses, val_losses=val_losses, train_accs=train_accs, val_accs=val_accs)

    return model, test_accuracy


def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    accuracy = 100 * correct / total
    return accuracy, total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description='Train Emotion Recognition Models')
    parser.add_argument('--dataset', type=str, choices=['ravdess', 'cremad'], required=True)
    parser.add_argument('--modality', type=str, choices=['audio', 'video', 'both'], required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using {device} for training")

    data_preprocessor = DataPreprocessor()
    if args.modality == "audio":
        train_data, val_data, test_data = data_preprocessor.prepare_audio_dataset(args.dataset)
    elif args.modality == "video":
        train_data, val_data, test_data = data_preprocessor.prepare_video_dataset(args.dataset)
    else:
        raise ValueError("Invalid modality. Choose 'audio' or 'video'.")

    labels = data_preprocessor.get_emotion_labels()
    num_classes = len(labels)

    if args.modality in ['audio', 'both']:
        audio_model = AudioEmotionModel(num_classes).to(device)
        print(f"\n🔊 Training Audio Model on {args.dataset.upper()}...")
        audio_model, acc = train_model(audio_model, train_data, val_data, test_data, labels, "Audio", args.dataset, device)
        print(f"🎯 Final Audio Model Accuracy: {acc:.2f}%")

    if args.modality in ['video', 'both']:
        video_model = VideoEmotionModel(num_classes).to(device)
        print(f"\n🎥 Training Video Model on {args.dataset.upper()}...")
        video_model, acc = train_model(video_model, train_data, val_data, test_data, labels, "Video", args.dataset, device)
        print(f"🎯 Final Video Model Accuracy: {acc:.2f}%")

def evaluate_model(model, test_loader, labels, model_type, dataset, device, 
                   train_losses=None, val_losses=None, train_accs=None, val_accs=None):
    """ Evaluate model and generate confusion matrix + classification report """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # Save confusion matrix and classification report
    save_confusion_matrix(all_labels, all_preds, labels, model_type, dataset)
    save_classification_report(all_labels, all_preds, labels, model_type, dataset)

    print(f"Labels length: {len(labels)}, Unique classes in y_true: {len(set(all_labels))}")
    print("Predictions:", np.unique(all_preds))
    print("Actual labels:", np.unique(all_labels))

    # ✅ Save training curves if the data is provided
    if train_losses is not None and val_losses is not None and train_accs is not None and val_accs is not None:
        save_training_curves(train_losses, val_losses, train_accs, val_accs, model_type, dataset)
        print("📈 Training curves saved!")

    return np.mean(np.array(all_preds) == np.array(all_labels))


def save_confusion_matrix(y_true, y_pred, labels, model_type, dataset):
    """ Save confusion matrix visualization """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(np.array(y_true), np.array(y_pred), labels=list(range(len(labels))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'{model_type} Model - {dataset.upper()} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'metrics/{model_type}_{dataset}_confusion_matrix.png')
    plt.close()

def save_classification_report(y_true, y_pred, labels, model_type, dataset):
    """ Save classification report as JSON """
    num_classes = len(set(y_true))  # Get the actual number of classes
    if len(labels) != num_classes:
        print(f"⚠️ Warning: Adjusting label size from {len(labels)} to {num_classes}")
        labels = [f"Class {i}" for i in range(num_classes)]  # Auto-generate correct labels

    report = classification_report(y_true, y_pred, target_names=labels[:num_classes], output_dict=True)

    with open(f'metrics/{model_type}_{dataset}_classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)


def save_training_curves(train_losses, val_losses, train_accs, val_accs, model_type, dataset):
    """ Save training curves for loss and accuracy """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_type} Model - {dataset.upper()} Loss Curve')
    plt.legend()
    plt.grid()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy', marker='o')
    plt.plot(epochs, val_accs, label='Validation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_type} Model - {dataset.upper()} Accuracy Curve')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'metrics/{model_type}_{dataset}_training_curves.png')
    plt.close()

def main():
    """ Main function to train models interactively """
    parser = argparse.ArgumentParser(description='Train Emotion Recognition Models')
    parser.add_argument('--dataset', type=str, choices=['ravdess', 'cremad'], required=True)
    parser.add_argument('--modality', type=str, choices=['audio', 'video', 'both'], required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using {device} for training")

    data_preprocessor = DataPreprocessor()
    if args.modality == "audio":
        train_data, val_data, test_data = data_preprocessor.prepare_audio_dataset(args.dataset)
    elif args.modality == "video":
        train_data, val_data, test_data = data_preprocessor.prepare_video_dataset(args.dataset)
    else:
        raise ValueError("Invalid modality. Choose 'audio' or 'video'.")

    labels = data_preprocessor.get_emotion_labels()
    num_classes = len(labels)

    if args.modality in ['audio', 'both']:
        audio_model = AudioEmotionModel(num_classes).to(device)
        print(f"\n🔊 Training Audio Model on {args.dataset.upper()}...")
        audio_model, acc = train_model(audio_model, train_data, val_data, test_data, labels, "Audio", args.dataset, device)
        print(f"🎯 Final Audio Model Accuracy: {acc:.2f}%")

    if args.modality in ['video', 'both']:
        video_model = VideoEmotionModel(num_classes).to(device)
        print(f"\n🎥 Training Video Model on {args.dataset.upper()}...")
        video_model, acc = train_model(video_model, train_data, val_data, test_data, labels, "Video", args.dataset, device)
        print(f"🎯 Final Video Model Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()
