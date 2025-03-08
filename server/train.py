import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import json
import argparse
import sys
from models.audio_model import AudioEmotionModel
from models.video_model import VideoEmotionModel
from utils.data_preprocessor import DataPreprocessor
from config import Config

def create_data_loaders(data_dict, batch_size=Config.BATCH_SIZE):
    """Create data loaders from dictionary containing features and labels."""
    features = data_dict['features']
    labels = data_dict['labels']
    
    # Create dataset
    dataset = TensorDataset(features, labels)
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True if 'train' in str(data_dict) else False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )
    
    return data_loader

def display_menu():
    """Display interactive menu for dataset and modality selection."""
    print("\nEmotion Recognition Model Training")
    print("=================================")
    
    # Dataset selection
    print("\nAvailable Datasets:")
    print("1. RAVDESS")
    print("2. CREMA-D")
    
    while True:
        try:
            dataset_choice = input("\nSelect dataset (1-2): ").strip()
            if dataset_choice == '1':
                selected_dataset = 'ravdess'
                break
            elif dataset_choice == '2':
                selected_dataset = 'cremad'
                break
            else:
                print("Invalid choice. Please select 1 or 2.")
        except Exception:
            print("Invalid input. Please try again.")
    
    # Modality selection
    print("\nAvailable Modalities:")
    print("1. Audio only")
    print("2. Video only")
    print("3. Both audio and video")
    
    while True:
        try:
            modality_choice = input("\nSelect modality (1-3): ").strip()
            if modality_choice == '1':
                selected_modality = 'audio'
                break
            elif modality_choice == '2':
                selected_modality = 'video'
                break
            elif modality_choice == '3':
                selected_modality = 'both'
                break
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
        except Exception:
            print("Invalid input. Please try again.")
    
    return selected_dataset, selected_modality


def train_model(model, train_data, val_data, test_data, labels, model_type, dataset, device, 
                num_epochs=Config.NUM_EPOCHS, learning_rate=Config.LEARNING_RATE):
    """Train the model with improved GPU utilization and monitoring."""
    model = model.to(device)
    
    # Create data loaders
    train_loader = create_data_loaders(train_data)
    val_loader = create_data_loaders(val_data)
    test_loader = create_data_loaders(test_data)
    
    # Use weighted cross entropy loss for imbalanced classes
    if 'class_weights' in train_data:
        class_weights = train_data['class_weights'].to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print(f"Using weighted cross entropy loss with weights: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Use AdamW optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=Config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=Config.LR_SCHEDULER_FACTOR,
        patience=Config.LR_SCHEDULER_PATIENCE,
        min_lr=Config.LR_SCHEDULER_MIN_LR,
        verbose=True
    )
    
    # Enable cuDNN benchmarking for faster training
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # Use automatic mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    best_model_state = None
    patience = Config.EARLY_STOPPING_PATIENCE
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Track metrics per class
    num_classes = len(labels)
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Clear gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Use automatic mixed precision if available
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward pass and optimization with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP_VALUE)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training without mixed precision
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimization
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRADIENT_CLIP_VALUE)
                
                optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f}')
        
        train_loss /= batch_count
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Per-class accuracy
                c = (predicted == targets).squeeze()
                for i in range(targets.size(0)):
                    label = targets[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)
        
        # Print per-class accuracy
        print('\nValidation Accuracy per class:')
        for i in range(num_classes):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f'{labels[i]}: {class_acc:.2f}%')
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            
            # Save best model checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }
            
            # Save with dataset-specific name
            if model_type.lower() == 'audio':
                if dataset.lower() == 'ravdess':
                    model_path = os.path.join(os.path.dirname(__file__), 'models', 'audio_model_ravdess.pth')
                else:
                    model_path = os.path.join(os.path.dirname(__file__), 'models', 'audio_model_cremad.pth')
            else:  # video model
                model_path = os.path.join(os.path.dirname(__file__), 'models', 'video_model_cremad.pth')
                
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    # Load best model for evaluation
    model.load_state_dict(best_model_state)
    test_accuracy = evaluate_model(model, test_loader, labels, model_type, dataset, device)
    
    # Save metrics
    metrics = {
        "accuracy": float(test_accuracy),
        "val_accuracy": float(best_val_accuracy / 100),
        "val_loss": float(best_val_loss)
    }
    
    metrics_dir = os.path.join(os.path.dirname(__file__), 'metrics')
    metrics_file = os.path.join(metrics_dir, 'model_metrics.json')
    
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    
    model_key = f"{model_type.lower()}_{dataset.lower()}"
    all_metrics[model_key] = metrics
    
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    return model, test_accuracy

def evaluate_model(model, test_loader, labels, model_type, dataset, device):
    """Evaluate model performance on test set."""
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
    
    # Save evaluation metrics
    save_confusion_matrix(all_labels, all_preds, labels, model_type, dataset)
    save_classification_report(all_labels, all_preds, labels, model_type, dataset)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return accuracy

def save_confusion_matrix(y_true, y_pred, labels, model_type, dataset):
    """Save confusion matrix visualization."""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )
    
    plt.title(f'{model_type} Model Confusion Matrix ({dataset.upper()})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save figure
    metrics_dir = os.path.join(os.path.dirname(__file__), 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    plt.savefig(os.path.join(metrics_dir, f'{model_type.lower()}_{dataset.lower()}_confusion_matrix.png'))
    plt.close()

def save_classification_report(y_true, y_pred, labels, model_type, dataset):
    """Save classification report as JSON."""
    report = classification_report(
        y_true, 
        y_pred, 
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0
    )
    
    # Save report as JSON
    metrics_dir = os.path.join(os.path.dirname(__file__), 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, f'{model_type.lower()}_{dataset.lower()}_classification_report.json'), 'w') as f:
        json.dump(report, f, indent=4)

def main():
    """Main function to train models with interactive menu."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train emotion recognition models')
    parser.add_argument('--dataset', type=str, choices=['ravdess', 'cremad'],
                        help='Dataset to use for training (ravdess or cremad)')
    parser.add_argument('--modality', type=str, choices=['audio', 'video', 'both'],
                        help='Modality to train (audio, video, or both)')
    parser.add_argument('--non-interactive', action='store_true',
                        help='Run in non-interactive mode using command line arguments')
    args = parser.parse_args()
    
    # Determine if we should use interactive mode
    if args.non_interactive and args.dataset and args.modality:
        selected_dataset = args.dataset
        selected_modality = args.modality
    else:
        selected_dataset, selected_modality = display_menu()
    
    # Set active dataset in Config
    Config.set_active_dataset(selected_dataset)
    
    # Create directories
    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'metrics'), exist_ok=True)
    
    # Initialize data preprocessor
    data_preprocessor = DataPreprocessor()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dictionary to store model accuracies
    model_accuracies = {}
    
    # Train audio model if requested
    if selected_modality in ['audio', 'both']:
        print(f"\nTraining audio model on {selected_dataset.upper()} dataset...")
        
        try:
            # Prepare audio dataset
            audio_data = data_preprocessor.prepare_audio_dataset(selected_dataset)
            
            if audio_data:
                train_data, val_data, test_data = audio_data
                
                # Get emotion labels
                emotion_labels = data_preprocessor.get_emotion_labels()
                
                # Initialize and train audio model
                num_emotions = len(emotion_labels)
                audio_model = AudioEmotionModel(num_emotions=num_emotions)
                
                audio_model, audio_accuracy = train_model(
                    audio_model,
                    train_data,
                    val_data,
                    test_data,
                    emotion_labels,
                    'Audio',
                    selected_dataset,
                    device
                )
                
                model_accuracies[f'audio_{selected_dataset}'] = audio_accuracy
                print(f"\nAudio model ({selected_dataset}) test accuracy: {audio_accuracy:.4f}")
            else:
                print(f"Error: Failed to prepare audio dataset for {selected_dataset}")
        
        except Exception as e:
            print(f"Error training audio model: {str(e)}")
    
    # Train video model if requested and using CREMA-D
    if selected_modality in ['video', 'both'] and selected_dataset == 'cremad':
        print("\nTraining video model on CREMA-D dataset...")
        
        try:
            # Prepare video dataset
            video_data = data_preprocessor.prepare_video_dataset('cremad')
            
            if video_data:
                train_data, val_data, test_data = video_data
                
                # Get emotion labels
                emotion_labels = data_preprocessor.get_emotion_labels()
                
                # Initialize and train video model
                num_emotions = len(emotion_labels)
                video_model = VideoEmotionModel(num_emotions=num_emotions)
                
                video_model, video_accuracy = train_model(
                    video_model,
                    train_data,
                    val_data,
                    test_data,
                    emotion_labels,
                    'Video',
                    'cremad',
                    device
                )
                
                model_accuracies['video_cremad'] = video_accuracy
                print(f"\nVideo model (CREMA-D) test accuracy: {video_accuracy:.4f}")
            else:
                print("Error: Failed to prepare video dataset for CREMA-D")
        
        except Exception as e:
            print(f"Error training video model: {str(e)}")
    
    # Print summary of trained models
    print("\nModel Accuracy Summary:")
    for model_name, accuracy in model_accuracies.items():
        print(f"{model_name}: {accuracy:.4f}")

if __name__ == "__main__":
    main()