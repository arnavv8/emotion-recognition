import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from models.audio_model import AudioEmotionModel
from models.video_model import VideoEmotionModel
from utils.data_preprocessor import DataPreprocessor
from config import Config
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import torch.backends.cudnn as cudnn

def plot_confusion_matrix(y_true, y_pred, labels, model_type):
    """Plot confusion matrix with better visualization."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    # Ensure we're using only the labels that appear in the data
    valid_labels = sorted(set(y_true))
    label_names = [labels[i] for i in valid_labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=valid_labels)
    
    plt.figure(figsize=(12, 10), dpi=300)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        square=True,
        annot_kws={'size': 12, 'weight': 'bold'},
        cbar_kws={'shrink': .8}
    )
    
    plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')
    plt.yticks(rotation=0, fontsize=10, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=15)
    plt.ylabel('True Label', fontsize=12, fontweight='bold', labelpad=15)
    plt.title(f'Confusion Matrix - {model_type}', pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    metrics_dir = os.path.join(os.path.dirname(__file__), 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    plt.savefig(
        os.path.join(metrics_dir, f'{model_type.lower()}_confusion_matrix.png'),
        dpi=300,
        bbox_inches='tight',
        pad_inches=0.5
    )
    plt.close()

def save_classification_report(y_true, y_pred, labels, model_type):
    """Save detailed classification report to JSON file."""
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    
    # Ensure we're using only the labels that appear in the data
    valid_labels = sorted(set(y_true))
    label_names = [labels[i] for i in valid_labels]
    
    report = classification_report(
        y_true,
        y_pred,
        labels=valid_labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )
    
    metrics_dir = os.path.join(os.path.dirname(__file__), 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    report_path = os.path.join(metrics_dir, f'{model_type.lower()}_classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)


def evaluate_model(model, test_loader, labels, model_type, device):
    """Evaluate the model and generate comprehensive metrics."""
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
    
    # Debug information
    unique_true = np.unique(all_labels)
    unique_pred = np.unique(all_preds)
    print(f"Unique true labels: {unique_true}")
    print(f"Unique predicted labels: {unique_pred}")
    print("\nLabel distribution in test set:")
    for label in sorted(set(all_labels)):
        count = all_labels.count(label)
        if label < len(labels):
            print(f"{labels[label]}: {count}")
        else:
            print(f"Unknown label {label}: {count}")
    
    # Ensure we have the correct number of labels
    valid_labels = sorted(set(all_labels))
    plot_confusion_matrix(all_labels, all_preds, [labels[i] for i in valid_labels], model_type)
    save_classification_report(all_labels, all_preds, [labels[i] for i in valid_labels], model_type)
    
def train_model(model, train_loader, val_loader, test_loader, labels, model_type, device, 
                num_epochs=Config.NUM_EPOCHS, learning_rate=Config.LEARNING_RATE):
    """Train the model with improved GPU utilization and monitoring."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Cosine annealing scheduler with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,  # Restart every 5 epochs
        T_mult=2,  # Double the restart interval after each restart
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Enable cuDNN benchmarking for faster training
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    scaler = torch.amp.GradScaler('cuda')  # Automatic mixed precision
    best_val_loss = float('inf')
    best_model_state = None
    patience = 10
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
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Use automatic mixed precision
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
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
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        print(f'Learning Rate: {current_lr:.6f}')
        print(f'GPU Memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f}MB')
        
        if val_loss < best_val_loss:
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
            torch.save(
                checkpoint,
                os.path.join(os.path.dirname(__file__), 'models', f'best_{model_type.lower()}_model.pth')
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    # Plot training curves
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    metrics_dir = os.path.join(os.path.dirname(__file__), 'metrics')
    plt.savefig(os.path.join(metrics_dir, f'{model_type.lower()}_training_curves.png'))
    plt.close()
    
    # Load best model for evaluation
    model.load_state_dict(best_model_state)
    evaluate_model(model, test_loader, labels, model_type, device)
    
    return model

def main():
    # Enable deterministic behavior
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Get dataset selection from user
    print("\nAvailable datasets:")
    print("1. RAVDESS (Audio only)")
    print("2. CREMA-D (Audio and Video)")
    
    dataset_choice = input("Select dataset (1/2): ").strip()
    if dataset_choice == '1':
        dataset = 'ravdess'
        modality_options = ['1']  # Audio only
        print("\nRAVDESS dataset selected - Audio training only")
    elif dataset_choice == '2':
        dataset = 'cremad'
        print("\nSelect modality to train:")
        print("1. Audio")
        print("2. Video")
        print("3. Both Audio and Video")
        modality_options = ['1', '2', '3']
    else:
        print("Invalid selection. Please choose 1 or 2.")
        return
    
    modality_choice = input(f"Select modality ({'/'.join(modality_options)}): ").strip()
    if modality_choice not in modality_options:
        print("Invalid modality selection.")
        return
    
    # Train audio model
    if modality_choice in ['1', '3']:
        print(f"\nPreparing audio dataset from {dataset.upper()}...")
        audio_features, audio_labels = preprocessor.prepare_audio_dataset(dataset)
        
        # Move data to GPU if available
        audio_features = audio_features.to(device)
        audio_labels = audio_labels.to(device)
        
        # Create dataset
        audio_dataset = TensorDataset(audio_features, audio_labels)
        
        # Split dataset (70% train, 15% val, 15% test)
        total_size = len(audio_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            audio_dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=1,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False, 
            num_workers=1
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            num_workers=1
        )
        
        # Initialize and train audio model
        print("\nTraining audio model...")
        num_emotions = len(preprocessor.get_emotion_mapping())
        audio_model = AudioEmotionModel(num_emotions=num_emotions)
        
        trained_audio_model = train_model(
            model=audio_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            labels=preprocessor.get_emotion_labels(),
            model_type='Audio',
            device=device
        )
    
    # Train video model (CREMA-D only)
    if dataset == 'cremad' and modality_choice in ['2', '3']:
        print("\nPreparing video dataset from CREMA-D...")
        video_features, video_labels = preprocessor.prepare_video_dataset(dataset)
        
        # Move data to GPU if available
        video_features = video_features.to(device)
        video_labels = video_labels.to(device)
        
        # Create dataset
        video_dataset = TensorDataset(video_features, video_labels)
        
        # Split dataset (70% train, 15% val, 15% test)
        total_size = len(video_dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            video_dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            pin_memory=True,
            num_workers=1
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            pin_memory=True,
            num_workers=1
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            pin_memory=True,
            num_workers=1
        )
        
        # Initialize and train video model
        print("\nTraining video model...")
        num_emotions = len(preprocessor.get_emotion_mapping())
        video_model = VideoEmotionModel(num_emotions=num_emotions)
        
        trained_video_model = train_model(
            model=video_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            labels=preprocessor.get_emotion_labels(),
            model_type='Video',
            device=device
        )
    
    print("\nTraining complete! Models and metrics have been saved.")

if __name__ == '__main__':
    main()