# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 16:58:22 2025

@author: BlueSky
"""
import argparse
import os
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from data.myloader import create_loader
from confusion_matrix import plot_confusion_matrix
import pyramid_vig

def parse_args():
    parser = argparse.ArgumentParser(description='Document Forgery Detection Training')

    # Essential parameters
    parser = argparse.ArgumentParser(description="Simplified Training Script")
    parser.add_argument('--data', default='C:/Users/BlueSky/Documents/Spyder/docnet/document/', type=str, help="Path to dataset")
    parser.add_argument('--output', default='C:/Users/BlueSky/Documents/Spyder/docnet/result/', type=str, help="Path to save results")
    parser.add_argument('--img-size', type=int, default=224, metavar='N', help='Image patch size (default: None => model default)')

    parser.add_argument('--model', type=str,metavar='MODEL', default='custom_link_predictor', help="Model name")
    
    # Essential parameters
    parser.add_argument('--num-classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--workers', type=int, default=14, help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--eval-metric', default='top1', type=str, help='Evaluation metric')
    parser.add_argument('--device', default='cuda', help='Device for training (e.g., "cuda" or "cpu")')
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer type (e.g., "adam", "sgd")')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='weight decay (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--sched', default='cosine', type=str, help='LR scheduler (cosine, step, ...)')
    return parser.parse_args()

def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []
    print(device)
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    accuracy = correct / total
    conf_matrix = confusion_matrix(all_targets, all_preds)
    return total_loss / len(loader), accuracy, conf_matrix

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = correct / total
    print("Validation Labels:")
    print("Targets:", all_targets)
    print("Predictions:", all_preds)
    return total_loss / len(loader), accuracy, confusion_matrix(all_targets, all_preds), classification_report(all_targets, all_preds)

def plot_losses(history, output_dir):
    """
    Plot training and validation losses over epochs.
    """
    epochs = len(history['train_loss'])
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    output_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(output_path)
    plt.show()
    
def plot_accuracies(history, output_dir):
    """
    Plot training and validation accuracies over epochs.
    """
    epochs = len(history['train_acc'])
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Accuracy')
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid()
    output_path = os.path.join(output_dir, 'accuracy_plot.png')
    plt.savefig(output_path)
    plt.show()
    

def main():
    args = parse_args()

    # Create model
    model = create_model(
        args.model,
        num_classes=args.num_classes
    ).to(args.device)

    # Optimizer and scheduler
    optimizer = create_optimizer(args, model)
    scheduler, _ = create_scheduler(args, optimizer)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Data loaders
    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')
    train_loader = create_loader(
        torchvision.datasets.ImageFolder(train_dir),
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        is_training=True,
        num_workers=args.workers,
        distributed=False
    )
    
    val_loader = create_loader(
        torchvision.datasets.ImageFolder(val_dir),
        input_size=(3, 224, 224),
        batch_size=args.batch_size,
        is_training=False,
        num_workers=args.workers,
        distributed=False
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_metric = -1
    all_train_conf_matrices = []
    all_val_conf_matrices = []
    
    for epoch in range(args.epochs):
        # Training
        train_loss, train_acc, train_conf_matrix = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        all_train_conf_matrices.append(train_conf_matrix)
        
        # Validation
        val_loss, val_acc, val_conf_matrix, class_report = validate(model, val_loader, criterion, args.device)
        all_val_conf_matrices.append(val_conf_matrix)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        print("Confusion Matrix:")
        print("Classification Report:")
        print(class_report)

        if val_acc > best_metric:
            best_metric = val_acc
            os.makedirs(args.output, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pth'))
    
    """classes = ['Authentic', 'Forgery']
    for epoch in range(args.epochs):
        plot_confusion_matrix(all_train_conf_matrices[epoch], classes, args.output, epoch, is_train=True)
        plot_confusion_matrix(all_val_conf_matrices[epoch], classes, args.output, epoch, is_train=False)"""

        
    plot_losses(history,  args.output)
    plot_accuracies(history,  args.output)
    
    

    print(f"Best Validation Accuracy: {best_metric:.4f}")

    # Save best model
    torch.save(model.state_dict(), os.path.join(args.output, 'final_model.pth'))

if __name__ == '__main__':
    main()









"""
import argparse
import os
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from data.myloader import create_loader

def parse_args():
    parser = argparse.ArgumentParser(description='Document Forgery Detection Training')

    # Essential parameters
    parser.add_argument('--data', required=True, metavar='DIR', help='Path to dataset')
    parser.add_argument('--model', default='custom_link_predictor', type=str, help='Model name')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--eval-metric', default='top1', type=str, help='Evaluation metric')
    parser.add_argument('--device', default='cuda', help='Device for training (e.g., "cuda" or "cpu")')
    parser.add_argument('--output', default='./output', type=str, help='Directory to save outputs')

    return parser.parse_args()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    accuracy = correct / total
    return total_loss / len(loader), accuracy

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = correct / total
    return total_loss / len(loader), accuracy, all_targets, all_preds

def plot_metrics(history, output_dir):
    epochs = len(history['train_loss'])
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Accuracy')
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_plot.png'))
    plt.show()

def plot_confusion_matrix(targets, preds, classes, output_dir):
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.show()

def cross_validate(model_fn, dataset, args):
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}")
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = create_loader(train_subset, (3, 224, 224), args.batch_size, True, args.workers, False)
        val_loader = create_loader(val_subset, (3, 224, 224), args.batch_size, False, args.workers, False)

        model = model_fn().to(args.device)
        optimizer = create_optimizer(args, model)
        criterion = nn.CrossEntropyLoss()

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, args.device)

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch + 1}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

        val_loss, val_acc, targets, preds = validate(model, val_loader, criterion, args.device)
        results.append((val_loss, val_acc, targets, preds))

        plot_metrics(history, args.output)
        plot_confusion_matrix(targets, preds, dataset.classes, args.output)

    return results

def main():
    args = parse_args()

    # Create model
    model_fn = lambda: create_model(
        args.model,
        num_classes=args.num_classes
    )

    # Data loaders
    dataset = torchvision.datasets.ImageFolder(args.data)
    cross_val_results = cross_validate(model_fn, dataset, args)

    for fold, (val_loss, val_acc, targets, preds) in enumerate(cross_val_results):
        print(f"Fold {fold + 1} Results:")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
        print("Confusion Matrix:")
        cm = confusion_matrix(targets, preds)
        print(cm)
        print("Classification Report:")
        print(classification_report(targets, preds, target_names=dataset.classes))

if __name__ == '__main__':
    main()


"""