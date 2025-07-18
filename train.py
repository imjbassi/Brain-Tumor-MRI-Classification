# train.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import TumorClassifier
from data_loader import get_dataloaders

def train_one_epoch(model, device, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss

def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            # Count correct predictions
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
    return val_loss, correct

def main():
    parser = argparse.ArgumentParser(description='Train a brain tumor classifier.')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the directory containing training images (one subfolder per class).')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32).')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10).')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate (default: 0.001).')
    parser.add_argument('--output_model', type=str, default='tumor_model.pth',
                        help='Path to save the best model checkpoint.')
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    train_loader, val_loader, class_names = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    print(f'Classes: {class_names}')

    # Initialize model, loss, optimizer
    model = TumorClassifier(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, device, train_loader, criterion, optimizer)
        val_loss, correct = validate(model, device, val_loader, criterion)
        total_val = len(val_loader.dataset)
        val_acc = correct / total_val
        print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}')
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.output_model)
            print(f'Best model saved to {args.output_model}')

if __name__ == '__main__':
    main()
