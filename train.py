import torch
import torch.nn as nn
import os  # Import the os module
from torch.optim import AdamW

from models.transformer import ViTModel
from utils.data_processing import get_train_val_loaders, get_test_loader
from config import Config

# Set device
device = torch.device(Config.device)

def train_model():
    # Set the seed for reproducibility
    torch.manual_seed(Config.seed)

    # Prepare DataLoaders
    train_loader, val_loader, _ = get_train_val_loaders(Config.batch_size, Config.val_size, Config.seed)

    # Initialize the model
    model = ViTModel(pretrained=Config.pretrained, num_classes=Config.num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)

    # Check if checkpoint directory exists, if not create it
    if not os.path.exists(Config.checkpoint_path):
        os.makedirs(Config.checkpoint_path)

    # Training Loop
    for epoch in range(Config.max_epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch+1}/{Config.max_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Validation step
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_acc = 100. * correct / total
        print(f'Validation Accuracy: {val_acc:.2f} %, Avg Loss: {val_loss/total:.4f}')

        # Save checkpoint
        if epoch % Config.save_frequency == 0:
            checkpoint_path = os.path.join(Config.checkpoint_path, f'model_epoch_{epoch}.pth')
            torch.save(model.state_dict(), checkpoint_path)

    print('Finished Training')

if __name__ == '__main__':
    train_model()
