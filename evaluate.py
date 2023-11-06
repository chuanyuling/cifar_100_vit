# evaluate.py
import torch
import torch.nn.functional as F
import os  # Make sure to import os module
from torch.utils.data import DataLoader
from models.transformer import ViTModel
from utils.data_processing import get_test_loader
from config import Config

def evaluate_model(model, test_loader, device):
    model.eval()  # Set the model to evaluation mode
    total = 0
    correct = 0
    with torch.no_grad():  # No need to track gradients for evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def main():
    device = torch.device(Config.device)

    # Prepare the test data loader
    test_loader = get_test_loader(batch_size=Config.test_batch_size)

    # Initialize the model
    model = ViTModel(num_classes=Config.num_classes, pretrained=False)  # No need to load pretrained weights
    model = model.to(device)

    # Load the trained model checkpoint
    # Ensure that this matches the saving pattern in your train.py
    checkpoint_filename = f'model_epoch_{Config.max_epochs - 1}.pth'
    checkpoint_path = os.path.join(Config.checkpoint_path, checkpoint_filename)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # Evaluate the model
    accuracy = evaluate_model(model, test_loader, device)
    print(f'Test Accuracy of the model on the test images: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    main()
