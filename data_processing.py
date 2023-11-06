import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


# Define a function to load CIFAR-100 dataset
def get_train_val_loaders(batch_size, val_size=0.1, random_seed=42):
    # Define standard transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by ViT
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Download and load the training dataset
    train_dataset = datasets.CIFAR100(root='datasets', train=True, download=True, transform=transform)

    # Split the dataset into training and validation sets
    train_indices, val_indices = train_test_split(
        list(range(len(train_dataset))),
        test_size=val_size,
        random_state=random_seed  # This should be random_state, not random_seed
    )

    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(train_dataset, val_indices)

    # Download and load the test dataset
    test_dataset = datasets.CIFAR100(root='datasets', train=False, download=True, transform=transform)

    # Prepare the data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_dataset


# Get the data loaders
def get_test_loader(batch_size):
    # Define standard transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size expected by ViT
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Download and load the test dataset
    test_dataset = datasets.CIFAR100(root='datasets', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
