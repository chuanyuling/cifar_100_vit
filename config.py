# config.py
import torch
class Config:
    # Dataset
    dataset_path = './data/cifar-100-python'  # Path where CIFAR-100 dataset is stored
    batch_size = 64                           # Batch size for training and evaluation
    num_workers = 4                           # Number of subprocesses to use for data loading
    shuffle_dataset = True                    # Whether to shuffle the dataset for each epoch
    
    # Model
    model_name = 'ViTModel'                   # Name of the model class
    pretrained = True                         # Whether to use pretrained weights
    num_classes = 100                         # Number of classes in CIFAR-100
    image_size = 224                          # Input image size for the model
    
    # Training
    learning_rate = 3e-4                      # Learning rate for the optimizer
    weight_decay = 1e-4                       # Weight decay (L2 penalty)
    max_epochs = 30                          # Number of epochs to train
    
    # Checkpoints
    checkpoint_path = 'checkpoints'         # Directory to save model checkpoints
    save_frequency = 1                       # Frequency of saving model checkpoints (by epoch)

    # Evaluation
    test_batch_size = 64 # Batch size for evaluation on the test set
    val_size = 0.1  # 验证集的比例
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device to run the model on

    # Other
    seed = 42                                 # Seed for reproducibility

    # You can add more configuration options as needed for your project.
