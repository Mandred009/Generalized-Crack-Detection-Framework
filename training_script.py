
import yaml
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter as tb_SummaryWriter

class SummaryWriter:
    def __init__(self, log_dir: Path):
        log_dir.mkdir(exist_ok=True, parents=True)

        self.writer = tb_SummaryWriter(log_dir)
        self.iter = 0

    def add_scalars(self, named_scalars: dict, update_iter: bool = True):
        for scalar_name, scalar_value in named_scalars.items():
            self.writer.add_scalar(scalar_name, scalar_value, self.iter)
        
        if update_iter:
            self.iter += 1
            
    def close(self):
        self.writer.close()

class ImageDataset(Dataset):
    def __init__(self, dir: Path, label: int, transform=None):
        self.data_dir = dir
        self.images = list(dir.iterdir())
        self.transform = transforms.ToTensor() if transform is None else transform
        self.label = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')  # Ensure RGB format
        image = self.transform(image)
            
        return image, self.label

def train(
    model: torch.nn.Module, 
    train_loader: DataLoader, 
    criterion: nn.CrossEntropyLoss, 
    optimizer: optim.Optimizer, 
    device: str,
    writer: SummaryWriter
):
    '''
    Training function for ResNet
    '''
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        batch_correct = (predicted == labels).sum().item()
        correct += batch_correct
        update_iter = batch_idx < len(train_loader)-1
        writer.add_scalars({'train/batch/loss': float(loss.item()), 'train/batch/acc': float(batch_correct / labels.size(0))}, update_iter=update_iter)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    model.eval()

    return epoch_loss, epoch_acc

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module, 
    test_loader: DataLoader, 
    criterion: nn.CrossEntropyLoss, 
    device: str
):
    '''
    Evaluation function
    '''
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def visualize_predictions(model: torch.nn.Module, test_loader: DataLoader, device: str, num_samples:int=6):
    '''
    Test the model on a few samples
    '''
    is_training = model.training
    model.eval()
    images, labels = next(iter(test_loader))
    images = images[:num_samples].to(device)
    labels = labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = images[i].cpu()
        img = img * std + mean  # Denormalize
        img = torch.clamp(img, 0, 1)
        
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(f'True: {"Crack" if labels[i] == 1 else "No Crack"}\n'
                        f'Pred: {"Crack" if predicted[i] == 1 else "No Crack"}',
                        color='green' if labels[i] == predicted[i] else 'red')
        axes[i].axis('off')
    
    plt.tight_layout()
    fig.savefig('vis_pred.png')

    if is_training:
        model.train()
        
def create_image_dataset(
    data_dir: Path, 
    split_name: str, 
    dataset_name: str, 
    transform: transforms.Compose
)->Dataset:
    negative_data_path = data_dir / f'{dataset_name}/{split_name}/negative'
    positive_data_path = data_dir / f'{dataset_name}/{split_name}/positive'
    
    # Creating dataset objects with transforms
    negative_dataset = ImageDataset(negative_data_path, label=0, transform=transform)
    positive_dataset = ImageDataset(positive_data_path, label=1, transform=transform)

    # Combine datasets
    full_dataset = ConcatDataset([negative_dataset, positive_dataset])
    return full_dataset
        
def main(
    data_dir: Path,
    run_dir: Path,
    datasets: list[str],
    batch_size: int,
    lr: float,
    pretrained: bool,
    epochs: int,
    weight_decay: float,
    warm_epochs: int,
    lr_decay_step: int,
    lr_decay_gamma: float,
    seed: int,
    num_workers: int
):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed(seed)

    # Define transforms for preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    datasets_objects = {
        'train': [],
        'test': [],
        'val': [],
    }

    if 'CCIC' in datasets or 'all' in datasets:
        # Replace the paths with your dataset paths
        for split_name in datasets_objects:
            datasets_objects[split_name].append(create_image_dataset(data_dir, split_name, 'CCIC', transform))
            
    if 'SouthAfricanHighway' in datasets or 'all' in datasets:
        # Replace the paths with your dataset paths
        for split_name in datasets_objects:
            datasets_objects[split_name].append(create_image_dataset(data_dir, split_name, 'SouthAfricanHighway', transform))

    if 'GAPSv1' in datasets or 'GAPSv2' in datasets or 'all' in datasets:
        pass


    # Create data loaders
    train_loader = DataLoader(ConcatDataset(datasets_objects['train']), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(ConcatDataset(datasets_objects['val']), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(ConcatDataset(datasets_objects['test']), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load pretrained ResNet50 model
    model = models.resnet18(pretrained=pretrained)
    
    # Modify the final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: crack and no crack

    # Freeze early layers (optional - helps with transfer learning)
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the final layer
        for param in model.fc.parameters():
            param.requires_grad = True

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"Model loaded and moved to {device}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    if pretrained:
        optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=weight_decay) # Only optimize final layer initially
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # train the entire model

    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=lr, total_iters=warm_epochs)
    decay_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, decay_scheduler], milestones=[warm_epochs])

    # Training loop
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    best_val_acc = 0.0
    best_model_state = None

    out_file = run_dir / 'best_model.pth'
    out_file.parent.mkdir(exist_ok=True, parents=True)

    train_config = {
        'data_dir': str(data_dir.resolve()),
        'run_dir': str(run_dir.resolve()),
        'datasets': datasets,
        'batch_size': batch_size,
        'lr': lr,
        'pretrained': pretrained,
        'epochs': epochs,
        'weight_decay': weight_decay,
        'warm_epochs': warm_epochs,
        'lr_decay_step': lr_decay_step,
        'lr_decay_gamma': lr_decay_gamma,
        'seed': seed,
        'num_workers': num_workers,
    }
    train_config_path = run_dir / 'config.yaml'
    with train_config_path.open('w') as f:
        yaml.dump(train_config, f)

    writer = SummaryWriter(run_dir / 'tensorboard_events')

    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        
        # Unfreeze more layers after 5 epochs (optional)
        if epoch == 5 and pretrained:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Train
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, writer)
        writer.add_scalars({'train/loss': train_loss, 'train/acc': train_acc}, update_iter=False)

        # Evaluate
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        writer.add_scalars({'val/loss': val_loss, 'val/acc': val_acc}, update_iter=False)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
            torch.save(best_model_state, out_file)
        
        writer.add_scalars({'lr': scheduler.get_last_lr()[0]})
        scheduler.step()
    writer.close()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f'\nBest model loaded with validation accuracy: {best_val_acc:.2f}%')

    print('Evaluating model on test set')
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training, Validation, and Test Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.axhline(y=test_acc, color='r', linestyle='--', label='Test Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training, Validation, and Test Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig('metrics.png')

    # Visualize some predictions
    visualize_predictions(model, test_loader, device)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True, help='Path to top level dataset directory')
    parser.add_argument('--run-dir', type=Path, required=True, help='Path to training run output directory')
    parser.add_argument('--datasets', nargs='+', choices=['CCIC', 'GAPSv1', 'GAPSv2', 'SouthAfricanHighway', 'all'], default=['all'],
                        help='List of datasets to run (choices: CCIC, GAPSv1, GAPSv2, SouthAfricanHighway, all)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-workers', type=int, default=os.cpu_count()-1, help='Number of worker threads')
    
    # training params
    parser.add_argument('--batch-size', default=128, type=int, help='Training/eval batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--pretrained', action='store_true', help='Finetune last layer with IMAGENET1K_V1 backbone weights')
    parser.add_argument('--epochs', default=100, type=int, help='Total training epochs')
    parser.add_argument('--weight-decay', type=float, default=0, help='Weight decay, leave empty to not apply weight decay')
    parser.add_argument('--warmup-epochs', type=float, default=5, help='Learning rate warmup epochs')
    parser.add_argument('--lr-decay-step', type=int, default=5, help='Reduce learning rate after `step` epochs')
    parser.add_argument('--lr-decay-gamma', type=int, default=5, help='Learning rate decay rate (multiplicative factor)')
    args = parser.parse_args()
    
    main(
        args.data_dir, 
        args.run_dir,
        args.datasets,
        args.batch_size,
        args.lr,
        args.pretrained,
        args.epochs,
        args.weight_decay,
        args.warmup_epochs,
        args.lr_decay_step,
        args.lr_decay_gamma,
        args.seed,
        args.num_workers,
    )