import os
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from argparse import ArgumentParser

from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim

class CCIC(Dataset):
    def __init__(self, dir, label, transform=None):
        self.data_dir = dir
        self.images = os.listdir(dir)
        self.transform = transform
        self.label = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_dir, self.images[index])
        image = Image.open(image_path).convert('RGB')  # Ensure RGB format
        
        if self.transform:
            image = self.transform(image)
        else:
            # If no transform is provided, convert to tensor
            image = transforms.ToTensor()(image)
            
        return image, self.label
    
def main(
    data_dir: Path,
    run_dir: Path,
    datasets: list[str],
    batch_size: int,
    lr: float,
    pretrained: bool,
    epochs: int,
    val_epochs: int,
    weight_decay: float,
    warm_epochs: int,
    lr_decay_step: float,
    lr_decay_gamma: float
):
    # Define transforms for preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])  # ImageNet normalization
    ])

# Replace the paths with your dataset paths
negative_data_path = 'CCIC/Negative'
positive_data_path = 'CCIC/Positive'

# Creating dataset objects with transforms
negative_dataset = ImageDataset(negative_data_path, label=0, transform=transform)
positive_dataset = ImageDataset(positive_data_path, label=1, transform=transform)

# Getting the length of the datasets
negative_dataset_length = len(negative_dataset)
positive_dataset_length = len(positive_dataset)

print('Number of No crack examples:', negative_dataset_length)
print('Number of crack examples:', positive_dataset_length)

# Display sample images (without normalization for visualization)
display_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create temporary datasets for visualization
temp_negative = ImageDataset(negative_data_path, label=0, transform=display_transform)
temp_positive = ImageDataset(positive_data_path, label=1, transform=display_transform)

# Display random negative sample
random_index = random.randint(0, negative_dataset_length - 1)
img, label = temp_negative[random_index]
fig = plt.figure(figsize=(5, 5))
plt.imshow(img.permute(1, 2, 0))
plt.title(f'No Crack (Label: {label})')
plt.axis('off')
fig.savefig('no_crack.png')

# Display random positive sample
random_index = random.randint(0, positive_dataset_length - 1)
img, label = temp_positive[random_index]
fig = plt.figure(figsize=(5, 5))
plt.imshow(img.permute(1, 2, 0))
plt.title(f'Crack (Label: {label})')
plt.axis('off')
fig.savefig('crack.png')

# Combine datasets
full_dataset = ConcatDataset([negative_dataset, positive_dataset])

# Create indices for train/test split
indices = list(range(len(full_dataset)))
train_indices, test_indices = train_test_split(
    indices, 
    test_size=0.2, 
    random_state=42,
    shuffle=True
)

train_dataset = Subset(full_dataset, train_indices)
test_dataset = Subset(full_dataset, test_indices)

print(f'\nTotal dataset size: {len(full_dataset)}')
print(f'Training set size: {len(train_indices)}')
print(f'Test set size: {len(test_indices)}')

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Training function for ResNet
def train(model, train_loader, criterion, optimizer, device):
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
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
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

# Load pretrained ResNet50 model
model = models.resnet50(pretrained=True)

# Freeze early layers (optional - helps with transfer learning)
for param in model.parameters():
    param.requires_grad = False

# Modify the final layer for binary classification
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # 2 classes: crack and no crack

# Unfreeze the final layer
for param in model.fc.parameters():
    param.requires_grad = True

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

print(f"Model loaded and moved to {device}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)  # Only optimize final layer initially
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
num_epochs = 20
train_losses = []
train_accs = []
val_losses = []
val_accs = []

best_val_acc = 0.0
best_model_state = None

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    
    # Unfreeze more layers after 5 epochs (optional)
    if epoch == 5:
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Train
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    
    # Evaluate
    val_loss, val_acc = evaluate_model(model, test_loader, criterion, device)
    
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
    
    scheduler.step()

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f'\nBest model loaded with validation accuracy: {best_val_acc:.2f}%')

# Save the model
from datetime import datetime
from pathlib import Path
out_file = Path(f'runs/run_{datetime.now()}/crack_detection_resnet50.pth')
out_file.parent.mkdir(exist_ok=True, parents=True)
torch.save(model.state_dict(), out_file)

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(train_accs, label='Train Acc')
ax2.plot(val_accs, label='Val Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
fig.savefig('loss.png')

# Test the model on a few samples
def visualize_predictions(model, test_loader, device, num_samples=6):
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

# Visualize some predictions
visualize_predictions(model, test_loader, device)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=Path, required=True, help='Path to top level dataset directory')
    parser.add_argument('--run-dir', type=Path, required=True, help='Path to training run output directory')
    parser.add_argument('--datasets', nargs='+', default=['all'], help='List of datasets to run (choices: CCIC, GAPSv1, GAPSv2, SouthAfricanHighway, all)')
    
    # training params
    parser.add_argument('--batch-size', default=128, type=int, help='Training/eval batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--pretrained', action='store_true', help='Finetune last layer with IMAGENET1K_V1 backbone weights')
    parser.add_argument('--epochs', default=100, type=int, help='Total training epochs')
    parser.add_argument('--weight-decay', type=float, help='Weight decay, leave empty to not apply weight decay')
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
        args.val_epochs,
        args.weight_decay,
        args.warm_epochs,
        args.lr_decay_step,
        args.lr_decay_gamma,
    )