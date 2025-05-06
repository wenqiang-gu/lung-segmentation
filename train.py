import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from tqdm import tqdm


# Dataset
class LungSegmentationDataset(Dataset):
    def __init__(self, list_file, transform=None):
        with open(list_file, 'r') as f:
            self.samples = [line.strip().split(',') for line in f]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path.strip()).convert("L")
        mask = Image.open(mask_path.strip()).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# Model
# The SimpleUNet class is now removed, UNet will be imported.
from model.unet import UNet


# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Datasets and loaders
train_dataset = LungSegmentationDataset('data/list_train.txt', transform)
test_dataset = LungSegmentationDataset('data/list_test.txt', transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Instantiate the new UNet model
# n_channels=1 for grayscale input, n_classes=1 for binary segmentation (lung/not lung)
model = UNet(n_channels=1, n_classes=1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Store IoU scores, Training Losses, and Evaluation Losses
epoch_ious = []
epoch_train_losses = []
epoch_eval_losses = []

# Configuration for saving predictions
num_epochs = 10
output_prediction_dir = "data/test/predictions"
os.makedirs(output_prediction_dir, exist_ok=True) # Create directory if it doesn't exist

# Training loop
for epoch in range(num_epochs): # Use num_epochs here
    model.train()
    total_loss = 0
    for img, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    epoch_train_losses.append(avg_loss) # Store training loss
    print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    ious = []
    total_eval_loss = 0.0
    with torch.no_grad():
        for i, (img_eval, mask_eval) in enumerate(test_loader): # Renamed for clarity
            img_eval = img_eval.to(device)
            mask_eval_device = mask_eval.to(device) # Mask to device for loss calculation

            raw_output_tensor = model(img_eval) # Get raw tensor output

            # Calculate evaluation loss
            eval_loss = criterion(raw_output_tensor, mask_eval_device)
            total_eval_loss += eval_loss.item()

            # For IoU calculation (using original mask_eval on CPU)
            output_for_iou = (raw_output_tensor.cpu().numpy() > 0.5)
            true_for_iou = mask_eval.cpu().numpy() > 0.5 # Ensure mask_eval is on CPU for numpy conversion
            iou = jaccard_score(true_for_iou.flatten(), output_for_iou.flatten())
            ious.append(iou)

            # Save predictions in the last epoch
            if epoch == num_epochs - 1:
                pred_np = raw_output_tensor.cpu().squeeze().numpy()
                pred_img_data = (pred_np * 255).astype(np.uint8)
                pred_image = Image.fromarray(pred_img_data, mode='L')
                pred_image.save(os.path.join(output_prediction_dir, f"prediction_{i}.png"))

    avg_eval_loss = total_eval_loss / len(test_loader)
    epoch_eval_losses.append(avg_eval_loss) # Store evaluation loss
    mean_iou = np.mean(ious)
    epoch_ious.append(mean_iou)
    print(f"Epoch {epoch+1} - Mean IoU: {mean_iou:.4f} - Eval Loss: {avg_eval_loss:.4f}")

# Plot IoU and Losses per epoch
plt.figure(figsize=(16, 6)) # Adjusted figure size for two subplots

# Subplot 1: Mean IoU
plt.subplot(1, 2, 1)
plt.plot(range(1, len(epoch_ious) + 1), epoch_ious, marker='o', color='b')
plt.title('Mean IoU per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Mean IoU')
plt.grid(True)

# Subplot 2: Losses
plt.subplot(1, 2, 2)
plt.plot(range(1, len(epoch_train_losses) + 1), epoch_train_losses, marker='o', linestyle='-', color='r', label='Training Loss')
plt.plot(range(1, len(epoch_eval_losses) + 1), epoch_eval_losses, marker='x', linestyle='--', color='g', label='Evaluation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
