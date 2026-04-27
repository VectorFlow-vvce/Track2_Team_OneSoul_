"""
Improved Segmentation Training Script
Changes vs original:
  1. All 11 classes including Flowers (600)
  2. Data augmentation (flip, rotation, colour jitter)
  3. AdamW optimizer instead of SGD
  4. Cosine LR scheduler
  5. Weighted CrossEntropyLoss for rare classes
  6. Saves BEST model (not just last epoch)
  7. 30 epochs
  8. Gradient clipping for stable training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import os
import random
from tqdm import tqdm

plt.switch_backend('Agg')


# ============================================================================
# Mask Conversion — ALL 11 classes including Flowers (600)
# ============================================================================

value_map = {
    0:     0,   # Background
    100:   1,   # Trees
    200:   2,   # Lush Bushes
    300:   3,   # Dry Grass
    500:   4,   # Dry Bushes
    550:   5,   # Ground Clutter
    600:   6,   # Flowers      <- was missing in original
    700:   7,   # Logs
    800:   8,   # Rocks
    7100:  9,   # Landscape
    10000: 10,  # Sky
}
n_classes = len(value_map)  # 11

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

# IMPROVEMENT: Weighted loss — rare classes get higher weight
class_weights = torch.tensor([
    0.5,   # Background     — not a real class, downweight
    1.2,   # Trees
    1.2,   # Lush Bushes
    1.0,   # Dry Grass
    1.2,   # Dry Bushes
    2.0,   # Ground Clutter — rare
    3.0,   # Flowers        — very rare
    3.0,   # Logs           — very rare
    2.0,   # Rocks
    0.5,   # Landscape      — dominant class, downweight
    0.8,   # Sky
], dtype=torch.float32)


def convert_mask(mask):
    """Convert raw mask pixel values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# IMPROVEMENT: Dataset with paired augmentation
# Original had ZERO augmentation — just resize + normalize
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, img_h, img_w, augment=False):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.img_h     = img_h
        self.img_w     = img_w
        self.augment   = augment
        self.data_ids  = sorted(os.listdir(self.image_dir))
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id   = self.data_ids[idx]
        img_path  = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask  = convert_mask(Image.open(mask_path))

        # Resize both to same size
        image = image.resize((self.img_w, self.img_h), Image.BILINEAR)
        mask  = mask.resize( (self.img_w, self.img_h), Image.NEAREST)

        # Augmentation (training only)
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask  = TF.hflip(mask)

            # Random rotation +-15 degrees
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
                mask  = TF.rotate(mask,  angle, interpolation=Image.NEAREST)

            # Color jitter on image only (NOT mask)
            if random.random() > 0.5:
                image = transforms.ColorJitter(
                    brightness=0.3, contrast=0.3,
                    saturation=0.3, hue=0.05
                )(image)

        image = self.normalize(TF.to_tensor(image))
        mask  = torch.from_numpy(np.array(mask)).long()

        return image, mask


# ============================================================================
# Model: Segmentation Head (ConvNeXt-style) — same architecture as original
# ============================================================================

class SegmentationHeadConvNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=7, padding=3),
            nn.GELU()
        )
        self.block = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=7, padding=3, groups=128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=1),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.block(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=11):
    pred   = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_per_class = []
    for c in range(num_classes):
        inter = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()
        iou_per_class.append((inter / union).item() if union > 0 else float('nan'))
    return np.nanmean(iou_per_class)


def compute_dice(pred, target, num_classes=11, smooth=1e-6):
    pred   = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    dice_per_class = []
    for c in range(num_classes):
        pi = (pred == c).float()
        ti = (target == c).float()
        dice_per_class.append(
            ((2 * (pi * ti).sum() + smooth) / (pi.sum() + ti.sum() + smooth)).item()
        )
    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    return (torch.argmax(pred, dim=1) == target).float().mean().item()


def evaluate_metrics(model, backbone, loader, device, num_classes=11):
    ious, dices, accs = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  Evaluating", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            out     = backbone.forward_features(imgs)["x_norm_patchtokens"]
            logits  = model(out)
            outputs = F.interpolate(logits, size=imgs.shape[2:],
                                    mode="bilinear", align_corners=False)
            ious.append(compute_iou(outputs, labels, num_classes))
            dices.append(compute_dice(outputs, labels, num_classes))
            accs.append(compute_pixel_accuracy(outputs, labels))
    model.train()
    return np.nanmean(ious), np.mean(dices), np.mean(accs)


# ============================================================================
# Plotting
# ============================================================================

def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.title('Loss vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png')); plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history['train_iou'], label='Train IoU')
    plt.plot(epochs, history['val_iou'],   label='Val IoU')
    plt.axhline(0.2478, color='red', linestyle='--', label='Baseline (0.2478)')
    plt.title('IoU vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('IoU')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png')); plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history['train_dice'], label='Train Dice')
    plt.plot(epochs, history['val_dice'],   label='Val Dice')
    plt.title('Dice vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('Dice')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png')); plt.close()

    print(f"Saved plots to '{output_dir}/'")


def save_history_to_file(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n" + "=" * 50 + "\n\n")
        f.write("Best Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou'])+1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice'])+1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc'])+1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss'])+1})\n")
        f.write("=" * 50 + "\n\nPer-Epoch History:\n" + "-" * 100 + "\n")
        headers = ['Epoch','Train Loss','Val Loss','Train IoU','Val IoU',
                   'Train Dice','Val Dice','Train Acc','Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")
        for i in range(len(history['train_loss'])):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i+1,
                history['train_loss'][i], history['val_loss'][i],
                history['train_iou'][i],  history['val_iou'][i],
                history['train_dice'][i], history['val_dice'][i],
                history['train_pixel_acc'][i], history['val_pixel_acc'][i]
            ))
    print(f"Saved metrics to {filepath}")


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Config
    batch_size = 2
    w          = int(((960 / 2) // 14) * 14)   # 476
    h          = int(((540 / 2) // 14) * 14)   # 266
    n_epochs   = 5                      # IMPROVED: was 10
    lr         = 6e-5                           # IMPROVED: AdamW lr
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    # Datasets
    data_dir = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir  = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'val')

    trainset     = MaskDataset(data_dir, img_h=h, img_w=w, augment=True)
    valset       = MaskDataset(val_dir,  img_h=h, img_w=w, augment=False)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(valset,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Training samples:   {len(trainset)}")
    print(f"Validation samples: {len(valset)}")
    print(f"Classes:            {n_classes}")
    print(f"Image size:         {h} x {w}")

    # Backbone
    print("Loading DINOv2 backbone...")
    backbone_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone_model.eval().to(device)
    print("Backbone loaded!")

    sample_imgs, _ = next(iter(train_loader))
    with torch.no_grad():
        out = backbone_model.forward_features(sample_imgs.to(device))["x_norm_patchtokens"]
    n_embedding = out.shape[2]
    print(f"Embedding dim: {n_embedding}")

    # Model
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding, out_channels=n_classes,
        tokenW=w // 14, tokenH=h // 14
    ).to(device)

    # IMPROVED: weighted loss + AdamW + cosine scheduler
    loss_fct  = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(classifier.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

    history = {k: [] for k in [
        'train_loss','val_loss','train_iou','val_iou',
        'train_dice','val_dice','train_pixel_acc','val_pixel_acc'
    ]}

    best_val_iou    = 0.0
    best_model_path = os.path.join(script_dir, "segmentation_head.pth")

    print(f"\nStarting training for {n_epochs} epochs...")
    print("=" * 80)

    for epoch in range(1, n_epochs + 1):
        # Train
        classifier.train()
        train_losses = []
        for imgs, labels in tqdm(train_loader,
                                 desc=f"Epoch {epoch:02d}/{n_epochs} [Train]",
                                 leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feat = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
            logits  = classifier(feat)
            outputs = F.interpolate(logits, size=imgs.shape[2:],
                                    mode="bilinear", align_corners=False)
            loss = loss_fct(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        classifier.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader,
                                     desc=f"Epoch {epoch:02d}/{n_epochs} [Val]",
                                     leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                feat    = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                logits  = classifier(feat)
                outputs = F.interpolate(logits, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)
                val_losses.append(loss_fct(outputs, labels).item())

        # Metrics
        train_iou, train_dice, train_acc = evaluate_metrics(
            classifier, backbone_model, train_loader, device, n_classes)
        val_iou, val_dice, val_acc = evaluate_metrics(
            classifier, backbone_model, val_loader, device, n_classes)

        scheduler.step()

        tl = np.mean(train_losses)
        vl = np.mean(val_losses)
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_acc)
        history['val_pixel_acc'].append(val_acc)

        # IMPROVED: save best model
        tag = ""
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(classifier.state_dict(), best_model_path)
            tag = "  <- BEST SAVED"

        print(f"Epoch {epoch:02d}/{n_epochs} | "
              f"Loss {tl:.4f}/{vl:.4f} | "
              f"IoU {train_iou:.4f}/{val_iou:.4f} | "
              f"Acc {train_acc:.4f}/{val_acc:.4f}{tag}")

    print("\nSaving plots and history...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"  Best Val IoU:  {best_val_iou:.4f}")
    print(f"  Baseline IoU:  0.2478")
    print(f"  Improvement:   {best_val_iou - 0.2478:+.4f}")
    print(f"  Model saved:   {best_model_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()