"""
Improved Segmentation Test/Evaluation Script
Fixes vs original:
  1. All 11 classes including Flowers (600)
  2. Fixed plotting crash (FixedLocator mismatch bug)
  3. Prints per-class IoU clearly in terminal
  4. Shows improvement vs baseline
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
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

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

n_classes = len(value_map)  # 11

color_palette = np.array([
    [0,   0,   0  ],  # Background     - black
    [34,  139, 34 ],  # Trees          - forest green
    [0,   255, 0  ],  # Lush Bushes    - lime
    [210, 180, 140],  # Dry Grass      - tan
    [139, 90,  43 ],  # Dry Bushes     - brown
    [128, 128, 0  ],  # Ground Clutter - olive
    [255, 182, 193],  # Flowers        - pink
    [139, 69,  19 ],  # Logs           - saddle brown
    [128, 128, 128],  # Rocks          - gray
    [160, 82,  45 ],  # Landscape      - sienna
    [135, 206, 235],  # Sky            - sky blue
], dtype=np.uint8)


def convert_mask(mask):
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir     = os.path.join(data_dir, 'Color_Images')
        self.masks_dir     = os.path.join(data_dir, 'Segmentation')
        self.transform     = transform
        self.mask_transform = mask_transform
        self.data_ids      = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id   = self.data_ids[idx]
        image     = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask      = convert_mask(Image.open(os.path.join(self.masks_dir, data_id)))

        if self.transform:
            image = self.transform(image)
            mask  = self.mask_transform(mask) * 255

        return image, mask, data_id


# ============================================================================
# Model — must match training architecture exactly
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
    return np.nanmean(iou_per_class), iou_per_class


def compute_dice(pred, target, num_classes=11, smooth=1e-6):
    pred   = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    dice_per_class = []
    for c in range(num_classes):
        pi = (pred == c).float()
        ti = (target == c).float()
        dice_per_class.append(
            ((2*(pi*ti).sum() + smooth) / (pi.sum() + ti.sum() + smooth)).item()
        )
    return np.mean(dice_per_class), dice_per_class


def compute_pixel_accuracy(pred, target):
    return (torch.argmax(pred, dim=1) == target).float().mean().item()


# ============================================================================
# Visualization
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    img = img_tensor.cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    gt_color   = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);        axes[0].set_title('Input Image');  axes[0].axis('off')
    axes[1].imshow(gt_color);   axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(pred_color); axes[2].set_title('Prediction');   axes[2].axis('off')
    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_summary(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Save text file
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("EVALUATION RESULTS\n" + "=" * 50 + "\n")
        f.write(f"Mean IoU:      {results['mean_iou']:.4f}\n")
        f.write(f"Baseline IoU:  0.2478\n")
        f.write(f"Improvement:   {results['mean_iou'] - 0.2478:+.4f}\n")
        f.write("=" * 50 + "\n\nPer-Class IoU:\n" + "-" * 40 + "\n")
        for name, iou in zip(class_names, results['class_iou']):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A (not in test set)"
            f.write(f"  {name:<20}: {iou_str}\n")
    print(f"Saved evaluation metrics to {filepath}")

    # FIX: Per-class bar chart — was crashing due to 10 vs 11 class mismatch
    valid_iou  = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    num_bars   = len(valid_iou)   # always use actual length, not hardcoded number

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(
        range(num_bars),
        valid_iou,
        color=[color_palette[i] / 255 for i in range(num_bars)],
        edgecolor='black'
    )
    ax.set_xticks(range(num_bars))
    ax.set_xticklabels(class_names[:num_bars], rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU  |  Mean IoU: {results["mean_iou"]:.4f}  |  Baseline: 0.2478')
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_iou'], color='red',    linestyle='--', label=f'Mean ({results["mean_iou"]:.4f})')
    ax.axhline(y=0.2478,              color='orange', linestyle=':',  label='Baseline (0.2478)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'per_class_metrics.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class chart to '{chart_path}'")


# ============================================================================
# Main
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  default=os.path.join(script_dir, 'segmentation_head.pth'))
    parser.add_argument('--data_dir',    default=os.path.join(script_dir, '..', 'Offroad_Segmentation_testImages'))
    parser.add_argument('--output_dir',  default='./predictions')
    parser.add_argument('--batch_size',  type=int, default=2)
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device:     {device}")
    print(f"Number of classes: {n_classes}")

    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    print(f"Loading dataset from {args.data_dir}...")
    valset     = MaskDataset(args.data_dir, transform=transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(valset)} samples")

    print("Loading DINOv2 backbone...")
    backbone_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone_model.eval().to(device)
    print("Backbone loaded!")

    sample_img, _, _ = valset[0]
    with torch.no_grad():
        out = backbone_model.forward_features(sample_img.unsqueeze(0).to(device))["x_norm_patchtokens"]
    n_embedding = out.shape[2]
    print(f"Embedding dim: {n_embedding}")

    print(f"Loading model from {args.model_path}...")
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding, out_channels=n_classes,
        tokenW=w // 14, tokenH=h // 14
    )
    classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    classifier.to(device).eval()
    print("Model loaded!")

    masks_dir       = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    for d in [masks_dir, masks_color_dir, comparisons_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\nRunning evaluation on {len(valset)} images...")

    iou_scores, dice_scores, pixel_accuracies = [], [], []
    all_class_iou, all_class_dice = [], []
    sample_count = 0

    with torch.no_grad():
        for imgs, labels, data_ids in tqdm(val_loader, desc="Processing"):
            imgs, labels = imgs.to(device), labels.to(device)

            feat    = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
            logits  = classifier(feat)
            outputs = F.interpolate(logits, size=imgs.shape[2:],
                                    mode="bilinear", align_corners=False)

            labels_sq       = labels.squeeze(1).long()
            predicted_masks = torch.argmax(outputs, dim=1)

            iou,  class_iou  = compute_iou(outputs,  labels_sq, n_classes)
            dice, class_dice = compute_dice(outputs,  labels_sq, n_classes)
            pixel_acc        = compute_pixel_accuracy(outputs, labels_sq)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            all_class_iou.append(class_iou)
            all_class_dice.append(class_dice)

            for i in range(imgs.shape[0]):
                base = os.path.splitext(data_ids[i])[0]
                pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)

                Image.fromarray(pred_mask).save(
                    os.path.join(masks_dir, f'{base}_pred.png'))
                cv2.imwrite(
                    os.path.join(masks_color_dir, f'{base}_pred_color.png'),
                    cv2.cvtColor(mask_to_color(pred_mask), cv2.COLOR_RGB2BGR))

                if sample_count < args.num_samples:
                    save_prediction_comparison(
                        imgs[i], labels_sq[i], predicted_masks[i],
                        os.path.join(comparisons_dir, f'sample_{sample_count}_comparison.png'),
                        data_ids[i]
                    )
                sample_count += 1

    mean_iou      = float(np.nanmean(iou_scores))
    avg_class_iou = np.nanmean(all_class_iou, axis=0)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean IoU:     {mean_iou:.4f}")
    print(f"Baseline IoU: 0.2478")
    print(f"Improvement:  {mean_iou - 0.2478:+.4f}")
    print("=" * 50)
    print("\nPer-Class IoU:")
    for name, iou in zip(class_names, avg_class_iou):
        bar = "█" * int(iou * 20) if not np.isnan(iou) else ""
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        print(f"  {name:<20} {iou_str}  {bar}")

    results = {'mean_iou': mean_iou, 'class_iou': avg_class_iou}
    save_metrics_summary(results, args.output_dir)

    print(f"\nDone! Outputs saved to {args.output_dir}/")
    print(f"  masks/           - raw prediction masks")
    print(f"  masks_color/     - coloured prediction masks")
    print(f"  comparisons/     - {args.num_samples} side-by-side comparison images")
    print(f"  evaluation_metrics.txt + per_class_metrics.png")


if __name__ == "__main__":
    main()