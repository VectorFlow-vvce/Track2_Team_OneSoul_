"""
CPU SPEED-RUN TEST SCRIPT (MATCHES TRAINING)

- Uses cached DINOv2 features (NO backbone here)
- Uses FastSegHead (same as training)
- Fast CPU evaluation
"""

import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import argparse

# ─────────────────────────────────────────────────────────────
# SAME SETTINGS AS TRAINING (MUST MATCH)
# ─────────────────────────────────────────────────────────────
IMG_W = int(((480) // 14) * 14)
IMG_H = int(((270) // 14) * 14)
TOKEN_W = IMG_W // 14
TOKEN_H = IMG_H // 14
N_CLASSES = 11

class_names = [
    'Background','Trees','Lush Bushes','Dry Grass','Dry Bushes',
    'Ground Clutter','Flowers','Logs','Rocks','Landscape','Sky'
]

# ─────────────────────────────────────────────────────────────
# MODEL (IDENTICAL TO TRAIN)
# ─────────────────────────────────────────────────────────────
class FastSegHead(nn.Module):
    def __init__(self, in_channels, n_classes, token_h, token_w):
        super().__init__()
        self.H, self.W = token_h, token_w

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(16, 256),
            nn.GELU(),

            nn.Conv2d(256, 128, 1),
            nn.GELU(),

            nn.Conv2d(128, n_classes, 1),
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# DATASET (CACHED FEATURES)
# ─────────────────────────────────────────────────────────────
class CachedFeatureDataset(Dataset):
    def __init__(self, cache_path):
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)

        self.features = data['features']
        self.masks    = data['masks']

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]   # (N, C)
        mask = self.masks[idx]      # (H, W)

        # Resize mask to token grid
        mask = torch.from_numpy(
            np.array(Image.fromarray(mask).resize(
                (TOKEN_W, TOKEN_H), Image.NEAREST))
        ).long()

        return feat, mask


# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────
def compute_iou(pred_logits, target, n_classes=N_CLASSES):
    pred   = torch.argmax(pred_logits, dim=1).view(-1)
    target = target.view(-1)

    ious = []
    per_class = []

    for c in range(n_classes):
        inter = ((pred == c) & (target == c)).sum().float()
        union = ((pred == c) | (target == c)).sum().float()

        if union > 0:
            iou = (inter / union).item()
            ious.append(iou)
            per_class.append(iou)
        else:
            per_class.append(np.nan)

    mean_iou = float(np.nanmean(ious)) if ious else 0.0
    return mean_iou, per_class


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='segmentation_head_cpu.pth')
    parser.add_argument('--cache_path', default='feature_cache/val_features.pkl')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--save_preds', action='store_true')
    parser.add_argument('--output_dir', default='test_outputs')
    args = parser.parse_args()

    device = torch.device('cpu')
    print(f"Device: {device}")

    # ── Load dataset ──────────────────────────────────────────
    dataset = CachedFeatureDataset(args.cache_path)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Loaded {len(dataset)} samples")

    # ── Get embedding dimension ───────────────────────────────
    with open(args.cache_path, 'rb') as f:
        sample = pickle.load(f)

    n_embed = sample['features'][0].shape[1]
    print(f"Embedding dim: {n_embed}")

    # ── Load model ────────────────────────────────────────────
    model = FastSegHead(
        in_channels=n_embed,
        n_classes=N_CLASSES,
        token_h=TOKEN_H,
        token_w=TOKEN_W
    )

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    print("Model loaded successfully!")

    # ── Optional output dir ───────────────────────────────────
    if args.save_preds:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── Evaluation ────────────────────────────────────────────
    all_ious = []
    class_ious_accum = []

    print("\nRunning evaluation...\n")

    with torch.no_grad():
        for idx, (feats, masks) in enumerate(tqdm(loader)):
            feats, masks = feats.to(device), masks.to(device)

            logits = model(feats)

            mean_iou, per_class = compute_iou(logits, masks)

            all_ious.append(mean_iou)
            class_ious_accum.append(per_class)

            # Optional: save predictions
            if args.save_preds:
                preds = torch.argmax(logits, dim=1)

                for b in range(preds.shape[0]):
                    pred = preds[b].cpu().numpy().astype(np.uint8)

                    Image.fromarray(pred).save(
                        os.path.join(args.output_dir, f"pred_{idx}_{b}.png")
                    )

    # ── Final metrics ─────────────────────────────────────────
    mean_iou = float(np.mean(all_ious))
    class_ious = np.nanmean(class_ious_accum, axis=0)

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Mean IoU: {mean_iou:.4f}")
    print("="*50)

    print("\nPer-Class IoU:")
    for name, iou in zip(class_names, class_ious):
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        print(f"{name:<20}: {iou_str}")

    print("\nDone.")


if __name__ == "__main__":
    main()
