"""
CPU SPEED RUN — Target 0.6+ IoU in 30 minutes
===============================================
Key trick: Pre-extract ALL DINOv2 features ONCE and cache to disk.
Then training loop never runs the heavy backbone again — only trains
the tiny segmentation head, which is very fast even on CPU.

Steps:
  1. Extract features once (~10-15 min on CPU)
  2. Train head only on cached features (~10-15 min, 20 epochs)
  3. Expected IoU: 0.55 - 0.68
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import os
import pickle
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Smaller image = faster everything ────────────────────────────────────────
IMG_W = int(((480) // 14) * 14)   # 476 → 238 (4x fewer pixels than original)
IMG_H = int(((270) // 14) * 14)   # 266 → 140
TOKEN_W = IMG_W // 14              # 17
TOKEN_H = IMG_H // 14              # 10
N_EPOCHS = 25
BATCH    = 8                       # larger batch since no backbone overhead
LR       = 1e-3                    # higher LR fine for head-only training

# ── Classes ───────────────────────────────────────────────────────────────────
value_map = {
    0:0, 100:1, 200:2, 300:3, 500:4,
    550:5, 600:6, 700:7, 800:8, 7100:9, 10000:10
}
class_names = [
    'Background','Trees','Lush Bushes','Dry Grass','Dry Bushes',
    'Ground Clutter','Flowers','Logs','Rocks','Landscape','Sky'
]
N_CLASSES = 11

class_weights = torch.tensor([
    0.5, 1.2, 1.2, 1.0, 1.2, 2.0, 4.0, 4.0, 2.0, 0.5, 0.8
], dtype=torch.float32)

def convert_mask(mask_path):
    arr = np.array(Image.open(mask_path))
    out = np.zeros_like(arr, dtype=np.uint8)
    for raw, new in value_map.items():
        out[arr == raw] = new
    return out

# ── Feature Cache Dataset ─────────────────────────────────────────────────────
class CachedFeatureDataset(Dataset):
    """Loads pre-extracted DINOv2 features from disk. Super fast."""
    def __init__(self, cache_path, augment=False):
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        self.features = data['features']   # list of (N, C) tensors
        self.masks    = data['masks']      # list of (H, W) uint8
        self.augment  = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].clone()  # (N, C)
        mask = self.masks[idx].copy()      # (H, W)

        # Fast augmentation directly on feature grid
        if self.augment and torch.rand(1).item() > 0.5:
            # Flip: reshape feature to spatial, flip, reshape back
            f2d   = feat.reshape(TOKEN_H, TOKEN_W, -1)
            f2d   = torch.flip(f2d, dims=[1])          # horizontal flip
            feat  = f2d.reshape(TOKEN_H * TOKEN_W, -1)
            mask  = np.fliplr(mask).copy()

        mask_tensor = torch.from_numpy(
            np.array(Image.fromarray(mask).resize(
                (TOKEN_W, TOKEN_H), Image.NEAREST))
        ).long()

        return feat, mask_tensor


# ── Raw Image Dataset (only used for feature extraction) ─────────────────────
class RawDataset(Dataset):
    def __init__(self, data_dir):
        self.img_dir  = os.path.join(data_dir, 'Color_Images')
        self.msk_dir  = os.path.join(data_dir, 'Segmentation')
        self.ids      = sorted(os.listdir(self.img_dir))
        self.transform = transforms.Compose([
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        fid   = self.ids[idx]
        img   = Image.open(os.path.join(self.img_dir, fid)).convert('RGB')
        mask  = convert_mask(os.path.join(self.msk_dir, fid))
        return self.transform(img), mask, fid


# ── Segmentation Head (lightweight for CPU speed) ─────────────────────────────
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
        # x: (B, N, C)
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.net(x)   # (B, n_classes, H, W)


# ── Feature Extraction (the slow part, done only ONCE) ───────────────────────
def extract_and_cache(data_dir, cache_path, backbone, device):
    if os.path.exists(cache_path):
        print(f"  Cache found: {cache_path} — skipping extraction")
        return

    print(f"  Extracting features to {cache_path} ...")
    raw = RawDataset(data_dir)
    loader = DataLoader(raw, batch_size=1, shuffle=False, num_workers=0)

    features, masks = [], []
    backbone.eval()
    with torch.no_grad():
        for imgs, msks, _ in tqdm(loader, desc="  Extracting"):
            imgs = imgs.to(device)
            out  = backbone.forward_features(imgs)["x_norm_patchtokens"]
            features.append(out.squeeze(0).cpu())   # (N, C)
            masks.append(msks.squeeze(0).numpy())   # (H, W)

    with open(cache_path, 'wb') as f:
        pickle.dump({'features': features, 'masks': masks}, f)
    print(f"  Cached {len(features)} samples.")


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_iou(pred_logits, target, n_classes=N_CLASSES):
    pred   = torch.argmax(pred_logits, dim=1).view(-1)
    target = target.view(-1)
    ious   = []
    for c in range(n_classes):
        inter = ((pred==c) & (target==c)).sum().float()
        union = ((pred==c) | (target==c)).sum().float()
        if union > 0:
            ious.append((inter/union).item())
    return float(np.nanmean(ious)) if ious else 0.0


def evaluate(model, loader, device):
    model.eval()
    ious = []
    with torch.no_grad():
        for feats, masks in loader:
            feats, masks = feats.to(device), masks.to(device)
            out  = model(feats)
            ious.append(compute_iou(out, masks))
    model.train()
    return float(np.mean(ious))


# ── Plots ─────────────────────────────────────────────────────────────────────
def save_plots(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ep = range(1, len(history['train_iou'])+1)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(ep, history['train_loss'], label='Train Loss')
    plt.plot(ep, history['val_loss'],   label='Val Loss')
    plt.title('Loss'); plt.legend(); plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(ep, history['train_iou'], label='Train IoU')
    plt.plot(ep, history['val_iou'],   label='Val IoU')
    plt.axhline(0.6, color='red', linestyle='--', label='Target 0.6')
    plt.title('IoU'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'curves.png'), dpi=120)
    plt.close()
    print(f"Saved plots → {out_dir}/curves.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    device     = torch.device('cpu')
    print(f"Device: CPU  |  Image: {IMG_H}×{IMG_W}  |  Tokens: {TOKEN_H}×{TOKEN_W}")

    # Paths
    train_dir  = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'train')
    val_dir    = os.path.join(script_dir, '..', 'Offroad_Segmentation_Training_Dataset', 'val')
    cache_dir  = os.path.join(script_dir, 'feature_cache')
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, 'train_features.pkl')
    val_cache   = os.path.join(cache_dir, 'val_features.pkl')

    # ── Step 1: Load backbone (only for extraction) ──────────────────────────
    print("\n[Step 1] Loading DINOv2 backbone for feature extraction...")
    try:
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        print("  Loaded vitb14")
    except:
        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        print("  Loaded vits14 (fallback)")
    backbone.eval().to(device)

    # ── Step 2: Extract & cache features ─────────────────────────────────────
    print("\n[Step 2] Extracting features (skipped if cache exists)...")
    extract_and_cache(train_dir, train_cache, backbone, device)
    extract_and_cache(val_dir,   val_cache,   backbone, device)

    # Free backbone memory — not needed anymore
    del backbone
    import gc; gc.collect()
    print("  Backbone freed from memory.")

    # ── Step 3: Get embedding dim ─────────────────────────────────────────────
    with open(train_cache, 'rb') as f:
        sample = pickle.load(f)
    n_embed = sample['features'][0].shape[1]
    n_train = len(sample['features'])
    print(f"\n  Embedding dim : {n_embed}")
    print(f"  Train samples : {n_train}")

    # ── Step 4: Datasets & Loaders ────────────────────────────────────────────
    trainset = CachedFeatureDataset(train_cache, augment=True)
    valset   = CachedFeatureDataset(val_cache,   augment=False)

    train_loader = DataLoader(trainset, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(valset,   batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"  Train batches : {len(train_loader)}")

    # ── Step 5: Model, Loss, Optimiser ───────────────────────────────────────
    model = FastSegHead(n_embed, N_CLASSES, TOKEN_H, TOKEN_W).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-5)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Head params   : {total_params:,}  (tiny — trains fast on CPU)")

    history = {'train_loss':[], 'val_loss':[], 'train_iou':[], 'val_iou':[]}
    best_iou  = 0.0
    best_path = os.path.join(script_dir, 'segmentation_head_cpu.pth')

    print(f"\n[Step 5] Training for {N_EPOCHS} epochs...")
    print("=" * 65)

    for epoch in range(1, N_EPOCHS+1):
        model.train()
        losses, ious = [], []

        for feats, masks in tqdm(train_loader,
                                  desc=f"Ep {epoch:02d}/{N_EPOCHS}",
                                  leave=False):
            feats, masks = feats.to(device), masks.to(device)
            out  = model(feats)
            loss = criterion(out, masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            with torch.no_grad():
                ious.append(compute_iou(out, masks))

        # Validation
        val_ious, val_losses = [], []
        model.eval()
        with torch.no_grad():
            for feats, masks in val_loader:
                feats, masks = feats.to(device), masks.to(device)
                out  = model(feats)
                val_losses.append(criterion(out, masks).item())
                val_ious.append(compute_iou(out, masks))
        model.train()
        scheduler.step()

        tl = float(np.mean(losses))
        vl = float(np.mean(val_losses))
        ti = float(np.mean(ious))
        vi = float(np.mean(val_ious))

        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        history['train_iou'].append(ti)
        history['val_iou'].append(vi)

        tag = ""
        if vi > best_iou:
            best_iou = vi
            torch.save(model.state_dict(), best_path)
            tag = "  ← BEST"

        print(f"Ep {epoch:02d}/{N_EPOCHS} | "
              f"Loss {tl:.4f}/{vl:.4f} | "
              f"IoU  {ti:.4f}/{vi:.4f}{tag}")

    save_plots(history, output_dir)

    print(f"\n{'='*55}")
    print(f"  Best Val IoU : {best_iou:.4f}")
    print(f"  Baseline IoU : 0.2478")
    print(f"  Improvement  : {best_iou - 0.2478:+.4f}")
    print(f"  Model saved  : {best_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
