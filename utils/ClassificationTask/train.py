#!/usr/bin/env python3
# kfold_resnet6.py
"""
K-Fold CV (Stratified) – ResNet-50 fine-tune 6 lớp:
dish_empty, dish_kakigori, dish_not_empty,
tray_empty, tray_kakigori, tray_not_empty
"""

import argparse, random, numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch, torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.model_selection import StratifiedKFold

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            # alpha là list/ndarray độ dài = num_classes
            self.register_buffer("alpha",
                                 torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(logits, targets,
                                               reduction="none")
        pt = torch.exp(-ce)                    # = prob nhãn đúng
        if self.alpha is not None:
            # chuyển alpha về đúng device & lấy phần tử theo batch index
            at = self.alpha.to(logits.device)[targets]
            ce = at * ce
        fl = ((1 - pt) ** self.gamma) * ce
        return fl.mean() if self.reduction == "mean" else fl.sum()

# -------- 1. Mapping nhãn -------- #
LABEL_ORDER = [
    "dish_empty", "dish_kakigori", "dish_not_empty",
    "tray_empty", "tray_kakigori", "tray_not_empty"
]
LABEL2ID = {n: i for i, n in enumerate(LABEL_ORDER)}

# -------- 2. Dataset đơn giản -------- #
class SixClsDataset(Dataset):
    def __init__(self, paths, labels, tf=None):
        self.paths, self.labels, self.tf = paths, labels, tf
    def __len__(self):  return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.tf:
            img = self.tf(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)  # 👈 ép kiểu int64
        return img, label

# -------- 3. Tạo danh sách file + label một lần -------- #
def collect_dataset(root="Classification"):
    root = Path(root)
    paths, labels = [], []
    for container in ("dish", "tray"):
        for sub in ("empty", "kakigori", "not_empty"):
            cls_name = f"{container}_{sub}"
            cls_id = LABEL2ID[cls_name]
            for p in (root / container / sub).glob("*.*"):
                paths.append(p.as_posix())
                labels.append(cls_id)
    return np.array(paths), np.array(labels)

# -------- 4. Transform -------- #
from torchvision.transforms import RandAugment       # 🔧 NEW

train_tf = T.Compose([
    T.Resize((224, 224)),
    # T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    # T.RandomHorizontalFlip(),
    T.ColorJitter(0.3, 0.3, 0.3, 0.15),             # 🔧 thêm
    RandAugment(num_ops=2, magnitude=9),            # 🔧 thêm
    T.ToTensor(),
    T.RandomErasing(p=0.5, scale=(0.02, 0.15)),     # 🔧 thêm
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])
val_tf = T.Compose([
    T.Resize((224, 224)),
    # T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# -------- 5. Hàm train + evaluate một fold -------- #

def run_fold(train_idx, val_idx, args, paths, labels, fold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = SixClsDataset(paths[train_idx], labels[train_idx], train_tf)
    val_ds   = SixClsDataset(paths[val_idx],   labels[val_idx],   val_tf)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    
    # sau collect_dataset …
    class_counts = np.bincount(labels, minlength=6)
    alpha = (class_counts.max() / class_counts).astype(np.float32)
    alpha /= alpha.sum()

    # ─── Model ───
    # model = Net(p=3, q=5, channels=4, classes=6)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for p in model.parameters(): p.requires_grad = False
    for p in model.layer4.parameters(): p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 6)
    model.to(device)

    optim = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": args.lr*0.1},
        {"params": model.fc.parameters(),     "lr": args.lr}
    ], 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(optim, 5, 0.1)
    
    # loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_fn = FocalLoss(alpha=alpha, gamma=2.0)

    best_acc, patience = 0.0, 0
    for ep in range(1, args.epochs+1):
        # ------ train ------
        model.train(); correct = 0
        for x, y in tqdm(train_ld, desc=f"Fold{fold} Epoch{ep}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x); loss = loss_fn(out, y)
            optim.zero_grad(); loss.backward(); optim.step()
            correct += (out.argmax(1)==y).sum().item()
        train_acc = correct/len(train_ds)

        # ------ val ------
        model.eval(); correct = 0
        with torch.inference_mode():
            for x, y in val_ld:
                x,y = x.to(device), y.to(device)
                correct += (model(x).argmax(1)==y).sum().item()
        val_acc = correct/len(val_ds)
        sched.step()

        print(f"Fold {fold}  Epoch {ep}: Train {train_acc:.3%}  Val {val_acc:.3%}")

        if val_acc > best_acc + 1e-4:     # cải thiện thật sự
            best_acc, patience = val_acc, 0
            torch.save(model.state_dict(), f"best_fold{fold}.pt")
        else:
            patience += 1
            if patience >= 5:             # 🔧 early stop nếu 5 epoch không cải thiện
                print(f"⏹️  Early stop at epoch {ep}")
                break

    return best_acc

# -------- 6. Main: Stratified K-Fold loop -------- #
def main(args):
    paths, labels = collect_dataset(args.data)
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)

    fold_results = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(paths, labels), 1):
        best = run_fold(train_idx, val_idx, args, paths, labels, fold)
        fold_results.append(best)
        print(f"Best val acc Fold{fold}: {best:.3%}\n")

    print("===== CV Results =====")
    for i, acc in enumerate(fold_results,1):
        print(f" Fold {i}: {acc:.3%}")
    print(f" Mean  : {np.mean(fold_results):.3%}")
    print(f" Std   : {np.std(fold_results):.3%}")

# -------- 7. CLI -------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Classification")
    ap.add_argument("--k", type=int, default=5, help="k-folds")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=0)   # Windows → 0
    args = ap.parse_args()

    main(args)
