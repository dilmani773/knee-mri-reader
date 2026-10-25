"""
RSNA Knee: 2.5D multi-view CNN on the preprocessed cache.

Input per study: (3 views, 24 slices, 224, 224) uint8.
Every 3 neighbouring slices -> one 3-channel image, so each view = 8 images.
A shared 2D backbone (ResNet34) embeds every image, attention pooling merges the 8 per view,
the 3 views are joined and a head predicts the 12 labels.

Targets: human labels where they exist (58 studies), otherwise the LLM average (soft 0..1).

Kaggle: GPU T4 x2, Internet ON. Attach as input:
  - the competition data
  - the preprocessing notebook output (has cache/meta.csv + shard_*.npz)
  - the labeling notebook output (has llm_labels_avg.csv)

    !pip install -q timm
    !python train.py --fold 0 --epochs 10
"""
import argparse, os, time, math, random

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
COMP = "/kaggle/input/competitions/rsna-knee-abnormality-detection"


def find_file(name, root="/kaggle/input", max_depth=6):
    """Find a file under /kaggle/input without walking the huge DICOM folders."""
    root_depth = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = [d for d in dirnames if d not in ("competitions", "train_series", "test_series")]
        if dirpath.count("/") - root_depth >= max_depth:
            dirnames[:] = []
        if name in filenames:
            return os.path.join(dirpath, name)
    raise FileNotFoundError(f"{name} not found under {root}. Did you attach the notebook output?")


# ---------------------------------------------------------------- data
class KneeDS(Dataset):
    def __init__(self, uids, targets, meta, cache_dir, train, ram=None, ram_idx=None, flip=False):
        self.uids, self.targets, self.train, self.flip = uids, targets, train, flip
        self.shard = meta.set_index("StudyInstanceUID")["shard"].to_dict()
        self.cache_dir = cache_dir
        self.files = {}  # opened lazily inside each worker
        self.ram, self.ram_idx = ram, ram_idx  # optional: all studies already in memory

    def __len__(self):
        return len(self.uids)

    def load(self, uid):
        if self.ram is not None:
            return self.ram[self.ram_idx[uid]]
        s = self.shard[uid]
        if s not in self.files:
            self.files[s] = np.load(os.path.join(self.cache_dir, f"shard_{s:03d}.npz"))
        return self.files[s][uid]  # (3, D, S, S) uint8

    def augment(self, x):
        x = x.copy()
        if self.flip and random.random() < 0.5:  # mirror left-right in patient space, same for all views
            x[0] = x[0][::-1]           # sagittal: slice axis is left-right
            x[1] = x[1][:, :, ::-1]     # coronal: width is left-right
            x[2] = x[2][:, :, ::-1]     # axial: width is left-right
        S = x.shape[-1]
        for v in range(3):
            if random.random() < 0.7:  # random zoom / shift
                sc = random.uniform(0.8, 1.0)
                c = int(S * sc)
                y0, x0 = random.randint(0, S - c), random.randint(0, S - c)
                crop = x[v][:, y0:y0 + c, x0:x0 + c]
                x[v] = np.stack([cv2.resize(sl, (S, S), interpolation=cv2.INTER_LINEAR) for sl in crop])
        return x

    def __getitem__(self, i):
        uid = self.uids[i]
        x = self.load(uid)
        if self.train:
            x = self.augment(x)
        x = torch.from_numpy(np.ascontiguousarray(x)).float() / 255.0  # (3, D, S, S)
        if self.train:  # brightness / contrast per view
            a = torch.empty(3, 1, 1, 1).uniform_(0.8, 1.2)
            b = torch.empty(3, 1, 1, 1).uniform_(-0.1, 0.1)
            x = (x * a + b).clamp(0, 1)
        V, D, S, _ = x.shape
        x = x.view(V, D // 3, 3, S, S)  # (3 views, 8 images, 3 channels, S, S)
        x = (x - 0.45) / 0.25
        return x, torch.tensor(self.targets[i], dtype=torch.float32)


def load_all(uids, meta, cache_dir):
    """Decompress every study once into one big uint8 array (about 3.6 MB per study)."""
    shard = meta.set_index("StudyInstanceUID")["shard"]
    idx = {u: i for i, u in enumerate(uids)}
    first = None
    t0 = time.time()
    for s in sorted(shard.loc[uids].unique()):
        z = np.load(os.path.join(cache_dir, f"shard_{s:03d}.npz"))
        for u in shard.index[shard == s]:
            if u in idx:
                x = z[u]
                if first is None:
                    first = np.empty((len(uids),) + x.shape, np.uint8)
                first[idx[u]] = x
        print(f"  loaded shard {s}  {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"RAM cache: {first.nbytes / 1e9:.1f} GB")
    return first, idx


# ---------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, backbone, pretrained=True, n_out=12):
        super().__init__()
        import timm
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        C = self.enc.num_features
        self.att = nn.ModuleList([nn.Sequential(nn.Linear(C, 128), nn.Tanh(), nn.Linear(128, 1))
                                  for _ in range(3)])
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(3 * 2 * C, 512), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(512, n_out))

    def forward(self, x):  # x: (B, 3, N, 3, S, S)
        B, V, N, Cin, S, _ = x.shape
        f = self.enc(x.reshape(B * V * N, Cin, S, S)).view(B, V, N, -1)
        feats = []
        for v in range(V):
            fv = f[:, v]                                   # (B, N, C)
            w = torch.softmax(self.att[v](fv).float(), dim=1).to(fv.dtype)
            feats.append(torch.cat([(w * fv).sum(1), fv.max(1).values], dim=1))
        return self.head(torch.cat(feats, dim=1))


# ---------------------------------------------------------------- eval
def macro_auc(y, p, mask=None):
    aucs = []
    for j in range(y.shape[1]):
        yj, pj = y[:, j], p[:, j]
        if mask is not None:
            yj, pj = yj[mask], pj[mask]
        if 0 < yj.sum() < len(yj):
            aucs.append(roc_auc_score(yj, pj))
        else:
            aucs.append(np.nan)
    return np.nanmean(aucs), aucs


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(torch.sigmoid(model(x.to(device, non_blocking=True)).float()).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--backbone", default="resnet34.a1_in1k")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="use only N studies (quick test)")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--no_ram", action="store_true", help="read shards from disk instead of RAM")
    ap.add_argument("--no_flip", action="store_true", help="turn off left-right mirror augmentation")
    a = ap.parse_args()

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    print("device:", device, "gpus:", torch.cuda.device_count())

    meta_path = find_file("meta.csv")
    cache_dir = os.path.dirname(meta_path)
    labels_path = find_file("llm_labels_avg.csv")
    print("cache:", cache_dir, "\nlabels:", labels_path)

    meta = pd.read_csv(meta_path)
    llm = pd.read_csv(labels_path).set_index("StudyInstanceUID")
    train = pd.read_csv(os.path.join(COMP, "train.csv")).set_index("StudyInstanceUID")
    hum = train[LABELS].apply(pd.to_numeric, errors="coerce")
    is_h = hum.isin([0, 1]).all(axis=1)

    uids = [u for u in meta.StudyInstanceUID if u in llm.index]
    if a.limit:
        uids = uids[:a.limit]
    uids = np.array(sorted(uids))
    tgt = llm.loc[uids, LABELS].values.astype(np.float32)
    human_mask = is_h.reindex(uids).fillna(False).values
    tgt[human_mask] = hum.loc[uids[human_mask], LABELS].values  # trust human labels
    print(f"{len(uids)} studies, {human_mask.sum()} with human labels")

    kf = KFold(a.n_folds, shuffle=True, random_state=42)
    tr_idx, va_idx = list(kf.split(uids))[a.fold]

    ram, ram_idx = (None, None) if a.no_ram else load_all(list(uids), meta, cache_dir)
    ds_tr = KneeDS(uids[tr_idx], tgt[tr_idx], meta, cache_dir, train=True, ram=ram, ram_idx=ram_idx, flip=not a.no_flip)
    ds_va = KneeDS(uids[va_idx], tgt[va_idx], meta, cache_dir, train=False, ram=ram, ram_idx=ram_idx)
    dl_tr = DataLoader(ds_tr, a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                       pin_memory=True, persistent_workers=a.workers > 0)
    dl_va = DataLoader(ds_va, a.bs, shuffle=False, num_workers=a.workers,
                       pin_memory=True, persistent_workers=a.workers > 0)

    model = Net(a.backbone, pretrained=not a.no_pretrained).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    total = a.epochs * len(dl_tr)
    warm = len(dl_tr) // 2
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    y_va_bin = (tgt[va_idx] >= 0.5).astype(int)
    h_va = human_mask[va_idx]
    best, t0 = -1, time.time()
    for ep in range(a.epochs):
        model.train()
        tot, n, t_data, t_step = 0.0, 0, 0.0, 0.0
        tic = time.time()
        for x, y in dl_tr:
            t_data += time.time() - tic; tic = time.time()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(x); n += len(x)
            t_step += time.time() - tic; tic = time.time()

        p = predict(model, dl_va, device)
        auc_llm, per = macro_auc(y_va_bin, p)
        auc_h, _ = macro_auc(y_va_bin, p, h_va) if h_va.sum() >= 5 else (float("nan"), None)
        print(f"ep {ep + 1}/{a.epochs}  loss {tot / n:.4f}  val AUC vs LLM {auc_llm:.4f}  "
              f"vs human ({h_va.sum()} studies) {auc_h:.4f}  {(time.time() - t0) / 60:.1f} min  "
              f"[data wait {t_data / 60:.1f} min, gpu {t_step / 60:.1f} min]", flush=True)
        if auc_llm > best:
            best = auc_llm
            sd = (model.module if hasattr(model, "module") else model).state_dict()
            torch.save(sd, os.path.join(a.out, f"model_fold{a.fold}.pt"))
            oof = pd.DataFrame(p, columns=LABELS)
            oof.insert(0, "StudyInstanceUID", uids[va_idx])
            oof["human"] = h_va
            oof.to_csv(os.path.join(a.out, f"oof_fold{a.fold}.csv"), index=False)
            best_per = per

    print(f"\nbest val AUC vs LLM: {best:.4f}")
    for c, v in zip(LABELS, best_per):
        print(f"  {c:18s} {v:.3f}")


if __name__ == "__main__":
    main()
