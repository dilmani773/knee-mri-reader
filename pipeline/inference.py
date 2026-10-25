"""
RSNA Knee: submission notebook (Internet OFF).

Needs in the same folder: preprocess.py and train.py (written with %%writefile).
Attach as input:
  - the competition data
  - the training notebook output (model_fold*.pt)
  - the wheels notebook output (wheels/ folder) for the DICOM decoders

    !pip install -q --no-index --find-links=<wheels dir> pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm
    !python inference.py
"""
import os, time, glob
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from preprocess import process_study, ROOT
from train import Net, LABELS

D, S, BS = 24, 256, 8
TTA_FLIP = False  # mirror TTA doubles GPU time for a tiny gain; off for the efficiency track
BACKBONE = "resnet34.a1_in1k"


def find_all(pattern, root="/kaggle/input", max_depth=6):
    out = []
    root_depth = root.rstrip("/").count("/")
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = [d for d in dirnames if d not in ("competitions", "train_series", "test_series")]
        if dirpath.count("/") - root_depth >= max_depth:
            dirnames[:] = []
        out += [os.path.join(dirpath, f) for f in filenames if glob.fnmatch.fnmatch(f, pattern)]
    return sorted(out)


def to_tensor(arrs):
    x = torch.from_numpy(np.stack(arrs)).float() / 255.0      # (B, 3, D, S, S)
    B, V, Dd, Ss, _ = x.shape
    x = x.view(B, V, Dd // 3, 3, Ss, Ss)
    return (x - 0.45) / 0.25


def flip(x):  # same left-right mirror as training augmentation
    x = x.clone()
    sag = x[:, 0].reshape(x.shape[0], -1, x.shape[-2], x.shape[-1]).flip(1)  # reverse all slices
    x[:, 0] = sag.reshape(x[:, 0].shape)
    x[:, 1:] = x[:, 1:].flip(-1)
    return x


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = find_all("model_fold*.pt")
    print("models:", paths)
    models = []
    for p in paths:
        m = Net(BACKBONE, pretrained=False)
        m.load_state_dict(torch.load(p, map_location="cpu"))
        models.append(m.to(device).eval().half() if device == "cuda" else m.eval())

    ser = pd.read_csv(os.path.join(ROOT, "test_series.csv"))
    test = pd.read_csv(os.path.join(ROOT, "test.csv"))
    groups = {u: g for u, g in ser.groupby("StudyInstanceUID")}
    uids = [u for u in test.StudyInstanceUID if u in groups]
    jobs = [(u, groups[u], "test", D, S) for u in uids]
    print(f"{len(test)} test studies, {len(jobs)} with series")

    preds = {}
    buf_u, buf_x = [], []

    @torch.no_grad()
    def run():
        x = to_tensor(buf_x).to(device)
        if device == "cuda":
            x = x.half()
        p = 0
        for m in models:
            p = p + torch.sigmoid(m(x).float())
            if TTA_FLIP:
                p = p + torch.sigmoid(m(flip(x)).float())
        p = (p / ((2 if TTA_FLIP else 1) * len(models))).cpu().numpy()
        for u, row in zip(buf_u, p):
            preds[u] = row
        buf_u.clear(); buf_x.clear()

    with Pool(os.cpu_count()) as pool:
        for i, (uid, arr, mask, meta) in enumerate(pool.imap(process_study, jobs, chunksize=1)):
            buf_u.append(uid); buf_x.append(arr)
            if len(buf_u) == BS:
                run()
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(jobs)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    if buf_u:
        run()

    sub = pd.DataFrame({"StudyInstanceUID": test.StudyInstanceUID})
    fallback = np.full(len(LABELS), 0.5)
    sub[LABELS] = np.stack([preds.get(u, fallback) for u in sub.StudyInstanceUID])
    sub.to_csv("submission.csv", index=False)
    print(sub.head())
    print(f"done in {(time.time() - t0) / 60:.1f} min, missing preds: {len(sub) - len(preds)}")


if __name__ == "__main__":
    main()
