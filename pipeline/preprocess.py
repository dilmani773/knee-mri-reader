"""
RSNA Knee: turn each study's DICOMs into a small fixed-size array.

For every study we keep 3 views (fluid-sensitive / fat-suppressed series):
    0 = sagittal, 1 = coronal, 2 = axial
Each view -> (D, S, S) uint8, default D=24 slices, S=256 pixels.
v2: 130 mm crop around the knee, outer slices dropped. Laterality is saved in meta.csv (no mirroring).
One study -> array (3, D, S, S) + mask (3,) saying which views exist.

Run in a CPU notebook (no GPU needed), Internet ON, as a committed version:
    !pip install -q pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg python-gdcm opencv-python-headless
    !python preprocess.py --limit 20        # quick test
    !python preprocess.py                   # all studies

Outputs in /kaggle/working/cache/:
    shard_000.npz ... (each holds ~200 studies, key = StudyInstanceUID)
    meta.csv          (which series was used, slice counts, laterality, errors)
"""
import argparse, os, re, glob, time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import cv2
import pydicom

ROOT = "/kaggle/input/competitions/rsna-knee-abnormality-detection"
PLANES = ["Sagittal", "Coronal", "Axial"]
# For each plane: which patient axis is (slice, up, right). Patient axes: 0=x(L) 1=y(P) 2=z(S)
TARGET = {"Sagittal": (0, 2, 1), "Coronal": (1, 2, 0), "Axial": (2, 1, 0)}


# ---------------------------------------------------------------- series choice
def choose_series(study_rows, series_dir):
    """Pick one series per plane: fluid-sensitive first, most slices wins."""
    out = {}
    for plane in PLANES:
        rows = study_rows[study_rows.Anatomical_Plane == plane]
        best = None
        for fs in (1, 0):  # fallback to non-FS if no FS series in this plane
            cand = rows[rows.Fluid_Sensitive == fs]
            if len(cand):
                counts = [(len(os.listdir(os.path.join(series_dir, s))), s)
                          for s in cand.SeriesInstanceUID
                          if os.path.isdir(os.path.join(series_dir, s))]
                if counts:
                    n, s = max(counts)
                    best = (s, fs, n)
                    break
        out[plane] = best
    return out


# ---------------------------------------------------------------- reading
LAT_RE = [(re.compile(r"\b(LEFT|LT|LINKS|IZQ\w*|GAUCHE|SOL)\b", re.I), "L"),
          (re.compile(r"\b(RIGHT|RT|RECHTS|DERECH\w*|DROIT\w*)\b", re.I), "R")]


def guess_laterality(ds):
    for tag in ("Laterality", "ImageLaterality"):
        v = str(ds.get(tag, "")).strip().upper()
        if v in ("L", "R"):
            return v
    text = " ".join(str(ds.get(t, "")) for t in
                    ("SeriesDescription", "ProtocolName", "StudyDescription", "BodyPartExamined"))
    for rx, lab in LAT_RE:
        if rx.search(text):
            return lab
    return ""


CROP_MM = 130.0          # keep a 130 mm square around the image centre (the knee)
SLICE_BAND = (0.08, 0.92)  # drop the outermost slices (mostly outside the joint)
LAT_MIN_MM = 5.0         # knee centre must be this far from x=0 to trust geometry
FLIP_RIGHT = False       # mirror right knees? Off: a top public team measured it as worse (LB 0.909 vs 0.915)


def crop_mm(img, ds):
    ps = ds.get("PixelSpacing", None)
    if ps is None or len(ps) < 2:
        return img
    rs, cs = float(ps[0]), float(ps[1])
    if rs <= 0 or cs <= 0:
        return img
    h, w = img.shape
    ch, cw = min(h, int(round(CROP_MM / rs))), min(w, int(round(CROP_MM / cs)))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def centre_x(ds, img_shape):
    """Patient x (mm) of the image centre. x < 0 -> right side of the patient."""
    try:
        ipp = np.array(ds.ImagePositionPatient, float)
        iop = np.array(ds.ImageOrientationPatient, float)
        ps = [float(v) for v in ds.PixelSpacing]
        rows, cols = img_shape
        c = ipp + iop[:3] * ps[1] * cols / 2 + iop[3:] * ps[0] * rows / 2
        return float(c[0])
    except Exception:
        return None


def read_series(folder):
    files = glob.glob(os.path.join(folder, "*.dcm"))
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            img = ds.pixel_array.astype(np.float32)
            if img.ndim != 2:
                continue
            slices.append((ds, img))
        except Exception:
            continue
    if not slices:
        raise RuntimeError("no readable slices")
    # keep the most common image size (some series mix sizes)
    shapes = pd.Series([s[1].shape for s in slices]).value_counts()
    main = shapes.index[0]
    slices = [s for s in slices if s[1].shape == main]
    mid = slices[len(slices) // 2]
    cx = centre_x(mid[0], mid[1].shape)
    slices = [(ds, crop_mm(img, ds)) for ds, img in slices]
    return slices, cx


def build_volume(slices, plane):
    """Sort slices in space and reorient to a standard view. Returns (D,H,W) float32."""
    ds0 = slices[0][0]
    iop = ds0.get("ImageOrientationPatient", None)
    have_geo = iop is not None and all(s[0].get("ImagePositionPatient", None) is not None for s in slices)

    if have_geo:
        r = np.array(iop[:3], float)   # direction of increasing column index
        c = np.array(iop[3:], float)   # direction of increasing row index
        n = np.cross(r, c)
        pos = [float(np.dot(np.array(s[0].ImagePositionPatient, float), n)) for s in slices]
        order = np.argsort(pos)
        vol = np.stack([slices[i][1] for i in order])          # axes: (n, c, r)
        dirs = [n, c, r]
        ax = [int(np.argmax(np.abs(d))) for d in dirs]
        if sorted(ax) == [0, 1, 2]:
            # make each volume axis point toward + patient axis
            for k, d in enumerate(dirs):
                if d[ax[k]] < 0:
                    vol = np.flip(vol, axis=k)
            # permute to (slice, up, right) for this plane
            want = TARGET[plane]
            vol = np.transpose(vol, [ax.index(p) for p in want])
            if want[1] == 2:            # up axis is z: put superior at the top
                vol = np.flip(vol, axis=1)
            return np.ascontiguousarray(vol), True
    # fallback: sort by InstanceNumber, keep raw orientation
    order = np.argsort([int(s[0].get("InstanceNumber", i)) for i, s in enumerate(slices)])
    return np.stack([slices[i][1] for i in order]), False


def resample(vol, D, S):
    n = vol.shape[0]
    lo, hi = SLICE_BAND[0] * (n - 1), SLICE_BAND[1] * (n - 1)
    idx = np.linspace(lo, hi, D).round().astype(int)
    vol = vol[idx]
    lo, hi = np.percentile(vol, [1, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    out = np.empty((D, S, S), np.uint8)
    for i in range(D):
        out[i] = (cv2.resize(vol[i], (S, S), interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- one study
def process_study(args):
    uid, study_rows, split, D, S = args
    series_dir = os.path.join(ROOT, f"{split}_series", uid)
    arr = np.zeros((3, D, S, S), np.uint8)
    mask = np.zeros(3, np.uint8)
    meta = {"StudyInstanceUID": uid}
    tags, xs, vols = [], [], {}
    picks = choose_series(study_rows, series_dir)
    for v, plane in enumerate(PLANES):
        pick = picks.get(plane)
        if pick is None:
            meta[f"{plane}_err"] = "missing"
            continue
        sid, fs, n = pick
        meta.update({f"{plane}_series": sid, f"{plane}_fs": fs, f"{plane}_n": n})
        try:
            slices, cx = read_series(os.path.join(series_dir, sid))
            tags.append(guess_laterality(slices[0][0]))
            if cx is not None:
                xs.append(cx)
            vol, ok = build_volume(slices, plane)
            vols[v] = (vol, ok)
            meta[f"{plane}_geo"] = int(ok)
        except Exception as e:
            meta[f"{plane}_err"] = str(e)[:100]

    # laterality: geometry first (knee centre x < 0 = right), then DICOM tags
    lat, how = "", ""
    if xs and abs(np.median(xs)) >= LAT_MIN_MM:
        lat, how = ("R" if np.median(xs) < 0 else "L"), "geo"
    else:
        t = [x for x in tags if x]
        if t:
            lat, how = max(set(t), key=t.count), "tag"
    meta.update({"laterality": lat, "lat_how": how, "centre_x": np.median(xs) if xs else np.nan})

    for v, (vol, ok) in vols.items():
        if FLIP_RIGHT and lat == "R" and ok:  # mirror right knees so medial/lateral sit on the same side
            vol = vol[::-1] if v == 0 else vol[:, :, ::-1]
        arr[v] = resample(np.ascontiguousarray(vol), D, S)
        mask[v] = 1
    return uid, arr, mask, meta


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--out", default="/kaggle/working/cache")
    ap.add_argument("--D", type=int, default=24)
    ap.add_argument("--S", type=int, default=256)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--shard", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    ser = pd.read_csv(os.path.join(ROOT, f"{a.split}_series.csv"))
    uids = ser.StudyInstanceUID.unique().tolist()
    if a.limit:
        uids = uids[:a.limit]
    groups = {u: g for u, g in ser.groupby("StudyInstanceUID")}
    jobs = [(u, groups[u], a.split, a.D, a.S) for u in uids]
    print(f"{len(jobs)} studies, {a.workers} workers, D={a.D}, S={a.S}")

    metas, buf, n_shard, t0 = [], {}, 0, time.time()

    def flush():
        nonlocal buf, n_shard
        if buf:
            np.savez_compressed(os.path.join(a.out, f"shard_{n_shard:03d}.npz"), **buf)
            n_shard += 1
            buf = {}

    with Pool(a.workers) as pool:
        for i, (uid, arr, mask, meta) in enumerate(pool.imap_unordered(process_study, jobs, chunksize=2)):
            buf[uid] = arr
            buf[uid + "__mask"] = mask
            meta["shard"] = n_shard
            metas.append(meta)
            if len(buf) // 2 >= a.shard:
                flush()
            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"{i + 1}/{len(jobs)}  {el / 60:.1f} min  (~{el / (i + 1) * (len(jobs) - i - 1) / 60:.0f} min left)", flush=True)
    flush()

    m = pd.DataFrame(metas)
    m.to_csv(os.path.join(a.out, "meta.csv"), index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min, {n_shard} shards")
    for p in PLANES:
        err = m.get(f"{p}_err")
        print(f"{p:9s} missing/failed: {0 if err is None else err.notna().sum()}"
              f"   geo-oriented: {m.get(f'{p}_geo', pd.Series(dtype=float)).sum():.0f}")
    print("laterality:", m.laterality.fillna("").replace("", "unknown").value_counts().to_dict(),
          "| source:", m.lat_how.fillna("").replace("", "none").value_counts().to_dict())


if __name__ == "__main__":
    main()
