"""
RSNA Knee: label reports with a local open LLM on Kaggle GPU (no API, no limits).
Self-contained: no other file needed.

Kaggle settings: Accelerator = GPU T4 x2, Internet = ON

    !pip install -q vllm
    !python label_reports_local.py --subset labeled --check
    !python label_reports_local.py --subset all
"""
import argparse, json, os, re

import numpy as np
import pandas as pd

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
KEYS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa",
        "lateral_oa", "pf_oa", "effusion", "synovitis", "bakers",
        "contusion", "fracture"]
VALUE = {"yes": 1.0, "no": 0.0, "uncertain": 0.5}

PROMPT = """You are a musculoskeletal radiologist. Read this knee MRI report.
It may be in any language. Decide if each finding below is PRESENT in this knee.

Findings:
- acl: anterior cruciate ligament injury (tear, partial tear, sprain)
- mcl: medial collateral ligament injury (tear, partial tear, sprain)
- medial_meniscus: medial meniscus tear
- lateral_meniscus: lateral meniscus tear
- medial_oa: osteoarthritis of the medial tibiofemoral compartment
- lateral_oa: osteoarthritis of the lateral tibiofemoral compartment
- pf_oa: patellofemoral osteoarthritis (patella or trochlea)
- effusion: joint effusion (more fluid than physiological)
- synovitis: synovitis / synovial thickening or inflammation
- bakers: Baker's (popliteal) cyst
- contusion: bone contusion / bone bruise / bone marrow edema from trauma
- fracture: fracture of any bone in the knee

Rules:
- "yes" if the report states the finding is present.
- "no" if the report says it is absent, normal, or never mentions it.
- "uncertain" only if the report itself is unsure ("possible", "cannot exclude",
  "suspected") or the finding is unclear (for example a reconstructed ACL graft).
- "physiological" or "trace" fluid counts as "no" for effusion.
- Read the WHOLE report, including the conclusion. Findings can be written in any
  language (for example Spanish "derrame articular" = joint effusion).
- OA: count as "yes" for a compartment if it has osteoarthritis, degenerative change,
  osteophytes, chondrosis, chondromalacia, cartilage thinning, cartilage loss, or a
  cartilage / osteochondral defect or ulcer in that compartment.
  Patella or trochlea cartilage problems = pf_oa.
- "Tricompartmental" or "all three compartments" = medial_oa, lateral_oa AND pf_oa
  are all "yes".

Return JSON only.

REPORT:
{report}
"""


def load_cache(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                done[d["StudyInstanceUID"]] = d["labels"]
    return done


def is_labeled(df):
    lab = df[LABELS].apply(pd.to_numeric, errors="coerce")
    return lab.isin([0, 1]).all(axis=1)


def check_agreement(df, llm):
    from sklearn.metrics import roc_auc_score, cohen_kappa_score
    m = df[is_labeled(df)].merge(llm, on="StudyInstanceUID", suffixes=("", "_llm"))
    print(f"\nAgreement on {len(m)} human-labeled studies")
    print(f"{'label':18s} {'prev':>5s} {'AUC':>6s} {'kappa':>6s} {'sens':>6s} {'spec':>6s} {'unc':>5s}")
    rows = []
    for c in LABELS:
        y = m[c].astype(int).values
        p = m[c + "_llm"].values
        hard = (p >= 0.5).astype(int)  # uncertain counted as positive here
        auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
        kap = cohen_kappa_score(y, hard)
        sens = hard[y == 1].mean() if (y == 1).any() else np.nan
        spec = 1 - hard[y == 0].mean() if (y == 0).any() else np.nan
        unc = (p == 0.5).mean()
        rows.append(auc)
        print(f"{c:18s} {y.mean():5.2f} {auc:6.3f} {kap:6.3f} {sens:6.3f} {spec:6.3f} {unc:5.2f}")
    print(f"{'macro AUC':18s} {'':5s} {np.nanmean(rows):6.3f}")
    # Save disagreements so you can read them and fix the prompt
    bad = []
    for c in LABELS:
        d = m[(m[c].astype(float) - m[c + "_llm"]).abs() >= 0.5]
        for _, r in d.iterrows():
            bad.append({"StudyInstanceUID": r.StudyInstanceUID, "label": c,
                        "human": r[c], "llm": r[c + "_llm"], "Report": r.Report})
    pd.DataFrame(bad).to_csv("disagreements.csv", index=False)
    print(f"\n{len(bad)} disagreements saved to disagreements.csv")


MAX_CHARS = 6000  # cut very long reports so they fit the context


def parse(text):
    """Pull the JSON object out of the model reply. Returns dict or None."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in KEYS:
        v = str(d.get(k, "no")).strip().lower()
        out[k] = v if v in VALUE else "no"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--subset", choices=["labeled", "all"], default="labeled")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache", default="llm_cache_local.jsonl")
    ap.add_argument("--out", default="llm_labels_local.csv")
    ap.add_argument("--tp", type=int, default=2, help="number of GPUs")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    df = pd.read_csv(a.train)
    todo_df = df[is_labeled(df)] if a.subset == "labeled" else df
    todo_df = todo_df[todo_df["Report"].fillna("").str.strip() != ""]
    if a.limit:
        todo_df = todo_df.head(a.limit)

    done = load_cache(a.cache)
    todo = [(u, r) for u, r in zip(todo_df.StudyInstanceUID, todo_df.Report) if u not in done]
    print(f"{len(todo_df)} studies, {len(done)} cached, {len(todo)} to run ({a.model})")

    if todo:
        from vllm import LLM, SamplingParams
        llm = LLM(model=a.model, dtype="half", tensor_parallel_size=a.tp,
                  max_model_len=4096, gpu_memory_utilization=0.90, seed=0)
        sp = SamplingParams(temperature=0.0, max_tokens=300)

        for attempt in range(3):  # re-run the ones whose JSON could not be parsed
            if not todo:
                break
            convs = [[{"role": "user", "content": PROMPT.format(report=str(r)[:MAX_CHARS])}]
                     for _, r in todo]
            outs = llm.chat(convs, sp, use_tqdm=True)
            failed = []
            with open(a.cache, "a") as f:
                for (uid, rep), o in zip(todo, outs):
                    lab = parse(o.outputs[0].text)
                    if lab is None:
                        failed.append((uid, rep))
                        continue
                    done[uid] = lab
                    f.write(json.dumps({"StudyInstanceUID": uid, "labels": lab}) + "\n")
            print(f"pass {attempt + 1}: {len(todo) - len(failed)} ok, {len(failed)} unparsed")
            todo = failed
            sp = SamplingParams(temperature=0.3, max_tokens=300)  # small change so retries differ

    rows = [{"StudyInstanceUID": u, **{c: VALUE[l[k]] for c, k in zip(LABELS, KEYS)}}
            for u, l in done.items()]
    llm_df = pd.DataFrame(rows)
    llm_df.to_csv(a.out, index=False)
    print(f"Saved {len(llm_df)} rows to {a.out}")

    if a.check:
        check_agreement(df, llm_df)


if __name__ == "__main__":
    main()
