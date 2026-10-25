"""Knee MRI reader: a research demo of my RSNA 2026 knee model.

Run:  streamlit run app.py
Needs model_fold0.pt in this folder (or model_fold0.zip from Kaggle).
"""
import base64, glob, io, json, os, re, zipfile

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

st.set_page_config(page_title="Knee MRI reader", layout="wide", initial_sidebar_state="collapsed")

HERE = os.path.dirname(os.path.abspath(__file__))
D, S = 24, 256                 # slices per view, pixels (same as training)
SLICE_BAND = (0.08, 0.92)      # same as preprocessing v2
BACKBONE = "resnet34.a1_in1k"
VIEWS = [("sag", "Sagittal"), ("cor", "Coronal"), ("ax", "Axial")]

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# Validation AUC of this exact model (fold 0, v2 preprocessing)
VAL_AUC = {"ACL": .863, "MCL": .742, "Medial Meniscus": .853, "Lateral Meniscus": .748,
           "Medial OA": .866, "Lateral OA": .757, "PF OA": .829, "Effusion": .821,
           "Synovitis": .792, "Baker's": .869, "Contusion": .742, "Fracture": .770}
GROUPS = [
    ("Ligaments", [("ACL", "Anterior cruciate ligament"), ("MCL", "Medial collateral ligament")]),
    ("Menisci", [("Medial Meniscus", "Medial meniscus tear"), ("Lateral Meniscus", "Lateral meniscus tear")]),
    ("Osteoarthritis", [("Medial OA", "Medial compartment"), ("Lateral OA", "Lateral compartment"),
                        ("PF OA", "Patellofemoral")]),
    ("Joint", [("Effusion", "Joint effusion"), ("Synovitis", "Synovitis"), ("Baker's", "Baker's cyst")]),
    ("Bone", [("Contusion", "Bone contusion"), ("Fracture", "Fracture")]),
]

# ---------------------------------------------------------------- style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{
  --paper:#11161C; --panel:#182029; --ink:#E4E9EE; --muted:#8B97A4; --line:#29323D;
  --steel:#78A9CC; --amber:#E3A33D; --amber-soft:rgba(227,163,61,.16); --well:#000000;
  --track:#26303A; --idle:#55626F;
}
html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, button, input, textarea {
  font-family:'IBM Plex Sans', system-ui, sans-serif !important;
}
.stApp{ background:var(--paper); color:var(--ink); }
#MainMenu, footer, header[data-testid="stHeader"]{ visibility:hidden; height:0; }
.block-container{ padding-top:1.6rem; padding-bottom:2rem; max-width:1440px; }

.topbar{ display:flex; justify-content:space-between; align-items:flex-end;
  border-bottom:1px solid var(--line); padding-bottom:14px; margin-bottom:6px; }
.topbar h1{ font-size:26px; font-weight:600; letter-spacing:-0.01em; margin:0; color:var(--ink); }
.topbar .sub{ font-size:14px; color:var(--muted); margin-top:4px; }
.topbar .warn{ font-size:13px; color:var(--muted); text-align:right; line-height:1.45; }

.stTabs [data-baseweb="tab-list"]{ gap:28px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{ padding:10px 2px; font-size:15px; color:var(--muted); background:none; }
.stTabs [aria-selected="true"]{ color:var(--ink) !important; font-weight:600; }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--steel); height:2px; }

.well{ background:var(--well); border-radius:6px; overflow:hidden; aspect-ratio:1/1;
  display:flex; align-items:center; justify-content:center; }
.well img{ width:100%; height:100%; object-fit:contain; display:block; }
.well.empty{ color:#7C8794; font-size:14px; }
.viewhead{ display:flex; justify-content:space-between; align-items:baseline; margin:2px 0 8px; }
.viewhead .name{ font-size:16px; font-weight:600; color:var(--ink); }
.viewhead .pos{ font-size:13px; color:var(--muted); font-variant-numeric:tabular-nums; }

.focus{ display:flex; gap:2px; height:14px; margin-top:2px; }
.focus span{ flex:1; border-radius:1px; background:var(--steel); }
.focus span.cur{ outline:2px solid var(--amber); outline-offset:1px; }
.focuscap{ font-size:12px; color:var(--muted); margin-top:6px; }

.findings{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px 20px 8px; }
.findings h3{ font-size:16px; font-weight:600; margin:0 0 2px; color:var(--ink); }
.findings .note{ font-size:12.5px; color:var(--muted); margin-bottom:10px; }
.grp{ font-size:13px; font-weight:600; color:var(--steel); margin:11px 0 4px; }
.row{ display:grid; grid-template-columns:1fr 52px; align-items:center; column-gap:10px; margin:0 0 6px; }
.row .lab{ font-size:14px; color:var(--ink); }
.row .lab small{ color:var(--muted); font-size:11.5px; margin-left:6px; }
.row .val{ font-size:14px; text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); }
.row.hit .lab, .row.hit .val{ color:var(--ink); font-weight:600; }
.bar{ grid-column:1 / span 2; height:5px; background:var(--track); border-radius:3px; margin-top:3px; }
.bar i{ display:block; height:100%; border-radius:3px; background:var(--idle); }
.row.hit .bar i{ background:var(--amber); }

.chip{ display:inline-block; font-size:12.5px; padding:2px 9px; border-radius:10px; font-weight:500; }
.chip.yes{ background:var(--amber-soft); color:var(--amber); }
.chip.unc{ background:rgba(120,169,204,.15); color:var(--steel); }
.chip.no{ background:var(--track); color:var(--muted); }

.credit{ font-size:12.5px; color:var(--muted); margin-top:8px; }
.summary{ font-size:15.5px; line-height:1.5; color:var(--ink); margin-top:22px; padding-top:14px;
  border-top:1px solid var(--line); max-width:72ch; }
.summary b{ font-weight:600; }
.kv, .kv tr, .kv td, .kv th{ border:none !important; background:none !important; }
.kv td{ border-bottom:1px solid var(--line) !important; padding:10px 0 !important; }
.stTextArea textarea{ background:var(--panel) !important; border:1px solid var(--line) !important; color:var(--ink) !important; }
.stTextInput input{ background:var(--panel) !important; color:var(--ink) !important; }
[data-testid="stExpander"] details{ background:var(--panel); border-color:var(--line) !important; }
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"]{ display:none !important; }
.prose{ font-size:15px; line-height:1.6; max-width:68ch; color:var(--ink); }
.kv{ width:100%; border-collapse:collapse; font-size:14.5px; }
.kv td{ padding:9px 0; border-bottom:1px solid var(--line); }
.kv td:last-child{ text-align:right; font-variant-numeric:tabular-nums; font-weight:600; }
.steps{ counter-reset:s; list-style:none; padding:0; margin:0; }
.steps li{ counter-increment:s; position:relative; padding:0 0 16px 40px; font-size:14.5px; line-height:1.5; }
.steps li::before{ content:counter(s); position:absolute; left:0; top:0; width:26px; height:26px;
  border-radius:50%; border:1.5px solid var(--steel); color:var(--steel); font-size:13px; font-weight:600;
  display:flex; align-items:center; justify-content:center; }
.steps li b{ display:block; font-weight:600; }
.stButton button{ border-radius:6px; border:1px solid var(--steel); color:var(--steel); background:transparent; font-weight:500; }
.stButton button[kind="primary"]{ background:var(--steel); color:#0B1117; border-color:var(--steel); }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- model
class Net(nn.Module):
    """Same architecture as train.py. forward() also returns the slice attention."""
    def __init__(self):
        super().__init__()
        import timm
        self.enc = timm.create_model(BACKBONE, pretrained=False, num_classes=0)
        C = self.enc.num_features
        self.att = nn.ModuleList([nn.Sequential(nn.Linear(C, 128), nn.Tanh(), nn.Linear(128, 1))
                                  for _ in range(3)])
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(3 * 2 * C, 512), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(512, 12))

    def forward(self, x):
        B, V, N, Cin, Sz, _ = x.shape
        f = self.enc(x.reshape(B * V * N, Cin, Sz, Sz)).view(B, V, N, -1)
        feats, atts = [], []
        for v in range(V):
            fv = f[:, v]
            w = torch.softmax(self.att[v](fv).float(), dim=1)
            atts.append(w[..., 0])
            feats.append(torch.cat([(w * fv).sum(1), fv.max(1).values], dim=1))
        return self.head(torch.cat(feats, dim=1)), torch.stack(atts, 1)  # (B,12), (B,3,N)


def _weights_path():
    pt = os.path.join(HERE, "model_fold0.pt")
    if os.path.isfile(pt):
        return pt
    zp = os.path.join(HERE, "model_fold0.zip")
    if os.path.isfile(zp):  # Kaggle download that Windows unpacked and re-zipped
        with zipfile.ZipFile(zp) as zin, zipfile.ZipFile(pt, "w", zipfile.ZIP_STORED) as zout:
            for info in zin.infolist():
                if info.is_dir():
                    continue
                zi = zipfile.ZipInfo(info.filename, (2026, 1, 1, 0, 0, 0))
                zout.writestr(zi, zin.read(info.filename))
        return pt
    return None


@st.cache_resource(show_spinner=False)
def load_model():
    path = _weights_path()
    if path is None:
        return None
    m = Net()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


# ---------------------------------------------------------------- images
def natural_key(name):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(name))]


def read_gray(src):
    img = Image.open(src).convert("L")
    return np.asarray(img, dtype=np.float32)


def load_folder(folder):
    out = {}
    for key, _ in VIEWS:
        files = sorted(glob.glob(os.path.join(folder, f"{key}_*.*")) +
                       glob.glob(os.path.join(folder, key, "*.*")), key=natural_key)
        files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if files:
            out[key] = [read_gray(f) for f in files]
    return out


def model_input(slices, crop):
    """Slices of one view -> (D, S, S) uint8, plus which input slice each model slice came from."""
    import cv2
    h, w = slices[0].shape
    vol = np.stack([s if s.shape == (h, w) else cv2.resize(s, (w, h)) for s in slices])
    n = len(vol)
    idx = np.linspace(SLICE_BAND[0] * (n - 1), SLICE_BAND[1] * (n - 1), D).round().astype(int)
    vol = vol[idx]
    ch, cw = int(h * crop), int(w * crop)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    vol = vol[:, y0:y0 + ch, x0:x0 + cw]
    lo, hi = np.percentile(vol, [1, 99.5])
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    out = np.stack([cv2.resize(sl, (S, S), interpolation=cv2.INTER_AREA) for sl in vol])
    return (out * 255).astype(np.uint8), idx


@torch.no_grad()
def analyse(model, study, crop):
    arrs, idxs = [], []
    for key, _ in VIEWS:
        a, i = model_input(study[key], crop)
        arrs.append(a); idxs.append(i)
    x = torch.from_numpy(np.stack(arrs)).float() / 255.0           # (3, D, S, S)
    x = ((x.view(3, D // 3, 3, S, S) - 0.45) / 0.25).unsqueeze(0)
    logits, att = model(x)
    probs = torch.sigmoid(logits)[0].numpy()
    focus = {}
    for v, (key, _) in enumerate(VIEWS):
        w = att[0, v].numpy()                      # one weight per group of 3 model slices
        per = np.zeros(len(study[key]))
        for g, wg in enumerate(w):
            for j in idxs[v][3 * g:3 * g + 3]:
                per[j] = max(per[j], wg)
        focus[key] = per / per.max() if per.max() > 0 else per
    return dict(zip(LABELS, probs.tolist())), focus


def to_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- report labeling
KEYS = ["acl", "mcl", "medial_meniscus", "lateral_meniscus", "medial_oa", "lateral_oa",
        "pf_oa", "effusion", "synovitis", "bakers", "contusion", "fracture"]
PROMPT = """You are a musculoskeletal radiologist. Read this knee MRI report.
It may be in any language. Decide if each finding is PRESENT in this knee.
acl: anterior cruciate ligament injury. mcl: medial collateral ligament injury.
medial_meniscus / lateral_meniscus: meniscus tear. medial_oa / lateral_oa: tibiofemoral
osteoarthritis or cartilage loss in that compartment. pf_oa: patellofemoral osteoarthritis or
patella/trochlea cartilage loss. effusion: joint effusion. synovitis: synovitis.
bakers: Baker's (popliteal) cyst. contusion: bone contusion / bone bruise. fracture: fracture.
Answer "yes", "no", or "uncertain" (only if the report itself is unsure). Return JSON only.

REPORT:
{report}"""

SAMPLES = {
    "English": "MRI right knee. There is a complete tear of the anterior cruciate ligament with "
               "bone bruising of the lateral femoral condyle and posterolateral tibial plateau. "
               "Horizontal tear of the posterior horn of the medial meniscus. Lateral meniscus intact. "
               "MCL is thickened with surrounding edema, in keeping with a grade 2 sprain. "
               "Moderate joint effusion. Small Baker's cyst. Cartilage is preserved.",
    "Spanish": "RM de rodilla izquierda. Rotura completa del ligamento cruzado anterior. "
               "Edema óseo en el cóndilo femoral lateral. Meniscos sin signos de rotura. "
               "Ligamento colateral medial íntegro. Leve derrame articular. "
               "Condropatía rotuliana grado II. No se observan quistes poplíteos.",
    "German": "MRT Kniegelenk rechts. Degenerativer Horizontalriss im Hinterhorn des Innenmeniskus. "
              "Fortgeschrittene Chondropathie im medialen Kompartiment mit Osteophyten. "
              "Kreuzbänder und Seitenbänder intakt. Mäßiger Gelenkerguss mit Synovialitis. "
              "Baker-Zyste von 2 cm. Keine Fraktur.",
}


def label_report(text, api_key):
    from google import genai
    from google.genai import types
    enum = {"type": "STRING", "enum": ["yes", "no", "uncertain"]}
    schema = {"type": "OBJECT", "properties": {k: enum for k in KEYS}, "required": KEYS}
    client = genai.Client(api_key=api_key)
    r = client.models.generate_content(
        model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
        contents=PROMPT.format(report=text),
        config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json",
                                           response_schema=schema))
    out = json.loads(r.text)
    return {lab: out.get(k, "no") for lab, k in zip(LABELS, KEYS)}


# ---------------------------------------------------------------- render helpers
def findings_html(probs, title, note, as_auc=False):
    h = [f'<div class="findings"><h3>{title}</h3><div class="note">{note}</div>']
    for grp, items in GROUPS:
        h.append(f'<div class="grp">{grp}</div>')
        for key, name in items:
            p = probs[key]
            hit = " hit" if (p >= 0.5 and not as_auc) else ""
            small = "" if as_auc else f"<small>AUC {VAL_AUC[key]:.2f}</small>"
            val = f"{p:.2f}" if as_auc else f"{p * 100:.0f}%"
            h.append(f'<div class="row{hit}"><div class="lab">{name}{small}</div>'
                     f'<div class="val">{val}</div>'
                     f'<div class="bar"><i style="width:{max(p * 100, 1.5):.1f}%"></i></div></div>')
    h.append("</div>")
    return "".join(h)


def report_html(res):
    h = ['<div class="findings"><h3>Labels found in the report</h3>'
         '<div class="note">These are the labels my pipeline used to train the image model.</div>']
    names = dict(n for _, items in GROUPS for n in items)
    for grp, items in GROUPS:
        h.append(f'<div class="grp">{grp}</div>')
        for key, name in items:
            v = res[key]
            cls, txt = {"yes": ("yes", "Present"), "uncertain": ("unc", "Uncertain")}.get(v, ("no", "Absent"))
            h.append(f'<div class="row"><div class="lab">{names[key]}</div>'
                     f'<div class="val"><span class="chip {cls}">{txt}</span></div></div>')
    h.append("</div>")
    return "".join(h)


# ---------------------------------------------------------------- page
st.markdown("""
<div class="topbar">
  <div><h1>Knee MRI reader</h1>
  <div class="sub">Twelve knee findings from three MRI views, with the slices the model looked at.</div></div>
  <div class="warn">Research demo built for the RSNA 2026 Kaggle challenge.<br>Not for clinical use.</div>
</div>""", unsafe_allow_html=True)

tab_scan, tab_report, tab_model = st.tabs(["Read a scan", "Read a report", "How it works"])

# ---- tab 1: scan
with tab_scan:
    model = load_model()
    if model is None:
        st.error("Model weights not found. Put model_fold0.pt (or model_fold0.zip from Kaggle) "
                 "in the same folder as app.py, then reload the page.")
        st.stop()

    with st.expander("Load a study", expanded="study" not in st.session_state):
        c1, c2 = st.columns([3, 1])
        folder = c1.text_input("Folder with the images", value=os.path.join(HERE, "demo_case"),
                               help="Name the files sag_01.jpg, cor_01.jpg, ax_01.jpg and so on, "
                                    "in slice order.")
        c2.write(""); c2.write("")
        if c2.button("Load folder", use_container_width=True):
            s = load_folder(folder)
            if len(s) < 3:
                st.error(f"Found {', '.join(s) or 'no'} views in that folder. Each of sag_, cor_ and "
                         f"ax_ needs at least 3 images.")
            else:
                st.session_state.study = s
                st.session_state.pop("result", None)
        st.caption("Or upload the slices of each view. Files are sorted by name.")
        up = st.columns(3)
        uploads = {key: up[i].file_uploader(title, type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                            key=f"up_{key}") for i, (key, title) in enumerate(VIEWS)}
        if all(uploads.values()) and st.button("Use uploaded images"):
            st.session_state.study = {k: [read_gray(f) for f in sorted(v, key=lambda f: natural_key(f.name))]
                                      for k, v in uploads.items()}
            st.session_state.pop("result", None)
        credit = st.text_input("Image credit shown under the viewer",
                               value=st.session_state.get("credit",
                                     "Case courtesy of Mostafa Elfeky, Radiopaedia.org, rID: 71883"))
        st.session_state.credit = credit
        crop = st.slider("Field of view kept for the model", 0.6, 1.0, 0.85, 0.05,
                         help="The model was trained on a 130 mm square around the knee. "
                              "These images have no pixel size, so this crop approximates it.")
        st.session_state.crop = crop

    study = st.session_state.get("study")
    if not study:
        st.markdown('<p class="prose" style="margin-top:28px">Load a knee MRI study to begin. '
                    'You need sagittal, coronal and axial slices, ideally fat suppressed or '
                    'fluid sensitive, the same kind of series the model was trained on.</p>',
                    unsafe_allow_html=True)
        st.stop()

    left, right = st.columns([3.1, 1.25], gap="large")
    with right:
        if "result" not in st.session_state:
            st.markdown('<div class="findings"><h3>Findings</h3><div class="note">'
                        'Run the model to see the probability of each finding.</div></div>',
                        unsafe_allow_html=True)
            st.write("")
            if st.button("Analyse study", type="primary", use_container_width=True):
                with st.spinner("Reading 72 slices"):
                    st.session_state.result = analyse(model, study, st.session_state.get("crop", 0.85))
                st.rerun()
        else:
            probs, _ = st.session_state.result
            st.markdown(findings_html(probs, "Findings",
                                      "Probability from the model. AUC is its validation score "
                                      "for that finding."), unsafe_allow_html=True)
    with left:
        cols = st.columns(3, gap="medium")
        focus = st.session_state.result[1] if "result" in st.session_state else None
        for col, (key, title) in zip(cols, VIEWS):
            sl = study[key]
            with col:
                if focus is not None:
                    default = int(np.argmax(focus[key]))
                else:
                    default = len(sl) // 2
                frame = st.empty()
                i = st.slider(title, 1, len(sl), default + 1, key=f"s_{key}_{focus is not None}",
                              label_visibility="collapsed") - 1
                frame.markdown(f'<div class="viewhead"><span class="name">{title}</span>'
                               f'<span class="pos">Slice {i + 1} of {len(sl)}</span></div>'
                               f'<div class="well"><img src="data:image/png;base64,{to_b64(sl[i])}"></div>',
                               unsafe_allow_html=True)
                if focus is not None:
                    cells = "".join(f'<span class="{"cur" if j == i else ""}" '
                                    f'style="opacity:{0.12 + 0.88 * f:.2f}"></span>'
                                    for j, f in enumerate(focus[key]))
                    st.markdown(f'<div class="focus">{cells}</div>'
                                f'<div class="focuscap">Model focus across slices. '
                                f'Strongest at slice {int(np.argmax(focus[key])) + 1}.</div>',
                                unsafe_allow_html=True)
        if "result" in st.session_state:
            names = dict(n for _, items in GROUPS for n in items)
            probs = st.session_state.result[0]
            top = sorted([(p, k) for k, p in probs.items() if p >= 0.5], reverse=True)
            short = {**{k: n.lower() for k, n in names.items()},
                     "Medial OA": "medial osteoarthritis", "Lateral OA": "lateral osteoarthritis",
                     "PF OA": "patellofemoral osteoarthritis", "ACL": "ACL injury", "MCL": "MCL injury",
                     "Baker's": "Baker's cyst"}
            line = (", ".join(f"{short[k]} ({p * 100:.0f}%)" for p, k in top)
                    if top else "nothing above 50%")
            st.markdown(f'<div class="summary"><b>Most likely:</b> {line}.</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="credit">{st.session_state.get("credit", "")}</div>', unsafe_allow_html=True)

# ---- tab 2: report
with tab_report:
    st.markdown('<p class="prose">Only 58 of the 4,407 training studies had labels. Every study had its '
                'radiology report, written in many languages. I used open LLMs (Qwen2.5 7B and 14B) to '
                'turn each report into the 12 labels, and checked them against the 58 labeled studies '
                '(macro AUC 0.84). This page runs the same prompt through Gemini so it works on a laptop.</p>',
                unsafe_allow_html=True)
    l, r = st.columns([1.35, 1], gap="large")
    with l:
        pick = st.radio("Example report", list(SAMPLES), horizontal=True)
        text = st.text_area("Report text", value=SAMPLES[pick], height=230, key=f"rep_{pick}")
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            key = st.text_input("Gemini API key (free from Google AI Studio)", type="password")
        go = st.button("Extract labels", type="primary", disabled=not key)
        st.caption("The examples are written for this demo. They are not from the challenge data.")
    with r:
        if go:
            try:
                with st.spinner("Reading the report"):
                    st.session_state.rep = label_report(text, key)
            except Exception as e:
                st.error(f"The model call failed: {str(e)[:200]}")
        if "rep" in st.session_state:
            st.markdown(report_html(st.session_state.rep), unsafe_allow_html=True)
        else:
            st.markdown('<div class="findings"><h3>Labels found in the report</h3><div class="note">'
                        'Choose an example or paste a report, then extract the labels.</div></div>',
                        unsafe_allow_html=True)

# ---- tab 3: how it works
with tab_model:
    a, b = st.columns([1.25, 1], gap="large")
    with a:
        st.markdown("""
<ol class="steps">
<li><b>Labels from reports</b>Two open LLMs read all 4,407 reports and produced soft labels (yes, no,
uncertain). Their average agreed with the 58 human-labeled studies at macro AUC 0.84.</li>
<li><b>One series per plane</b>For each study, pick the fluid-sensitive series in the sagittal,
coronal and axial planes.</li>
<li><b>Orientation and crop</b>Use the DICOM geometry so every scan faces the same way, keep a 130 mm
square around the knee, drop the outer slices, and resample to 24 slices of 256 px.</li>
<li><b>2.5D network</b>Every 3 neighbouring slices form one image. A ResNet34 reads all 24 images,
attention picks the important slices in each plane, and one head predicts the 12 findings.</li>
<li><b>Training</b>10 epochs on two T4 GPUs on Kaggle, about 94 minutes, with mirror augmentation.</li>
</ol>""", unsafe_allow_html=True)
        st.markdown("""
<table class="kv">
<tr><td>Private leaderboard AUC, this model</td><td>0.835</td></tr>
<tr><td>Public leaderboard AUC, this model</td><td>0.853</td></tr>
<tr><td>First version (no crop, 224 px), private</td><td>0.823</td></tr>
<tr><td>Report labels against human labels</td><td>0.84</td></tr>
<tr><td>Training studies</td><td>4,407</td></tr>
</table>""", unsafe_allow_html=True)
    with b:
        st.markdown(findings_html(VAL_AUC, "Validation AUC per finding",
                                  "Held-out fold of 882 studies, scored against the report labels. "
                                  "Macro AUC 0.804.", as_auc=True), unsafe_allow_html=True)