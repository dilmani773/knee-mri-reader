# Knee MRI reader

A research demo of my model for the RSNA 2026 Knee Abnormality Detection challenge on Kaggle.
It predicts 12 knee findings from sagittal, coronal and axial MRI slices and shows which
slices the model paid attention to. Not for clinical use.

## Run on Windows

1. Put `model_fold0.pt` (or `model_fold0.zip` downloaded from Kaggle) in this folder.
2. Put the study images in `demo_case`, named in slice order:
   `sag_01.jpg ... sag_24.jpg`, `cor_01.jpg ...`, `ax_01.jpg ...`
3. In a terminal opened in this folder:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

For the report tab, set a Gemini API key first (free from Google AI Studio):

```
$env:GEMINI_API_KEY="your-key"
streamlit run app.py
```

## Notes

The model weights and images are not included. The challenge data cannot be shared, so the demo
uses public cases from Radiopaedia.org (CC BY-NC-SA 3.0), credited in the app.
