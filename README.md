# Skin Cancer Detection — Streamlit App

Best model: **DenseNet121** (test accuracy 0.790, F1 0.817, ROC-AUC 0.874)

## Before deploying
Copy `best_skin_cancer_model.keras` (from the zip your Colab notebook downloaded,
`skin_cancer_streamlit_app.zip`) into this folder, next to `app.py`.

## Deploy on Streamlit Community Cloud
1. Push the contents of this folder to a GitHub repository.
2. Go to https://share.streamlit.io/ -> **Create app** -> choose the repo, main file `app.py`.
3. Advanced settings -> Python **3.11** -> Deploy.

Run locally: `pip install -r requirements.txt && streamlit run app.py`
