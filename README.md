# Feinstein & Zhang Risk-Neutral Option pricing

Local Streamlit app: CRR binomial tree for a European call, volatility from historical **Adj Close** returns, put price from put–call parity, and optional PNG export of the trees.

## Run locally

```bash
cd "e:\Babson\Risk Neutral Tree"
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Yahoo Finance copy-paste data

If you paste history from Yahoo Finance, dividend-only rows (for example a cell like `0.25 Dividend`) are detected automatically: those rows are removed from the return series, and the amount is stored in a **`dividend`** column on the matching date so regular price rows stay aligned.

## Deploy on Streamlit Community Cloud

Deployment is done from a **GitHub** repository (Streamlit cannot deploy directly from only a local folder).

1. Create a GitHub repository and push this folder (include `app.py`, `requirements.txt`, and optionally `Apple stock Price.xlsx` if you want a default file on the server).
2. Sign in at [Streamlit Community Cloud](https://streamlit.io/cloud).
3. **New app** → select the repo → **Main file path**: `app.py`.
4. Deploy. If the default Excel file is not in the repo, users can still upload a `.xlsx` in the app.

`st.graphviz_chart` needs Graphviz support in the host environment; if the on-screen graph fails on Cloud, the **Download** PNG trees (Pillow-based) should still work.
