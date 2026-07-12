# 📊 End-to-End Sales Forecasting & Demand Intelligence System

**Xylofy AI — Week 3 & 4 Internship Project**  
**Author:** Dhruv | **Date:** July 2026

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 🎯 Problem Statement

Build an intelligent sales forecasting system that predicts future product demand, detects unusual sales spikes/drops, segments products by demand pattern, and presents everything through a deployed interactive dashboard.

## 📁 Project Structure

```
├── analysis.ipynb          # Complete Jupyter Notebook (8 tasks)
├── analysis_with_output.ipynb  # Executed notebook with all outputs
├── app.py                  # Streamlit Dashboard (4 pages)
├── train.csv               # Superstore Sales Dataset (9,800 rows)
├── vgsales.csv             # Video Game Sales Dataset (supplementary)
├── requirements.txt        # Python dependencies
├── summary.pdf             # 2-page Executive Business Report
├── charts/                 # All chart images as .png
└── .streamlit/config.toml  # Streamlit theme configuration
```

## 📋 Tasks Completed

| Task | Description | Status |
|------|-------------|--------|
| **Task 1** | Data Loading, Merging & Deep Exploration | ✅ |
| **Task 2** | Time Series Analysis & Decomposition | ✅ |
| **Task 3** | Sales Forecasting — SARIMA, Prophet, XGBoost | ✅ |
| **Task 4** | Product Category & Region Level Forecasting | ✅ |
| **Task 5** | Anomaly Detection (Isolation Forest + Z-Score) | ✅ |
| **Task 6** | Product Demand Segmentation (K-Means) | ✅ |
| **Task 7** | Interactive Streamlit Dashboard | ✅ |
| **Task 8** | Executive Business Report | ✅ |

## 🛠️ Tech Stack

- **Python 3.x** — Core language
- **Pandas & NumPy** — Data manipulation
- **Statsmodels** — SARIMA, decomposition, ADF test
- **Prophet** — Facebook's forecasting library
- **XGBoost** — ML-based time series forecasting
- **Scikit-learn** — Isolation Forest, K-Means, PCA
- **Matplotlib & Seaborn** — Static visualizations
- **Plotly** — Interactive charts
- **Streamlit** — Dashboard deployment

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📊 Dashboard Pages

1. **Sales Overview** — KPIs, yearly bar chart, monthly trends, region/category filters
2. **Forecast Explorer** — Select category/region, choose horizon, see XGBoost predictions
3. **Anomaly Report** — Flagged anomalous weeks with sales values
4. **Product Demand Segments** — Cluster visualization with stocking recommendations

---

*Built as part of the Xylofy AI Data Science Internship Program*
