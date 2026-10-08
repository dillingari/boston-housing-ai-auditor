# 🛡️ Boston Housing AI Auditor

An enterprise-grade, automated pre-modeling data-quality audit, governance, and rectification platform built for the Boston Housing dataset.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

---

## 📌 Overview

Before training machine learning models on the Boston Housing dataset, data scientists must diagnose data quality anomalies, distributional distortions, severe multicollinearity, and leakage hazards. 

The **Boston Housing AI Auditor** executes this entire pre-modeling audit workflow across 6 structured steps:
1. **Variable Inventory & Role Inference:** Automatic mapping of variables against the data dictionary, inferring `MEDV` as target and 9 features as predictors.
2. **Dataset-Level Checks:** Integrity verification across 506 census tracts and 10 columns (0 duplicates, 0.10% overall missingness).
3. **Variable Scorecards & Deep-Dives:** Univariate distributions, 1.5×IQR and $|z| > 3$ outlier screening, boundary/censoring detection (e.g., top-coded home values at $50k).
4. **Multicollinearity & Risk Diagnosis:** Pairwise correlation matrix, Variance Inflation Factors (VIF), and variance scale disparities across 5 orders of magnitude.
5. **Prioritized Issue Matrix:** Prioritization of all issues by severity (High, Medium, Low) with empirical evidence and mathematical justifications.
6. **Executive PDF & Corrected Dataset Deliverables:**
   - **Executive PDF Report:** Publication-grade multi-page audit report generated via ReportLab (`Boston_Housing_Audit_Report.pdf`).
   - **Rectified Dataset Workbook:** Multi-sheet Excel workbook (`Boston_Housing_Corr.xlsx`) with `DATA_CORR`, `CORRECTIONS_LOG`, `BEFORE_AFTER`, and `DATA DICT`, plus `Boston_Housing_Corr.csv`.

---

## 🚀 Deployment to Streamlit Community Cloud

This repository is fully configured and ready for 1-click deployment on [Streamlit Community Cloud](https://share.streamlit.io/):

1. **Push this repository to GitHub.**
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **"New app"**.
3. Select your repository and branch (`main`).
4. Set **Main file path** to:
   ```text
   app.py
   ```
5. Click **"Deploy!"**.

All file references in the codebase use dynamic relative paths (`pathlib.Path(__file__).resolve().parent`) and will seamlessly resolve on Streamlit Cloud Linux containers.

---

## 💻 Local Testing & Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<your-username>/boston-housing-ai-auditor.git
cd boston-housing-ai-auditor
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📂 Repository File Structure

```text
boston-housing-ai-auditor/
├── app.py                           # Main Streamlit web application
├── pdf_generator.py                 # ReportLab executive PDF compilation engine
├── corrector.py                     # Automated preprocessing rectification engine
├── Boston_Housing.xlsx              # Raw input dataset & data dictionary
├── requirements.txt                 # Pinned dependencies for Streamlit Cloud
├── .gitignore                       # Clean Git configuration
├── README.md                        # Documentation & setup guide
├── Boston_Housing_Audit_Report.pdf  # Generated executive PDF audit deliverable
├── Boston_Housing_Corr.xlsx         # Generated 4-sheet corrected Excel deliverable
└── Boston_Housing_Corr.csv          # Generated corrected CSV deliverable
```

---

## 📋 Preprocessing Order Protocol (Leakage Prevention)

To guarantee 100% prevention of data leakage:
1. **Split First:** Partition dataset into train and test splits (e.g. 80/20).
2. **Fit on Training Set Only:** Compute `RM` imputation median ($6.208$) and `CRIM` winsorization percentiles ($41.37$) strictly on the training partition.
3. **Transform Predictors:** Apply fitted imputers and `log1p` feature transformations downstream to test features without re-fitting.
4. **Target Handling:** Train models on `log1p(MEDV)`; during test evaluation, invert predictions via $\exp(\hat{y}) - 1$ to compute un-biased test metrics in original dollar terms.
