# 🏦 Credit Scoring AI — Machine Learning Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-189947?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-5.22-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

**A production-ready, end-to-end Machine Learning system for credit risk assessment**  
Built as part of the **CodeAlpha Machine Learning Internship**

[🚀 Live Demo](#deployment) • [📊 Dataset](#dataset) • [🛠️ Installation](#installation) • [📖 Documentation](#project-structure)

</div>

---

## 📌 Project Overview

This project implements a complete, professional **Credit Scoring System** that predicts whether a loan applicant is **creditworthy** or likely to **default** — using real financial and personal data.

It covers the **full ML lifecycle**:
- Real dataset ingestion from UCI ML Repository
- Data cleaning, EDA, and feature engineering
- Training & comparing 5 ML classification models
- Hyperparameter tuning with GridSearchCV
- SHAP-based model explainability
- A polished, interactive Streamlit web dashboard
- Downloadable prediction reports

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Real Dataset** | UCI German Credit Dataset — 1,000 real applicants, 20 features |
| 🔬 **Deep EDA** | Correlation heatmaps, 3D scatter plots, risk trend analysis |
| ⚙️ **Feature Engineering** | Debt ratios, age groups, credit tiers, encoding & scaling |
| 🤖 **5 ML Models** | Logistic Regression, Decision Tree, Random Forest, GBM, XGBoost |
| 📈 **Full Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
| 🎛️ **Hyperparameter Tuning** | GridSearchCV with 5-fold cross-validation |
| 🔮 **Live Prediction** | Interactive form with risk gauge and confidence breakdown |
| 🧠 **Explainable AI** | SHAP values + feature importance for every prediction |
| 📥 **Export Reports** | Download prediction as CSV or JSON |
| 🌙 **Dark Mode UI** | Modern, polished Streamlit dashboard |

---

## 🛠️ Tech Stack

```
Backend:          Python 3.10+
ML Framework:     Scikit-learn, XGBoost
Data:             Pandas, NumPy, SciPy
Visualization:    Plotly, Seaborn, Matplotlib
Web App:          Streamlit
Explainability:   SHAP
Model Saving:     Joblib
```

---

## 📊 Dataset

**German Credit Dataset** — UCI Machine Learning Repository  
Created by: Prof. Dr. Hans Hofmann, Universität Hamburg (1994)

| Property | Value |
|----------|-------|
| Samples | 1,000 |
| Features | 20 (financial + personal) |
| Target | 1 = Creditworthy, 0 = Default |
| Class Split | 70% Good / 30% Bad |
| Missing Values | None |
| Source | [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) |

**Key Features:**
- Checking account status
- Credit history
- Loan purpose & amount
- Savings account balance
- Employment duration
- Age, housing, job category
- Number of dependents

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/CodeAlpha_CreditScoring.git
cd CodeAlpha_CreditScoring
```

### Step 2: Create virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the model
```bash
python train_model.py
```
> This downloads the real dataset, trains 5 models, tunes the best one, and saves `credit_model.pkl`.

### Step 5: Launch the app
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📁 Project Structure

```
CodeAlpha_CreditScoring/
│
├── app.py                      # Main Streamlit app (entry point)
├── train_model.py              # Standalone training pipeline
├── credit_model.pkl            # Saved model bundle (after training)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── utils/                      # Core ML modules
│   ├── __init__.py
│   ├── data_loader.py          # Dataset loading, cleaning, feature engineering
│   ├── model_trainer.py        # Training, evaluation, tuning, prediction
│   └── visualizations.py      # All Plotly chart functions
│
├── pages/                      # Streamlit page modules
│   ├── __init__.py
│   ├── home.py                 # Landing page
│   ├── dataset_insights.py     # Data preview & statistics
│   ├── eda.py                  # Exploratory data analysis
│   ├── model_training.py       # Training UI & metrics
│   ├── prediction.py           # Live prediction form
│   ├── explainability.py       # SHAP & feature importance
│   └── about.py                # Project info & deployment
│
├── data/                       # Dataset files
│   ├── german.data             # Raw UCI dataset (auto-downloaded)
│   └── german_credit_cleaned.csv
│
├── models/                     # Saved model files
│   ├── random_forest_bundle.pkl
│   └── model_metadata.json
│
├── notebooks/                  # Jupyter notebooks (EDA & experiments)
│   └── credit_scoring_eda.ipynb
│
├── screenshots/                # App screenshots for README
└── assets/                     # Static assets
```

---

## 📈 Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~75% | ~0.83 | ~0.84 | ~0.84 | ~0.78 |
| Decision Tree | ~72% | ~0.80 | ~0.82 | ~0.81 | ~0.72 |
| **Random Forest** | **~79%** | **~0.86** | **~0.87** | **~0.87** | **~0.83** |
| Gradient Boosting | ~78% | ~0.85 | ~0.86 | ~0.86 | ~0.82 |
| XGBoost | ~80% | ~0.87 | ~0.87 | ~0.87 | ~0.84 |

> Results vary slightly based on random seed and tuning. XGBoost or Random Forest typically performs best.

---

## 🖥️ Screenshots

> Add screenshots to the `screenshots/` folder and link them here after running the app.

```
screenshots/
├── home_page.png
├── eda_charts.png
├── model_comparison.png
├── live_prediction.png
└── shap_explanation.png
```

---

## 🚀 Deployment

### Streamlit Cloud (Recommended — Free)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as the entry point
4. Deploy!

### Render
```bash
# Start command:
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### HuggingFace Spaces
- Choose **Streamlit** SDK when creating a Space
- Upload project files or connect GitHub
- Add `sdk: streamlit` to README YAML header

---

## 🔮 Future Improvements

- [ ] Deep learning model (PyTorch/Keras) comparison
- [ ] SMOTE oversampling for class imbalance
- [ ] FastAPI REST endpoint for programmatic access
- [ ] Batch prediction (upload CSV of applicants)
- [ ] Real-time drift monitoring
- [ ] Docker containerization
- [ ] CI/CD with GitHub Actions

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**CodeAlpha Internship Project**  
Machine Learning Track | Credit Scoring Task

---

## 🙏 Acknowledgements

- **UCI ML Repository** for the German Credit Dataset
- **Prof. Dr. Hans Hofmann** for creating the benchmark dataset
- **CodeAlpha** for the internship opportunity
- **Scikit-learn, XGBoost, SHAP, Plotly, Streamlit** open source communities

---

<div align="center">
  <b>⭐ Star this repo if you found it helpful!</b><br>
  CodeAlpha ML Internship
</div>
