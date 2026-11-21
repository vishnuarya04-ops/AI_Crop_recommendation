# User Guide — Crop Recommendation System

A step-by-step guide to setting up and running the AI-based Crop Recommendation System.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| **Python** | 3.10 or higher |
| **pip** | Latest (comes with Python) |
| **Git** | Optional (for cloning) |
| **OS** | Windows 10/11, macOS, or Linux |

---

## 1. Get the Code

**Option A — Clone from GitHub:**
```bash
git clone https://github.com/<your-username>/Crop-Recommendation-System.git
cd Crop-Recommendation-System
```

**Option B — Download ZIP:**
Download the repository as a ZIP file and extract it.

---

## 2. Set Up the Virtual Environment

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your terminal prompt after activation.

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages including scikit-learn, XGBoost, pandas, matplotlib, seaborn, SHAP, Streamlit, and Jupyter.

---

## 4. Running the Notebooks (Phases 2–6)

The analysis notebooks should be run in order. Each notebook builds on outputs from the previous phase.

### Start Jupyter

```bash
jupyter notebook
```

This opens the Jupyter interface in your browser. Navigate to the project folder and open each notebook.

### Execution Order

| Order | Notebook | Purpose | Estimated Time |
|-------|----------|---------|----------------|
| 1 | `Phase2_EDA.ipynb` | Data exploration & preprocessing | ~2 min |
| 2 | `Phase3_Feature_Engineering.ipynb` | Feature scaling, encoding, train/test split | ~1 min |
| 3 | `Phase4_Model_Development.ipynb` | Train 7 ML models | ~3 min |
| 4 | `Phase5_Model_Evaluation.ipynb` | Evaluate & compare all models | ~5 min |
| 5 | `Phase6_Feature_Importance.ipynb` | MDI, Permutation, SHAP analysis | ~5 min |

**To run a notebook:** Open it → Kernel → Restart & Run All

> **Note:** Phase 3 creates `preprocessing_pipeline.joblib`, `X_train.csv`, `X_test.csv`, etc. These files are required by subsequent phases. If you skip Phase 3, later notebooks will fail.

---

## 5. Launching the Streamlit App (Phase 7)

The Decision Support System is a web application built with Streamlit.

### Start the App

```bash
streamlit run app.py
```

The app will launch and display:
```
  Local URL: http://localhost:8501
  Network URL: http://<your-ip>:8501
```

Open `http://localhost:8501` in your browser.

### Using the App

1. **Select an Example Scenario** *(optional)*
   - Use the dropdown at the top to auto-fill values for a known crop (e.g., "Rice — tropical lowlands")

2. **Enter Soil & Climate Data**
   - **Soil Nutrients:** Nitrogen (N), Phosphorus (P), Potassium (K) in kg/ha
   - **Climate:** Temperature (°C), Humidity (%)
   - **Soil:** pH value (3.5–10.0)
   - **Rainfall:** Annual rainfall in mm

3. **Click "🚀 Get Recommendation"**

4. **View Results**
   - **Primary recommendation** — The best crop with a confidence percentage
   - **Runner-up crops** — Alternative crops with their probabilities
   - **Probability chart** — Distribution across all 22 crops
   - **Input summary** — Recap of your entered values

5. **Sidebar** — Click the `>` icon (top-left) to view model details, accuracy stats, and how-it-works information.

### Stopping the App

Press `Ctrl+C` in the terminal where Streamlit is running.

---

## 6. Understanding the Input Fields

| Field | What to Enter | Range | Example |
|-------|---------------|-------|---------|
| **Nitrogen (N)** | Soil nitrogen content from a soil test report | 0–140 kg/ha | 80 |
| **Phosphorus (P)** | Soil phosphorus content | 5–145 kg/ha | 48 |
| **Potassium (K)** | Soil potassium content | 5–205 kg/ha | 40 |
| **Temperature** | Average temperature during the growing season | 8–44 °C | 23.5 |
| **Humidity** | Relative humidity in your area | 14–100% | 82.0 |
| **pH** | Soil pH from a soil test (acidic < 7 < alkaline) | 3.5–10.0 | 6.5 |
| **Rainfall** | Total annual rainfall in your region | 20–300 mm | 236.0 |

> **Tip:** If you don't know your soil values, contact your local agricultural extension office for a soil test report.

---

## 7. Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install streamlit` inside your activated venv |
| `❌ Model file not found` | Run Phase 3 & 4 notebooks first to generate model files |
| `❌ Pipeline file not found` | Run Phase 3 notebook to create `preprocessing_pipeline.joblib` |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| Jupyter kernel not found | Install: `pip install ipykernel && python -m ipykernel install --user` |
| Permission denied (Windows) | Use PowerShell as Administrator, or run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

### Getting Help

- **Streamlit docs:** https://docs.streamlit.io
- **Scikit-learn docs:** https://scikit-learn.org/stable/documentation.html
- **XGBoost docs:** https://xgboost.readthedocs.io

---

## 8. Project File Reference

| File/Folder | Description |
|-------------|-------------|
| `Crop_recommendation.csv` | Raw dataset (2,200 samples) |
| `Phase2_EDA.ipynb` | Exploratory Data Analysis |
| `Phase3_Feature_Engineering.ipynb` | Preprocessing & feature engineering |
| `Phase4_Model_Development.ipynb` | Model training (7 algorithms) |
| `Phase5_Model_Evaluation.ipynb` | Model evaluation & comparison |
| `Phase6_Feature_Importance.ipynb` | Feature importance analysis |
| `app.py` | Streamlit Decision Support System |
| `models/` | Trained model files (.joblib) |
| `evaluation/` | Evaluation metrics & rankings |
| `feature_importance/` | Importance scores (MDI, Permutation, SHAP) |
| `preprocessing_pipeline.joblib` | Saved scaler + encoder |
| `requirements.txt` | Python dependencies |
| `PROJECT_REPORT.md` | Full project report |
| `README.md` | GitHub README |

---

*Happy farming! 🌾*
