"""
🌾 Crop Recommendation — Decision Support System
==================================================
Multi-Page Streamlit Portal matching Academic Synopsis:
- Dedicated Page 1: Crop Recommendation (with top-3 alternatives & local SHAP factors)
- Dedicated Page 2: Data Analysis (Exploratory Data Analysis)
- Dedicated Page 3: Model Performance & Validation (Baseline + 5-Fold CV + Paired Statistical Tests)
- Dedicated Page 4: Explainability (MDI + Permutation + SHAP + Agronomic Disclaimer)
- Dedicated Page 5: Project Information (Figure 1 Pipeline Diagram + Objectives + References)

Usage:
    streamlit run app.py
"""

import streamlit as st
import importlib.util
import os

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS — premium dark glassmorphism theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0f1a 0%, #111b2e 40%, #0d1f2d 70%, #0a1628 100%);
        font-family: 'Inter', sans-serif;
    }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(67,206,162,0.3); border-radius: 3px; }

    section[data-testid="stSidebar"] {
        background: rgba(15,20,35,0.96);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .sidebar-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .sidebar-card h4 {
        color: #e2e8f0;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .sidebar-metric {
        display: flex;
        justify-content: space-between;
        padding: 0.25rem 0;
        font-size: 0.8rem;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .sidebar-metric:last-child { border-bottom: none; }
    .metric-label { color: #94a3b8; }
    .metric-value { color: #43cea2; font-weight: 600; }

    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 24px rgba(67,206,162,0.3) !important;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Helper to dynamically load page modules
# ──────────────────────────────────────────────
def load_page_module(relative_path, module_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, relative_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# ──────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <span style="font-size: 2.2rem;">🌾</span>
        <h2 style="font-size: 1.25rem; font-weight: 800; color: #f1f5f9; margin: 0.2rem 0 0.1rem 0;">CropAdvisor</h2>
        <span style="font-size: 0.75rem; color: #43cea2; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">AI Decision Support System</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧭 Navigation")
    page_selection = st.radio(
        "Select Portal Page:",
        [
            "🌾 Crop Recommendation",
            "📊 Data Analysis (EDA)",
            "📈 Model Performance",
            "🔍 Explainability (XAI)",
            "ℹ️ Project Information"
        ],
        index=0,
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🧠 Model Summary")
    st.markdown("""
    <div class="sidebar-card">
        <h4>Key Specifications</h4>
        <div class="sidebar-metric">
            <span class="metric-label">Best Model</span>
            <span class="metric-value">Random Forest</span>
        </div>
        <div class="sidebar-metric">
            <span class="metric-label">Test Accuracy</span>
            <span class="metric-value">99.55%</span>
        </div>
        <div class="sidebar-metric">
            <span class="metric-label">5-Fold Mean CV</span>
            <span class="metric-value">99.43%</span>
        </div>
        <div class="sidebar-metric">
            <span class="metric-label">Majority Baseline</span>
            <span class="metric-value">4.55%</span>
        </div>
        <div class="sidebar-metric">
            <span class="metric-label">Crops Supported</span>
            <span class="metric-value">22 Classes</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='color:#475569; font-size:0.72rem; text-align:center; margin-top:1rem;'>
        AI/ML Capstone Project · Academic Synopsis Compliance
    </p>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Render Selected Dedicated Page
# ──────────────────────────────────────────────
if page_selection == "🌾 Crop Recommendation":
    mod = load_page_module("pages/1_🌾_Crop_Recommendation.py", "page_crop_rec")
    mod.render_crop_recommendation_page()

elif page_selection == "📊 Data Analysis (EDA)":
    mod = load_page_module("pages/2_📊_Data_Analysis.py", "page_data_analysis")
    mod.render_data_analysis_page()

elif page_selection == "📈 Model Performance":
    mod = load_page_module("pages/3_📈_Model_Performance.py", "page_model_perf")
    mod.render_model_performance_page()

elif page_selection == "🔍 Explainability (XAI)":
    mod = load_page_module("pages/4_🔍_Explainability.py", "page_explainability")
    mod.render_explainability_page()

elif page_selection == "ℹ️ Project Information":
    mod = load_page_module("pages/5_ℹ️_Project_Information.py", "page_project_info")
    mod.render_project_info_page()

# ──────────────────────────────────────────────
# Global Footer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:0.75rem'>"
    "AI-Based Crop Recommendation System · Random Forest (99.55% Test Accuracy) · 5-Fold Stratified Validation · SHAP XAI Integration"
    "</p>",
    unsafe_allow_html=True,
)
