"""
Page 5: ℹ️ Project Information
System Architecture Pipeline (Mandatory Figure 1), Project Objectives,
Tools & Platform Used, and Academic References.
"""

import streamlit as st

import os
import pandas as pd
import matplotlib.pyplot as plt

def render_project_info_page():

    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.5rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.3rem;">ℹ️</div>
        <h1 style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #43cea2, #4facfe, #00f2fe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem;">
            Project Information &amp; Architecture
        </h1>
        <p style="color: #94a3b8; font-size: 0.95rem; max-width: 650px; margin: 0 auto 0.8rem auto;">
            Complete architectural methodology, academic project objectives, technological stack, and formal academic references.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Section 1: Architecture Pipeline ──
    st.markdown("### 🏗️ Figure 1: Architectural Pipeline Diagram (Mandatory)")
    st.markdown("""
    The system is built as an end-to-end sequential pipeline starting with raw agricultural data and terminating with an interpretable decision support application:
    """)

    st.markdown("""
    <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 1.5rem; margin: 1rem 0; text-align: center;">
        <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 0.6rem; font-size: 0.82rem; font-weight: 600;">
            <div style="background: rgba(67,206,162,0.15); border: 1px solid #43cea2; padding: 0.6rem 0.9rem; border-radius: 8px; color: #43cea2;">
                🌾 Agricultural<br>Dataset
            </div>
            <div style="color: #64748b;">➔</div>
            <div style="background: rgba(79,172,254,0.15); border: 1px solid #4facfe; padding: 0.6rem 0.9rem; border-radius: 8px; color: #4facfe;">
                📊 Data Cleaning<br>&amp; EDA
            </div>
            <div style="color: #64748b;">➔</div>
            <div style="background: rgba(129,140,248,0.15); border: 1px solid #818cf8; padding: 0.6rem 0.9rem; border-radius: 8px; color: #818cf8;">
                ⚙️ Preprocessing<br>&amp; Split (80/20)
            </div>
            <div style="color: #64748b;">➔</div>
            <div style="background: rgba(245,158,11,0.15); border: 1px solid #f59e0b; padding: 0.6rem 0.9rem; border-radius: 8px; color: #f59e0b;">
                🤖 Baseline +<br>ML Models
            </div>
            <div style="color: #64748b;">➔</div>
            <div style="background: rgba(236,72,153,0.15); border: 1px solid #ec4899; padding: 0.6rem 0.9rem; border-radius: 8px; color: #ec4899;">
                📈 Cross-Val. &amp;<br>Significance Test
            </div>
            <div style="color: #64748b;">➔</div>
            <div style="background: rgba(16,185,129,0.15); border: 1px solid #10b981; padding: 0.6rem 0.9rem; border-radius: 8px; color: #10b981;">
                🔍 Best Model +<br>Explainability
            </div>
            <div style="color: #64748b;">➔</div>
            <div style="background: rgba(67,206,162,0.25); border: 2px solid #43cea2; padding: 0.6rem 1rem; border-radius: 8px; color: #f1f5f9; font-weight: 700;">
                🚀 Streamlit<br>Application
            </div>
        </div>
        <p style="color: #94a3b8; font-size: 0.8rem; margin-top: 1rem; font-style: italic;">
            Figure 1: Architectural Diagram — AI-Based Crop Recommendation Pipeline (Mandatory)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 2: Model Performance Benchmark Graph & Variance Analysis ──
    st.markdown("### 📊 Model Performance Benchmark Graph & Variance Analysis")
    st.markdown("""
    High-resolution comparative evaluation across all 8 candidate models. 
    Select **High-Resolution Zoom (95%–100%)** or **Misclassification Error Count** below to reveal the distinct performance margins and accurate experimental differences between algorithms:
    """)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(base_dir, "evaluation", "metrics_comparison.csv")
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)

        col_v1, col_v2 = st.columns([3, 2])
        with col_v1:
            view_mode = st.radio(
                "Display Metric:",
                [
                    "🔍 High-Resolution Zoom (95% – 100%)",
                    "📉 Test Misclassifications (Errors out of 440)",
                    "🔬 Generalization Overfit Gap (Train - Test %)",
                    "📊 5-Fold Cross-Validation Stability (Std Dev %)",
                    "🌐 Full Scale (0% – 100%)"
                ],
                index=0,
                key="proj_info_view_mode"
            )
        with col_v2:
            st.markdown("""
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 0.8rem; margin-top: 0.5rem; font-size: 0.82rem; color: #94a3b8;">
                <strong style="color: #43cea2;">💡 Why allow scale zooming?</strong><br>
                Because top ML classifiers operate within high-accuracy thresholds (97%–99.55%), a standard 0–100% axis flattens differences. Zooming or viewing error counts reveals that <strong>Logistic Regression makes 6× more errors</strong> (12 vs 2) than <strong>Random Forest</strong>!
            </div>
            """, unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(9, 4.4))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#1e293b")

        if view_mode == "🔍 High-Resolution Zoom (95% – 100%)":
            # Exclude baseline to zoom into competitive ML models
            df_filtered = df_metrics[df_metrics["Model"] != "Majority Baseline"].copy()
            models = df_filtered["Model"].tolist()
            values = (df_filtered["Test Acc"] * 100).tolist()

            colors = ["#43cea2" if m == "Random Forest" else "#38bdf8" for m in models]
            bars = ax.barh(models[::-1], values[::-1], color=colors[::-1], height=0.55, edgecolor="none")
            ax.set_xlim(95.0, 100.2)
            ax.set_xlabel("Held-Out Test Accuracy (%) [Zoomed 95% – 100%]", color="#94a3b8", fontsize=9.5)
            ax.set_title("High-Resolution Performance Margins: Held-Out Test Accuracy", color="#f1f5f9", fontsize=11, fontweight="bold", pad=10)

            for bar, val in zip(bars, values[::-1]):
                ax.text(val + 0.08, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%",
                        va="center", color="#f1f5f9", fontsize=9, fontweight="bold" if val > 99 else "normal")

        elif view_mode == "📉 Test Misclassifications (Errors out of 440)":
            df_filtered = df_metrics[df_metrics["Model"] != "Majority Baseline"].copy()
            models = df_filtered["Model"].tolist()
            errors = [int(round((1.0 - acc) * 440)) for acc in df_filtered["Test Acc"]]

            colors = ["#43cea2" if m == "Random Forest" else "#f59e0b" if e < 8 else "#ef4444" for m, e in zip(models, errors)]
            bars = ax.barh(models[::-1], errors[::-1], color=colors[::-1], height=0.55, edgecolor="none")
            ax.set_xlim(0, max(errors) + 2.5)
            ax.set_xlabel("Number of Misclassified Samples (Lower is Better)", color="#94a3b8", fontsize=9.5)
            ax.set_title("Test Set Misclassification Errors (Out of 440 Held-Out Samples)", color="#f1f5f9", fontsize=11, fontweight="bold", pad=10)

            for bar, val in zip(bars, errors[::-1]):
                ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val} errors",
                        va="center", color="#f1f5f9", fontsize=9, fontweight="bold")

        elif view_mode == "🔬 Generalization Overfit Gap (Train - Test %)":
            df_filtered = df_metrics[df_metrics["Model"] != "Majority Baseline"].copy()
            models = df_filtered["Model"].tolist()
            gaps = (df_filtered["Gap"] * 100).tolist()

            colors = ["#43cea2" if g <= 0.5 else "#f59e0b" if g <= 1.5 else "#ef4444" for g in gaps]
            bars = ax.barh(models[::-1], gaps[::-1], color=colors[::-1], height=0.55, edgecolor="none")
            ax.set_xlim(-0.2, max(gaps) + 0.35)
            ax.set_xlabel("Generalization Gap (Train Acc - Test Acc %) [Lower is Better]", color="#94a3b8", fontsize=9.5)
            ax.set_title("Overfitting Risk: Discrepancy Between Training & Test Performance", color="#f1f5f9", fontsize=11, fontweight="bold", pad=10)

            for bar, val in zip(bars, gaps[::-1]):
                ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2, f"+{val:.2f}%" if val >= 0 else f"{val:.2f}%",
                        va="center", color="#f1f5f9", fontsize=9, fontweight="bold")

        elif view_mode == "📊 5-Fold Cross-Validation Stability (Std Dev %)":
            df_filtered = df_metrics[df_metrics["Model"] != "Majority Baseline"].copy()
            models = df_filtered["Model"].tolist()
            stds = (df_filtered["CV Std"] * 100).tolist()

            colors = ["#43cea2" if s < 0.5 else "#38bdf8" if s < 0.8 else "#ef4444" for s in stds]
            bars = ax.barh(models[::-1], stds[::-1], color=colors[::-1], height=0.55, edgecolor="none")
            ax.set_xlim(0, max(stds) + 0.22)
            ax.set_xlabel("5-Fold Cross-Validation Standard Deviation (±%) [Lower = More Stable]", color="#94a3b8", fontsize=9.5)
            ax.set_title("Cross-Validation Fold Variance & Algorithmic Stability", color="#f1f5f9", fontsize=11, fontweight="bold", pad=10)

            for bar, val in zip(bars, stds[::-1]):
                ax.text(val + 0.03, bar.get_y() + bar.get_height() / 2, f"±{val:.2f}%",
                        va="center", color="#f1f5f9", fontsize=9, fontweight="bold")

        else: # Full Scale (0% - 100%)
            models = df_metrics["Model"].tolist()
            values = (df_metrics["Test Acc"] * 100).tolist()
            colors = ["#43cea2" if m == "Random Forest" else "#ef4444" if m == "Majority Baseline" else "#38bdf8" for m in models]
            bars = ax.barh(models[::-1], values[::-1], color=colors[::-1], height=0.55, edgecolor="none")
            ax.set_xlim(0, 112)
            ax.set_xlabel("Held-Out Test Accuracy (%)", color="#94a3b8", fontsize=9.5)
            ax.set_title("Macro Comparison: All Classifiers vs Majority Baseline", color="#f1f5f9", fontsize=11, fontweight="bold", pad=10)

            for bar, val in zip(bars, values[::-1]):
                ax.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%",
                        va="center", color="#f1f5f9", fontsize=8.5, fontweight="bold" if val > 99 else "normal")

        ax.tick_params(colors="#cbd5e1", labelsize=9)
        ax.grid(axis="x", color="#334155", linestyle="--", alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color("#475569")

        plt.tight_layout()
        st.pyplot(fig)

        # Comparative breakdown table showing exact differences
        st.markdown("#### 📋 Detailed Model Variance & Accuracy Discrepancy Table")
        table_df = pd.DataFrame({
            "Rank": df_metrics["Rank"],
            "Model": df_metrics["Model"],
            "Test Accuracy": [f"{v*100:.2f}%" for v in df_metrics["Test Acc"]],
            "Test Errors (440)": [f"{int(round((1-v)*440))} errors" for v in df_metrics["Test Acc"]],
            "5-Fold CV Mean": [f"{v*100:.2f}%" for v in df_metrics["CV Accuracy"]],
            "CV Std Dev": [f"±{v*100:.2f}%" for v in df_metrics["CV Std"]],
            "Overfit Gap": [f"+{v*100:.2f}%" if v >= 0 else f"{v*100:.2f}%" for v in df_metrics["Gap"]],
            "Relative Error vs RF": [
                "— (Champion)" if m == "Random Forest" else
                f"{(round((1-acc)*440) / 2):.1f}× higher" if m != "Majority Baseline" else "210× higher"
                for m, acc in zip(df_metrics["Model"], df_metrics["Test Acc"])
            ]
        })
        st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Section 3: Objectives ──
    st.markdown("### 🎯 Formal Project Objectives")
    st.markdown("""
    1. **To collect, clean, and analyze an agricultural dataset** comprising soil nutrient (N, P, K), pH, temperature, humidity, and rainfall parameters through structured exploratory data analysis.
    2. **To define a naive baseline (majority-class predictor)** and train multiple machine learning classification algorithms — **Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Support Vector Machine, and XGBoost** — under an identical experimental procedure.
    3. **To evaluate all models** using Accuracy, Precision, Recall, F1-score, confusion matrices, and 5-fold cross-validation, and to determine whether performance differences between the top models are statistically significant.
    4. **To integrate explainable AI techniques** (tree-based feature importance and optionally SHAP) so that every recommendation is accompanied by a clear indication of which input factors most influenced it.
    5. **To select, tune, and save the best-performing model**, and deploy it through a Streamlit web application that allows a user to input soil and environmental values and receive a recommended crop, top-3 alternatives, and a plain-language explanation.
    6. **To document the complete methodology, experimental results, limitations, and conclusions** in a manner that is reproducible and suitable for academic evaluation.
    """)

    st.markdown("---")

    # ── Section 4: Tools & Platform Used ──
    st.markdown("### 🛠️ Tools & Platform Used")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("""
        - **Programming Language:** Python 3.10+
        - **Data Processing:** Pandas, NumPy
        - **Machine Learning:** Scikit-learn, XGBoost
        - **Statistical Testing:** SciPy (`scipy.stats.ttest_rel`, `scipy.stats.binomtest`)
        """)
    with col_t2:
        st.markdown("""
        - **Visualization:** Matplotlib, Seaborn, Plotly
        - **Explainability (XAI):** SHAP (SHapley Additive exPlanations)
        - **Application Framework:** Streamlit
        - **Model Serialization:** Joblib
        - **Version Control & IDE:** VS Code, Git, GitHub
        """)

    st.markdown("---")

    # ── Section 5: Academic References ──
    st.markdown("### 📚 Academic References")
    st.markdown("""
    1. Pudumalar, S., Ramanujam, E., Rajashree, R. H., Kavya, C., Kiruthika, T., & Nisha, J. (2017). Crop recommendation system for precision agriculture. In *2016 Eighth International Conference on Advanced Computing (ICOAC)*.
    2. Enhancing Agricultural Productivity: A Machine Learning Approach to Crop Recommendations. (2024). *Human-Centric Intelligent Systems*. Springer Nature.
    3. Crop recommendation using machine learning algorithms. (2023). *IEEE Conference Publication, IEEE Xplore*.
    4. Enhancing crop recommendation systems with explainable artificial intelligence: a study on agricultural decision-making (XAI-CROP). (2024). *Neural Computing and Applications*. Springer Nature.
    5. A Decision Support System for Crop Recommendation Using Machine Learning Classification Algorithms. (2024). *Agriculture*, 14(8), 1256. MDPI.
    6. Ommane, Y., Rhanbouri, M. A., Chouikh, H., Jbene, M., Chairi, I., Lachgar, M., & Benjelloun, S. (2022). Machine Learning based Recommender Systems for Crop Selection: A Systematic Literature Review.
    7. Crop Recommender System Using Machine Learning Approach. (2021). *IEEE Conference Publication, IEEE Xplore*.
    8. Mufamadi, T. O., & Ajoodha, R. Crop Recommendation using Machine Learning Algorithms and Soil Attributes Data.
    9. A. Ingle, “Crop Recommendation Dataset,” Kaggle, 2020. [Online]. Available: [https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset/data](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset/data)
    """)

if __name__ == "__main__":
    st.set_page_config(page_title="ℹ️ Project Information", page_icon="ℹ️", layout="wide")
    render_project_info_page()
