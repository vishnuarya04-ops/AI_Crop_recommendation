"""
Page 4: 🔍 Explainability (XAI)
Tree-based feature importance (MDI), permutation importance, SHAP global & per-crop analysis,
and mandatory academic disclaimer regarding agronomic causation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_xai_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mdi_p = os.path.join(base_dir, "feature_importance", "mdi_importance.csv")
    perm_p = os.path.join(base_dir, "feature_importance", "permutation_importance.csv")
    shap_g_p = os.path.join(base_dir, "feature_importance", "shap_global_importance.csv")
    shap_crop_p = os.path.join(base_dir, "feature_importance", "shap_per_crop_importance.csv")
    unified_p = os.path.join(base_dir, "feature_importance", "unified_importance_ranking.csv")

    df_mdi = pd.read_csv(mdi_p) if os.path.exists(mdi_p) else None
    df_perm = pd.read_csv(perm_p) if os.path.exists(perm_p) else None
    df_shap_g = pd.read_csv(shap_g_p) if os.path.exists(shap_g_p) else None
    df_shap_crop = pd.read_csv(shap_crop_p, index_col=0) if os.path.exists(shap_crop_p) else None
    df_uni = pd.read_csv(unified_p) if os.path.exists(unified_p) else None

    return df_mdi, df_perm, df_shap_g, df_shap_crop, df_uni

df_mdi, df_perm, df_shap_g, df_shap_crop, df_unified = load_xai_data()

def render_explainability_page():
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.5rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.3rem;">🔍</div>
        <h1 style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #43cea2, #4facfe, #00f2fe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem;">
            Model Explainability &amp; Interpretability (XAI)
        </h1>
        <p style="color: #94a3b8; font-size: 0.95rem; max-width: 650px; margin: 0 auto 0.8rem auto;">
            Multi-method explainability using Tree-Based Impurity Reduction (MDI), Permutation Importance, and SHapley Additive exPlanations (SHAP).
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Mandatory Academic Disclaimer
    st.markdown("""
    <div style="background: rgba(245,158,11,0.08); border-left: 4px solid #f59e0b; padding: 1rem 1.2rem; border-radius: 0 10px 10px 0; margin-bottom: 1.5rem; color: #fcd34d;">
        <h4 style="margin: 0 0 0.3rem 0; font-size: 0.95rem; color: #fbbf24;">⚠️ Academic Disclaimer on Model Interpretation</h4>
        <p style="margin: 0; font-size: 0.86rem; line-height: 1.5; color: #fde68a;">
            The feature importance values and SHAP attributions presented below are surfaced as <strong>model interpretation only</strong>, reflecting how mathematical splits separate classes in this dataset, and <strong>not as a claim of direct agronomic causation</strong>. True biological crop viability additionally depends on soil microbiology, seasonal pests, cultivar genetics, and regional management practices.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Unified Consensus Ranking",
        "🌲 Tree MDI vs Permutation",
        "📈 Global SHAP Summary",
        "🌾 Per-Crop SHAP Sensitivity"
    ])

    with tab1:
        st.markdown("### 📊 Multi-Method Consensus Importance Ranking")
        st.markdown("""
        Comparing three independent explainability techniques to prevent method-specific bias:
        1. **MDI (Mean Decrease Impurity):** Tree-native Gini reduction.
        2. **Permutation Importance:** Model-agnostic accuracy drop upon feature shuffling.
        3. **SHAP (Shapley Values):** Game-theoretic contribution to class probabilities.
        """)

        if df_unified is not None:
            st.dataframe(df_unified, use_container_width=True, hide_index=True)

            st.markdown("#### Consensus Importance Hierarchy")
            chart_df = df_unified.copy()
            chart_df["Inverse Rank Score"] = 8 - chart_df["Avg Rank"]
            st.bar_chart(chart_df.set_index("Feature")["Inverse Rank Score"], color="#43cea2", height=300)

    with tab2:
        st.markdown("### 🌲 Tree-Based MDI vs Permutation Importance")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Mean Decrease Impurity (MDI %)")
            if df_mdi is not None:
                mdi_chart = df_mdi.set_index("Feature")["Pct"].sort_values(ascending=True)
                st.bar_chart(mdi_chart, horizontal=True, color="#4facfe", height=320)
                st.caption("Top drivers: Rainfall (21.96%), Humidity (21.71%), Potassium (18.08%)")

        with c2:
            st.markdown("#### Permutation Importance")
            if df_perm is not None:
                perm_chart = df_perm.set_index("Feature")["Mean Importance"].sort_values(ascending=True)
                st.bar_chart(perm_chart, horizontal=True, color="#818cf8", height=320)
                st.caption("Top drivers: Humidity (0.322), Nitrogen (0.225), Rainfall (0.178)")

    with tab3:
        st.markdown("### 📈 SHAP (SHapley Additive exPlanations) Global Importance")
        st.markdown("""
        SHAP attributes fair Shapley values across all 2,200 samples, measuring the mean absolute magnitude of each feature's contribution across all classes.
        """)

        if df_shap_g is not None:
            col_s1, col_s2 = st.columns([1.2, 1])
            with col_s1:
                shap_chart = df_shap_g.set_index("Feature")["Mean |SHAP|"].sort_values(ascending=True)
                st.bar_chart(shap_chart, horizontal=True, color="#43cea2", height=340)
            with col_s2:
                st.dataframe(df_shap_g, use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("### 🌾 Per-Crop SHAP Sensitivity Matrix")
        st.markdown("Inspect how the Random Forest model uniquely weights features for any of the 22 specific crops.")

        if df_shap_crop is not None:
            crop_choice = st.selectbox("Select Crop to Inspect Sensitivity Profile:", list(df_shap_crop.index))
            crop_series = df_shap_crop.loc[crop_choice].sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#1f2937")
            crop_series.plot(kind="barh", color="#43cea2", ax=ax)
            ax.set_title(f"SHAP Feature Importance for {crop_choice.capitalize()}", color="#f1f5f9", fontsize=11, fontweight="bold")
            ax.set_xlabel("Mean |SHAP| Value", color="#cbd5e1")
            ax.tick_params(colors="#94a3b8")
            ax.grid(color="#374151", linestyle="--", alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color("#4b5563")
            st.pyplot(fig)

            st.caption(f"💡 For **{crop_choice.capitalize()}**, the #1 distinguishing factor is **{crop_series.index[-1].upper()}**.")

if __name__ == "__main__":
    st.set_page_config(page_title="🔍 Explainability", page_icon="🔍", layout="wide")
    render_explainability_page()
