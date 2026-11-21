"""
Page 2: 📊 Data Analysis (EDA)
Exploratory Data Analysis: feature distributions, histograms, box plots,
correlation heatmap, class-wise comparisons, and summary statistics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "Crop_recommendation.csv")
    df = pd.read_csv(csv_path)
    return df

df = load_dataset()

def render_data_analysis_page():
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.5rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.3rem;">📊</div>
        <h1 style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #43cea2, #4facfe, #00f2fe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem;">
            Exploratory Data Analysis (EDA)
        </h1>
        <p style="color: #94a3b8; font-size: 0.95rem; max-width: 600px; margin: 0 auto 0.8rem auto;">
            Structured inspection of feature distributions, correlation heatmaps, class distributions, and parameter separability across 22 crops.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Overview Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Samples", f"{len(df):,}")
    with c2:
        st.metric("Crop Classes", f"{df['label'].nunique()} (Balanced)")
    with c3:
        st.metric("Continuous Features", "7")
    with c4:
        st.metric("Missing / Duplicates", "0 / 0")

    st.markdown("---")

    # Tabs for structured analysis
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "🖼️ Visual Analytics Dashboard (15 Charts)",
        "📈 Feature Distributions",
        "🔥 Correlation Heatmap",
        "🌱 Class-Wise Comparisons",
        "📋 Summary Statistics"
    ])

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    feature_labels = {
        "N": "Nitrogen (N) - kg/ha",
        "P": "Phosphorus (P) - kg/ha",
        "K": "Potassium (K) - kg/ha",
        "temperature": "Temperature (°C)",
        "humidity": "Relative Humidity (%)",
        "ph": "Soil pH",
        "rainfall": "Annual Rainfall (mm)"
    }

    with tab0:
        st.markdown("### 📊 Comprehensive Exploratory Data Analysis & Model Matrix Gallery")
        st.markdown("Grid gallery of all 15 analytical visualizers, algorithm performance benchmarks, and 22×22 confusion matrices.")

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_dir = os.path.join(base_dir, "webapp", "frontend", "assets", "analysis")

        def render_chart_card(title, img_name):
            img_path = os.path.join(assets_dir, img_name)
            st.markdown(f"""
            <div style="background: #111d38; border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 8px 8px 0 0; padding: 0.5rem 0.8rem; display: flex; align-items: center; justify-content: space-between;">
                <span style="color: #38bdf8; font-weight: 700; font-size: 0.92rem;">📊 {title}</span>
            </div>
            """, unsafe_allow_html=True)
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
            else:
                st.info(f"Image {img_name} not found.")

        # Row 1
        r1_c1, r1_c2 = st.columns(2)
        with r1_c1:
            render_chart_card("Crop Distribution", "crop_distribution.png")
        with r1_c2:
            render_chart_card("Decision Tree Confusion Matrix", "decision_tree_confusion_matrix.png")

        st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

        # Row 2
        r2_c1, r2_c2 = st.columns(2)
        with r2_c1:
            render_chart_card("Feature Distributions", "feature_distributions.png")
        with r2_c2:
            render_chart_card("KNN Confusion Matrix", "knn_confusion_matrix.png")

        st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

        # Row 3
        r3_c1, r3_c2 = st.columns(2)
        with r3_c1:
            render_chart_card("Feature Correlation", "feature_correlation.png")
        with r3_c2:
            render_chart_card("F1 Score Comparison", "f1_score_comparison.png")

        st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

        # Row 4
        r4_c1, r4_c2 = st.columns(2)
        with r4_c1:
            render_chart_card("Accuracy Comparison", "accuracy_comparison.png")
        with r4_c2:
            render_chart_card("Baseline Confusion Matrix", "baseline_confusion_matrix.png")

        st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

        # Row 5
        r5_c1, r5_c2 = st.columns(2)
        with r5_c1:
            render_chart_card("Precision Comparison", "precision_comparison.png")
        with r5_c2:
            render_chart_card("Random Forest Confusion Matrix", "random_forest_confusion_matrix.png")

        st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

        # Row 6
        r6_c1, r6_c2 = st.columns(2)
        with r6_c1:
            render_chart_card("Recall Comparison", "recall_comparison.png")
        with r6_c2:
            render_chart_card("SVM Confusion Matrix", "svm_confusion_matrix.png")

        st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

        # Row 7
        r7_c1, r7_c2 = st.columns(2)
        with r7_c1:
            render_chart_card("Logistic Regression Confusion Matrix", "logistic_regression_confusion_matrix.png")
        with r7_c2:
            render_chart_card("Overall Model Comparison", "overall_model_comparison.png")

        st.markdown("<div style='height: 1.2rem;'></div>", unsafe_allow_html=True)

        # Row 8
        st.markdown("---")
        render_chart_card("XGBoost Confusion Matrix", "xgboost_confusion_matrix.png")

    with tab1:
        st.markdown("### 📊 Distribution of Soil & Climate Parameters")
        selected_feature = st.selectbox("Select Feature to Inspect Distribution:", features, format_func=lambda x: feature_labels[x])

        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#1f2937")
            sns.histplot(df[selected_feature], kde=True, color="#43cea2", ax=ax, bins=30)
            ax.set_title(f"Histogram & KDE: {feature_labels[selected_feature]}", color="#f1f5f9", fontsize=11, fontweight="bold")
            ax.tick_params(colors="#94a3b8")
            ax.grid(color="#374151", linestyle="--", alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color("#4b5563")
            st.pyplot(fig)

        with col_dist2:
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#1f2937")
            sns.boxplot(x=df[selected_feature], color="#4facfe", ax=ax)
            ax.set_title(f"Box Plot: {feature_labels[selected_feature]}", color="#f1f5f9", fontsize=11, fontweight="bold")
            ax.tick_params(colors="#94a3b8")
            ax.grid(color="#374151", linestyle="--", alpha=0.5)
            for spine in ax.spines.values():
                spine.set_color("#4b5563")
            st.pyplot(fig)

        st.caption(f"💡 **Distribution Insight:** Mean: {df[selected_feature].mean():.2f}, Std Dev: {df[selected_feature].std():.2f}, Min: {df[selected_feature].min():.2f}, Max: {df[selected_feature].max():.2f}")

    with tab2:
        st.markdown("### 🔥 Pearson Correlation Heatmap")
        st.markdown("Examines linear relationships and potential multicollinearity between environmental and soil parameters.")

        corr_matrix = df[features].corr()

        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#111827")
        ax.set_facecolor("#1f2937")
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="mako",
            vmin=-1,
            vmax=1,
            cbar=True,
            ax=ax,
            annot_kws={"size": 10, "color": "white"}
        )
        ax.set_title("Correlation Matrix of Agricultural Features", color="#f1f5f9", fontsize=12, fontweight="bold")
        ax.tick_params(colors="#94a3b8")
        st.pyplot(fig)

        st.info("ℹ️ **Collinearity Finding:** No pairwise correlation exceeds |r| > 0.75, confirming absence of severe multicollinearity across the 7 inputs.")

    with tab3:
        st.markdown("### 🌱 Class-Wise Feature Variations Across All 22 Crops")
        st.markdown("Identify which soil or climatic parameters most clearly differentiate individual crop requirements.")

        c_f1, c_f2 = st.columns([2, 1])
        with c_f1:
            comp_feature = st.selectbox("Select Parameter for Class-Wise Breakdown:", features, format_func=lambda x: feature_labels[x], key="comp_feat")
        with c_f2:
            view_mode = st.radio(
                "Axis Scaling Mode:",
                ["🔍 Focused Variance (Zoomed)", "📊 Deviation from Mean (Δ %)", "📏 Zero-Based Full Range"],
                index=0,
                help="Switch between zoomed variance view, percentage deviation from the average, and standard zero-based axis."
            )

        crop_stats = df.groupby("label")[comp_feature].agg(["mean", "std", "min", "max"]).reset_index()
        overall_mean = df[comp_feature].mean()
        crop_stats["deviation_pct"] = ((crop_stats["mean"] - overall_mean) / overall_mean) * 100

        if "Deviation from Mean" in view_mode:
            crop_sorted = crop_stats.sort_values(by="deviation_pct", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 7))
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#1f2937")

            colors = ["#43cea2" if d >= 0 else "#f87171" for d in crop_sorted["deviation_pct"]]
            bars = ax.barh(crop_sorted["label"], crop_sorted["deviation_pct"], color=colors, alpha=0.85, edgecolor="#374151")
            ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.set_title(f"Crop-Specific Demand vs Global Average: {feature_labels[comp_feature]}\n(Global Mean = {overall_mean:.2f})", color="#f1f5f9", fontsize=12, fontweight="bold")
            ax.set_xlabel("Relative Deviation from Mean (%)", color="#cbd5e1")
            ax.set_ylabel("Crop", color="#cbd5e1")
            ax.tick_params(colors="#94a3b8")
            ax.grid(color="#374151", linestyle="--", alpha=0.3)
            for spine in ax.spines.values():
                spine.set_color("#4b5563")

            for bar in bars:
                w = bar.get_width()
                offset = 1.0 if w >= 0 else -1.0
                ha = 'left' if w >= 0 else 'right'
                ax.text(w + offset, bar.get_y() + bar.get_height()/2, f"{w:+.1f}%", va='center', ha=ha, color="#e2e8f0", fontsize=8)

            st.pyplot(fig)
            st.caption(f"💡 **Interpretation:** Green bars indicate crops requiring **above-average** {feature_labels[comp_feature]}, while red bars indicate crops requiring **below-average** levels.")

        elif "Focused Variance" in view_mode:
            crop_sorted = crop_stats.sort_values(by="mean", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 7))
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#1f2937")

            f_min = crop_sorted["mean"].min()
            f_max = crop_sorted["mean"].max()
            padding = (f_max - f_min) * 0.12 if (f_max - f_min) > 0 else 1.0

            sns.barplot(data=crop_sorted, x="mean", y="label", palette="viridis", ax=ax, edgecolor="#374151")
            ax.set_xlim(max(0, f_min - padding), f_max + padding)
            ax.set_title(f"Focused Mean Differences: {feature_labels[comp_feature]} across Crops", color="#f1f5f9", fontsize=12, fontweight="bold")
            ax.set_xlabel(feature_labels[comp_feature], color="#cbd5e1")
            ax.set_ylabel("Crop", color="#cbd5e1")
            ax.tick_params(colors="#94a3b8")
            ax.grid(color="#374151", linestyle="--", alpha=0.4)
            for spine in ax.spines.values():
                spine.set_color("#4b5563")

            # Value labels on bars
            for p in ax.patches:
                w = p.get_width()
                if not np.isnan(w) and w > 0:
                    ax.text(w + (padding * 0.05), p.get_y() + p.get_height()/2, f"{w:.1f}", va='center', color="#cbd5e1", fontsize=8)

            st.pyplot(fig)
            st.caption(f"💡 **Focused Range:** Zoomed into [{max(0, f_min - padding):.1f} – {f_max + padding:.1f}] to clearly reveal fine distinctions between crops.")

        else:
            crop_sorted = crop_stats.sort_values(by="mean", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 7))
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#1f2937")
            sns.barplot(data=crop_sorted, x="mean", y="label", palette="viridis", ax=ax)
            ax.set_title(f"Average {feature_labels[comp_feature]} by Crop (Zero-Based)", color="#f1f5f9", fontsize=12, fontweight="bold")
            ax.set_xlabel(feature_labels[comp_feature], color="#cbd5e1")
            ax.set_ylabel("Crop", color="#cbd5e1")
            ax.tick_params(colors="#94a3b8")
            ax.grid(color="#374151", linestyle="--", alpha=0.4)
            for spine in ax.spines.values():
                spine.set_color("#4b5563")
            st.pyplot(fig)

        # Multi-Crop Contrast Tool
        with st.expander("⚖️ Multi-Crop Side-by-Side Contrast Tool", expanded=False):
            st.markdown("Select 2 to 4 crops to directly inspect and contrast their physiological requirement differences.")
            all_crop_names = sorted(df["label"].unique())
            selected_crops = st.multiselect(
                "Pick crops to contrast:",
                options=all_crop_names,
                default=["rice", "chickpea", "maize"],
                max_selections=4
            )
            if selected_crops:
                contrast_df = df[df["label"].isin(selected_crops)].groupby("label")[features].mean().round(2).T
                st.dataframe(contrast_df, use_container_width=True)

                # Relative comparison heatmap or radar view
                c_min = contrast_df.min(axis=1)
                c_max = contrast_df.max(axis=1)
                c_denom = (c_max - c_min).replace(0, 1e-6)
                norm_contrast = contrast_df.sub(c_min, axis=0).div(c_denom, axis=0)
                fig_c, ax_c = plt.subplots(figsize=(8, 4))
                fig_c.patch.set_facecolor("#111827")
                ax_c.set_facecolor("#1f2937")
                sns.heatmap(norm_contrast, annot=contrast_df, fmt=".1f", cmap="mako", cbar=False, ax=ax_c, annot_kws={"size": 10, "color": "white"})
                ax_c.set_title("Normalized Factor Contrast (Relative Requirement Heatmap)", color="#f1f5f9", fontsize=11, fontweight="bold")
                ax_c.tick_params(colors="#94a3b8")
                st.pyplot(fig_c)

    with tab4:
        st.markdown("### 📋 Descriptive Statistics Summary")
        stats_df = df[features].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
        stats_df = stats_df.rename(columns={"count": "Count", "mean": "Mean", "std": "Std Dev", "min": "Min", "25%": "Q1 (25%)", "50%": "Median", "75%": "Q3 (75%)", "max": "Max"})
        st.dataframe(stats_df.round(2), use_container_width=True)

        st.markdown("#### Sample Dataset Records")
        st.dataframe(df.head(10), use_container_width=True)

if __name__ == "__main__":
    st.set_page_config(page_title="📊 Data Analysis", page_icon="📊", layout="wide")
    render_data_analysis_page()
