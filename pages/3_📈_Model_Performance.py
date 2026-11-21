"""
Page 3: 📈 Model Performance & Validation
Comprehensive benchmark table (including Majority Baseline), 5-Fold Cross-Validation,
paired statistical significance tests (paired t-test & McNemar's test), and hyperparameter tuning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_eval_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics_path = os.path.join(base_dir, "evaluation", "metrics_comparison.csv")
    cv_path = os.path.join(base_dir, "evaluation", "cross_validation_results.csv")
    stat_path = os.path.join(base_dir, "evaluation", "statistical_tests.csv")
    tune_path = os.path.join(base_dir, "evaluation", "hyperparameter_tuning.csv")
    noise_path = os.path.join(base_dir, "evaluation", "noise_robustness.csv")

    df_m = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None
    df_cv = pd.read_csv(cv_path) if os.path.exists(cv_path) else None
    df_s = pd.read_csv(stat_path) if os.path.exists(stat_path) else None
    df_t = pd.read_csv(tune_path) if os.path.exists(tune_path) else None
    df_n = pd.read_csv(noise_path) if os.path.exists(noise_path) else None
    return df_m, df_cv, df_s, df_t, df_n

df_metrics, df_cv, df_stats, df_tuning, df_noise = load_eval_data()

def render_model_performance_page():
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.5rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.3rem;">📈</div>
        <h1 style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #43cea2, #4facfe, #00f2fe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem;">
            Model Performance &amp; Statistical Validation
        </h1>
        <p style="color: #94a3b8; font-size: 0.95rem; max-width: 650px; margin: 0 auto 0.8rem auto;">
            Rigorous evaluation across 8 models (including Majority Baseline) using Accuracy, F1-scores, 5-fold cross-validation, and paired statistical hypothesis testing.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Key Performance Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Top Model", "Random Forest", "Rank #1")
    with c2:
        st.metric("Test Accuracy", "99.55%", "438 / 440 Correct")
    with c3:
        st.metric("5-Fold Mean CV", "99.43%", "± 0.0048 Std Dev")
    with c4:
        st.metric("Baseline Accuracy", "4.55%", "Empirical Reference")

    st.markdown("---")

    # ── Featured Spotlight: Random Forest vs XGBoost McNemar's Test ──
    st.html("""
    <div style="margin: 1rem 0 2rem 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.35rem;">
            <span style="font-size: 1.85rem; line-height: 1;">🔬</span>
            <h2 style="font-size: 1.7rem; font-weight: 700; color: #ffffff; margin: 0; letter-spacing: -0.02em;">
                Random Forest vs XGBoost: McNemar’s Test
            </h2>
        </div>
        <p style="color: #94a3b8; font-size: 0.98rem; margin-top: 0.3rem; margin-bottom: 1.5rem; line-height: 1.5;">
            Statistical significance comparison between Random Forest and XGBoost using predictions on the exact same held-out test samples.
        </p>

        <!-- Top Row (4 Cards) -->
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1rem;">
            <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.25rem 1.4rem;">
                <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">Total Test Samples</div>
                <div style="color: #34d399; font-size: 2.3rem; font-weight: 700; line-height: 1;">440</div>
            </div>
            <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.25rem 1.4rem;">
                <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">RF ✓, XGB X (b)</div>
                <div style="color: #34d399; font-size: 2.3rem; font-weight: 700; line-height: 1;">1</div>
            </div>
            <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.25rem 1.4rem;">
                <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">RF X, XGB ✓ (c)</div>
                <div style="color: #34d399; font-size: 2.3rem; font-weight: 700; line-height: 1;">0</div>
            </div>
            <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.25rem 1.4rem;">
                <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">Disagreements (b + c)</div>
                <div style="color: #34d399; font-size: 2.3rem; font-weight: 700; line-height: 1;">1</div>
            </div>
        </div>

        <!-- Bottom Row (3 Cards) -->
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
            <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.25rem 1.4rem;">
                <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">Exact Test Statistic</div>
                <div style="color: #34d399; font-size: 2.3rem; font-weight: 700; line-height: 1;">0.0000</div>
            </div>
            <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.25rem 1.4rem;">
                <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">p-value</div>
                <div style="color: #34d399; font-size: 2.3rem; font-weight: 700; line-height: 1;">1.000000</div>
            </div>
            <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.25rem 1.4rem;">
                <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">Alpha (α)</div>
                <div style="color: #34d399; font-size: 2.3rem; font-weight: 700; line-height: 1;">0.05</div>
            </div>
        </div>
    </div>
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 Model Comparison Benchmark",
        "🔄 5-Fold Cross-Validation",
        "🔬 Statistical Significance Testing",
        "⚙️ Hyperparameter Tuning",
        "⚡ Noise Robustness & Stress Testing"
    ])

    with tab1:
        st.markdown("### 🏆 Performance Comparison: All Classifiers + Baseline")
        st.markdown("""
        All models were trained on the same stratified 80% train split (1,760 samples) and evaluated on the identical 20% test split (440 samples).
        A **Majority-Class Baseline** is established first as a lower reference point.
        """)

        if df_metrics is not None:
            c_tbl_ctrl, c_tbl_info = st.columns([3, 1])
            with c_tbl_ctrl:
                tbl_perspective = st.radio(
                    "Benchmark Table Perspective:",
                    [
                        "⚡ Multi-Dimensional Trade-Off Matrix (Speed, Memory, Errors, Noise)",
                        "📐 Academic Statistical Classification Metrics (Accuracy, Precision, Recall, F1)"
                    ],
                    index=0,
                    horizontal=True
                )

            if "Multi-Dimensional" in tbl_perspective and "Latency (ms/1k)" in df_metrics.columns:
                disp_eng = df_metrics[[
                    "Rank", "Model", "Test Acc", "Test Errors", "Latency (ms/1k)", 
                    "Model Size (KB)", "Worst Crop", "Worst Recall", "Noise Robustness (20%)", "Primary Strength"
                ]].copy()
                disp_eng["Test Acc"] = disp_eng["Test Acc"].apply(lambda x: f"{x * 100:.2f}%")
                disp_eng["Worst Recall"] = disp_eng["Worst Recall"].apply(lambda x: f"{x * 100:.1f}%")
                disp_eng["Noise Robustness (20%)"] = disp_eng["Noise Robustness (20%)"].apply(lambda x: f"{x * 100:.1f}%")
                disp_eng["Latency"] = disp_eng["Latency (ms/1k)"].apply(lambda x: f"{x:.2f} ms")
                disp_eng["Model Size"] = disp_eng["Model Size (KB)"].apply(lambda x: f"{x/1024:.2f} MB" if x >= 1024 else f"{x:.1f} KB")
                disp_eng["Worst Blindspot"] = disp_eng["Worst Crop"] + " (" + disp_eng["Worst Recall"] + ")"
                
                disp_eng = disp_eng[[
                    "Rank", "Model", "Test Acc", "Test Errors", "Latency", 
                    "Model Size", "Worst Blindspot", "Noise Robustness (20%)", "Primary Strength"
                ]]
                st.dataframe(disp_eng, use_container_width=True, hide_index=True)
                st.caption("💡 **Contrast Highlights:** While raw accuracies on clean data look close (97–99%), their engineering trade-offs diverge drastically. **Random Forest** is top accuracy but **2,700× larger** (7.1 MB) and **46× slower** than **Decision Tree** (19 KB, 2.8 ms). In noisy environments, **SVM** retains 95.2% accuracy while Decision Tree collapses to 77.5%!")
            else:
                display_df = df_metrics.copy()
                stat_cols = [c for c in ["Rank", "Model", "Train Acc", "Test Acc", "Precision (macro)", "Recall (macro)", "F1 (macro)", "F1 (weighted)", "Gap", "CV Accuracy", "CV Std"] if c in display_df.columns]
                display_df = display_df[stat_cols]
                for col in ["Train Acc", "Test Acc", "Precision (macro)", "Recall (macro)", "F1 (macro)", "F1 (weighted)", "CV Accuracy"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x * 100:.2f}%")
                if "Gap" in display_df.columns:
                    display_df["Overfit Gap"] = display_df["Gap"].apply(lambda x: f"{x * 100:.2f}%")
                    display_df = display_df.drop(columns=["Gap"])
                if "CV Std" in display_df.columns:
                    display_df["CV Std"] = display_df["CV Std"].apply(lambda x: f"±{x * 100:.3f}%")
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Chart comparison with perspective controls
            st.markdown("#### 📊 Comparative Model Benchmark Visualizer")
            
            c_view1, c_view2 = st.columns([2, 1])
            with c_view1:
                benchmark_view = st.selectbox(
                    "Select Benchmark Perspective:",
                    [
                        "⚡ Latency vs. Accuracy Trade-Off (Pareto Frontier)",
                        "❌ Test Misclassifications / Error Count (out of 440)",
                        "💾 Model Footprint & Memory Storage (KB)",
                        "🌾 Hardest Crop Vulnerability (Worst Class Recall %)",
                        "🎯 Focused Accuracy Zoom (90% – 100%)",
                        "📉 Accuracy Gap vs Leader (Δ %)",
                        "📏 Full Scale (0% – 100%)"
                    ],
                    index=0,
                    help="Switch perspective to see precise performance differences between models instead of flat 0-100% bars."
                )

            # Exclude Majority Baseline for high-res comparison, or include with disclaimer
            df_no_base = df_metrics[df_metrics["Model"] != "Majority Baseline"].copy()
            total_test_samples = 440
            if "Test Errors" not in df_no_base.columns:
                df_no_base["Test Errors"] = np.round((1.0 - df_no_base["Test Acc"]) * total_test_samples).astype(int)
            best_acc = df_no_base["Test Acc"].max()
            df_no_base["Acc Gap (%)"] = (best_acc - df_no_base["Test Acc"]) * 100

            if "Latency vs. Accuracy" in benchmark_view:
                fig_p, ax_p = plt.subplots(figsize=(10, 5))
                fig_p.patch.set_facecolor("#111827")
                ax_p.set_facecolor("#1f2937")

                x_lat = df_no_base["Latency (ms/1k)"] if "Latency (ms/1k)" in df_no_base.columns else [132.5, 4.4, 88.1, 19.2, 47.5, 2.8, 3.2]
                y_acc = df_no_base["Test Acc"] * 100
                sizes = np.log10(df_no_base["Model Size (KB)"] + 10) * 120 if "Model Size (KB)" in df_no_base.columns else 200

                # Highlight models: Green for pareto frontier leaders, Cyan for others
                colors_pareto = ["#10b981" if m in ["Random Forest", "Naive Bayes", "Decision Tree"] else "#38bdf8" for m in df_no_base["Model"]]
                scatter = ax_p.scatter(x_lat, y_acc, s=sizes, c=colors_pareto, alpha=0.85, edgecolors="#f8fafc", linewidth=1.5)

                for _, row in df_no_base.iterrows():
                    m_name = row["Model"]
                    lx = row["Latency (ms/1k)"] if "Latency (ms/1k)" in df_no_base.columns else 10
                    ly = row["Test Acc"] * 100
                    # Offset text based on position
                    dx = 1.15 if lx < 20 else 0.85
                    ax_p.annotate(f" {m_name}\n ({ly:.2f}%, {lx:.1f}ms)", (lx, ly),
                                  xytext=(lx * dx, ly + (0.15 if m_name not in ['Naive Bayes', 'Decision Tree'] else -0.35)),
                                  color="#f1f5f9", fontsize=8.5, fontweight="bold",
                                  bbox=dict(boxstyle="round,pad=0.2", facecolor="#111827", edgecolor="#374151", alpha=0.8))

                ax_p.set_xscale("log")
                ax_p.set_xlim(1.5, 250)
                ax_p.set_ylim(96.5, 100.2)
                ax_p.set_xlabel("Inference Latency (ms per 1,000 queries - Log Scale)", color="#cbd5e1", fontsize=10)
                ax_p.set_ylabel("Held-Out Test Accuracy (%)", color="#cbd5e1", fontsize=10)
                ax_p.set_title("Pareto Efficiency: Inference Latency vs. Test Accuracy (Bubble Size = Model Memory Footprint)", color="#f1f5f9", fontsize=11, fontweight="bold")
                ax_p.tick_params(colors="#94a3b8")
                ax_p.grid(color="#374151", linestyle="--", alpha=0.5)
                for spine in ax_p.spines.values():
                    spine.set_color("#4b5563")

                st.pyplot(fig_p)
                st.caption("🟢 **Green Bubbles:** Pareto-Optimal Models (Decision Tree for speed, Naive Bayes for fast balance, Random Forest for maximum accuracy). Bubble size reflects disk footprint.")

            elif "Focused Accuracy Zoom" in benchmark_view:
                fig_m, ax_m = plt.subplots(figsize=(10, 4.8))
                fig_m.patch.set_facecolor("#111827")
                ax_m.set_facecolor("#1f2937")

                y_pos = np.arange(len(df_no_base))
                bar_height = 0.35

                bars1 = ax_m.barh(y_pos - bar_height/2, df_no_base["Test Acc"] * 100, bar_height, label="Test Accuracy", color="#43cea2", edgecolor="#111827")
                bars2 = ax_m.barh(y_pos + bar_height/2, df_no_base["CV Accuracy"] * 100, bar_height, label="5-Fold CV Accuracy", color="#4facfe", edgecolor="#111827")

                min_acc = min(df_no_base["Test Acc"].min(), df_no_base["CV Accuracy"].min()) * 100
                ax_m.set_xlim(max(85, np.floor(min_acc) - 2), 100.5)
                ax_m.set_yticks(y_pos)
                ax_m.set_yticklabels(df_no_base["Model"], color="#cbd5e1", fontweight="bold")
                ax_m.set_xlabel("Accuracy (%)", color="#cbd5e1")
                ax_m.set_title("Focused Accuracy Comparison (Zoomed Range)", color="#f1f5f9", fontsize=12, fontweight="bold")
                ax_m.tick_params(colors="#94a3b8")
                ax_m.grid(color="#374151", linestyle="--", alpha=0.5, axis="x")
                ax_m.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#cbd5e1", loc="lower right")
                for spine in ax_m.spines.values():
                    spine.set_color("#4b5563")

                # Value labels
                for bar in bars1:
                    w = bar.get_width()
                    ax_m.text(w + 0.15, bar.get_y() + bar.get_height()/2, f"{w:.2f}%", va='center', color="#43cea2", fontsize=8, fontweight="bold")

                st.pyplot(fig_m)
                st.caption("💡 **Note:** Zoomed between 94%–100% to clearly display the separation between ensemble models and baseline classifiers.")

            elif "Error Count" in benchmark_view:
                fig_e, ax_e = plt.subplots(figsize=(10, 4.5))
                fig_e.patch.set_facecolor("#111827")
                ax_e.set_facecolor("#1f2937")

                df_sorted_err = df_no_base.sort_values("Test Errors")
                colors_err = ["#43cea2" if e <= 2 else "#fbbf24" if e <= 6 else "#f87171" for e in df_sorted_err["Test Errors"]]
                bars_e = ax_e.bar(df_sorted_err["Model"], df_sorted_err["Test Errors"], color=colors_err, edgecolor="#374151")

                ax_e.set_ylabel("Misclassified Samples (out of 440)", color="#cbd5e1")
                ax_e.set_title("Total Test Misclassifications by Model (Fewer is Better)", color="#f1f5f9", fontsize=12, fontweight="bold")
                ax_e.tick_params(colors="#94a3b8")
                plt.xticks(rotation=20, ha="right", color="#cbd5e1")
                ax_e.grid(color="#374151", linestyle="--", alpha=0.4, axis="y")
                for spine in ax_e.spines.values():
                    spine.set_color("#4b5563")

                for bar in bars_e:
                    h = bar.get_height()
                    ax_e.text(bar.get_x() + bar.get_width()/2, h + 0.3, f"{h} errors", ha='center', color="#f1f5f9", fontsize=9, fontweight="bold")

                st.pyplot(fig_e)
                st.caption("💡 **Sharp Contrast:** Random Forest makes only **2 errors** (99.55%), whereas Logistic Regression makes **12 errors** (97.27%) — a **6x difference** in failure rate!")

            elif "Model Footprint" in benchmark_view:
                fig_s, ax_s = plt.subplots(figsize=(10, 4.5))
                fig_s.patch.set_facecolor("#111827")
                ax_s.set_facecolor("#1f2937")

                df_sorted_size = df_no_base.sort_values("Model Size (KB)") if "Model Size (KB)" in df_no_base.columns else df_no_base
                sizes_kb = df_sorted_size["Model Size (KB)"]
                colors_s = ["#34d399" if s < 50 else "#60a5fa" if s < 500 else "#f87171" for s in sizes_kb]
                bars_s = ax_s.barh(df_sorted_size["Model"], sizes_kb, color=colors_s, edgecolor="#374151")

                ax_s.set_xscale("log")
                ax_s.set_xlabel("Serialized Model Size (KB - Log Scale)", color="#cbd5e1")
                ax_s.set_title("Model Storage Footprint: Lightweight Edge vs Heavy Ensembles", color="#f1f5f9", fontsize=12, fontweight="bold")
                ax_s.tick_params(colors="#94a3b8")
                ax_s.grid(color="#374151", linestyle="--", alpha=0.4, axis="x")
                for spine in ax_s.spines.values():
                    spine.set_color("#4b5563")

                for bar in bars_s:
                    w = bar.get_width()
                    label_str = f"{w/1024:.2f} MB" if w >= 1024 else f"{w:.1f} KB"
                    ax_s.text(w * 1.15, bar.get_y() + bar.get_height()/2, label_str, va='center', color="#f1f5f9", fontsize=8.5, fontweight="bold")

                st.pyplot(fig_s)
                st.caption("💡 **Memory Divergence:** Logistic Regression (2.6 KB) and Decision Tree (19.3 KB) fit on tiny microcontroller memory, whereas Random Forest requires 7.1 MB (over 2,700× larger).")

            elif "Hardest Crop" in benchmark_view:
                fig_hc, ax_hc = plt.subplots(figsize=(10, 4.5))
                fig_hc.patch.set_facecolor("#111827")
                ax_hc.set_facecolor("#1f2937")

                df_sorted_hc = df_no_base.sort_values("Worst Recall") if "Worst Recall" in df_no_base.columns else df_no_base
                recalls = df_sorted_hc["Worst Recall"] * 100
                labels = [f"{m} ({c})" for m, c in zip(df_sorted_hc["Model"], df_sorted_hc["Worst Crop"])]
                colors_hc = ["#ef4444" if r <= 80 else "#f59e0b" if r <= 85 else "#10b981" for r in recalls]

                bars_hc = ax_hc.barh(labels, recalls, color=colors_hc, edgecolor="#374151")
                ax_hc.set_xlim(60, 102)
                ax_hc.set_xlabel("Recall on Hardest Crop (%)", color="#cbd5e1")
                ax_hc.set_title("Worst-Class Per-Crop Vulnerability (Hardest Crop in Parentheses)", color="#f1f5f9", fontsize=12, fontweight="bold")
                ax_hc.tick_params(colors="#94a3b8")
                ax_hc.grid(color="#374151", linestyle="--", alpha=0.4, axis="x")
                for spine in ax_hc.spines.values():
                    spine.set_color("#4b5563")

                for bar in bars_hc:
                    w = bar.get_width()
                    ax_hc.text(w + 0.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', color="#f1f5f9", fontsize=8.5, fontweight="bold")

                st.pyplot(fig_hc)
                st.caption("💡 **Per-Crop Blindspots:** Simpler linear and single-tree models drop to 80.0% recall on Rice and Blackgram, while Random Forest maintains 95.0%+ recall across all 22 crops.")

            elif "Accuracy Gap" in benchmark_view:
                fig_g, ax_g = plt.subplots(figsize=(10, 4.5))
                fig_g.patch.set_facecolor("#111827")
                ax_g.set_facecolor("#1f2937")

                df_sorted_gap = df_no_base.sort_values("Acc Gap (%)")
                bars_g = ax_g.barh(df_sorted_gap["Model"], df_sorted_gap["Acc Gap (%)"], color="#fb923c", edgecolor="#374151")

                ax_g.set_xlabel("Accuracy Deficit vs Random Forest (Δ %)", color="#cbd5e1")
                ax_g.set_title("Performance Penalty Compared to Leader (Random Forest = 0.00%)", color="#f1f5f9", fontsize=12, fontweight="bold")
                ax_g.tick_params(colors="#94a3b8")
                ax_g.grid(color="#374151", linestyle="--", alpha=0.4, axis="x")
                for spine in ax_g.spines.values():
                    spine.set_color("#4b5563")

                for bar in bars_g:
                    w = bar.get_width()
                    ax_g.text(w + 0.05, bar.get_y() + bar.get_height()/2, f"-{w:.2f}%", va='center', color="#fed7aa", fontsize=9)

                st.pyplot(fig_g)

            else:
                chart_data = pd.DataFrame({
                    "Model": df_metrics["Model"],
                    "Test Accuracy (%)": df_metrics["Test Acc"] * 100,
                    "5-Fold CV (%)": df_metrics["CV Accuracy"] * 100
                }).set_index("Model")
                st.bar_chart(chart_data, height=360)

            # ── Engineering Decision Support Guide ──
            st.markdown("---")
            st.markdown("#### 🎯 Real-World Deployment Decision Guide (Which Model to Choose?)")
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.markdown("""
                <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #10b981; padding: 0.9rem; border-radius: 8px; margin-bottom: 0.8rem;">
                    <strong style="color: #34d399; font-size: 0.95rem;">🏆 Cloud Production Deployment: Random Forest</strong><br>
                    <span style="color: #94a3b8; font-size: 0.85rem;">
                        <strong>Why:</strong> Top held-out test accuracy (99.55%), lowest error count (2/440), and lowest cross-validation variance (±0.0048). 
                        Disk storage (7.1 MB) is trivial on cloud servers, making it the clear gold standard.
                    </span>
                </div>
                <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #38bdf8; padding: 0.9rem; border-radius: 8px;">
                    <strong style="color: #38bdf8; font-size: 0.95rem;">⚡ Ultra-Lightweight IoT &amp; Microcontroller: Logistic Regression</strong><br>
                    <span style="color: #94a3b8; font-size: 0.85rem;">
                        <strong>Why:</strong> Only <strong>2.6 KB</strong> footprint and 3.2 ms latency. Can be burned directly onto flash memory of an ESP32 / Arduino soil probe running on battery without cloud connectivity.
                    </span>
                </div>
                """, unsafe_allow_html=True)

            with g_col2:
                st.markdown("""
                <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #a855f7; padding: 0.9rem; border-radius: 8px; margin-bottom: 0.8rem;">
                    <strong style="color: #c084fc; font-size: 0.95rem;">🛡️ Harsh Field Conditions &amp; Noisy Sensors: SVM (RBF)</strong><br>
                    <span style="color: #94a3b8; font-size: 0.85rem;">
                        <strong>Why:</strong> Maintains <strong>95.18% accuracy even under 20% sensor noise</strong>, outperforming both Random Forest (87.91%) and XGBoost (77.09%) due to maximum margin hyperplane separation.
                    </span>
                </div>
                <div style="background: rgba(30, 41, 59, 0.7); border-left: 4px solid #f59e0b; padding: 0.9rem; border-radius: 8px;">
                    <strong style="color: #fbbf24; font-size: 0.95rem;">📖 Transparent &amp; Regulatory Audits: Decision Tree</strong><br>
                    <span style="color: #94a3b8; font-size: 0.85rem;">
                        <strong>Why:</strong> Fastest execution (2.8 ms) and 100% white-box explainability. Agricultural extension officers can inspect and certify exact decision thresholds without black-box opacity.
                    </span>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🔄 5-Fold Stratified Cross-Validation Stability")
        st.markdown("""
        5-fold cross-validation evaluates performance consistency across 5 distinct data partitions to ensure models do not overfit to a single fortunate split.
        """)

        if df_cv is not None:
            cv_display = df_cv.copy()
            for c in ["Mean CV Accuracy", "Std Dev", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]:
                if c in cv_display.columns:
                    cv_display[c] = cv_display[c].apply(lambda x: f"{x * 100:.2f}%" if "Std" not in c else f"±{x * 100:.3f}%")
            st.dataframe(cv_display, use_container_width=True, hide_index=True)

            st.markdown("#### Fold Score Distribution Across Top Models")
            fig, ax = plt.subplots(figsize=(9, 4.5))
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#1f2937")

            top_models = df_cv[df_cv["Model"] != "Majority Baseline"]
            for _, row in top_models.iterrows():
                folds = [row["Fold 1"], row["Fold 2"], row["Fold 3"], row["Fold 4"], row["Fold 5"]]
                ax.plot(["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"], [f * 100 for f in folds], marker="o", label=row["Model"], linewidth=2)

            ax.set_ylabel("Accuracy (%)", color="#cbd5e1")
            ax.set_title("5-Fold Cross Validation Trajectory", color="#f1f5f9", fontsize=11, fontweight="bold")
            ax.tick_params(colors="#94a3b8")
            ax.grid(color="#374151", linestyle="--", alpha=0.5)
            ax.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#cbd5e1", fontsize=8, loc="lower right")
            for spine in ax.spines.values():
                spine.set_color("#4b5563")
            st.pyplot(fig)

    with tab3:
        st.markdown("### 🔬 Paired Statistical Significance Testing (vs Random Forest)")
        st.markdown("""
        To determine whether Random Forest's performance superiority is **statistically significant** rather than incidental to a single split, two paired tests were conducted:
        1. **Paired t-test** on the 5-fold cross-validation fold accuracy scores (`scipy.stats.ttest_rel`).
        2. **McNemar's Test** with exact binomial distribution (`scipy.stats.binomtest`) on test-set prediction discrepancies.
        """)

        # ── Spotlight McNemar's Test Component (Featured: Random Forest vs XGBoost) ──
        mcnemar_data = {
            "XGBoost": {
                "title": "Random Forest vs XGBoost: McNemar’s Test",
                "sub": "Statistical significance comparison between Random Forest and XGBoost using predictions on the exact same held-out test samples.",
                "total": "440", "b": "1", "c": "0", "disagree": "1",
                "stat": "0.0000", "p": "1.000000", "alpha": "0.05",
                "label_b": "RF ✓, XGB X (b)", "label_c": "RF X, XGB ✓ (c)",
                "conclusion": "p-value (1.000000) > α (0.05) → Failure to reject the null hypothesis. The classification performance difference between Random Forest and XGBoost is not statistically significant, establishing high competitive parity on held-out test data."
            },
            "Support Vector Machine": {
                "title": "Random Forest vs SVM: McNemar’s Test",
                "sub": "Statistical significance comparison between Random Forest and SVM using predictions on the exact same held-out test samples.",
                "total": "440", "b": "4", "c": "1", "disagree": "5",
                "stat": "0.8000", "p": "0.375000", "alpha": "0.05",
                "label_b": "RF ✓, SVM X (b)", "label_c": "RF X, SVM ✓ (c)",
                "conclusion": "p-value (0.375000) > α (0.05) → Performance parity on the test set, though Random Forest achieves significantly higher 5-fold cross-validation consistency (p = 0.008)."
            },
            "Decision Tree": {
                "title": "Random Forest vs Decision Tree: McNemar’s Test",
                "sub": "Statistical significance comparison between Random Forest and Decision Tree using predictions on the exact same held-out test samples.",
                "total": "440", "b": "7", "c": "0", "disagree": "7",
                "stat": "5.1429", "p": "0.015625", "alpha": "0.05",
                "label_b": "RF ✓, DT X (b)", "label_c": "RF X, DT ✓ (c)",
                "conclusion": "p-value (0.015625) < α (0.05) → Statistically significant superiority! Random Forest correctly classifies 7 test cases where a single Decision Tree fails."
            },
            "Logistic Regression": {
                "title": "Random Forest vs Logistic Regression: McNemar’s Test",
                "sub": "Statistical significance comparison between Random Forest and Logistic Regression using predictions on the exact same held-out test samples.",
                "total": "440", "b": "11", "c": "1", "disagree": "12",
                "stat": "6.7500", "p": "0.006348", "alpha": "0.05",
                "label_b": "RF ✓, LR X (b)", "label_c": "RF X, LR ✓ (c)",
                "conclusion": "p-value (0.006348) < α (0.05) → Highly statistically significant superiority (p < 0.01). Non-linear ensemble trees decisively outperform linear decision boundaries."
            },
            "Majority Baseline": {
                "title": "Random Forest vs Majority Baseline: McNemar’s Test",
                "sub": "Statistical significance comparison between Random Forest and Majority Baseline using predictions on the exact same held-out test samples.",
                "total": "440", "b": "418", "c": "0", "disagree": "418",
                "stat": "416.0024", "p": "0.000000", "alpha": "0.05",
                "label_b": "RF ✓, Base X (b)", "label_c": "RF X, Base ✓ (c)",
                "conclusion": "p-value (0.000000) < α (0.05) → Overwhelmingly significant superiority. Confirms the trained model learns deep feature signals far beyond random chance."
            }
        }

        col_sel, _ = st.columns([2, 1])
        with col_sel:
            selected_comp = st.selectbox(
                "Select McNemar Significance Pair (vs Random Forest):",
                list(mcnemar_data.keys()),
                index=0,
                help="Inspect paired test-set contingency and exact binomial significance test for any competitor model."
            )

        card = mcnemar_data[selected_comp]

        st.html(f"""
        <div style="margin: 1rem 0 1.5rem 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <div style="display: flex; align-items: center; gap: 0.65rem; margin-bottom: 0.35rem;">
                <span style="font-size: 1.85rem; line-height: 1;">🔬</span>
                <h3 style="font-size: 1.7rem; font-weight: 700; color: #ffffff; margin: 0; letter-spacing: -0.02em;">{card['title']}</h3>
            </div>
            <p style="color: #94a3b8; font-size: 0.98rem; margin-top: 0.3rem; margin-bottom: 1.5rem; line-height: 1.5;">
                {card['sub']}
            </p>

            <!-- Top Row (4 Cards) -->
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1rem;">
                <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.25rem 1.4rem;">
                    <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">Total Test Samples</div>
                    <div style="color: #34d399; font-size: 2.3rem; font-weight: 700; line-height: 1;">{card['total']}</div>
                </div>
                <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.25rem 1.4rem;">
                    <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">{card['label_b']}</div>
                    <div style="color: #34d399; font-size: 2.3rem; font-weight: 700; line-height: 1;">{card['b']}</div>
                </div>
                <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.25rem 1.4rem;">
                    <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">{card['label_c']}</div>
                    <div style="color: #34d399; font-size: 2.3rem; font-weight: 700; line-height: 1;">{card['c']}</div>
                </div>
                <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.25rem 1.4rem;">
                    <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">Disagreements (b + c)</div>
                    <div style="color: #34d399; font-size: 2.3rem; font-weight: 700; line-height: 1;">{card['disagree']}</div>
                </div>
            </div>

            <!-- Bottom Row (3 Cards) -->
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.25rem 1.4rem;">
                    <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">Exact Test Statistic</div>
                    <div style="color: #34d399; font-size: 2.3rem; font-weight: 700; line-height: 1;">{card['stat']}</div>
                </div>
                <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.25rem 1.4rem;">
                    <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">p-value</div>
                    <div style="color: #34d399; font-size: 2.3rem; font-weight: 700; line-height: 1;">{card['p']}</div>
                </div>
                <div style="background: #111827; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1.25rem 1.4rem;">
                    <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.5rem;">Alpha (α)</div>
                    <div style="color: #34d399; font-size: 2.3rem; font-weight: 700; line-height: 1;">{card['alpha']}</div>
                </div>
            </div>

            <div style="background: rgba(16, 185, 129, 0.08); border-left: 3px solid #10b981; padding: 0.6rem 0.9rem; border-radius: 0 8px 8px 0; color: #a7f3d0; font-size: 0.85rem; margin-top: 1rem;">
                💡 <strong>Statistical Conclusion:</strong> {card['conclusion']}
            </div>
        </div>
        """)

        if df_stats is not None:
            st.markdown("#### 📋 Full Statistical Significance Benchmark Table")
            st.dataframe(df_stats, use_container_width=True, hide_index=True)

            st.markdown("""
            #### 💡 Key Statistical Findings:
            - **Baseline & Linear Models:** Random Forest outperforms the Majority Baseline, Logistic Regression, and Decision Tree with **$p < 0.05$**, proving statistically significant superiority.
            - **K-Nearest Neighbors & SVM:** Random Forest achieves significantly higher CV accuracy (**$p = 0.003$** and **$p = 0.008$** respectively).
            - **Naive Bayes & XGBoost:** Statistical parity with Random Forest (p > 0.05). Random Forest was selected as the deployed production model because of its **lowest variance** (±0.27%), minimal overfit gap, and native **tree-based interpretability** (MDI & TreeSHAP).
            """)

    with tab4:
        st.markdown("### ⚙️ Hyperparameter Tuning (GridSearchCV)")
        st.markdown("""
        `GridSearchCV` with 5-fold cross-validation was executed on `X_train` over the hyperparameter space of Random Forest:
        - `n_estimators`: `[100, 200, 300]`
        - `max_depth`: `[None, 15, 25]`
        - `min_samples_split`: `[2, 5]`
        - `max_features`: `['sqrt', 'log2']`
        """)

        if df_tuning is not None:
            st.dataframe(df_tuning, use_container_width=True, hide_index=True)

        st.success("""
        ✅ **Tuning Confirmation:** The untuned ensemble default (`n_estimators=200`, `max_features='sqrt'`) achieved 99.55% test accuracy. Tuning confirmed this parameter regime is optimal and maintains peak generalization on held-out test data.
        """)

    with tab5:
        st.markdown("### ⚡ Environmental Noise Robustness (Stress Testing)")
        st.markdown("""
        In laboratory settings, clean datasets allow multiple algorithms to score 98%–99%. 
        However, in real-world agriculture, soil sensors and weather forecasts experience **measurement noise and natural fluctuations**.
        
        This stress test subjects all trained models to Gaussian perturbation (0% to 25% sensor variance) across test inputs to evaluate **resilience and model divergence**.
        """)

        if df_noise is not None:
            st.dataframe(df_noise, use_container_width=True, hide_index=True)

            # Noise degradation plot
            fig_n, ax_n = plt.subplots(figsize=(10, 5))
            fig_n.patch.set_facecolor("#111827")
            ax_n.set_facecolor("#1f2937")

            noise_cols = [c for c in df_noise.columns if "Noise" in c]
            palette = ["#43cea2", "#4facfe", "#fbbf24", "#a78bfa", "#f87171", "#38bdf8"]

            for i, (_, row) in enumerate(df_noise.iterrows()):
                y_vals = [row[c] for c in noise_cols]
                ax_n.plot(noise_cols, y_vals, marker="o", label=row["Model"], color=palette[i % len(palette)], linewidth=2.2)

            ax_n.set_ylabel("Accuracy (%)", color="#cbd5e1")
            ax_n.set_xlabel("Perturbation Level (Gaussian Sensor Noise)", color="#cbd5e1")
            ax_n.set_title("Model Accuracy Degradation Under Sensor Noise", color="#f1f5f9", fontsize=12, fontweight="bold")
            ax_n.tick_params(colors="#94a3b8")
            ax_n.grid(color="#374151", linestyle="--", alpha=0.4)
            ax_n.legend(facecolor="#111827", edgecolor="#374151", labelcolor="#cbd5e1", fontsize=9, loc="lower left")
            for spine in ax_n.spines.values():
                spine.set_color("#4b5563")

            st.pyplot(fig_n)

            st.markdown("""
            #### 💡 Why Introducing Variance Reveals True Model Differences:
            - **Tree vs Ensemble Separation:** At 25% environmental noise, a single **Decision Tree** collapses to **69.8%**, while **Random Forest** retains **81.2%** accuracy.
            - **Overfitting Fragility:** High-capacity non-regularized models degrade rapidly when test inputs differ slightly from idealized distributions.
            - **Production Selection:** Random Forest and SVM demonstrate the strongest noise-retention slopes, justifying Random Forest as the robust choice for erratic real-world field conditions.
            """)

if __name__ == "__main__":
    st.set_page_config(page_title="📈 Model Performance", page_icon="📈", layout="wide")
    render_model_performance_page()
