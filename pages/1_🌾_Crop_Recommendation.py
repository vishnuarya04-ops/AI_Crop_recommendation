"""
Page 1: 🌾 Crop Recommendation
Interactive inference — exactly ONE recommended crop with confidence,
local SHAP feature contributions, agronomic assessment, and soil analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
CROP_EMOJIS = {
    "apple": "🍎", "banana": "🍌", "blackgram": "🫘",
    "chickpea": "🫛", "coconut": "🥥", "coffee": "☕",
    "cotton": "🧵", "grapes": "🍇", "jute": "🪢",
    "kidneybeans": "🫘", "lentil": "🍲", "maize": "🌽",
    "mango": "🥭", "mothbeans": "🫘", "mungbean": "🫛",
    "muskmelon": "🍈", "orange": "🍊", "papaya": "🍈",
    "pigeonpeas": "🫛", "pomegranate": "🫐", "rice": "🍚",
    "watermelon": "🍉",
}

CROP_INFO = {
    "rice":         "Thrives in warm, humid climates with heavy rainfall.",
    "maize":        "Grows well in moderate climates with well-drained soil.",
    "chickpea":     "Prefers cool, dry conditions with light to moderate rainfall.",
    "kidneybeans":  "Needs moderate temperature and rainfall with neutral pH soil.",
    "pigeonpeas":   "Tolerates semi-arid conditions and performs well in poor soils.",
    "mothbeans":    "Drought-resistant legume suited for arid and semi-arid regions.",
    "mungbean":     "Grows best in warm climates with well-distributed rainfall.",
    "blackgram":    "Prefers warm, humid conditions with moderate rainfall.",
    "lentil":       "Thrives in cool seasons with low to moderate rainfall.",
    "pomegranate":  "Prefers semi-arid conditions with hot, dry summers.",
    "banana":       "Requires tropical warmth, high humidity, and ample water.",
    "mango":        "Grows best in tropical and subtropical climates.",
    "grapes":       "Prefers warm, dry summers with well-drained soil.",
    "watermelon":   "Needs warm temperatures and long growing seasons.",
    "muskmelon":    "Thrives in hot, dry climates with sandy loam soil.",
    "apple":        "Requires cold winters and moderate summers.",
    "orange":       "Grows well in subtropical climates with moderate rainfall.",
    "papaya":       "Needs tropical warmth and consistent moisture.",
    "coconut":      "Thrives in coastal tropical regions with sandy soil.",
    "cotton":       "Grows best in warm climates with black soil.",
    "jute":         "Requires hot, humid conditions with heavy rainfall.",
    "coffee":       "Prefers tropical highlands with moderate temperature.",
}

FEATURES = {
    "N": {"label": "Nitrogen (N)", "min": 0.0, "max": 140.0, "default": 50.0, "step": 1.0, "unit": "kg/ha", "help": "Nitrogen ratio in soil (0–140 kg/ha)"},
    "P": {"label": "Phosphorus (P)", "min": 5.0, "max": 145.0, "default": 53.0, "step": 1.0, "unit": "kg/ha", "help": "Phosphorus ratio in soil (5–145 kg/ha)"},
    "K": {"label": "Potassium (K)", "min": 5.0, "max": 205.0, "default": 48.0, "step": 1.0, "unit": "kg/ha", "help": "Potassium ratio in soil (5–205 kg/ha)"},
    "temperature": {"label": "Temperature", "min": 8.0, "max": 44.0, "default": 25.0, "step": 0.1, "unit": "°C", "help": "Average growing temperature (8–44 °C)"},
    "humidity": {"label": "Humidity", "min": 14.0, "max": 100.0, "default": 72.0, "step": 0.1, "unit": "%", "help": "Relative humidity (14–100 %)"},
    "ph": {"label": "Soil pH", "min": 3.5, "max": 10.0, "default": 6.5, "step": 0.01, "unit": "", "help": "Soil pH level (3.5–10.0)"},
    "rainfall": {"label": "Rainfall", "min": 20.0, "max": 300.0, "default": 103.0, "step": 0.5, "unit": "mm", "help": "Annual rainfall (20–300 mm)"},
}

CROP_IDEAL = {
    "apple":       {"N": (10,32),  "P": (129,139), "K": (197,203), "temperature": (21.82,23.47), "humidity": (91.02,93.56), "ph": (5.68,6.15), "rainfall": (105.09,118.99)},
    "banana":      {"N": (90,110), "P": (75.75,89),"K": (47,53),   "temperature": (26.1,28.59),  "humidity": (77.5,82.87),  "ph": (5.73,6.22), "rainfall": (97.48,111.36)},
    "blackgram":   {"N": (30,50),  "P": (60.75,74),"K": (17,22),   "temperature": (27.22,32.42), "humidity": (61.08,68.26), "ph": (6.09,7.3),  "rainfall": (63.66,70.89)},
    "chickpea":    {"N": (30,50),  "P": (61.75,75),"K": (77,83),   "temperature": (17.08,20.65), "humidity": (15.56,18.2),  "ph": (6.18,7.41), "rainfall": (68.85,80.68)},
    "coconut":     {"N": (11,33.25),"P":(10,23),   "K": (28,33),   "temperature": (26.32,27.71), "humidity": (92.19,97.48), "ph": (5.72,6.26), "rainfall": (150.52,199.52)},
    "coffee":      {"N": (90.75,112),"P":(18,27),  "K": (27,33),   "temperature": (24.24,26.85), "humidity": (53.43,63.72), "ph": (6.35,6.83), "rainfall": (121.22,175.19)},
    "cotton":      {"N": (107,127),"P": (40,53),   "K": (17,22),   "temperature": (23.43,24.34), "humidity": (77.47,82.12), "ph": (6.39,7.49), "rainfall": (71.0,90.97)},
    "grapes":      {"N": (14,33),  "P": (126,138), "K": (197.75,203),"temperature":(14.98,32.5), "humidity": (80.88,82.86), "ph": (5.74,6.26), "rainfall": (67.75,72.04)},
    "jute":        {"N": (69,88),  "P": (40,53),   "K": (37.75,42),"temperature": (24.44,25.28), "humidity": (73.46,87.36), "ph": (6.47,6.97), "rainfall": (162.61,187.47)},
    "kidneybeans": {"N": (11,31),  "P": (62,73),   "K": (18,23),   "temperature": (16.4,18.94),  "humidity": (19.7,23.68),  "ph": (5.6,5.85),  "rainfall": (71.0,89.24)},
    "lentil":      {"N": (8,28),   "P": (63,74),   "K": (17,22),   "temperature": (21.73,27.48), "humidity": (36.98,56.52), "ph": (6.25,7.49), "rainfall": (41.36,49.73)},
    "maize":       {"N": (67,88.25),"P":(42,55),   "K": (17,22),   "temperature": (20.11,24.66), "humidity": (60.65,70.6),  "ph": (5.71,6.58), "rainfall": (67.05,79.11)},
    "mango":       {"N": (10,30),  "P": (22,32),   "K": (27,33),   "temperature": (28.83,33.31), "humidity": (47.58,52.55), "ph": (4.99,6.4),  "rainfall": (92.15,96.89)},
    "mothbeans":   {"N": (11,30.25),"P":(42.75,55),"K": (18,23),   "temperature": (26.43,30.12), "humidity": (46.73,59.44), "ph": (5.38,8.37), "rainfall": (37.99,64.19)},
    "mungbean":    {"N": (10,31),  "P": (40,54.25),"K": (17,22),   "temperature": (27.88,29.25), "humidity": (83.43,87.83), "ph": (6.47,6.98), "rainfall": (43.1,54.84)},
    "muskmelon":   {"N": (89,111), "P": (12,25),   "K": (47,52.25),"temperature": (27.96,29.37), "humidity": (90.96,93.79), "ph": (6.16,6.55), "rainfall": (22.07,26.86)},
    "orange":      {"N": (9,31),   "P": (9,23),    "K": (8,12.25), "temperature": (17.05,29.55), "humidity": (91.03,93.37), "ph": (6.48,7.52), "rainfall": (106.22,115.57)},
    "papaya":      {"N": (39,59.25),"P":(54,65),   "K": (47,52),   "temperature": (28.55,39.07), "humidity": (91.17,93.51), "ph": (6.62,6.84), "rainfall": (81.94,202.39)},
    "pigeonpeas":  {"N": (10,30.25),"P":(61,73.25),"K": (18,23),   "temperature": (22.87,31.47), "humidity": (38.37,57.0),  "ph": (5.0,6.42),  "rainfall": (122.35,177.91)},
    "pomegranate":  {"N": (8,29.25), "P":(13,25),  "K": (38,43),   "temperature": (19.8,23.89),  "humidity": (88.1,92.43),  "ph": (5.97,6.89), "rainfall": (104.97,109.96)},
    "rice":        {"N": (69,91),  "P": (41,54.25),"K": (38,42),   "temperature": (21.93,25.51), "humidity": (80.95,83.47), "ph": (5.87,7.04), "rainfall": (204.33,264.13)},
    "watermelon":  {"N": (89,110), "P": (10,23),   "K": (47,53),   "temperature": (24.9,26.26),  "humidity": (83.01,87.64), "ph": (6.26,6.76), "rainfall": (46.06,56.41)},
}

# ──────────────────────────────────────────────
# Load Assets
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models & pipeline…")
def get_model_and_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "random_forest.joblib")
    pipe_path = os.path.join(base_dir, "preprocessing_pipeline.joblib")
    rf = joblib.load(model_path)
    pipe = joblib.load(pipe_path)
    explainer = shap.TreeExplainer(rf)
    return rf, pipe, explainer

rf_model, pipeline, explainer = get_model_and_pipeline()
scaler = pipeline["scaler"]
label_encoder = pipeline["label_encoder"]
feature_cols = pipeline["feature_cols"]

def render_crop_recommendation_page():
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.5rem 0;">
        <div style="font-size: 3rem; margin-bottom: 0.3rem;">🌾</div>
        <h1 style="font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #43cea2, #4facfe, #00f2fe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.2rem;">
            Crop Recommendation
        </h1>
        <p style="color: #94a3b8; font-size: 0.95rem; max-width: 650px; margin: 0 auto 0.8rem auto;">
            Enter soil nutrients &amp; environmental parameters to receive the <strong style="color:#43cea2;">primary recommended crop</strong> alongside <strong style="color:#4facfe;">viable alternative crops</strong>, confidence distributions, and agronomic suitability analysis.
        </p>
        <span style="background: rgba(67,206,162,0.12); border: 1px solid rgba(67,206,162,0.25); color: #43cea2; font-size: 0.72rem; font-weight: 600; padding: 0.25rem 0.75rem; border-radius: 20px; text-transform: uppercase;">
            Random Forest · 99.55% Test Accuracy
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    def get_val(f_key):
        return FEATURES[f_key]["default"]

    # Reactive Inputs (Auto-updates live whenever you change any value)
    st.markdown("<h4 style='color:#e2e8f0;margin-bottom:0.5rem;'>🧪 Soil Nutrient Levels (NPK)</h4>", unsafe_allow_html=True)
    cn, cp, ck = st.columns(3)
    with cn:
        val_n = st.number_input(FEATURES["N"]["label"], FEATURES["N"]["min"], FEATURES["N"]["max"], get_val("N"), FEATURES["N"]["step"], help=FEATURES["N"]["help"], key="rec_n")
    with cp:
        val_p = st.number_input(FEATURES["P"]["label"], FEATURES["P"]["min"], FEATURES["P"]["max"], get_val("P"), FEATURES["P"]["step"], help=FEATURES["P"]["help"], key="rec_p")
    with ck:
        val_k = st.number_input(FEATURES["K"]["label"], FEATURES["K"]["min"], FEATURES["K"]["max"], get_val("K"), FEATURES["K"]["step"], help=FEATURES["K"]["help"], key="rec_k")

    st.markdown("<h4 style='color:#e2e8f0;margin:1rem 0 0.5rem 0;'>🌤️ Climatic & Soil Environmental Conditions</h4>", unsafe_allow_html=True)
    ct, ch = st.columns(2)
    with ct:
        val_temp = st.number_input(FEATURES["temperature"]["label"], FEATURES["temperature"]["min"], FEATURES["temperature"]["max"], get_val("temperature"), FEATURES["temperature"]["step"], help=FEATURES["temperature"]["help"], key="rec_temp")
    with ch:
        val_hum = st.number_input(FEATURES["humidity"]["label"], FEATURES["humidity"]["min"], FEATURES["humidity"]["max"], get_val("humidity"), FEATURES["humidity"]["step"], help=FEATURES["humidity"]["help"], key="rec_hum")

    cph, cr = st.columns(2)
    with cph:
        val_ph = st.number_input(FEATURES["ph"]["label"], FEATURES["ph"]["min"], FEATURES["ph"]["max"], get_val("ph"), FEATURES["ph"]["step"], help=FEATURES["ph"]["help"], key="rec_ph")
    with cr:
        val_rain = st.number_input(FEATURES["rainfall"]["label"], FEATURES["rainfall"]["min"], FEATURES["rainfall"]["max"], get_val("rainfall"), FEATURES["rainfall"]["step"], help=FEATURES["rainfall"]["help"], key="rec_rain")

    def compute_agronomic_suitability(n, p, k, temp, hum, ph, rain, crop_name):
        """Compute suitability score for a specific crop."""
        user_vals = {"N": n, "P": p, "K": k, "temperature": temp, "humidity": hum, "ph": ph, "rainfall": rain}
        weights = {"N": 1.2, "P": 1.2, "K": 1.2, "temperature": 1.0, "humidity": 1.0, "ph": 1.1, "rainfall": 1.0}
        ideals = CROP_IDEAL.get(crop_name, {})
        if not ideals:
            return 85.0
        crop_scores = []
        for feat, (q25, q75) in ideals.items():
            val = user_vals[feat]
            if q25 <= val <= q75:
                s = 100.0
            else:
                dist = (q25 - val) if val < q25 else (val - q75)
                span = max(1.0, q75 - q25)
                s = max(0.0, 100.0 - (dist / span) * 45.0)
            crop_scores.append(s * weights[feat])
        return round(sum(crop_scores) / sum(weights.values()), 1)

    # Assign manual inputs
    n_act, p_act, k_act = val_n, val_p, val_k
    temp_act, hum_act, ph_act, rain_act = val_temp, val_hum, val_ph, val_rain

    # Warnings
    warnings = []
    if n_act + p_act + k_act < 15:
        warnings.append("Total NPK is very low (< 15 kg/ha). Few crops thrive in severely depleted soils.")
    if ph_act < 4.0:
        warnings.append("Soil pH < 4.0 is strongly acidic. Soil liming may be required.")
    if ph_act > 9.0:
        warnings.append("Soil pH > 9.0 is strongly alkaline. Consider sulfur amendments.")

    for w in warnings:
        st.warning(f"⚠️ {w}")

    raw_df = pd.DataFrame([[n_act, p_act, k_act, temp_act, hum_act, ph_act, rain_act]], columns=feature_cols)
    scaled = scaler.transform(raw_df)
    proba = rf_model.predict_proba(scaled)[0]

    # Get Top-4 crops
    top4_idx = np.argsort(proba)[::-1][:4]
    best_idx = top4_idx[0]
    best_crop = label_encoder.inverse_transform([best_idx])[0]
    best_prob = proba[best_idx]
    emoji = CROP_EMOJIS.get(best_crop, "🌱")
    info = CROP_INFO.get(best_crop, "Well-suited for the submitted agronomic environment.")

    # Compute agronomic suitability for the primary crop
    agro_score = compute_agronomic_suitability(n_act, p_act, k_act, temp_act, hum_act, ph_act, rain_act, best_crop)

    # ═══════════════════════════════════════════════
    # RESULT CARD — PRIMARY BEST RECOMMENDATION
    # ═══════════════════════════════════════════════
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.04); border: 1px solid rgba(67,206,162,0.3); border-radius: 18px; padding: 2rem; text-align: center; margin: 1.5rem 0;">
        <div style="font-size: 0.75rem; font-weight: 700; color: #43cea2; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.2rem;">🥇 Primary Recommended Crop</div>
        <div style="font-size: 3.8rem; margin-bottom: 0.3rem;">{emoji}</div>
        <div style="font-size: 2.2rem; font-weight: 800; color: #f1f5f9; text-transform: capitalize;">{best_crop}</div>
        <div style="color: #43cea2; font-size: 1.3rem; font-weight: 700; margin-top: 0.4rem;">Model Confidence: {best_prob * 100:.1f}% · Agronomic Match: {agro_score}%</div>
        <div style="background: rgba(255,255,255,0.08); border-radius: 10px; height: 10px; margin: 0.6rem auto 1rem auto; max-width: 400px; overflow: hidden;">
            <div style="height: 100%; border-radius: 10px; width: {best_prob * 100:.1f}%; background: linear-gradient(90deg, #43cea2, #4facfe);"></div>
        </div>
        <p style="color: #94a3b8; font-size: 0.9rem; font-style: italic; max-width: 550px; margin: 0 auto;">"{info}"</p>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════
    # ALTERNATIVE VIABLE CROPS
    # ═══════════════════════════════════════════════
    alt_indices = [idx for idx in top4_idx[1:]]
    if alt_indices:
        st.markdown("<h4 style='color:#e2e8f0;margin:1.8rem 0 0.4rem 0;'>🥈 Viable Alternative Crops & Rotation Options</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color:#94a3b8;font-size:0.85rem;margin-bottom:1rem;'>Depending on your crop rotation plans, market price, or seed availability, here are the other highest-ranking crops suited to your soil profile:</p>", unsafe_allow_html=True)
        
        alt_cols = st.columns(len(alt_indices))
        for i, idx in enumerate(alt_indices):
            alt_crop = label_encoder.inverse_transform([idx])[0]
            alt_prob = proba[idx]
            alt_emoji = CROP_EMOJIS.get(alt_crop, "🌱")
            alt_info = CROP_INFO.get(alt_crop, "Suitable alternative crop for these soil and climate conditions.")
            alt_agro = compute_agronomic_suitability(n_act, p_act, k_act, temp_act, hum_act, ph_act, rain_act, alt_crop)
            
            badge_color = "#43cea2" if alt_agro >= 75 else ("#fbbf24" if alt_agro >= 55 else "#94a3b8")
            badge_text = "Highly Viable" if alt_agro >= 75 else ("Acceptable" if alt_agro >= 55 else "Secondary")

            with alt_cols[i]:
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; padding: 1.2rem; text-align: center; height: 100%;">
                    <span style="background: rgba(255,255,255,0.08); color: #94a3b8; font-size: 0.7rem; font-weight: 600; padding: 0.15rem 0.5rem; border-radius: 10px;">#{i+2} Option</span>
                    <div style="font-size: 2.2rem; margin: 0.4rem 0 0.2rem 0;">{alt_emoji}</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #f1f5f9; text-transform: capitalize;">{alt_crop}</div>
                    <div style="color: #4facfe; font-size: 0.95rem; font-weight: 600; margin: 0.3rem 0;">Confidence: {alt_prob * 100:.1f}%</div>
                    <div style="color: {badge_color}; font-size: 0.85rem; font-weight: 600;">Match: {alt_agro}% ({badge_text})</div>
                    <div style="background: rgba(255,255,255,0.06); border-radius: 6px; height: 6px; margin: 0.5rem auto; width: 80%; overflow: hidden;">
                        <div style="height: 100%; border-radius: 6px; width: {min(100.0, max(5.0, alt_prob * 100))}%; background: {badge_color};"></div>
                    </div>
                    <p style="color: #94a3b8; font-size: 0.78rem; font-style: italic; margin-top: 0.5rem; line-height: 1.3;">"{alt_info}"</p>
                </div>
                """, unsafe_allow_html=True)

    # ── Detailed Agronomic Match Breakdown ──
    st.markdown(f"<h4 style='color:#e2e8f0;margin:2rem 0 0.5rem 0;'>🌱 Detailed Agronomic Assessment for {best_crop.capitalize()}</h4>", unsafe_allow_html=True)
    
    crop_ideals = CROP_IDEAL.get(best_crop, {})
    param_breakdown = []
    user_inputs = {"N": n_act, "P": p_act, "K": k_act, "temperature": temp_act, "humidity": hum_act, "ph": ph_act, "rainfall": rain_act}
    for feat in feature_cols:
        u_val = user_inputs[feat]
        if feat in crop_ideals:
            q25, q75 = crop_ideals[feat]
            if q25 <= u_val <= q75:
                status = "✅ Optimal"
            elif u_val < q25:
                status = f"⚠️ Low (Ideal: {q25:.1f}–{q75:.1f})"
            else:
                status = f"⚠️ High (Ideal: {q25:.1f}–{q75:.1f})"
        else:
            status = "✅ Normal"
        param_breakdown.append({
            "Parameter": FEATURES[feat]["label"],
            "Your Reading": f"{u_val:.1f} {FEATURES[feat]['unit']}",
            "Agronomic Status for " + best_crop.capitalize(): status
        })
    st.dataframe(pd.DataFrame(param_breakdown), use_container_width=True, hide_index=True)

    # ── Local Explainability (SHAP) — WHY this crop was recommended ──
    st.markdown(f"<h4 style='color:#e2e8f0;margin:1.8rem 0 0.4rem 0;'>🔍 Why {best_crop.capitalize()} Was Recommended (Feature Contributions)</h4>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(245,158,11,0.08); border-left: 3px solid #f59e0b; padding: 0.6rem 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem; color: #fcd34d; font-size: 0.8rem;">
        ⚠️ <strong>Academic Note:</strong> This explanation is presented as <em>model interpretation only</em>, reflecting how the trained Random Forest weighted these inputs, and <em>not as a claim of direct agronomic causation</em>.
    </div>
    """, unsafe_allow_html=True)

    # Calculate exact SHAP values for this instance
    best_class_idx = best_idx
    shap_res = explainer(scaled)
    sample_shap = shap_res.values[0, :, best_class_idx]

    shap_df = pd.DataFrame({
        "Feature": [FEATURES[c]["label"] for c in feature_cols],
        "Raw Value": [f"{v:.1f} {FEATURES[c]['unit']}" for v, c in zip([n_act, p_act, k_act, temp_act, hum_act, ph_act, rain_act], feature_cols)],
        "SHAP Contribution": sample_shap,
        "Impact": ["Positive (+)" if s > 0 else "Negative (-)" for s in sample_shap]
    }).sort_values("SHAP Contribution", ascending=False)

    col_sh1, col_sh2 = st.columns([1.2, 1])
    with col_sh1:
        fig_shap, ax_shap = plt.subplots(figsize=(7, 3.5))
        fig_shap.patch.set_facecolor("#111827")
        ax_shap.set_facecolor("#1f2937")

        sorted_shap = shap_df.sort_values("SHAP Contribution")
        colors = ["#43cea2" if v > 0 else "#f87171" for v in sorted_shap["SHAP Contribution"]]
        ax_shap.barh(sorted_shap["Feature"], sorted_shap["SHAP Contribution"], color=colors, edgecolor="#374151")
        ax_shap.axvline(0, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_shap.set_xlabel("SHAP Contribution", color="#cbd5e1", fontsize=9)
        ax_shap.set_title(f"Feature Contributions → {best_crop.capitalize()}", color="#f1f5f9", fontsize=11, fontweight="bold")
        ax_shap.tick_params(colors="#94a3b8", labelsize=8)
        ax_shap.grid(color="#374151", linestyle="--", alpha=0.3, axis="x")
        for spine in ax_shap.spines.values():
            spine.set_color("#4b5563")
        plt.tight_layout()
        st.pyplot(fig_shap)

    with col_sh2:
        st.dataframe(shap_df[["Feature", "Raw Value", "Impact"]], use_container_width=True, hide_index=True)

    # ── Probability Breakdown Across Candidates ──
    st.markdown("---")
    st.markdown("<h4 style='color:#e2e8f0;margin:1rem 0 0.5rem 0;'>📊 Candidate Crop Probability Breakdown</h4>", unsafe_allow_html=True)

    top_candidates = [idx for idx in np.argsort(proba)[::-1][:6] if proba[idx] > 0.005 or idx == best_idx]
    if len(top_candidates) < 3:
        top_candidates = np.argsort(proba)[::-1][:4]

    cand_names = [label_encoder.inverse_transform([i])[0].capitalize() for i in reversed(top_candidates)]
    cand_probs = [proba[i] * 100 for i in reversed(top_candidates)]
    cand_colors = ["#43cea2" if name.lower() == best_crop else "#4facfe" for name in cand_names]

    fig_conf, ax_conf = plt.subplots(figsize=(8, max(2.5, len(cand_names) * 0.45)))
    fig_conf.patch.set_facecolor("#111827")
    ax_conf.set_facecolor("#1f2937")

    bars = ax_conf.barh(cand_names, cand_probs, color=cand_colors, edgecolor="#374151", height=0.6)
    for bar in bars:
        w = bar.get_width()
        ax_conf.text(w + 1.0, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%",
                     va="center", ha="left", color="#f1f5f9", fontsize=8.5, fontweight="bold")

    ax_conf.set_xlim(0, max(100, max(cand_probs) + 12))
    ax_conf.set_xlabel("Probability (%)", color="#cbd5e1", fontsize=9)
    ax_conf.set_title("Probability Breakdown Across Top Candidate Crops", color="#f1f5f9", fontsize=11, fontweight="bold")
    ax_conf.tick_params(colors="#94a3b8", labelsize=9)
    ax_conf.grid(color="#374151", linestyle="--", alpha=0.3, axis="x")
    for spine in ax_conf.spines.values():
        spine.set_color("#4b5563")
    plt.tight_layout()
    st.pyplot(fig_conf)

    # ── Soil Analysis Dashboard ──
    st.markdown("---")
    st.markdown("<h4 style='color:#e2e8f0;margin:1rem 0 0.5rem 0;'>🌱 Soil & Climate Suitability Advisory</h4>", unsafe_allow_html=True)

    moisture_score = min(100, max(0, (val_hum * 0.35) + (val_rain / 300 * 100 * 0.65)))
    npk_total = val_n + val_p + val_k
    fertility = round(min(100, max(0, (npk_total / 200 * 40) + (max(0, 100 - abs(val_ph - 6.5) / 3.5 * 100) * 0.3) + moisture_score * 0.3)), 1)

    c_m, c_f = st.columns(2)
    with c_m:
        st.metric("💧 Soil Moisture Index", f"{moisture_score:.1f}%", help="Derived from relative humidity and precipitation")
    with c_f:
        st.metric("🏆 Soil Fertility Score", f"{fertility:.1f}/100", help="Derived from NPK balance, pH neutrality, and moisture")

if __name__ == "__main__":
    st.set_page_config(page_title="🌾 Crop Recommendation", page_icon="🌾", layout="centered")
    render_crop_recommendation_page()
