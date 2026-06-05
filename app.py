import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from datetime import datetime

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FitCalc · Calorie Predictor",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Global CSS  (light bg, dark font, coral accent)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono&display=swap');

/* ── Base ── */
html, body, .stApp { background:#f7f8fa; color:#1a1f2e; font-family:'DM Sans',sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background:#ffffff;
    border-right:1px solid #e8eaf0;
}
[data-testid="stSidebar"] * { color:#1a1f2e !important; }
[data-testid="stSidebar"] .sidebar-logo {
    font-size:1.5rem; font-weight:700; letter-spacing:-0.5px;
    color:#ff5c5c !important; padding:0 0 0.5rem 0;
}
[data-testid="stSidebar"] hr { border-color:#e8eaf0; }

/* ── Radio nav ── */
div[data-testid="stRadio"] label {
    display:flex; align-items:center; gap:10px;
    padding:10px 14px; border-radius:10px;
    font-weight:500; font-size:0.95rem;
    color:#4a5568 !important; cursor:pointer;
    transition:all .15s;
}
div[data-testid="stRadio"] label:hover { background:#fff0f0; color:#ff5c5c !important; }
div[data-testid="stRadio"] [aria-checked="true"] ~ div label,
div[data-testid="stRadio"] input:checked + div label {
    background:#fff0f0 !important; color:#ff5c5c !important;
}

/* ── Page headers ── */
.page-header {
    font-size:1.75rem; font-weight:700; color:#1a1f2e;
    border-left:4px solid #ff5c5c; padding-left:14px;
    margin-bottom:1.5rem;
}
.page-sub { color:#6b7280; font-size:0.97rem; margin-top:-1rem; margin-bottom:1.5rem; padding-left:18px; }

/* ── Cards ── */
.card {
    background:#ffffff; border-radius:14px;
    padding:24px 28px; margin-bottom:1rem;
    box-shadow:0 1px 4px rgba(0,0,0,.07);
    border:1px solid #e8eaf0;
}
.card-label { font-size:0.78rem; font-weight:600; letter-spacing:.06em; text-transform:uppercase; color:#9ca3af; margin-bottom:4px; }
.card-value { font-size:2rem; font-weight:700; color:#1a1f2e; }
.card-unit  { font-size:0.9rem; color:#6b7280; margin-top:2px; }

/* ── Result box ── */
.result-box {
    background:linear-gradient(135deg,#ff5c5c,#ff8c42);
    border-radius:16px; padding:28px 32px; text-align:center;
    color:#fff; margin:1.5rem 0;
    box-shadow:0 6px 24px rgba(255,92,92,.25);
}
.result-box .kcal { font-size:3.5rem; font-weight:700; line-height:1; }
.result-box .label { font-size:1rem; opacity:.85; margin-top:6px; }

/* ── Insight chip ── */
.chip {
    display:inline-block; padding:8px 18px; border-radius:999px;
    font-weight:600; font-size:0.88rem; margin-top:8px;
}
.chip.low    { background:#fff3cd; color:#92600a; }
.chip.mid    { background:#d1fae5; color:#065f46; }
.chip.high   { background:#dbeafe; color:#1e40af; }

/* ── Table ── */
.stDataFrame { border-radius:12px; overflow:hidden; }
thead th { background:#f7f8fa !important; font-weight:600; color:#1a1f2e !important; }
tbody td { color:#374151 !important; }

/* ── Buttons ── */
.stButton > button {
    background:#ff5c5c; color:#fff !important;
    border:none; border-radius:10px;
    padding:10px 28px; font-weight:600; font-size:1rem;
    transition:all .15s; width:100%;
}
.stButton > button:hover { background:#e04848; box-shadow:0 4px 14px rgba(255,92,92,.35); }

/* ── Inputs ── */
.stSlider [data-baseweb="slider"] div[role="slider"] { background:#ff5c5c !important; border-color:#ff5c5c !important; }
.stTextInput input, .stNumberInput input {
    border-radius:8px !important; border-color:#e8eaf0 !important;
    background:#f7f8fa !important; color:#1a1f2e !important;
}
label[data-testid="stWidgetLabel"] > div { color:#374151 !important; font-weight:500; }

/* ── Alert overrides ── */
.stAlert { border-radius:10px; }

/* ── Section divider ── */
.sect-div { border:none; border-top:1px solid #e8eaf0; margin:1.5rem 0; }

/* ── Metric badge (sidebar) ── */
.badge {
    background:#fff0f0; color:#ff5c5c;
    border-radius:8px; padding:10px 14px;
    font-size:0.88rem; font-weight:600;
    margin-top:12px; text-align:center;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
HISTORY_FILE = "prediction_history.csv"

feature_names = [
    "Id", "TotalSteps", "TotalDistance", "TrackerDistance",
    "VeryActiveDistance", "ModeratelyActiveDistance", "LightActiveDistance",
    "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes",
    "SedentaryMinutes",
]

def init_history():
    if not os.path.exists(HISTORY_FILE):
        pd.DataFrame(columns=feature_names + ["Predicted_Calories", "Timestamp"]).to_csv(HISTORY_FILE, index=False)

def save_prediction(row: dict, cal: float):
    init_history()
    row["Predicted_Calories"] = round(cal)
    row["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.concat([pd.read_csv(HISTORY_FILE), pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

def get_history(uid):
    init_history()
    df = pd.read_csv(HISTORY_FILE)
    df["Id"] = df["Id"].astype(str)
    return df[df["Id"] == str(uid)]

@st.cache_resource
def load_model():
    if os.path.exists("catboost_model.pkl"):
        return joblib.load("catboost_model.pkl")
    return None

model = load_model()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>🔥 FitCalc</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6b7280;font-size:.85rem;margin-top:-6px;'>Calorie Burn Predictor</p>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "",
        ["📊  Predict Calorie", "📈  Dashboard", "📋  Model Details"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("<div class='badge'>🏆 Best Model: CatBoost<br><span style='font-size:1.1rem;'>R² = 0.905</span></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Predictions stored locally in `prediction_history.csv`")

# ─────────────────────────────────────────────
# Matplotlib style helper
# ─────────────────────────────────────────────
def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f7f8fa",
        "axes.edgecolor": "#e8eaf0",
        "axes.labelcolor": "#374151",
        "xtick.color": "#6b7280",
        "ytick.color": "#6b7280",
        "grid.color": "#e8eaf0",
        "grid.linestyle": "--",
        "font.family": "sans-serif",
        "text.color": "#1a1f2e",
    })

# ═════════════════════════════════════════════
# PAGE 1 — PREDICT
# ═════════════════════════════════════════════
if page == "📊  Predict Calorie":
    st.markdown("<div class='page-header'>📊 Predict Calorie Burn</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Fill in your daily activity stats and hit Predict.</div>", unsafe_allow_html=True)

    input_mode = st.toggle("Use sliders instead of number inputs", value=True)

    with st.form("predict_form"):
        col_l, col_r = st.columns(2, gap="large")

        def _input(label, min_v, max_v, default, step, col, use_slider, is_float=False):
            with col:
                if use_slider:
                    return st.slider(label, min_value=min_v, max_value=max_v, value=default, step=step)
                else:
                    return st.number_input(label, min_value=min_v, max_value=max_v, value=default, step=step)

        with col_l:
            st.markdown("#### 🧍 User & Distance")
            user_id          = st.number_input("User ID", value=12345, step=1)
            total_steps      = _input("Total Steps", 0, 30000, 8000, 100,   col_l, input_mode)
            total_dist       = _input("Total Distance (km)", 0.0, 30.0, 6.0, 0.1, col_l, input_mode, True)
            tracker_dist     = _input("Tracker Distance (km)", 0.0, 30.0, 6.0, 0.1, col_l, input_mode, True)
            very_active_dist = _input("Very Active Distance (km)", 0.0, 15.0, 2.0, 0.1, col_l, input_mode, True)
            mod_active_dist  = _input("Moderately Active Distance (km)", 0.0, 10.0, 1.0, 0.1, col_l, input_mode, True)

        with col_r:
            st.markdown("#### ⏱️ Activity Minutes")
            light_active_dist  = _input("Light Active Distance (km)", 0.0, 20.0, 4.0, 0.1, col_r, input_mode, True)
            very_active_min    = _input("Very Active Minutes", 0, 300, 60, 1, col_r, input_mode)
            fairly_active_min  = _input("Fairly Active Minutes", 0, 200, 30, 1, col_r, input_mode)
            lightly_active_min = _input("Lightly Active Minutes", 0, 500, 200, 1, col_r, input_mode)
            sedentary_min      = _input("Sedentary Minutes", 0, 1440, 720, 10, col_r, input_mode)

        st.markdown("<hr class='sect-div'>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔥 Predict Calories")

    if submitted:
        input_df = pd.DataFrame([[
            user_id, total_steps, total_dist, tracker_dist,
            very_active_dist, mod_active_dist, light_active_dist,
            very_active_min, fairly_active_min, lightly_active_min, sedentary_min,
        ]], columns=feature_names)

        if model:
            prediction = float(model.predict(input_df)[0])
        else:
            # demo prediction if model not found
            prediction = 300 + total_steps * 0.04 + very_active_min * 8.2 + fairly_active_min * 5 + lightly_active_min * 1.5 - sedentary_min * 0.2
            st.warning("⚠️ `catboost_model.pkl` not found – showing a demo estimate.")

        # Result display
        st.markdown(f"""
        <div class='result-box'>
            <div class='label'>Estimated Calorie Burn</div>
            <div class='kcal'>{prediction:,.0f}</div>
            <div class='label'>kcal / day</div>
        </div>
        """, unsafe_allow_html=True)

        if prediction < 1500:
            chip = "<span class='chip low'>⚡ Low burn – try increasing active minutes</span>"
        elif prediction < 2500:
            chip = "<span class='chip mid'>✅ Moderate burn – solid daily activity</span>"
        else:
            chip = "<span class='chip high'>🎉 High burn – excellent work!</span>"
        st.markdown(chip, unsafe_allow_html=True)

        # Quick summary cards
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, lbl, val, unit in [
            (c1, "Steps",            f"{total_steps:,}",       "steps"),
            (c2, "Active Minutes",   f"{very_active_min}",     "min"),
            (c3, "Distance",         f"{total_dist:.1f}",      "km"),
            (c4, "Sedentary",        f"{sedentary_min}",       "min"),
        ]:
            col.markdown(f"""
            <div class='card'>
                <div class='card-label'>{lbl}</div>
                <div class='card-value'>{val}</div>
                <div class='card-unit'>{unit}</div>
            </div>""", unsafe_allow_html=True)

        # Save
        row = dict(zip(feature_names, [
            user_id, total_steps, total_dist, tracker_dist,
            very_active_dist, mod_active_dist, light_active_dist,
            very_active_min, fairly_active_min, lightly_active_min, sedentary_min,
        ]))
        save_prediction(row, prediction)
        st.success("✅ Saved to history — check the **Dashboard** tab.")

# ═════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ═════════════════════════════════════════════
elif page == "📈  Dashboard":
    st.markdown("<div class='page-header'>📈 User Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>View real Fitbit history and your prediction trends for any user.</div>", unsafe_allow_html=True)

    @st.cache_data
    def load_original_data():
        if os.path.exists("dailyActivity_merged.csv"):
            return pd.read_csv("dailyActivity_merged.csv")
        return pd.DataFrame()

    original_df = load_original_data()

    uid = st.text_input("Enter User ID", value="1503960366", placeholder="e.g. 1503960366")

    if uid:
        set_plot_style()
        CORAL = "#ff5c5c"
        BLUE  = "#4f8ef7"

        # ── SECTION A: Real Fitbit data ──────────────────────────────────
        orig_user = pd.DataFrame()
        if not original_df.empty:
            orig_user = original_df[original_df["Id"].astype(str) == str(uid)].copy()

        if not orig_user.empty:
            orig_user = orig_user.sort_values("ActivityDate").reset_index(drop=True)

            # KPI cards from real data
            st.markdown("### 📅 Real Fitbit History")
            k1, k2, k3, k4 = st.columns(4)
            for col, lbl, val, unit in [
                (k1, "Days Recorded",  str(len(orig_user)),                    "days"),
                (k2, "Avg Calories",   f"{orig_user['Calories'].mean():,.0f}", "kcal"),
                (k3, "Peak Calories",  f"{orig_user['Calories'].max():,.0f}",  "kcal"),
                (k4, "Avg Steps",      f"{orig_user['TotalSteps'].mean():,.0f}", "steps"),
            ]:
                col.markdown(f"""
                <div class='card'>
                    <div class='card-label'>{lbl}</div>
                    <div class='card-value'>{val}</div>
                    <div class='card-unit'>{unit}</div>
                </div>""", unsafe_allow_html=True)

            # Actual calorie trend
            st.markdown("#### 🔵 Actual Calories Burned (Fitbit Dataset)")
            fig1, ax1 = plt.subplots(figsize=(10, 3.5))
            ax1.fill_between(orig_user["ActivityDate"], orig_user["Calories"], alpha=0.1, color=BLUE)
            ax1.plot(orig_user["ActivityDate"], orig_user["Calories"], marker="o", color=BLUE,
                     linewidth=2, markersize=4, label="Actual")
            avg_real = orig_user["Calories"].mean()
            ax1.axhline(avg_real, color=BLUE, linestyle="--", linewidth=1, alpha=0.5, label=f"Avg {avg_real:.0f} kcal")
            ax1.set_ylabel("Calories (kcal)")
            ax1.legend(frameon=False)
            plt.xticks(rotation=35, ha="right", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig1)

            # Steps vs Calories + Active minutes pie — side by side
            col_a, col_b = st.columns(2, gap="large")
            with col_a:
                st.markdown("#### 👟 Steps vs Actual Calories")
                fig2, ax2 = plt.subplots(figsize=(5, 3.5))
                ax2.scatter(orig_user["TotalSteps"], orig_user["Calories"],
                            color=BLUE, alpha=0.7, edgecolors="white", linewidths=0.5, s=55)
                if len(orig_user) > 1:
                    z = np.polyfit(orig_user["TotalSteps"], orig_user["Calories"], 1)
                    xs = np.linspace(orig_user["TotalSteps"].min(), orig_user["TotalSteps"].max(), 100)
                    ax2.plot(xs, np.poly1d(z)(xs), "--", color="#aaa", linewidth=1)
                ax2.set_xlabel("Total Steps")
                ax2.set_ylabel("Calories (kcal)")
                plt.tight_layout()
                st.pyplot(fig2)

            with col_b:
                st.markdown("#### ⏱️ Avg Active Minutes Breakdown")
                act_cols = ["VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes"]
                # check which columns exist
                available = [c for c in act_cols if c in orig_user.columns]
                if available:
                    avgs   = [orig_user[c].mean() for c in available]
                    labels = [c.replace("Minutes", "").replace("Lightly", "Light ").replace("Fairly", "Fairly ") for c in available]
                    colors = ["#ff5c5c", "#ff8c42", "#ffd166", "#e8eaf0"][:len(available)]
                    fig3, ax3 = plt.subplots(figsize=(5, 3.5))
                    wedges, _, autotexts = ax3.pie(
                        avgs, labels=None, colors=colors, autopct="%1.0f%%",
                        startangle=140, pctdistance=0.75,
                        wedgeprops={"linewidth": 2, "edgecolor": "white"},
                    )
                    for at in autotexts: at.set_fontsize(9); at.set_color("#1a1f2e")
                    ax3.legend(labels, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=9, frameon=False)
                    plt.tight_layout()
                    st.pyplot(fig3)

            # Raw table (expandable)
            with st.expander("📄 View raw Fitbit records"):
                show = [c for c in ["ActivityDate", "TotalSteps", "VeryActiveMinutes",
                                    "SedentaryMinutes", "Calories"] if c in orig_user.columns]
                st.dataframe(orig_user[show], use_container_width=True)

        else:
            if not original_df.empty:
                st.info(f"User ID **{uid}** not found in `dailyActivity_merged.csv`. Showing prediction history only.")
            else:
                st.warning("`dailyActivity_merged.csv` not found in app folder. Place it alongside `app.py` to see real Fitbit data.")

        # ── SECTION B: App prediction history ────────────────────────────
        st.markdown("---")
        st.markdown("### 🔴 Your Prediction History (from this app)")

        pred_df = get_history(uid)
        if pred_df.empty:
            st.info(f"No predictions saved yet for User ID **{uid}**. Go to **Predict Calorie** and make some predictions.")
        else:
            pred_df["Timestamp"] = pd.to_datetime(pred_df["Timestamp"])
            pred_df = pred_df.sort_values("Timestamp").reset_index(drop=True)
            n = len(pred_df)
            avg_pred = pred_df["Predicted_Calories"].mean()

            # KPI cards for predictions
            p1, p2, p3, p4 = st.columns(4)
            for col, lbl, val, unit in [
                (p1, "Predictions",   str(n),                        "total"),
                (p2, "Avg Predicted", f"{avg_pred:,.0f}",            "kcal"),
                (p3, "Peak Predicted",f"{pred_df['Predicted_Calories'].max():,.0f}", "kcal"),
                (p4, "Avg Steps",     f"{pred_df['TotalSteps'].mean():,.0f}", "steps"),
            ]:
                col.markdown(f"""
                <div class='card'>
                    <div class='card-label'>{lbl}</div>
                    <div class='card-value'>{val}</div>
                    <div class='card-unit'>{unit}</div>
                </div>""", unsafe_allow_html=True)

            # Predicted calorie trend
            st.markdown("#### 🔴 Predicted Calorie Trend")
            fig_p, ax_p = plt.subplots(figsize=(10, 3.5))
            ax_p.fill_between(pred_df["Timestamp"], pred_df["Predicted_Calories"], alpha=0.1, color=CORAL)
            ax_p.plot(pred_df["Timestamp"], pred_df["Predicted_Calories"], marker="o", color=CORAL,
                      linewidth=2, markersize=5, label="Predicted")
            ax_p.axhline(avg_pred, color=CORAL, linestyle="--", linewidth=1, alpha=0.5, label=f"Avg {avg_pred:.0f} kcal")
            ax_p.set_ylabel("Calories (kcal)")
            ax_p.legend(frameon=False)
            plt.xticks(rotation=35, ha="right", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_p)

            # Distribution histogram
            st.markdown("#### 📊 Distribution of Predicted Calories")
            fig_h, ax_h = plt.subplots(figsize=(10, 3))
            sns.histplot(pred_df["Predicted_Calories"], bins=max(5, n // 2), kde=True,
                         color=CORAL, ax=ax_h, edgecolor="white", linewidth=0.5)
            ax_h.set_xlabel("Calories (kcal)")
            ax_h.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig_h)

            # Raw prediction table
            with st.expander("📄 View raw prediction history"):
                show_cols = ["Timestamp", "Predicted_Calories", "TotalSteps",
                             "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes"]
                st.dataframe(pred_df[show_cols].sort_values("Timestamp", ascending=False).reset_index(drop=True),
                             use_container_width=True)

# ═════════════════════════════════════════════
# PAGE 3 — MODEL DETAILS
# ═════════════════════════════════════════════
elif page == "📋  Model Details":
    st.markdown("<div class='page-header'>📋 Model Details</div>", unsafe_allow_html=True)
    st.markdown("<div class='page-sub'>Performance comparison across all trained models.</div>", unsafe_allow_html=True)

    # ── Metrics table ──
    comparison_df = pd.DataFrame({
        "Model":       ["Decision Tree", "Linear Regression", "Random Forest",
                        "Gradient Boosting", "XGBoost", "Neural Network",
                        "Ensemble (XGB+NN)", "TabM", "CatBoost 🏆"],
        "R²":          [0.580, 0.710, 0.810, 0.880, 0.893, 0.761, 0.838, 0.767, 0.905],
        "MAE (kcal)":  [248,   289,   211,   174,   145,   232,   199,   224,   139],
        "RMSE (kcal)": [445,   369,   300,   235,   224,   336,   277,   332,   212],
    })

    st.markdown("#### 📊 Model Comparison Table")

    # Highlight best row
    def highlight_best(row):
        return ["background:#fff0f0;font-weight:700;color:#ff5c5c" if row["Model"] == "CatBoost 🏆"
                else "" for _ in row]

    st.dataframe(
        comparison_df.style.apply(highlight_best, axis=1).format({"R²": "{:.3f}"}),
        use_container_width=True, hide_index=True,
    )

    st.success("🏆 **CatBoost** achieved the highest R² (0.905) and lowest error (MAE = 139 kcal) on the held-out test set.")

    # ── R² bar chart ──
    set_plot_style()
    st.markdown("#### 📈 R² Score Comparison")
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#ff5c5c" if m == "CatBoost 🏆" else "#c5cfe8" for m in comparison_df["Model"]]
    bars = ax.barh(comparison_df["Model"], comparison_df["R²"], color=colors, height=0.55, edgecolor="white")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("R² Score")
    ax.axvline(0.9, color="#ff5c5c", linestyle="--", linewidth=1, alpha=0.4)
    for bar, val in zip(bars, comparison_df["R²"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
                va="center", fontsize=9, color="#374151")
    plt.tight_layout()
    st.pyplot(fig)

    # ── Evaluation graphs ──
    st.markdown("---")
    st.markdown("#### 🖼️ Evaluation Graphs")

    # ── UPDATE THESE PATHS ──
    graph_paths = {
        "Actual vs Predicted":         "actual_vs_predicted_nn.png",
        "Feature Importance":          "catboost_feature_importance.png",
        "Model Comparison Chart":      "model_comparison.png",
        "Ensemble Comparison":         "ensemble_comparison.png",
        "Training History":            "training_history.png",
    }

    cols = st.columns(2, gap="large")
    for i, (title, path) in enumerate(graph_paths.items()):
        with cols[i % 2]:
            if os.path.exists(path):
                st.markdown(f"**{title}**")
                st.image(path, use_container_width=True)
            else:
                st.markdown(f"""
                <div class='card' style='text-align:center;color:#9ca3af;min-height:140px;display:flex;align-items:center;justify-content:center;flex-direction:column;'>
                    <div style='font-size:2rem;'>🖼️</div>
                    <div style='font-size:.85rem;margin-top:6px;'><b>{title}</b></div>
                    <div style='font-size:.75rem;color:#d1d5db;margin-top:2px;'>{path}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("All models trained on the same 80/20 train-test split of Fitbit daily activity data.")