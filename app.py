
import streamlit as st
import pandas as pd
import pickle
import os
import html
import numpy as np
import streamlit.components.v1 as components

# =========================================================
# AGRIVISION AI — PREMIUM ANIMATED AGRICULTURE DASHBOARD
# Existing ML model and dataset are preserved.
# =========================================================

st.set_page_config(
    page_title="AgriVision AI | Smart Agriculture",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- THEME --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@700;800&display=swap');

:root {
    --forest:#102b20;
    --green:#176b45;
    --emerald:#28a66b;
    --gold:#d8b45c;
    --cream:#f5f7ef;
    --ink:#18372a;
}

.stApp {
    background:linear-gradient(135deg,#f5f7ef 0%,#e8f1e7 55%,#f8f5e9 100%);
    color:var(--ink);
    font-family:'DM Sans',sans-serif;
}

[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#102b20,#174b35);
}
[data-testid="stSidebar"] * {color:#f4f7ef !important;}
[data-testid="stSidebar"] hr {border-color:#ffffff30;}

.block-container {
    padding-top:1.5rem;
    padding-bottom:3rem;
    max-width:1450px;
}

.hero {
    position:relative;
    overflow:hidden;
    padding:48px 25px;
    border-radius:28px;
    text-align:center;
    color:white;
    background:
      linear-gradient(110deg,rgba(9,37,25,.96),rgba(20,94,57,.88),rgba(115,91,36,.78)),
      url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=85') center/cover;
    box-shadow:0 18px 45px #102b2030;
    animation:heroIn .9s ease both;
}
.hero:after {
    content:"";
    position:absolute;
    width:300px;height:300px;
    border:1px solid #ffffff30;
    border-radius:50%;
    top:-180px;right:-70px;
    box-shadow:0 0 0 30px #ffffff0c,0 0 0 65px #ffffff08;
}
.hero h1 {
    font-family:'Playfair Display',serif;
    font-size:clamp(38px,6vw,66px);
    margin:0;
    font-weight:800;
    background:linear-gradient(90deg,#fff,#f7d98b,#fff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.hero p {
    margin:12px 0 0;
    color:#e8f3e8;
    font-size:17px;
    letter-spacing:1.3px;
}
.hero-tag {
    display:inline-block;
    margin-top:20px;
    padding:8px 17px;
    border:1px solid #ffffff55;
    border-radius:30px;
    color:#f5d88b;
    font-size:11px;
    font-weight:800;
    letter-spacing:2px;
}

.section-head {
    margin:28px 0 16px;
    padding:0 0 12px;
    border-bottom:2px solid #d6e5d6;
    animation:rise .7s ease both;
}
.section-head h2 {
    font-family:'Playfair Display',serif;
    color:#143c2b;
    font-size:29px;
    margin:0;
}
.section-head p {
    color:#668071;
    font-size:14px;
    margin:5px 0 0;
}

.kpi {
    background:linear-gradient(145deg,#fff,#f4f8f1);
    border:1px solid #dce9dc;
    border-radius:19px;
    padding:21px;
    min-height:125px;
    box-shadow:0 8px 22px #18372a0c;
    animation:rise .7s ease both;
    transition:all .3s ease;
}
.kpi:hover {
    transform:translateY(-7px);
    box-shadow:0 15px 30px #176b4525;
    border-color:#8bc9a1;
}
.kpi-label {
    color:#71877a;
    font-size:11px;
    font-weight:800;
    letter-spacing:1.3px;
    text-transform:uppercase;
}
.kpi-value {
    color:#176b45;
    font-size:31px;
    font-weight:800;
    margin-top:9px;
}
.kpi-note {color:#87978d;font-size:12px;margin-top:3px;}

.panel {
    background:rgba(255,255,255,.88);
    border:1px solid #dfe9dc;
    border-radius:20px;
    padding:20px;
    box-shadow:0 8px 24px #18372a0b;
    margin-bottom:15px;
    animation:rise .7s ease both;
}
.panel h3 {
    color:#174b35;
    font-size:17px;
    font-weight:800;
    margin:0 0 12px;
}
.result {
    background:linear-gradient(120deg,#102b20,#176b45 65%,#947334);
    border-radius:24px;
    padding:28px;
    text-align:center;
    color:white;
    box-shadow:0 15px 32px #102b2030;
    animation:rise .7s ease both;
}
.result .eyebrow {
    color:#e5d39c;
    font-size:12px;
    font-weight:800;
    letter-spacing:2px;
}
.result h2 {
    color:#f6d98d;
    font-family:'Playfair Display',serif;
    font-size:40px;
    margin:8px 0;
}
.result p {color:#e8f2e8;margin:0;}

div.stButton > button {
    background:linear-gradient(100deg,#176b45,#28a66b);
    color:white;
    border:0;
    border-radius:13px;
    min-height:48px;
    font-weight:800;
    letter-spacing:.3px;
    transition:.25s;
}
div.stButton > button:hover {
    transform:translateY(-3px);
    box-shadow:0 9px 22px #176b4535;
    color:white;
}
div[data-testid="stMetric"] {
    background:white;
    padding:15px;
    border-radius:14px;
    border:1px solid #e0eadd;
}
div[data-testid="stDataFrame"] {
    border:1px solid #dce8dc;
    border-radius:14px;
    overflow:hidden;
}
.stProgress > div > div > div > div {
    background:linear-gradient(90deg,#176b45,#d8b45c);
}
div[data-testid="stAlert"] {border-radius:13px;}

@keyframes heroIn {
    from {opacity:0;transform:translateY(18px) scale(.985);}
    to {opacity:1;transform:translateY(0) scale(1);}
}
@keyframes rise {
    from {opacity:0;transform:translateY(18px);}
    to {opacity:1;transform:translateY(0);}
}
#MainMenu,footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD FILES --------------------
@st.cache_resource
def load_model():
    with open("models/crop_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_dataset():
    path = "data/crop_recommendation_10k.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.read_csv("data/crop_recommendation.csv")

model = load_model()
df = load_dataset()

# -------------------- HELPERS --------------------
def section(title, subtitle):
    st.markdown(
        f'<div class="section-head"><h2>{title}</h2><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )

def kpi(label, value, note, emoji):
    st.markdown(f"""
    <div class="kpi">
      <div class="kpi-label">{emoji} &nbsp; {label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)

def animated_bars(labels, values, colors=None, suffix=""):
    colors = colors or ["#28a66b","#d8b45c","#4d91bd","#9a78c5","#e28a55"]
    maximum = max(values) if values and max(values) > 0 else 1
    rows = ""
    for i, (label, value) in enumerate(zip(labels, values)):
        width = max(1, float(value) / maximum * 100)
        color = colors[i % len(colors)]
        rows += f"""
        <div style="margin:15px 0;">
          <div style="display:flex;justify-content:space-between;gap:10px;
                      color:#315642;font-size:13px;font-weight:700;">
            <span>{html.escape(str(label))}</span>
            <span>{value:,.2f}{suffix}</span>
          </div>
          <div style="height:12px;background:#e9f0e7;border-radius:20px;
                      overflow:hidden;margin-top:7px;">
            <div style="height:100%;width:{width}%;background:{color};
                        border-radius:20px;animation:grow 1.2s ease both;">
            </div>
          </div>
        </div>
        """
    components.html(f"""
    <html><body style="margin:0;background:transparent;font-family:Arial;">
    <div style="padding:5px 3px;">{rows}</div>
    <style>
    @keyframes grow {{from{{width:0;}} to{{width:var(--w);}}}}
    </style>
    </body></html>
    """, height=max(130, len(labels)*62+15))


def circular_card(name, probability, color, index):
    from string import Template

    pct = float(probability) * 100
    radius = 43
    circumference = 2 * np.pi * radius
    offset = circumference * (1 - pct / 100)

    safe_name = html.escape(str(name).title())
    safe_color = html.escape(str(color), quote=True)

    template = Template("""
    <html>
    <head>
    <style>
      body { margin:0; background:transparent; font-family:Arial; }
      .crop-card {
        text-align:center; padding:12px 4px 15px;
        background:linear-gradient(145deg,#fff,#f4f8f1);
        border:1px solid #dfebdd; border-radius:18px;
        box-shadow:0 7px 20px rgba(16,43,32,.08);
        transition:transform .3s,box-shadow .3s;
      }
      .crop-card:hover {
        transform:translateY(-7px) scale(1.025);
        box-shadow:0 13px 25px rgba(23,107,69,.2);
      }
      .ring-wrap { display:flex; justify-content:center; }
      .progress-ring {
        transform:rotate(-90deg); transform-origin:60px 60px;
        animation:ringReveal 1.5s ease forwards;
      }
      @keyframes ringReveal {
        from { stroke-dashoffset:${circumference}; }
        to { stroke-dashoffset:${offset}; }
      }
      .crop-name {
        color:#174b35; font-size:15px; font-weight:800; margin-top:3px;
      }
      .crop-rank {
        display:inline-block; margin-top:7px; padding:4px 10px;
        border-radius:20px; background:#e8f3e9; color:#26734b;
        font-size:9px; font-weight:800; letter-spacing:1px;
      }
    </style>
    </head>
    <body>
      <div class="crop-card">
        <div class="ring-wrap">
          <svg width="132" height="132" viewBox="0 0 120 120">
            <circle cx="60" cy="60" r="${radius}" fill="none"
              stroke="#e8eee5" stroke-width="10"/>
            <circle class="progress-ring" cx="60" cy="60" r="${radius}"
              fill="none" stroke="${color}" stroke-width="10"
              stroke-linecap="round"
              stroke-dasharray="${circumference}"
              stroke-dashoffset="${circumference}"/>
            <text x="60" y="57" text-anchor="middle"
              font-size="19" font-weight="800" fill="#174b35">${pct}%</text>
            <text x="60" y="73" text-anchor="middle"
              font-size="8" fill="#789080">PROBABILITY</text>
          </svg>
        </div>
        <div class="crop-name">${name}</div>
        <div class="crop-rank">TOP ${index}</div>
      </div>
    </body>
    </html>
    """)

    rendered = template.substitute(
        radius=radius,
        circumference=f"{circumference:.3f}",
        offset=f"{offset:.3f}",
        color=safe_color,
        pct=f"{pct:.1f}",
        name=safe_name,
        index=int(index)
    )

    components.html(rendered, height=195, scrolling=False)
def donut_chart(labels, values, colors):
    total = sum(values) if sum(values) else 1
    stops = []
    current = 0
    for value, color in zip(values, colors):
        end = current + (value / total * 100)
        stops.append(f"{color} {current:.2f}% {end:.2f}%")
        current = end
    gradient = ", ".join(stops)
    legend = ""
    for label, value, color in zip(labels, values, colors):
        legend += f"""
        <div style="display:flex;align-items:center;gap:8px;margin:8px 0;
                    font-size:12px;color:#315642;">
          <span style="width:10px;height:10px;border-radius:3px;
                       background:{color};display:inline-block;"></span>
          <span style="flex:1;">{html.escape(str(label))}</span>
          <b>{value:,.0f}</b>
        </div>"""
    components.html(f"""
    <html><body style="margin:0;background:transparent;font-family:Arial;">
      <div style="display:flex;align-items:center;justify-content:center;
                  gap:24px;flex-wrap:wrap;padding:12px;">
        <div style="width:150px;height:150px;border-radius:50%;
          background:conic-gradient({gradient});
          display:flex;align-items:center;justify-content:center;
          animation:spinIn 1.2s ease both;box-shadow:0 6px 20px #18372a18;">
          <div style="width:94px;height:94px;border-radius:50%;background:#fff;
            display:flex;flex-direction:column;align-items:center;justify-content:center;">
            <b style="font-size:22px;color:#176b45;">{sum(values):,.0f}</b>
            <span style="font-size:9px;color:#789080;">TOTAL</span>
          </div>
        </div>
        <div style="min-width:145px;">{legend}</div>
      </div>
      <style>
      @keyframes spinIn {{from{{transform:rotate(-100deg) scale(.7);opacity:.3;}}
      to{{transform:rotate(0) scale(1);opacity:1;}}}}
      </style>
    </body></html>
    """, height=210)

# -------------------- HERO --------------------
st.markdown("""
<div class="hero">
  <h1>🌿 AgriVision AI</h1>
  <p>Smart Agriculture • Intelligent Crop Insights</p>
  <div class="hero-tag">DATA-DRIVEN FARMING · SMARTER DECISIONS</div>
</div>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.markdown("# 🌿 AgriVision AI")
st.sidebar.caption("SMART AGRICULTURE INTELLIGENCE")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "EXPLORE PLATFORM",
    ["Dashboard", "Crop Recommendation", "Dataset Insights",
     "Model Performance", "About Project"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**PROJECT TEAM**")
st.sidebar.write("Surya Namdeo")
st.sidebar.write("Mansi Tiwari")
st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning · Random Forest")

# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":

    section("🌾 Farm Intelligence Overview",
            "A live overview of the agricultural dataset and model.")

    total_crops = int(df["label"].nunique()) if "label" in df.columns else 0
    feature_count = len([c for c in df.columns if c != "label"])

    cols = st.columns(4)
    with cols[0]:
        kpi("Crop Categories", total_crops, "Unique crop classes", "🌱")
    with cols[1]:
        kpi("Dataset Records", f"{len(df):,}", "Available observations", "📊")
    with cols[2]:
        kpi("ML Algorithm", "Random Forest", "Trained classifier", "🤖")
    with cols[3]:
        kpi("Input Features", feature_count, "Soil and climate inputs", "🧪")

    section("📈 Agricultural Analytics",
            "Explore crop distribution and nutrient patterns in the dataset.")

    left, right = st.columns(2)

    with left:
        st.markdown('<div class="panel"><h3>🌿 Crop Distribution</h3>',
                    unsafe_allow_html=True)
        if "label" in df.columns:
            counts = df["label"].value_counts().head(6)
            donut_chart(
                counts.index.tolist(),
                counts.values.tolist(),
                ["#176b45","#d8b45c","#4d91bd","#9a78c5","#e28a55","#60a99a"]
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><h3>🧪 Average Soil Nutrients</h3>',
                    unsafe_allow_html=True)
        nutrient_cols = [c for c in ["N", "P", "K"] if c in df.columns]
        if nutrient_cols:
            means = df[nutrient_cols].mean()
            animated_bars(means.index.tolist(), means.values.tolist(),
                          ["#28a66b","#d8b45c","#4d91bd"])
        st.markdown("</div>", unsafe_allow_html=True)

    section("🌤️ Climate Intelligence",
            "Summary of the environmental variables available in your dataset.")

    climate_cols = [c for c in
                    ["temperature", "humidity", "ph", "rainfall"]
                    if c in df.columns]

    if climate_cols:
        c1, c2 = st.columns(2)
        summary = df[climate_cols].mean()
        with c1:
            st.markdown('<div class="panel"><h3>Climate Averages</h3>',
                        unsafe_allow_html=True)
            animated_bars(summary.index.tolist(), summary.values.tolist(),
                          ["#4d91bd","#28a66b","#d8b45c","#9a78c5"])
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="panel"><h3>Climate Statistics</h3>',
                        unsafe_allow_html=True)
            st.dataframe(df[climate_cols].describe().round(2),
                         use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.info("Open **Crop Recommendation** to analyse soil and weather inputs.")

# =========================================================
# CROP RECOMMENDATION
# =========================================================
elif page == "Crop Recommendation":

    section("🌱 Smart Crop Recommendation",
            "Enter your soil and climate values to receive an ML-based crop recommendation.")

    st.markdown('<div class="panel"><h3>🧪 Soil Parameters</h3>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        N = st.number_input("Nitrogen (N)", 0.0, 200.0, 90.0)
    with c2:
        P = st.number_input("Phosphorus (P)", 0.0, 200.0, 42.0)
    with c3:
        K = st.number_input("Potassium (K)", 0.0, 250.0, 43.0)
    with c4:
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, step=0.1)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><h3>🌤️ Weather Parameters</h3>',
                unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3)
    with w1:
        temperature = st.number_input("Temperature (°C)", -10.0, 60.0, 25.0)
    with w2:
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
    with w3:
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)
    st.markdown("</div>", unsafe_allow_html=True)

    analyse = st.button("🌾 ANALYZE & RECOMMEND CROP",
                        use_container_width=True)

    if analyse:
        try:
            input_df = pd.DataFrame([{
                "N": N, "P": P, "K": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph, "rainfall": rainfall
            }])

            # Preserve the trained model's expected feature order.
            if hasattr(model, "feature_names_in_"):
                input_df = input_df[list(model.feature_names_in_)]

            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            classes = model.classes_

            ranked = sorted(zip(classes, probabilities),
                            key=lambda item: item[1], reverse=True)[:5]
            confidence = float(max(probabilities) * 100)

            st.markdown("---")
            section("✨ Your AI Recommendation",
                    "Prediction generated by your existing trained Random Forest model.")

            st.markdown(f"""
            <div class="result">
              <div class="eyebrow">RECOMMENDED CROP</div>
              <h2>🌾 {html.escape(str(prediction).title())}</h2>
              <p>Highest model class probability: {confidence:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            section("🎯 Top 5 Crop Probability Insights",
                    "Each circular indicator represents the model probability for one crop.")

            colors = ["#28a66b", "#d8b45c", "#4d91bd",
                      "#9a78c5", "#e28a55"]
            ring_cols = st.columns(5)

            for i, (crop, probability) in enumerate(ranked):
                with ring_cols[i]:
                    circular_card(crop, probability, colors[i], i + 1)

            st.caption(
                "Important: These are Random Forest class probabilities. "
                "They are not calibrated certainty, crop-yield estimates, "
                "or guarantees of successful cultivation."
            )

            section("📊 Crop Probability Comparison",
                    "Compare the model's top five crop probabilities.")

            animated_bars(
                [str(c).title() for c, p in ranked],
                [float(p) * 100 for c, p in ranked],
                colors, "%"
            )

            section("🧪 Soil & Weather Input Summary",
                    "Values submitted for this prediction.")

            input_summary = pd.DataFrame({
                "Parameter": ["Nitrogen", "Phosphorus", "Potassium",
                              "Temperature", "Humidity", "Soil pH", "Rainfall"],
                "Value": [N, P, K, temperature, humidity, ph, rainfall]
            })
            st.dataframe(input_summary, use_container_width=True,
                         hide_index=True)

            st.download_button(
                "⬇️ Download Prediction Report",
                input_summary.to_csv(index=False),
                file_name="agrivision_prediction.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.exception(e)

# =========================================================
# DATASET INSIGHTS
# =========================================================
elif page == "Dataset Insights":

    section("📊 Dataset Intelligence",
            "Explore the records, crop categories and feature values.")

    a, b, c = st.columns(3)
    a.metric("Total Records", f"{len(df):,}")
    b.metric("Crop Categories",
             int(df["label"].nunique()) if "label" in df.columns else 0)
    c.metric("Total Columns", len(df.columns))

    if "label" in df.columns:
        section("🌿 Crop Category Distribution",
                "Distribution of crop labels in the available dataset.")

        counts = df["label"].value_counts()
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="panel"><h3>Crop Distribution — Donut</h3>',
                        unsafe_allow_html=True)
            top_counts = counts.head(8)
            donut_chart(
                top_counts.index.tolist(), top_counts.values.tolist(),
                ["#176b45","#d8b45c","#4d91bd","#9a78c5",
                 "#e28a55","#60a99a","#c46d83","#819b49"]
            )
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown('<div class="panel"><h3>Top Crop Categories</h3>',
                        unsafe_allow_html=True)
            animated_bars(counts.head(8).index.tolist(),
                          counts.head(8).values.tolist())
            st.markdown("</div>", unsafe_allow_html=True)

    section("🧾 Dataset Preview",
            "A sample of the actual records used by the project.")
    st.dataframe(df.head(30), use_container_width=True)

    st.download_button(
        "⬇️ Download Dataset CSV",
        df.to_csv(index=False),
        file_name="agrivision_dataset.csv",
        mime="text/csv"
    )

# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "Model Performance":

    section("🤖 Machine Learning Performance",
            "Review saved evaluation outputs from the existing model.")

    report_path = "models/classification_report.csv"
    matrix_path = "models/confusion_matrix.png"

    if os.path.exists(report_path):
        st.markdown('<div class="panel"><h3>📋 Classification Report</h3>',
                    unsafe_allow_html=True)
        report = pd.read_csv(report_path)
        st.dataframe(report, use_container_width=True)
        st.download_button(
            "⬇️ Download Classification Report",
            report.to_csv(index=False),
            file_name="classification_report.csv",
            mime="text/csv"
        )
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Classification report file was not found.")

    if os.path.exists(matrix_path):
        st.markdown('<div class="panel"><h3>🎯 Confusion Matrix</h3>',
                    unsafe_allow_html=True)
        st.image(matrix_path, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Confusion matrix image was not found.")

# =========================================================
# ABOUT PROJECT
# =========================================================
elif page == "About Project":

    section("🌿 About AgriVision AI",
            "A machine-learning prototype for intelligent crop recommendation.")

    st.markdown("""
    <div class="result">
      <div class="eyebrow">SMART AGRICULTURE PLATFORM</div>
      <h2>AgriVision AI</h2>
      <p>Turning soil and climate data into intelligent crop insights.</p>
    </div>
    """, unsafe_allow_html=True)

    section("⚙️ Technology Stack",
            "Tools and technologies used to build this prototype.")

    tech = st.columns(4)
    tech_items = [
        ("🐍", "Python", "Core programming"),
        ("🤖", "Scikit-learn", "Machine learning"),
        ("📊", "Pandas", "Dataset processing"),
        ("🌐", "Streamlit", "Interactive dashboard")
    ]
    for col, item in zip(tech, tech_items):
        with col:
            st.markdown(f"""
            <div class="kpi">
              <div style="font-size:27px;">{item[0]}</div>
              <div class="kpi-value" style="font-size:19px;">{item[1]}</div>
              <div class="kpi-note">{item[2]}</div>
            </div>
            """, unsafe_allow_html=True)

    section("🧪 Model Input Features",
            "Seven soil and climate parameters are used for prediction.")

    st.markdown("""
    <div class="panel">
      <h3>Soil Parameters</h3>
      <p>• Nitrogen (N) &nbsp; • Phosphorus (P) &nbsp; • Potassium (K)</p>
      <p>• Soil pH</p>
      <h3>Weather Parameters</h3>
      <p>• Temperature &nbsp; • Humidity &nbsp; • Rainfall</p>
    </div>
    """, unsafe_allow_html=True)

    section("👨‍💻 Project Team", "Project contributors.")
    st.markdown("""
    <div class="panel">
      <h3>Surya Namdeo</h3>
      <p>Project team member</p>
      <hr>
      <h3>Mansi Tiwari</h3>
      <p>Project team member</p>
    </div>
    """, unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:12px;color:#71877a;">
  <b style="color:#176b45;">🌿 AgriVision AI</b><br>
  Smart Agriculture • Intelligent Crop Insights<br>
  <small>Machine Learning Prototype</small>
</div>
""", unsafe_allow_html=True)
