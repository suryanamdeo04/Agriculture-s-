import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌿",
    layout="wide"
)

with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

st.markdown("""
<style>
.stApp {
    background: #081b13;
    color: #f3fff5;
}
[data-testid="stSidebar"] {
    background: #102b1e;
}
h1, h2, h3, p, label, span {
    color: #f3fff5;
}
.hero {
    background: linear-gradient(120deg,#174c32,#287c49);
    padding: 35px;
    border-radius: 22px;
    border: 1px solid #4c9b65;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 42px;
    font-weight: 800;
}
.card {
    background: #153a28;
    padding: 23px;
    border-radius: 18px;
    border: 1px solid #376d4b;
    text-align: center;
}
.card h2 {
    color: #a7ed76;
    font-size: 30px;
}
.stButton button {
    background: #a5e86f;
    color: #102719;
    border-radius: 12px;
    height: 52px;
    font-weight: 800;
    border: none;
}
.stButton button:hover {
    background: #c5f59b;
    color: #102719;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("# 🌿 AgriVision AI")
    st.caption("SMART AGRICULTURE PLATFORM")
    st.divider()
    st.markdown("### Dashboard")
    st.success("Crop Recommendation")
    st.markdown("### Model Details")
st.divider()

st.metric("🎯 Test Accuracy", "99.55%")
st.metric("🧪 Test Samples", "440")
st.markdown("## 🧠 Model Performance & Evaluation")

m1, m2 = st.columns(2)

with m1:
    st.metric("Test Accuracy", "99.55%")

with m2:
    st.metric("Test Samples", "440")

st.markdown("### 📊 Confusion Matrix")

st.image(
    "confusion_matrix.png",
    use_container_width=True
)

st.markdown("### 📋 Classification Report")

report = pd.read_csv(
    "classification_report.csv",
    index_col=0
)

st.dataframe(
    report.round(3),
    use_container_width=True
)

st.markdown("""
<div class="hero">
    <h1>🌱 AgriVision AI</h1>
    <h3>Smart Agriculture. Smarter Decisions.</h3>
    <p>AI-powered crop recommendations using soil and climate conditions.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("## 📊 Farm Intelligence Overview")

a, b, c = st.columns(3)

with a:
    st.markdown("""
    <div class="card">
    <h2>22</h2>
    <p>Crop Categories</p>
    </div>
    """, unsafe_allow_html=True)

with b:
    st.markdown("""
    <div class="card">
    <h2>99.55%</h2>
    <p>Test Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with c:
    st.markdown("""
    <div class="card">
    <h2>AI / ML</h2>
    <p>Random Forest Model</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown("## 🧪 Soil & Climate Analysis")
st.write("Enter your farm measurements to generate an AI-based crop recommendation.")

left, right = st.columns(2)

with left:
    N = st.number_input("🌿 Nitrogen (N)", 0, 200, 90)
    P = st.number_input("🌱 Phosphorus (P)", 0, 200, 42)
    K = st.number_input("🥬 Potassium (K)", 0, 250, 43)
    temperature = st.number_input("🌡️ Temperature (°C)", 0.0, 60.0, 20.8)

with right:
    humidity = st.number_input("💧 Humidity (%)", 0.0, 100.0, 82.0)
    ph = st.number_input("🧫 Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.number_input("🌧️ Rainfall (mm)", 0.0, 500.0, 202.0)

if st.button("🌾 ANALYZE & RECOMMEND CROP", use_container_width=True):

    inputs = pd.DataFrame([[
        N, P, K, temperature, humidity, ph, rainfall
    ]], columns=[
        "N", "P", "K", "temperature",
        "humidity", "ph", "rainfall"
    ])

    prediction = model.predict(inputs)[0]
    probabilities = model.predict_proba(inputs)[0]
    confidence = max(probabilities) * 100

    st.divider()
    st.markdown("## 🌟 AI Recommendation")

    st.markdown(f"""
    <div class="hero">
        <p>RECOMMENDED CROP</p>
        <h1>🌾 {prediction.upper()}</h1>
        <h3>Model confidence: {confidence:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)

    chart = pd.DataFrame({
        "Crop": model.classes_,
        "Probability": probabilities * 100
    }).sort_values("Probability", ascending=False).head(5)

   
    st.markdown("### 📈 Top 5 Crop Predictions")

    st.markdown("""
<style>
.crop-card {
    background: linear-gradient(120deg, #153a28, #204d32);
    border: 1px solid #4c9b65;
    border-radius: 14px;
    padding: 18px 22px;
    margin: 12px 0;
}
.crop-title {
    color: #ffffff !important;
    font-size: 23px !important;
    font-weight: 900 !important;
    text-transform: uppercase;
}
.crop-percent {
    color: #b8f28d !important;
    font-size: 21px !important;
    font-weight: 800 !important;
}
.crop-track {
    background: #355b43;
    height: 13px;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 12px;
}
.crop-fill {
    background: linear-gradient(90deg, #8bdc55, #d0ff91);
    height: 13px;
    border-radius: 20px;
}
</style>
""", unsafe_allow_html=True)

    for rank, row in enumerate(chart.itertuples(index=False), start=1):
        crop_name = str(row.Crop)
        probability = float(row.Probability)
        bar_width = max(0, min(probability, 100))

        st.markdown(f"""
    <div class="crop-card">
        <div style="display:flex; justify-content:space-between;
                    align-items:center; gap:10px;">
            <div class="crop-title">
                #{rank} &nbsp; {crop_name}
            </div>
            <div class="crop-percent">
                {probability:.2f}%
            </div>
        </div>
        <div class="crop-track">
            <div class="crop-fill" style="width:{bar_width}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "This is a model-based recommendation, not a yield guarantee. "
        "Confirm local soil and weather conditions before planting."
    )

    report = pd.DataFrame([{
        "Nitrogen": N,
        "Phosphorus": P,
        "Potassium": K,
        "Temperature": temperature,
        "Humidity": humidity,
        "Soil pH": ph,
        "Rainfall": rainfall,
        "Recommended Crop": prediction,
        "Model Confidence (%)": round(confidence, 2)
    }])

    csv_data = report.to_csv(index=False)

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv_data,
        file_name="AgriVision_Crop_Report.csv",
        mime="text/csv",
        use_container_width=True,
        key="agrivision_final_report_download"
    )

st.divider()
st.caption("© 2026 AgriVision AI | Built with Python, Scikit-learn & Streamlit")

st.markdown("""
<div style="
position: fixed;
bottom: 12px;
right: 22px;
background: #153a28;
color: #a5e86f;
padding: 10px 18px;
border: 1px solid #4c9b65;
border-radius: 12px;
font-size: 13px;
z-index: 999999;
">
<strong>Project By: Surya Namdeo & Mansi Tiwari</strong>
</div>
""", unsafe_allow_html=True)



st.markdown("---")
st.header("🌱 About AgriVision AI")

st.markdown("""
AgriVision AI is a machine-learning-based crop recommendation
system that helps farmers identify suitable crops using soil
and environmental conditions.

### Project Highlights

- **Dataset:** 2,200 records
- **Crop Categories:** 22
- **Algorithm:** Random Forest Classifier
- **Model Accuracy:** 99.55% on the test dataset
- **Input Parameters:** Nitrogen, Phosphorus, Potassium,
  Temperature, Humidity, pH and Rainfall

### Project Team
**Surya Namdeo & Mansi Tiwari**
""")


# DATASET_INSIGHTS_SECTION
st.markdown("---")
st.header("📊 Dataset Insights")
st.write("Explore the actual crop recommendation training dataset.")

dataset = pd.read_csv("data/crop_recommendation.csv")

d1, d2, d3 = st.columns(3)

with d1:
    st.metric("Total Records", f"{len(dataset):,}")

with d2:
    st.metric("Crop Categories", dataset["label"].nunique())

with d3:
    st.metric("Input Features", len(dataset.columns) - 1)

st.subheader("🌾 Crop-wise Dataset Distribution")
crop_counts = dataset["label"].value_counts().sort_values(ascending=False)
st.bar_chart(crop_counts)

st.subheader("🧪 Average Soil Nutrients by Crop")
nutrient_means = dataset.groupby("label")[["N", "P", "K"]].mean()
st.bar_chart(nutrient_means)

st.subheader("🌦️ Average Climate Conditions")
climate_means = dataset.groupby("label")[
    ["temperature", "humidity", "rainfall"]
].mean()
st.bar_chart(climate_means)

with st.expander("View Dataset Sample"):
    st.dataframe(dataset.head(10), use_container_width=True)

