
import streamlit as st
import pandas as pd
import pickle
import os
import html
import numpy as np
import streamlit.components.v1 as components
from crop_profiles import CROP_PROFILES, GENERAL_NOTE

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


/* AGRIVISION FINAL VISUAL POLISH */

.stApp,
.stApp [data-testid="stMarkdownContainer"],
.stApp input,
.stApp textarea,
.stApp button,
.stApp label {
    font-family: 'DM Sans', sans-serif !important;
}

h1, h2, h3, h4,
.section-head h2,
.panel h3 {
    font-family: 'Playfair Display', serif !important;
}

.section-head {
    position: relative;
    margin-top: 34px;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 1px solid #d5e4d4;
}

.section-head h2 {
    font-size: clamp(23px, 2.5vw, 30px);
    line-height: 1.3;
    letter-spacing: -.4px;
}

.section-head h2::after {
    content: "";
    display: block;
    width: 58px;
    height: 3px;
    margin-top: 9px;
    border-radius: 10px;
    background: linear-gradient(90deg, #28a66b, #d8b45c);
}

.section-head p {
    line-height: 1.7;
    color: #668071;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #dce9dc !important;
    border-radius: 19px !important;
    background: rgba(255,255,255,.68) !important;
    box-shadow: 0 7px 22px rgba(16,43,32,.055);
    transition: border-color .25s ease, box-shadow .25s ease;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #a8d2b1 !important;
    box-shadow: 0 11px 27px rgba(23,107,69,.10);
}

div[data-testid="stExpander"] {
    border: 1px solid #dce9dc !important;
    border-radius: 16px !important;
    background: rgba(255,255,255,.78);
    overflow: hidden;
}

div[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    color: #174b35 !important;
}

div[data-testid="stMetric"] {
    border-radius: 17px !important;
    box-shadow: 0 5px 17px rgba(16,43,32,.055);
}

div.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: .35px;
    border-radius: 14px !important;
    transition: all .25s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 9px 22px rgba(23,107,69,.20);
}

.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #176b45, #28a66b, #d8b45c);
    border-radius: 20px;
}

[data-testid="stMarkdownContainer"] p {
    line-height: 1.75;
}

[data-testid="stAlert"] {
    border-radius: 14px !important;
}


/* AGRIVISION PROFESSIONAL DASHBOARD V1 */

.stApp {
    background: #071b14 !important;
    color: #e8f2e9 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"] {
    background: rgba(7,27,20,.94) !important;
}

.block-container {
    max-width: 1600px !important;
    padding: 1.2rem 2rem 3rem !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#071b14 0%,#103b29 100%) !important;
    border-right: 1px solid #ffffff18;
}

[data-testid="stSidebar"] * {
    color: #e8f2e9 !important;
}

[data-testid="stSidebar"] hr {
    border-color: #ffffff20 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 9px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: #ffffff0a;
    border: 1px solid #ffffff12;
    border-radius: 11px;
    padding: 9px 12px;
    transition: .2s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #ffffff18;
    border-color: #d8b45c88;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
    font-family: 'Playfair Display',serif !important;
    color: #f5d98b !important;
    font-size: 25px !important;
}

.topbar {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:15px;
    padding:14px 20px;
    margin-bottom:25px;
    border:1px solid #ffffff16;
    border-radius:17px;
    background:linear-gradient(110deg,#102e21,#153e2b);
    box-shadow:0 10px 30px #00000020;
}


.brand-letter {
    display:inline-block;
    animation:letterJump .85s cubic-bezier(.2,.85,.3,1.35) both;
}
.brand-leaf {
    display:inline-block;
    margin-right:12px;
    animation:leafPop .9s ease both;
}
@keyframes letterJump {
    0% {opacity:0;transform:translateY(45px) scale(.65);}
    65% {opacity:1;transform:translateY(-9px) scale(1.08);}
    82% {transform:translateY(4px) scale(.98);}
    100% {opacity:1;transform:translateY(0) scale(1);}
}
@keyframes leafPop {
    0% {opacity:0;transform:scale(.2) rotate(-35deg);}
    70% {opacity:1;transform:scale(1.3) rotate(12deg);}
    100% {opacity:1;transform:scale(1) rotate(0);}
}

.topbrand {
    color:#f5d98b;
    font-size:19px;
    font-weight:800;
    letter-spacing:.2px;
}

.topmeta {
    color:#a9c4b0;
    font-size:12px;
    letter-spacing:.7px;
}


/* PREMIUM KPI CARDS */
.kpi {
    position:relative !important;
    overflow:hidden !important;
    background:linear-gradient(145deg,#174b35,#0c2b20) !important;
    border:1px solid rgba(218,190,116,.35) !important;
    border-radius:20px !important;
    padding:23px !important;
    min-height:135px !important;
    box-shadow:0 12px 30px rgba(0,0,0,.22) !important;
    transition:all .35s ease !important;
}
.kpi:before {
    content:"";
    position:absolute;
    top:-55px;
    right:-35px;
    width:130px;
    height:130px;
    border-radius:50%;
    background:radial-gradient(circle,rgba(220,190,105,.18),transparent 70%);
}
.kpi:hover {
    transform:translateY(-8px) scale(1.015) !important;
    border-color:#e5c875 !important;
    box-shadow:0 18px 38px rgba(0,0,0,.32),0 0 18px rgba(218,190,116,.12) !important;
}
.kpi-label {
    color:#b9cdbb !important;
    letter-spacing:1.6px !important;
    font-size:11px !important;
}
.kpi-value {
    color:#f3d785 !important;
    font-size:34px !important;
    font-weight:900 !important;
}
.kpi-note {
    color:#a8c0ad !important;
}

.page-heading {
    color:#f4f7ee;
    font-family:'Playfair Display',serif;
    font-size:30px;
    font-weight:800;
    margin:5px 0;
}

.page-subtitle {
    color:#9bb7a4;
    font-size:13px;
    margin-bottom:22px;
}

.kpi {
    background:linear-gradient(145deg,#173d2b,#102c20) !important;
    border:1px solid #ffffff1c !important;
    border-radius:17px !important;
    padding:19px !important;
    min-height:125px;
    box-shadow:0 10px 25px #00000020 !important;
}

.kpi-label {
    color:#a9c4b0 !important;
}

.kpi-value {
    color:#f3d785 !important;
    font-size:29px !important;
}

.kpi-note {
    color:#8fae99 !important;
}

.section-head {
    border-bottom:1px solid #ffffff20 !important;
    margin-top:30px !important;
}

.section-head h2 {
    color:#f1f5eb !important;
    font-family:'Playfair Display',serif !important;
}

.section-head p {
    color:#9bb7a4 !important;
}

.panel {
    background:linear-gradient(145deg,#153b29,#102c20) !important;
    border:1px solid #ffffff1c !important;
    border-radius:18px !important;
    box-shadow:0 12px 28px #00000020 !important;
}

.panel h3 {
    color:#f2d98f !important;
}

.farm-layout {
    display:grid;
    grid-template-columns:minmax(0,2fr) minmax(230px,.85fr);
    gap:16px;
    margin:22px 0 28px;
}

.farm-visual {
    min-height:310px;
    position:relative;
    overflow:hidden;
    border-radius:20px;
    border:1px solid #ffffff24;
    background:
      linear-gradient(90deg,rgba(5,24,16,.82),rgba(5,24,16,.12)),
      url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1500&q=85') center/cover;
    box-shadow:0 15px 35px #00000035;
    display:flex;
    flex-direction:column;
    justify-content:flex-end;
    padding:26px;
}

.farm-eyebrow {
    color:#f3d98c;
    font-size:10px;
    font-weight:800;
    letter-spacing:2px;
}

.farm-title {
    color:white;
    font-family:'Playfair Display',serif;
    font-size:30px;
    font-weight:800;
    margin:8px 0;
}

.farm-description {
    color:#e0eee1;
    font-size:13px;
    max-width:480px;
    line-height:1.7;
}

.intelligence-card {
    background:linear-gradient(160deg,#173f2b,#0c281c);
    border:1px solid #ffffff20;
    border-radius:20px;
    padding:22px;
    box-shadow:0 12px 30px #00000025;
}

.intelligence-card h3 {
    color:#f3d98c;
    font-family:'Playfair Display',serif;
    font-size:19px;
    margin:0 0 18px;
}

.intel-item {
    padding:13px 0;
    border-bottom:1px solid #ffffff18;
}

.intel-item:last-child {
    border-bottom:0;
}

.intel-label {
    color:#9db9a5;
    font-size:11px;
    letter-spacing:.8px;
    text-transform:uppercase;
}

.intel-value {
    color:#f1f5ed;
    font-size:17px;
    font-weight:800;
    margin-top:5px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background:#123523 !important;
    border:1px solid #ffffff20 !important;
    border-radius:17px !important;
}

div.stButton > button {
    background:linear-gradient(110deg,#176b45,#269b64) !important;
    color:white !important;
    border:1px solid #ffffff20 !important;
    border-radius:12px !important;
}

div[data-testid="stMetric"] {
    background:#153b29 !important;
    border:1px solid #ffffff20 !important;
    border-radius:15px !important;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color:#a9c4b0 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color:#f3d98c !important;
}

[data-testid="stDataFrame"] {
    border:1px solid #ffffff20;
    border-radius:13px;
}

[data-testid="stExpander"] {
    background:#123523 !important;
    border:1px solid #ffffff22 !important;
    border-radius:15px !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color:#f1f5ed !important;
}

[data-testid="stAlert"] {
    border-radius:13px !important;
}

.stProgress > div > div > div > div {
    background:linear-gradient(90deg,#28a66b,#d8b45c) !important;
}

@media(max-width:850px) {
    .farm-layout {grid-template-columns:1fr;}
    .block-container {padding:1rem !important;}
    .farm-visual {min-height:240px;}
    .farm-title {font-size:24px;}
}

/* END AGRIVISION PROFESSIONAL DASHBOARD V1 */

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
    colors = colors or ["#36c98b", "#e4c36c", "#65a9d6", "#ae91df", "#e99b70"]
    maximum = max(values) if len(values) and max(values) > 0 else 1
    rows = ""

    for i, (label, value) in enumerate(zip(labels, values)):
        width = max(1, float(value) / maximum * 100)
        color = colors[i % len(colors)]
        rows += f"""
        <div style="margin:17px 0;">
          <div style="display:flex;justify-content:space-between;gap:10px;
                      color:#e4eee5;font-size:13px;font-weight:700;">
            <span>{html.escape(str(label))}</span>
            <span style="color:#f0d48b;">{value:,.2f}{suffix}</span>
          </div>
          <div style="height:13px;background:#203d30;border-radius:20px;
                      overflow:hidden;margin-top:8px;border:1px solid #ffffff12;">
            <div style="height:100%;width:{width}%;--bar-width:{width}%;
                        background:linear-gradient(90deg,{color},#e4c36c);
                        border-radius:20px;animation:grow 1.2s ease both;">
            </div>
          </div>
        </div>
        """

    components.html(f"""
    <html><body style="margin:0;background:transparent;
    font-family:'DM Sans',sans-serif;">
    <div style="padding:8px 12px;background:linear-gradient(145deg,#123526,#0b241b);
    border:1px solid #d8b45c44;border-radius:16px;">
    {rows}
    </div>
    <style>
    @keyframes grow {{
      from {{width:0;}}
      to {{width:var(--bar-width);}}
    }}
    </style>
    </body></html>
    """, height=max(145, len(labels)*65+30))

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
      body { margin:0; background:transparent; font-family:'DM Sans',sans-serif; }
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
    palette = colors or ["#36c98b", "#e4c36c", "#65a9d6", "#ae91df", "#e99b70"]

    stops = []
    current = 0
    legend = ""

    for i, (label, value) in enumerate(zip(labels, values)):
        color = palette[i % len(palette)]
        end_angle = current + (value / total * 100)
        stops.append(f"{color} {current:.2f}% {end_angle:.2f}%")
        current = end_angle

        legend += f"""
        <div class="legend-row">
          <span class="dot" style="background:{color};box-shadow:0 0 10px {color}66;"></span>
          <span class="legend-name">{html.escape(str(label))}</span>
          <strong>{value:,.0f}</strong>
          <small>{value / total * 100:.1f}%</small>
        </div>
        """

    gradient = ", ".join(stops)

    components.html(f"""
    <html>
    <head>
    <style>
      * {{ box-sizing:border-box; }}
      body {{
        margin:0;
        background:transparent;
        font-family:'DM Sans',sans-serif;
        color:#e8f2e9;
      }}
      .chart-panel {{
        display:flex;
        align-items:center;
        justify-content:center;
        gap:30px;
        flex-wrap:wrap;
        padding:24px 18px;
        border-radius:20px;
        background:linear-gradient(145deg,#123526,#0a2119);
        border:1px solid #d8b45c55;
        box-shadow:0 12px 30px #06180f55;
      }}
      .donut {{
        width:190px;
        height:190px;
        flex-shrink:0;
        border-radius:50%;
        background:conic-gradient({gradient});
        display:flex;
        align-items:center;
        justify-content:center;
        position:relative;
        animation:donutReveal 1.3s ease both;
        box-shadow:0 0 28px #36c98b20,0 10px 25px #00000055;
      }}
      .donut::before {{
        content:'';
        position:absolute;
        inset:13px;
        border-radius:50%;
        background:linear-gradient(145deg,#173d2d,#0b251b);
        border:1px solid #ffffff18;
      }}
      .donut-center {{
        position:relative;
        z-index:1;
        text-align:center;
      }}
      .total {{
        font-size:27px;
        font-weight:800;
        color:#f0d48b;
      }}
      .total-label {{
        font-size:10px;
        letter-spacing:2px;
        color:#a9c5b0;
        margin-top:4px;
      }}
      .legend {{
        width:min(100%,300px);
        min-width:220px;
      }}
      .legend-title {{
        color:#f0d48b;
        font-size:12px;
        font-weight:800;
        letter-spacing:1.5px;
        margin-bottom:13px;
      }}
      .legend-row {{
        display:flex;
        align-items:center;
        gap:9px;
        padding:9px 10px;
        margin:5px 0;
        border-radius:10px;
        background:#ffffff08;
        border:1px solid #ffffff0c;
        transition:all .25s ease;
      }}
      .legend-row:hover {{
        background:#ffffff15;
        transform:translateX(4px);
      }}
      .dot {{
        width:9px;
        height:9px;
        border-radius:50%;
        flex-shrink:0;
      }}
      .legend-name {{
        flex:1;
        font-size:12px;
        color:#e0ece2;
      }}
      .legend-row strong {{
        color:#f0d48b;
        font-size:12px;
      }}
      .legend-row small {{
        color:#9ab5a0;
        font-size:10px;
        min-width:38px;
        text-align:right;
      }}
      @keyframes donutReveal {{
        from {{transform:rotate(-100deg) scale(.75);opacity:0;}}
        to {{transform:rotate(0) scale(1);opacity:1;}}
      }}
    </style>
    </head>
    <body>
      <div class="chart-panel">
        <div class="donut">
          <div class="donut-center">
            <div class="total">{sum(values):,.0f}</div>
            <div class="total-label">TOTAL RECORDS</div>
          </div>
        </div>
        <div class="legend">
          <div class="legend-title">CROP BREAKDOWN</div>
          {legend}
        </div>
      </div>
    </body>
    </html>
    """, height=max(270, min(470, 100 + len(labels) * 38)), scrolling=False)

# -------------------- HERO --------------------
st.markdown("""
<div class="topbar">
    <div class="topbrand">🌿 AgriVision AI</div>
    <div class="topmeta">SMART AGRICULTURE INTELLIGENCE &nbsp; • &nbsp; ML PLATFORM</div>
</div>
""", unsafe_allow_html=True)


# -------------------- PREMIUM CENTERED BRANDING --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

.topbar {
    width:100%;
    display:flex !important;
    flex-direction:column !important;
    justify-content:center !important;
    align-items:center !important;
    text-align:center !important;
    padding:35px 15px 28px !important;
    margin:0 auto 22px !important;
    border-bottom:1px solid rgba(218,190,116,.25);
    background:radial-gradient(ellipse at center,rgba(37,100,64,.20),transparent 70%);
}
.topbrand {
    width:100%;
    text-align:center !important;
    font-family:'Playfair Display',Georgia,serif !important;
    font-size:clamp(38px,5vw,64px) !important;
    font-weight:800 !important;
    letter-spacing:1.5px !important;
    line-height:1.2 !important;
    background:linear-gradient(90deg,#f4e3a5,#d5b65f,#fff1bd,#80bd8a,#f4e3a5);
    background-size:250% auto;
    -webkit-background-clip:text;
    background-clip:text;
    -webkit-text-fill-color:transparent;
    animation:brandShimmer 5s linear infinite, brandAppear 1s ease both;
    filter:drop-shadow(0 5px 15px rgba(0,0,0,.22));
}
.topmeta {
    text-align:center !important;
    width:100%;
    margin-top:12px !important;
    font-family:'Manrope',sans-serif !important;
    font-size:11px !important;
    font-weight:700 !important;
    letter-spacing:4px !important;
    color:#d8c991 !important;
    animation:brandAppear 1.3s ease both;
}
@keyframes brandShimmer {
    to {background-position:250% center;}
}
@keyframes brandAppear {
    from {opacity:0;transform:translateY(15px);}
    to {opacity:1;transform:translateY(0);}
}
@media(max-width:600px) {
    .topbrand {font-size:36px !important;}
    .topmeta {font-size:9px !important;letter-spacing:2px !important;}
}
</style>
""", unsafe_allow_html=True)

# -------------------- HORIZONTAL NAVIGATION --------------------
st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none !important;
}
.topbar {
    animation: navEnter .8s ease both;
    padding: 12px 0 20px;
}
.topbrand {
    font-size: 30px !important;
    font-weight: 900 !important;
    letter-spacing: .5px;
}
@keyframes navEnter {
    from {opacity:0; transform:translateY(-12px);}
    to {opacity:1; transform:translateY(0);}
}

.stRadio [role="radiogroup"] {
    display:flex !important;
    flex-direction:row !important;
    flex-wrap:wrap !important;
    justify-content:center !important;
    gap:12px !important;
}

.stRadio [role="radiogroup"] label {
    background:linear-gradient(145deg,#123c2b,#0b291e) !important;
    border:1px solid rgba(211,183,105,.35) !important;
    border-radius:30px !important;
    padding:10px 20px !important;
    min-height:42px !important;
    transition:all .25s ease !important;
    cursor:pointer !important;
}

.stRadio [role="radiogroup"] label > div:first-child {
    display:none !important;
}

.stRadio [role="radiogroup"] label:hover {
    transform:translateY(-3px);
    border-color:#e7ca79 !important;
    box-shadow:0 8px 22px #0005 !important;
}

.stRadio [role="radiogroup"] label:has(input:checked) {
    background:linear-gradient(120deg,#d9b85f,#a98735) !important;
    border-color:#f5df9b !important;
    box-shadow:0 5px 18px #c6a44a33 !important;
}

.stRadio [role="radiogroup"] label p {
    color:#f4f0df !important;
    font-size:13px !important;
    font-weight:700 !important;
    margin:0 !important;
}

.stRadio [role="radiogroup"] label:has(input:checked) p {
    color:#142b1d !important;
}

.page-heading {
    animation:navEnter .7s ease both;
}
.kpi,.panel {
    transition:transform .25s ease,box-shadow .25s ease;
}
.kpi:hover,.panel:hover {
    transform:translateY(-4px);
    box-shadow:0 15px 30px #0004 !important;
}
@media(max-width:700px) {
    .topbrand {font-size:23px !important;}
    .stRadio > div > label {padding:7px 10px;}
}
</style>
""", unsafe_allow_html=True)

page = st.radio(
    "EXPLORE PLATFORM",
    ["Dashboard", "Crop Recommendation", "Dataset Insights",
     "Model Performance", "About Project"],
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation"
)

# =========================================================
# DASHBOARD
# =========================================================
if page == "Dashboard":

    section("🌾 Farm Intelligence Overview",
            "A live overview of the agricultural dataset and model.")

    total_crops = int(df["label"].nunique()) if "label" in df.columns else 0
    feature_count = len([c for c in df.columns if c != "label"])

    st.markdown("""
    <div class="page-heading">Welcome to AgriVision AI</div>
    <div class="page-subtitle">Your intelligent agriculture analytics workspace</div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    with cols[0]:
        kpi("Crop Categories", total_crops, "Unique crop classes", "🌱")
    with cols[1]:
        kpi("Dataset Records", f"{len(df):,}", "Available observations", "📊")
    with cols[2]:
        kpi("ML Algorithm", "Random Forest", "Trained classifier", "🤖")
    with cols[3]:
        kpi("Input Features", feature_count, "Soil and climate inputs", "🧪")

    st.markdown(f"""
    <div class="farm-layout">
      <div class="farm-visual">
        <div class="farm-eyebrow">AGRICULTURE • DATA • INTELLIGENCE</div>
        <div class="farm-title">Smarter Insights.<br>Better Crop Decisions.</div>
        <div class="farm-description">
          Explore soil and climate patterns, understand crop categories,
          and generate data-driven crop recommendations with your
          trained machine learning model.
        </div>
      </div>
      <div class="intelligence-card">
        <h3>🌿 Intelligence Overview</h3>
        <div class="intel-item">
          <div class="intel-label">Dataset Status</div>
          <div class="intel-value">● Loaded successfully</div>
        </div>
        <div class="intel-item">
          <div class="intel-label">Crop Categories</div>
          <div class="intel-value">{total_crops} crop classes</div>
        </div>
        <div class="intel-item">
          <div class="intel-label">Available Records</div>
          <div class="intel-value">{len(df):,} observations</div>
        </div>
        <div class="intel-item">
          <div class="intel-label">Prediction Engine</div>
          <div class="intel-value">Random Forest ML</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

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
        st.session_state["show_crop_details"] = False

    if analyse or st.session_state.get("show_crop_details", False):
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

            crop_key = str(prediction).strip().lower()
            selected_crop = st.session_state.get(
                "selected_crop_profile", crop_key
            )
            profile = CROP_PROFILES.get(selected_crop)

            st.button(
                f"🌱 View {crop_key.title()} Crop Details",
                key="view_crop_details",
                use_container_width=True,
                on_click=lambda: st.session_state.update(
                    {"show_crop_details": True}
                )
            )

            if st.session_state.get("show_crop_details", False):
                if profile:
                    section(
                        f"🌿 {selected_crop.title()} Crop Profile",
                        "Detailed growing conditions and cultivation guidance."
                    )

                    with st.expander(
                        f"🌱 Complete {selected_crop.title()} Growing Guide",
                        expanded=True
                    ):
                        left, right = st.columns(2)

                        with left:
                            st.markdown("### 🌤️ Growing Conditions")
                            st.markdown(f"**Temperature:** {profile['temperature']}")
                            st.markdown(f"**Humidity:** {profile['humidity']}")
                            st.markdown(f"**Soil pH:** {profile['ph']}")
                            st.markdown(f"**Suitable Soil:** {profile['soil']}")

                        with right:
                            st.markdown("### 🚜 Cultivation Guide")
                            st.markdown(f"**Water / Irrigation:** {profile['water']}")
                            st.markdown(f"**Crop Duration:** {profile['duration']}")
                            st.markdown(f"**Fertilizer:** {profile['fertilizer']}")
                            st.markdown(f"**Pests & Diseases:** {profile['risks']}")

                        st.info(GENERAL_NOTE)
                else:
                    st.warning("Detailed crop information is not available yet.")

            section(
                "🌾 Top 5 Crop Explorer",
                "Discover and explore the five crops identified by your AI model."
            )

            colors = ["#28a66b", "#d8b45c", "#4d91bd",
                      "#9a78c5", "#e28a55"]

            crop_cols = st.columns(5)

            for i, (crop, probability) in enumerate(ranked):
                crop_name = str(crop).title()
                crop_value = float(probability) * 100

                with crop_cols[i]:
                    with st.container(border=True):
                        st.markdown(
                            f"""
                            <div style="
                                background:linear-gradient(145deg,#102b20,#176b45);
                                padding:16px 8px;
                                border-radius:16px;
                                text-align:center;
                                margin-bottom:12px;">
                                <div style="color:#f7d98b;font-size:11px;
                                    font-weight:700;letter-spacing:1px;">
                                    TOP {i + 1} CROP
                                </div>
                                <div style="color:white;font-size:19px;
                                    font-weight:800;margin-top:9px;">
                                    🌱 {html.escape(crop_name)}
                                </div>
                                <div style="color:#f7d98b;font-size:25px;
                                    font-weight:800;margin-top:9px;">
                                    {crop_value:.1f}%
                                </div>
                                <div style="color:#e5f0e8;font-size:10px;">
                                    AI MODEL PROBABILITY
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.progress(float(probability))

                        if st.button(
                            "Explore Profile",
                            key=f"explore_crop_{i}",
                            use_container_width=True
                        ):
                            st.session_state["selected_crop_profile"] = str(crop).strip().lower()
                            st.session_state["show_crop_details"] = True
                            st.rerun()

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

    section(
        "🤖 AI Model Performance",
        "A detailed evaluation of the trained Random Forest crop recommendation model."
    )

    report_path = "models/classification_report.csv"
    matrix_path = "models/confusion_matrix.png"

    if os.path.exists(report_path):
        report = pd.read_csv(report_path)

        label_col = report.columns[0]
        report[label_col] = report[label_col].astype(str)
        labels = report[label_col].str.strip().str.lower()

        accuracy_row = report.loc[labels == "accuracy"]
        macro_row = report.loc[labels == "macro avg"]

        crop_rows = report.loc[
            ~labels.isin(["accuracy", "macro avg", "weighted avg"])
        ].copy()

        accuracy = (
            float(accuracy_row.iloc[0]["precision"]) * 100
            if not accuracy_row.empty else 0.0
        )
        macro_precision = (
            float(macro_row.iloc[0]["precision"]) * 100
            if not macro_row.empty else 0.0
        )
        macro_recall = (
            float(macro_row.iloc[0]["recall"]) * 100
            if not macro_row.empty else 0.0
        )
        macro_f1 = (
            float(macro_row.iloc[0]["f1-score"]) * 100
            if not macro_row.empty else 0.0
        )

        total_samples = int(
            pd.to_numeric(crop_rows["support"], errors="coerce").fillna(0).sum()
        )
        crop_count = len(crop_rows)

        st.markdown("""
        <div style="
            background:linear-gradient(120deg,#0b261b,#145b3b,#102e22);
            border:1px solid rgba(213,184,105,.35);
            border-radius:22px;
            padding:28px;
            margin:12px 0 22px 0;
            box-shadow:0 12px 35px rgba(0,0,0,.20);
        ">
            <div style="color:#e7c978;font-size:11px;font-weight:800;
                letter-spacing:2px;">MODEL EVALUATION OVERVIEW</div>
            <div style="color:white;font-size:34px;font-weight:850;
                margin-top:8px;">Performance Analytics</div>
            <div style="color:#c5ddce;font-size:14px;margin-top:7px;">
                Real evaluation results from the saved classification report
            </div>
        </div>
        """, unsafe_allow_html=True)

        metrics = [
            ("🎯", "MODEL ACCURACY", f"{accuracy:.2f}%", "Overall test accuracy"),
            ("🌾", "CROP CLASSES", str(crop_count), "Evaluated crop categories"),
            ("🧪", "TEST SAMPLES", f"{total_samples:,}", "Total evaluated records"),
            ("🏆", "MACRO F1-SCORE", f"{macro_f1:.2f}%", "Balanced class performance"),
        ]

        metric_cols = st.columns(4)
        for col, (symbol, label, value, note) in zip(metric_cols, metrics):
            with col:
                st.markdown(f"""
                <div style="
                    background:linear-gradient(145deg,#102b20,#163d2c);
                    border:1px solid rgba(213,184,105,.24);
                    border-radius:17px;
                    padding:19px 13px;
                    min-height:145px;
                    text-align:center;
                    box-shadow:0 8px 22px rgba(0,0,0,.13);
                ">
                    <div style="font-size:23px;">{symbol}</div>
                    <div style="color:#d9c27f;font-size:10px;
                        font-weight:800;letter-spacing:1px;margin-top:9px;">
                        {label}
                    </div>
                    <div style="color:#ffffff;font-size:27px;
                        font-weight:850;margin:7px 0;">{value}</div>
                    <div style="color:#b9d0c1;font-size:10px;">{note}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

        section(
            "📈 Evaluation Scorecard",
            "Macro-average metrics calculated from all evaluated crop classes."
        )

        score_cols = st.columns(3)
        for col, label, value in zip(
            score_cols,
            ["Precision", "Recall", "F1-Score"],
            [macro_precision, macro_recall, macro_f1]
        ):
            with col:
                st.markdown(f"""
                <div style="background:#102b20;border:1px solid #28563d;
                    border-radius:15px;padding:17px;margin-bottom:12px;">
                    <div style="color:#d9c27f;font-size:12px;
                        font-weight:700;">{label}</div>
                    <div style="color:white;font-size:25px;
                        font-weight:800;margin-top:7px;">{value:.2f}%</div>
                    <div style="background:#294637;height:7px;
                        border-radius:10px;margin-top:12px;overflow:hidden;">
                        <div style="width:{max(0,min(value,100)):.2f}%;
                            height:7px;background:linear-gradient(90deg,
                            #27a66a,#e2c36e);border-radius:10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        section(
            "📋 Crop-wise Classification Report",
            "Precision, recall, F1-score and test support for each crop."
        )

        display_report = crop_rows.rename(
            columns={label_col: "Crop"}
        ).copy()

        for metric_col in ["precision", "recall", "f1-score"]:
            if metric_col in display_report.columns:
                display_report[metric_col] = display_report[metric_col].map(
                    lambda value: f"{float(value) * 100:.2f}%"
                )

        st.dataframe(
            display_report,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Classification Report",
            report.to_csv(index=False),
            file_name="agrivision_classification_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:
        st.warning("Classification report file was not found.")

    section(
        "🎯 Confusion Matrix",
        "Visual comparison of actual crop labels against model predictions."
    )

    if os.path.exists(matrix_path):
        st.image(matrix_path, use_container_width=True)
    else:
        st.warning("Confusion matrix image was not found.")

    st.caption(
        "Note: These metrics describe the saved evaluation results. "
        "They do not guarantee real-world crop recommendation accuracy."
    )

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

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:12px;color:#71877a;">
  <b style="color:#176b45;">🌿 AgriVision AI</b><br>
  Smart Agriculture • Intelligent Crop Insights<br>
  <small>Machine Learning Prototype</small>
</div>
""", unsafe_allow_html=True)
