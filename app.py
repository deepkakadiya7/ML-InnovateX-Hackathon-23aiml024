import streamlit as st
import joblib
import numpy as np

# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: -0.5rem;
    }

    .result-card {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1rem;
        animation: fadeIn 0.5s ease-in-out;
    }
    .fraud-card {
        background: linear-gradient(135deg, #ff416c22, #ff4b2b22);
        border: 1px solid #ff416c55;
    }
    .legit-card {
        background: linear-gradient(135deg, #00b09b22, #96c93d22);
        border: 1px solid #00b09b55;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 12px;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ── Load Model ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>💳 Credit Card Fraud Detection</h1>
    <p>AI-powered real-time transaction analysis</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Feature names in training order: Time, V1-V28, Amount ───────────
FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Options")

    # Generate random button (not checkbox — generates fresh values each click)
    if st.button("🎲 Generate Random Values"):
        rng = np.random.default_rng()  # fresh random each click
        for name in FEATURE_NAMES:
            if name == "Time":
                st.session_state[f"val_{name}"] = float(rng.uniform(0, 170000))
            elif name == "Amount":
                st.session_state[f"val_{name}"] = float(rng.uniform(0, 500))
            else:
                st.session_state[f"val_{name}"] = float(rng.standard_normal() * 2)

    if st.button("🧹 Clear All"):
        for name in FEATURE_NAMES:
            st.session_state[f"val_{name}"] = 0.0

    st.markdown("---")

    # Quick presets for demo
    st.markdown("### 🎯 Demo Presets")
    if st.button("⚠️ Suspicious Transaction"):
        # Values that look like fraud based on typical fraud patterns
        rng = np.random.default_rng(42)
        fraud_vals = {
            "Time": 50000.0, "V1": -3.5, "V2": 4.2, "V3": -7.1, "V4": 5.8,
            "V5": -2.1, "V6": -1.8, "V7": -5.3, "V8": 1.2, "V9": -4.5,
            "V10": -8.2, "V11": 3.1, "V12": -9.5, "V13": 0.3, "V14": -12.0,
            "V15": -1.5, "V16": -6.2, "V17": -8.7, "V18": -3.2, "V19": 1.7,
            "V20": 0.9, "V21": 3.5, "V22": 2.1, "V23": -1.8, "V24": -0.5,
            "V25": 0.7, "V26": -0.3, "V27": 2.8, "V28": 1.2, "Amount": 450.0,
        }
        for name, val in fraud_vals.items():
            st.session_state[f"val_{name}"] = val

    if st.button("✅ Normal Transaction"):
        normal_vals = {
            "Time": 80000.0, "V1": -1.3, "V2": 1.2, "V3": 0.8, "V4": -0.3,
            "V5": -0.5, "V6": -0.7, "V7": 0.6, "V8": -0.1, "V9": 0.3,
            "V10": -0.5, "V11": 1.1, "V12": 0.2, "V13": -0.8, "V14": -0.4,
            "V15": 0.1, "V16": 0.3, "V17": -0.2, "V18": -0.1, "V19": 0.5,
            "V20": 0.1, "V21": -0.2, "V22": -0.1, "V23": 0.0, "V24": -0.3,
            "V25": 0.2, "V26": 0.1, "V27": 0.0, "V28": -0.1, "Amount": 25.0,
        }
        for name, val in normal_vals.items():
            st.session_state[f"val_{name}"] = val

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "This app uses a **Random Forest** classifier trained on "
        "credit card transaction features to predict whether a "
        "transaction is **fraudulent** or **legitimate**."
    )
    st.markdown(
        "<div style='text-align:center; color:#94a3b8; font-size:0.85rem; margin-top:1rem;'>"
        "Built with ❤️ using Streamlit</div>",
        unsafe_allow_html=True,
    )

# ── Feature Inputs ──────────────────────────────────────────────────
st.markdown("### 📝 Enter Transaction Features")
st.caption("Provide values for 30 features (Time, V1–V28, Amount)")

cols = st.columns(3)
ui_values = {}

for idx, name in enumerate(FEATURE_NAMES):
    with cols[idx % 3]:
        default = st.session_state.get(f"val_{name}", 0.0)
        val = st.number_input(
            name,
            value=default,
            format="%.4f",
            key=f"input_{name}",
        )
        ui_values[name] = val

st.markdown("")

# ── Predict ─────────────────────────────────────────────────────────
if st.button("🔍  Analyze Transaction"):
    with st.spinner("Running prediction model..."):
        import time as _time
        _time.sleep(0.4)

        # Build feature array in EXACT training order: Time, V1-V28, Amount, Class(dummy=0)
        ordered = []
        for name in FEATURE_NAMES:
            ordered.append(ui_values[name])
        ordered.append(0.0)  # 31st feature — Class col was in scaler training data

        features_arr = np.array(ordered).reshape(1, -1)
        features_scaled = scaler.transform(features_arr)

        pred = model.predict(features_scaled)[0]
        prob = model.predict_proba(features_scaled)[0][1]

    st.markdown("")

    if pred == 1:
        st.markdown(f"""
        <div class="result-card fraud-card">
            <h2 style="color:#ff416c; margin:0;">⚠️ Fraud Detected</h2>
            <p style="font-size:1.2rem; color:#ff6b81; margin:0.5rem 0 0;">
                Fraud Probability: <strong>{prob:.2%}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card legit-card">
            <h2 style="color:#00b09b; margin:0;">✅ Legitimate Transaction</h2>
            <p style="font-size:1.2rem; color:#59d999; margin:0.5rem 0 0;">
                Fraud Probability: <strong>{prob:.2%}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Confidence Meter ────────────────────────────────────────────
    st.markdown("")
    st.markdown("##### 📊 Confidence Breakdown")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Legit Confidence", f"{(1 - prob):.2%}")
    with col2:
        st.metric("Fraud Confidence", f"{prob:.2%}")

    st.progress(prob, text=f"Fraud probability: {prob:.2%}")
