import streamlit as st
import joblib
import pandas as pd
from geometry_features import extract_features

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="Ln–TM Exchange Predictor",
    layout="centered"
)

st.title("🔬 Ln–TM Exchange Coupling Predictor")
st.markdown(
"""
Upload a **Cartesian XYZ file** (atomic numbers, no header).  
The app will compute geometric descriptors and predict the **exchange coupling constant J**.
"""
)

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    return joblib.load("rf_model.joblib")

model = load_model()

# ---------------- File upload ----------------
uploaded_file = st.file_uploader(
    "Upload XYZ file",
    type=["xyz"]
)

if uploaded_file is not None:
    with open("temp.xyz", "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        X_pred = extract_features("temp.xyz")
        J_pred = model.predict(X_pred)[0]

        st.success(f"✅ Predicted J value: **{J_pred:.3f} cm⁻¹**")

        with st.expander("Show extracted descriptors"):
            st.dataframe(X_pred)

    except Exception as e:
        st.error("❌ Error processing XYZ file")
        st.exception(e)

st.markdown("---")
st.caption(
"Model: Random Forest | Features: Ln–O–Tm, Ln–X–Tm, torsion | "
"Use for research purposes only."
)
