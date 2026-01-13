import streamlit as st
import joblib
import pandas as pd
from geometry_features import extract_features

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="Ln–TM Exchange Predictor",
    layout="centered"
)

st.title("🔬 Ln–TM Magnetic Exchange Predictor")

st.markdown("""
Upload a **Cartesian XYZ file** (atomic numbers only, no header).

- If **multiple Ln or TM atoms** are present, specify the atom indices.
- **Zn(II)** systems are automatically detected as **diamagnetic**.
""")

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

# ---------------- Optional indices ----------------
st.markdown("### Optional: Metal indices (for multinuclear systems)")

ln_index = st.number_input(
    "Lanthanide atom index (leave empty if only one Ln)",
    min_value=1,
    step=1,
    value=None
)

tm_index = st.number_input(
    "Transition metal atom index (leave empty if only one TM)",
    min_value=1,
    step=1,
    value=None
)

# ---------------- Prediction ----------------
if uploaded_file is not None:
    with open("temp.xyz", "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        X_pred = extract_features(
            "temp.xyz",
            ln_index=ln_index,
            tm_index=tm_index
        )

        J_pred = model.predict(X_pred)[0]

        st.success(f"✅ Predicted exchange coupling **J = {J_pred:.3f} cm⁻¹**")

        with st.expander("Show extracted geometric descriptors"):
            st.dataframe(X_pred)

    except ValueError as e:
        st.warning(f"⚠️ {str(e)}")

    except Exception as e:
        st.error("❌ Error processing XYZ file")
        st.exception(e)

st.markdown("---")
st.caption(
    "Random Forest model trained on geometry-based descriptors. "
    "For research use only."
)
