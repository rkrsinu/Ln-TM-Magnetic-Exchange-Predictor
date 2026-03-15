import streamlit as st
import joblib
import pandas as pd
from geometry_features import extract_features


# ---------------- Page setup ----------------
st.set_page_config(
    page_title="3d–Gd Magnetic Exchange Predictor",
    layout="centered"
)

st.title("🔬 3d–Gd Magnetic Exchange Predictor")

st.markdown("""
Upload a **Cartesian XYZ file**.

This tool calculates magnetic exchange coupling **J for 3d–Gd systems**.
""")


# ---------------- Load ML model ----------------
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

        # Extract descriptors
        X_pred = extract_features(
            "temp.xyz",
            ln_index=ln_index,
            tm_index=tm_index
        )

        # Ruiz prediction from ML model
        J_ruiz = model.predict(X_pred)[0]

        # ---------------- Spin information ----------------
        S_tm = float(X_pred["Spin"].values[0])
        S_gd = 3.5

        S1 = S_gd
        S2 = S_tm

        S_HS = S1 + S2
        S_BS = abs(S1 - S2)

        # ---------------- Calculate EBS − EHS ----------------
        denom_ruiz = 2*S1*S2 + S2
        deltaE = J_ruiz * denom_ruiz

        # ---------------- Convert to other J ----------------
        J_noodle = deltaE / (S_HS*(S_HS+1))

        J_yama = (2*deltaE) / (
            S_HS*(S_HS+1) - S_BS*(S_BS+1)
        )

        # ---------------- Error propagation ----------------
        err_ruiz = 0.27

        err_deltaE = err_ruiz * denom_ruiz

        err_noodle = err_deltaE / (S_HS*(S_HS+1))

        err_yama = (2*err_deltaE) / (
            S_HS*(S_HS+1) - S_BS*(S_BS+1)
        )

        # ---------------- Results table ----------------
        results = pd.DataFrame({
            "Method":[
                "Ruiz",
                "Noodleman",
                "Yamaguchi"
            ],
            "J (cm⁻¹)":[
                f"{J_ruiz:.3f} ± {err_ruiz:.2f}",
                f"{J_noodle:.3f} ± {err_noodle:.2f}",
                f"{J_yama:.3f} ± {err_yama:.2f}"
            ]
        })

        st.success("✅ Exchange coupling results")

        st.table(results)


        # ---------------- Show formulas ----------------
        st.markdown("### Formulas used")

        st.markdown("""
**Ruiz**

J = (EBS − EHS) / (2S₁S₂ + S₂)

---

**Noodleman**

J = (EBS − EHS) / [S_HS(S_HS + 1)]

---

**Yamaguchi**

J = 2(EBS − EHS) / (⟨S²⟩HS − ⟨S²⟩BS)

where

⟨S²⟩ = S(S+1)
""")


        # ---------------- Spin information ----------------
        with st.expander("Spin information"):

            spin_table = pd.DataFrame({
                "Parameter":[
                    "Spin(Gd)",
                    "Spin(3d metal)",
                    "S_HS",
                    "S_BS",
                    "EBS − EHS"
                ],
                "Value":[
                    S_gd,
                    S_tm,
                    S_HS,
                    S_BS,
                    f"{deltaE:.3f} cm⁻¹"
                ]
            })

            st.table(spin_table)


        # ---------------- Show descriptors ----------------
        with st.expander("Show extracted geometric descriptors"):
            st.dataframe(X_pred)


    except ValueError as e:
        st.warning(f"⚠️ {str(e)}")

    except Exception as e:
        st.error("❌ Error processing XYZ file")
        st.exception(e)


# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "Machine learning model for predicting magnetic exchange coupling in 3d–Gd systems."
)
