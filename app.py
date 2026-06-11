import streamlit as st
import joblib
import pandas as pd
import numpy as np

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


# ---------------- Prediction function ----------------
def predict_J_with_uncertainty(X):

    # Prediction from every RF tree
    tree_preds = np.array([
        tree.predict(X)[0]
        for tree in model.estimators_
    ])

    J_mean = tree_preds.mean()
    J_std = tree_preds.std(ddof=1)

    return J_mean, J_std


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

        # ---------------- Extract descriptors ----------------
        X_pred = extract_features(
            "temp.xyz",
            ln_index=ln_index,
            tm_index=tm_index
        )

        # ---------------- RF prediction + uncertainty ----------------
        J_ruiz, J_std = predict_J_with_uncertainty(X_pred)

        # Same philosophy as Co-SIM predictor
        err_ruiz = J_std / 2

        # Optional minimum uncertainty
        if err_ruiz < 0.10:
            err_ruiz = 0.10

        # ---------------- Spins ----------------
        S_tm = float(X_pred["Spin"].values[0])
        S_gd = 3.5

        S1 = S_gd
        S2 = S_tm

        S_HS = S1 + S2
        S_BS = abs(S1 - S2)

        # ---------------- Energy difference ----------------
        denom_ruiz = 2 * S1 * S2 + S2

        deltaE = J_ruiz * denom_ruiz

        # ---------------- Convert J ----------------
        J_noodle = deltaE / (
            S_HS * (S_HS + 1)
        )

        J_yama = deltaE / (
            S_HS * (S_HS + 1)
            - S_BS * (S_BS + 1)
        )

        # ---------------- Error propagation ----------------
        err_deltaE = err_ruiz * denom_ruiz

        err_noodle = err_deltaE / (
            S_HS * (S_HS + 1)
        )

        err_yama = err_deltaE / (
            S_HS * (S_HS + 1)
            - S_BS * (S_BS + 1)
        )

        # ---------------- Results ----------------
        results = pd.DataFrame({
            "Method": [
                "Ruiz",
                "Noodleman",
                "Yamaguchi"
            ],
            "J (cm⁻¹)": [
                f"{J_ruiz:.3f} ± {err_ruiz:.3f}",
                f"{J_noodle:.3f} ± {err_noodle:.3f}",
                f"{J_yama:.3f} ± {err_yama:.3f}"
            ]
        })

        st.success("✅ Exchange coupling results")

        st.table(results)

        # ---------------- Warning ----------------
        if J_std > 1.0:
            st.warning(
                "⚠️ High uncertainty: molecule may be outside the training domain."
            )

        # ---------------- Formulas ----------------
        st.markdown("### Formulas used")

        st.markdown("**Ruiz**")
        st.latex(
            r"J = \frac{E_{BS}-E_{HS}}{2S_1S_2 + S_2}"
        )

        st.markdown("**Noodleman**")
        st.latex(
            r"J = \frac{E_{BS}-E_{HS}}{S_{HS}(S_{HS}+1)}"
        )

        st.markdown("**Yamaguchi**")
        st.latex(
            r"J = \frac{E_{BS}-E_{HS}}{\langle S^2\rangle_{HS}-\langle S^2\rangle_{BS}}"
        )

        st.markdown("where")

        st.latex(
            r"\langle S^2\rangle = S(S+1)"
        )

        # ---------------- Spin information ----------------
        with st.expander("Spin information"):

            spin_table = pd.DataFrame({
                "Parameter": [
                    "Spin(Gd)",
                    "Spin(3d metal)",
                    "S_HS",
                    "S_BS",
                    "EBS − EHS"
                ],
                "Value": [
                    S_gd,
                    S_tm,
                    S_HS,
                    S_BS,
                    f"{deltaE:.3f} cm⁻¹"
                ]
            })

            st.table(spin_table)

        # ---------------- Model uncertainty ----------------
        with st.expander("Model uncertainty"):

            st.write(
                f"Random Forest standard deviation: {J_std:.3f} cm⁻¹"
            )

            st.write(
                f"Reported uncertainty (½σ): ±{err_ruiz:.3f} cm⁻¹"
            )

        # ---------------- Descriptors ----------------
        with st.expander(
            "Show extracted geometric descriptors"
        ):
            st.dataframe(X_pred)

    except ValueError as e:
        st.warning(f"⚠️ {str(e)}")

    except Exception as e:
        st.error("Error processing XYZ file")
        st.exception(e)


# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "Machine learning model for predicting magnetic exchange coupling in 3d–Gd systems."
)
