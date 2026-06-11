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

    tree_preds = np.array([
        tree.predict(X)[0]
        for tree in model.estimators_
    ])

    J_mean = tree_preds.mean()
    J_std = tree_preds.std(ddof=1)

    return J_mean, J_std


# ---------------- Element definitions ----------------
LN_ELEMENTS = {
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"
}

TM_ELEMENTS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "Hf", "Ta", "W", "Re",
    "Os", "Ir", "Pt", "Au", "Hg"
}


# ---------------- File upload ----------------
uploaded_file = st.file_uploader(
    "Upload XYZ file",
    type=["xyz"]
)


# ---------------- Prediction ----------------
if uploaded_file is not None:

    with open("temp.xyz", "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:

        # --------------------------------------------------
        # Detect Ln and TM centers
        # --------------------------------------------------
        with open("temp.xyz") as f:
            xyz_lines = f.readlines()[2:]

        ln_atoms = []
        tm_atoms = []

        for idx, line in enumerate(xyz_lines, start=1):

            parts = line.split()

            if len(parts) < 4:
                continue

            element = parts[0]

            if element in LN_ELEMENTS:
                ln_atoms.append(idx)

            if element in TM_ELEMENTS:
                tm_atoms.append(idx)

        if len(ln_atoms) == 0:
            st.error("No lanthanide atom detected.")
            st.stop()

        if len(tm_atoms) == 0:
            st.error("No transition-metal atom detected.")
            st.stop()

        # --------------------------------------------------
        # Select pair only if needed
        # --------------------------------------------------
        ln_index = None
        tm_index = None

        need_selection = (
            len(ln_atoms) > 1 or
            len(tm_atoms) > 1
        )

        if need_selection:

            st.info(
                f"Detected {len(ln_atoms)} Ln center(s) and "
                f"{len(tm_atoms)} TM center(s)."
            )

            st.caption(
                "Select the Ln–TM pair for J calculation."
            )

            ln_index = st.selectbox(
                "Lanthanide atom index",
                ln_atoms
            )

            tm_index = st.selectbox(
                "Transition metal atom index",
                tm_atoms
            )

        else:
            ln_index = ln_atoms[0]
            tm_index = tm_atoms[0]

        # --------------------------------------------------
        # Extract descriptors
        # --------------------------------------------------
        X_pred = extract_features(
            "temp.xyz",
            ln_index=ln_index,
            tm_index=tm_index
        )

        # --------------------------------------------------
        # RF prediction + uncertainty
        # --------------------------------------------------
        J_ruiz, J_std = predict_J_with_uncertainty(X_pred)

        err_ruiz = J_std / 2

        if err_ruiz < 0.10:
            err_ruiz = 0.10

        # --------------------------------------------------
        # Spins
        # --------------------------------------------------
        S_tm = float(X_pred["Spin"].values[0])
        S_gd = 3.5

        S1 = S_gd
        S2 = S_tm

        S_HS = S1 + S2
        S_BS = abs(S1 - S2)

        # --------------------------------------------------
        # Energy difference
        # --------------------------------------------------
        denom_ruiz = 2 * S1 * S2 + S2

        deltaE = J_ruiz * denom_ruiz

        # --------------------------------------------------
        # Convert J
        # --------------------------------------------------
        J_noodle = deltaE / (
            S_HS * (S_HS + 1)
        )

        J_yama = deltaE / (
            S_HS * (S_HS + 1)
            - S_BS * (S_BS + 1)
        )

        # --------------------------------------------------
        # Error propagation
        # --------------------------------------------------
        err_deltaE = err_ruiz * denom_ruiz

        err_noodle = err_deltaE / (
            S_HS * (S_HS + 1)
        )

        err_yama = err_deltaE / (
            S_HS * (S_HS + 1)
            - S_BS * (S_BS + 1)
        )

        # --------------------------------------------------
        # Results
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Warning
        # --------------------------------------------------
        if J_std > 1.0:
            st.warning(
                "⚠️ High uncertainty: molecule may be outside the training domain."
            )

        # --------------------------------------------------
        # Formulas
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Spin information
        # --------------------------------------------------
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

        # --------------------------------------------------
        # Model uncertainty
        # --------------------------------------------------
        with st.expander("Model uncertainty"):

            st.write(
                f"Random Forest standard deviation: {J_std:.3f} cm⁻¹"
            )

            st.write(
                f"Reported uncertainty (½σ): ±{err_ruiz:.3f} cm⁻¹"
            )

        # --------------------------------------------------
        # Descriptors
        # --------------------------------------------------
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
