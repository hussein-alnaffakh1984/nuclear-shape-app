import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Nuclear Shape Diagnostic (R4/2, R6/2)", layout="centered")

# -------------------------
# Helpers
# -------------------------
Z_MAP = {"Xe": 54, "Ba": 56, "Ce": 58, "Nd": 60}

def diagnose_shape(r42, r62):
    """
    Simple physics-informed diagnostic.
    Uses R4/2 primary, R6/2 secondary.
    """
    if np.isnan(r42):
        return "Unknown", "R4/2 is NaN"

    # Shell-closure / very spherical-like
    if r42 < 1.6:
        return "Near shell-closure (strongly spherical)", "R4/2 < 1.6"

    # Vibrational-like
    if 1.6 <= r42 < 2.1:
        return "Spherical / vibrational-like", "1.6 ≤ R4/2 < 2.1 (≈ 2.0)"

    # Transitional / gamma-soft
    if 2.1 <= r42 < 2.9:
        return "Transitional / gamma-soft", "2.1 ≤ R4/2 < 2.9"

    # Rotational-like (ideal rotor ~3.33)
    if r42 >= 2.9:
        return "Deformed / rotational-like", "R4/2 ≥ 2.9 (→ rotor limit ~3.33)"

    return "Unknown", "Out of rules"

def safe_float(x):
    try:
        return float(x)
    except:
        return np.nan

# -------------------------
# Load data (from repo folder)
# -------------------------
@st.cache_data
def load_data():
    # These CSV files must be in the same repo as app.py
    xe = pd.read_csv("xe_data.csv")
    ba = pd.read_csv("ba_data.csv")
    ce = pd.read_csv("ce_data.csv")
    nd = pd.read_csv("nd_data.csv")
    df = pd.concat([xe, ba, ce, nd], ignore_index=True)
    df = df.dropna(subset=["E2_1", "R4_2", "R6_2"]).copy()
    return df

df = load_data()

# -------------------------
# Train lightweight models at startup
# Model-1: predict E2 from (Z,N,A)
# Model-2: predict (R4/2, R6/2) from (Z,N,A,E2)
# -------------------------
@st.cache_resource
def train_models(df_):
    # E2 model
    X_e2 = df_[["Z", "N", "A"]].copy()
    y_e2 = df_["E2_1"].copy()

    e2_model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=600, random_state=42, min_samples_leaf=2, n_jobs=-1))
    ])
    e2_model.fit(X_e2, y_e2)

    # R ratios model
    X_r = df_[["Z", "N", "A", "E2_1"]].copy()
    y_r = df_[["R4_2", "R6_2"]].copy()

    r_model = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", MultiOutputRegressor(
            RandomForestRegressor(n_estimators=900, random_state=42, min_samples_leaf=2, n_jobs=-1)
        ))
    ])
    r_model.fit(X_r, y_r)

    return e2_model, r_model

e2_model, r_model = train_models(df)

# -------------------------
# UI
# -------------------------
st.title("Nuclear Structure Diagnostic Tool")
st.write("One interface with two modes: (1) Element + A/N (predict R) or (2) E2/E4/E6 (compute R).")

mode = st.radio(
    "Choose input mode:",
    ["1) Element + (A or N) → Predict R → Diagnose", "2) E2/E4/E6 → Compute R → Diagnose"],
    horizontal=False
)

st.divider()

if mode.startswith("1)"):
    st.subheader("Mode 1: Predict R from Element + A/N")
    col1, col2 = st.columns(2)

    with col1:
        element = st.selectbox("Element", ["Xe", "Ba", "Ce", "Nd"], index=0)
    with col2:
        which = st.selectbox("Provide", ["A (mass number)", "N (neutron number)"], index=0)

    Z = Z_MAP[element]

    if which.startswith("A"):
        A_in = st.number_input("A", min_value=90, max_value=250, value=132, step=1)
        A = int(A_in)
        N = A - Z
        st.caption(f"Computed N = A - Z = {A} - {Z} = {N}")
    else:
        N_in = st.number_input("N", min_value=40, max_value=200, value=78, step=1)
        N = int(N_in)
        A = N + Z
        st.caption(f"Computed A = N + Z = {N} + {Z} = {A}")

    # Predict E2 first (because R-model needs E2)
    E2_pred = e2_model.predict(pd.DataFrame([{"Z": Z, "N": N, "A": A}]))[0]

    # Predict R ratios
    R_pred = r_model.predict(pd.DataFrame([{"Z": Z, "N": N, "A": A, "E2_1": E2_pred}]))[0]
    r42_pred, r62_pred = float(R_pred[0]), float(R_pred[1])

    label, reason = diagnose_shape(r42_pred, r62_pred)

    st.success("Prediction completed")
    st.write("### Outputs")
    st.metric("Predicted E2_1 (keV)", f"{E2_pred:.3f}")
    st.metric("Predicted R4/2", f"{r42_pred:.3f}")
    st.metric("Predicted R6/2", f"{r62_pred:.3f}")

    st.write("### Diagnosis")
    st.info(f"**{label}**  \nReason: {reason}")

    with st.expander("Show nearest experimental entries (if exist)"):
        near = df[(df["Element"] == element)].copy()
        near["dA"] = (near["A"] - A).abs()
        near = near.sort_values("dA").head(5)[["Element","A","N","E2_1","R4_2","R6_2"]]
        st.dataframe(near, use_container_width=True)

else:
    st.subheader("Mode 2: Compute R from E2/E4/E6")
    element = st.selectbox("Element (optional for display)", ["Xe", "Ba", "Ce", "Nd"], index=0)
    col1, col2, col3 = st.columns(3)

    with col1:
        E2 = st.number_input("E(2+1) keV", min_value=0.0, value=667.715, step=1.0)
    with col2:
        E4 = st.number_input("E(4+1) keV", min_value=0.0, value=1440.323, step=1.0)
    with col3:
        E6 = st.number_input("E(6+1) keV", min_value=0.0, value=2111.880, step=1.0)

    if E2 <= 0:
        st.error("E2 must be > 0 to compute ratios.")
    else:
        r42 = E4 / E2
        r62 = E6 / E2
        label, reason = diagnose_shape(r42, r62)

        st.success("Computation completed")
        st.write("### Outputs")
        st.metric("R4/2", f"{r42:.3f}")
        st.metric("R6/2", f"{r62:.3f}")

        st.write("### Diagnosis")
        st.info(f"**{label}**  \nReason: {reason}")

st.divider()
st.caption("Data source: IAEA LiveChart (ENSDF-based). Models trained on Xe/Ba/Ce/Nd low-lying levels.")
