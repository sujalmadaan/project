import streamlit as st
from pr_model import solve_pressure
from rk_model import calculate_solubility

st.set_page_config(page_title="CO2 Thermodynamics", layout="centered")

st.title("CO₂–H₂O Thermodynamic Tool")

tab1, tab2 = st.tabs(["Pressure Solver", "Solubility Solver"])

# -------------------------
# MODEL 1
# -------------------------
with tab1:
    st.subheader("PR EOS + Henry Model")

    T = st.number_input("Temperature (K)", value=300.0)
    x = st.number_input("x_CO2 (liquid)", value=0.01)
    y = st.number_input("y_CO2 (vapor)", value=0.9)

    if st.button("Calculate Pressure"):
        P = solve_pressure(T, x, y)

        if P:
            st.success(f"Pressure = {P:.3f} MPa")
        else:
            st.error("No solution found")

# -------------------------
# MODEL 2
# -------------------------
with tab2:
    st.subheader("γ–φ Model")

    T2 = st.number_input("Temperature (°C)", value=200.0)
    P2 = st.number_input("Pressure (bar)", value=150.0)
    m = st.number_input("Salinity (mol/kg)", value=1.0)

    if st.button("Calculate Solubility"):
        x_CO2, y_H2O, molality = calculate_solubility(T2, P2, m)

        st.success("Results:")
        st.write(f"x_CO2: {x_CO2:.6f}")
        st.write(f"y_H2O: {y_H2O:.6f}")
        st.write(f"Molality: {molality:.3f} mol/kg")