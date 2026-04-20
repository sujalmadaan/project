import numpy as np
import matplotlib.pyplot as plt

R = 83.14

def log_K0_CO2(T):
    return 1.668 + 3.992e-3*T - 1.156e-5*T**2 + 1.593e-9*T**3

def log_K0_H2O(T):
    return -2.1077 + 2.8127e-2*T - 8.4298e-5*T**2 + 1.4969e-7*T**3 - 1.1812e-10*T**4

def V_CO2(T):
    return 32.6 + 3.413e-2*(T - 373.15)

def V_H2O(T):
    return 18.1 + 3.137e-2*(T - 373.15)

def compute_K(T, P):
    TK = T + 273.15
    K_CO2 = 10**log_K0_CO2(T) * np.exp((P - 1)*V_CO2(T)/(R*TK))
    K_H2O = 10**log_K0_H2O(T) * np.exp((P - 1)*V_H2O(T)/(R*TK))
    return K_CO2, K_H2O

def compute_gamma(x_CO2, T, m):

    TK = T + 273.15

    if T <= 100:
        gamma_CO2 = 1.0
        gamma_H2O = 1.0
    else:
        a = -3.084e-2
        b = 1.927e-5
        AM = a*(TK - 373.15) + b*(TK - 373.15)**2

        x_H2O = 1 - x_CO2

        ln_gamma_H2O = (AM - 2*AM*x_H2O)*(x_CO2**2)
        ln_gamma_CO2 = 2*AM*x_CO2*(x_H2O**2)

        gamma_CO2 = np.exp(ln_gamma_CO2)
        gamma_H2O = np.exp(ln_gamma_H2O)

    k_s = 0.12

    gamma_CO2 = gamma_CO2 * np.exp(k_s * m)

    gamma_H2O = gamma_H2O

    return gamma_CO2, gamma_H2O

def rk_params_CO2(TK):
    return 8.008e7 - 4.984e4 * TK, 28.25

def rk_params_H2O(TK):
    return 1.337e8 - 1.4e4 * TK, 15.70

def kij_dynamic(TK, y_CO2, y_H2O):
    K_CO2_H2O = 0.4228 - 7.422e-4 * TK
    K_H2O_CO2 = 1.427e-2 - 4.037e-4 * TK
    return K_CO2_H2O * y_CO2 + K_H2O_CO2 * y_H2O

def mixing_parameters(T, y_CO2):

    TK = T + 273.15
    y_H2O = 1 - y_CO2

    a1, b1 = rk_params_CO2(TK)
    a2, b2 = rk_params_H2O(TK)

    kij = kij_dynamic(TK, y_CO2, y_H2O)

    a12 = np.sqrt(a1 * a2) * (1 - kij)

    a_mix = (
        y_CO2**2 * a1 +
        2*y_CO2*y_H2O * a12 +
        y_H2O**2 * a2
    )

    b_mix = y_CO2*b1 + y_H2O*b2

    return a_mix, b_mix, a1, a2, a12, b1, b2

def solve_Z(A, B):
    coeffs = [1, -1, A - B - B**2, -A*B]
    roots = np.roots(coeffs)
    return max(np.real(roots[np.isreal(roots)]))

def compute_phi_mixture(T, P, y_CO2):

    TK = T + 273.15
    y_H2O = 1 - y_CO2

    a_mix, b_mix, a1, a2, a12, b1, b2 = mixing_parameters(T, y_CO2)

    A = a_mix * P / (R**2 * TK**2.5)
    B = b_mix * P / (R * TK)

    Z = solve_Z(A, B)

    def phi_component(a_i, b_i, sum_aij):
        term1 = b_i/b_mix * (Z - 1)
        term2 = -np.log(Z - B)
        term3 = (A/B) * (b_i/b_mix - 2*sum_aij/a_mix) * np.log(1 + B/Z)
        return np.exp(term1 + term2 + term3)

    sum_a_CO2 = y_CO2*a1 + y_H2O*a12
    sum_a_H2O = y_CO2*a12 + y_H2O*a2

    phi_CO2 = phi_component(a1, b1, sum_a_CO2)
    phi_H2O = phi_component(a2, b2, sum_a_H2O)

    return phi_CO2, phi_H2O

def solve_system(T, P, m, tol=1e-6):

    x_CO2 = 0.01
    y_H2O = 0.01

    for _ in range(100):

        K_CO2, K_H2O = compute_K(T, P)
        gamma_CO2, gamma_H2O = compute_gamma(x_CO2, T, m)

        phi_CO2, phi_H2O = compute_phi_mixture(T, P, 1 - y_H2O)

        A = (K_H2O * gamma_H2O) / (phi_H2O * P)
        B = (phi_CO2 * P) / (55.508 * gamma_CO2 * K_CO2)

        y_new = A * (1 - x_CO2)
        x_new = B * (1 - y_new)

        alpha = 0.3
        x_CO2 = alpha*x_new + (1-alpha)*x_CO2
        y_H2O = alpha*y_new + (1-alpha)*y_H2O

    return x_CO2, y_H2O

def run_pressure_sweep(T, m):

    pressures = np.linspace(50, 600, 40)

    x_vals = []
    y_vals = []

    for P in pressures:
        x, y = solve_system(T, P, m)
        x_vals.append(100*x)
        y_vals.append(100*y)

    return pressures, x_vals, y_vals

def mole_fraction_to_molality(x):
    return 55.508 * x / (1 - x)

def plot_paper_style():

    temps = [90, 150, 210, 270]
    salinities = [0, 1, 2, 4]

    fig, axes = plt.subplots(2, 2, figsize=(10,8))
    axes = axes.flatten()

    for i, T in enumerate(temps):

        ax = axes[i]

        for m in salinities:

            pressures = np.linspace(50, 600, 30)
            molality_vals = []

            for P in pressures:
                x, _ = solve_system(T, P, m)
                molality_vals.append(mole_fraction_to_molality(x))

            ax.plot(pressures, molality_vals)

            ax.text(
                pressures[-1],
                molality_vals[-1],
                f"{m} m",
                fontsize=8
            )

        ax.set_title(f"Temperature = {T}°C", fontsize=10)
        ax.set_xlabel("Pressure (bar)")
        ax.set_ylabel("CO2 (molal)")
        ax.grid(True, linewidth=0.5)

    plt.tight_layout()
    plt.show()

plot_paper_style()
import pandas as pd

def generate_results_table(T_list, P_list, m_list):
   
    results = []

    for T in T_list:
        for m in m_list:
            for P in P_list:

                # Solve system
                x_CO2, y_H2O = solve_system(T, P, m)

                # Convert
                molality = mole_fraction_to_molality(x_CO2)

                # Store everything
                results.append({
                    "Temperature (°C)": T,
                    "Pressure (bar)": P,
                    "Salinity (mol/kg)": m,
                    "x_CO2 (liquid)": x_CO2,
                    "y_H2O (vapour)": y_H2O,
                    "CO2 Molality": molality
                })

    df = pd.DataFrame(results)

    return df
    T_list = [90, 150, 210, 270]
P_list = np.linspace(50, 600, 20)
m_list = [0, 1, 2, 4]

df = generate_results_table(T_list, P_list, m_list)

print(df)
