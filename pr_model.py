import numpy as np
from scipy.optimize import brentq

R = 8.314
T0 = 298.15

components = {
    "CO2": {"Tc": 304.13, "Pc": 7.3773e6, "omega": 0.22394},
    "H2O": {"Tc": 647.1, "Pc": 22.064e6, "omega": 0.3443},
}

def PR_parameters(T, comp):
    Tc = comp["Tc"]
    Pc = comp["Pc"]
    omega = comp["omega"]

    kappa = 0.37464 + 1.54226*omega - 0.26992*omega**2
    alpha = (1 + kappa*(1 - np.sqrt(T/Tc)))**2

    a = 0.45724 * (R**2 * Tc**2 / Pc) * alpha
    b = 0.07780 * (R * Tc / Pc)

    return a, b

def kij_CO2_H2O(T):
    return 0.33810 - 0.46426*(T0/T)

def mixing_rules(y, a_list, b_list, kij):
    a_mix = 0.0
    for i in range(2):
        for j in range(2):
            kij_ij = 0 if i == j else kij
            a_mix += y[i]*y[j]*np.sqrt(a_list[i]*a_list[j])*(1 - kij_ij)

    b_mix = y[0]*b_list[0] + y[1]*b_list[1]
    return a_mix, b_mix

def solve_PR_Z(P, T, a_mix, b_mix):
    A = a_mix * P / (R**2 * T**2)
    B = b_mix * P / (R * T)

    coeffs = [1, -(1 - B), A - 3*B**2 - 2*B, -(A*B - B**2 - B**3)]
    roots = np.roots(coeffs)
    real_roots = np.real(roots[np.isreal(roots)])

    return max(real_roots)

def fugacity_CO2(y, P, T):
    kij = kij_CO2_H2O(T)

    a1, b1 = PR_parameters(T, components["CO2"])
    a2, b2 = PR_parameters(T, components["H2O"])

    a_mix, b_mix = mixing_rules(y, [a1,a2], [b1,b2], kij)
    Z = solve_PR_Z(P, T, a_mix, b_mix)

    A = a_mix * P / (R**2 * T**2)
    B = b_mix * P / (R * T)

    bi = b1
    a12 = np.sqrt(a1*a2)*(1 - kij)
    sum_aij = y[0]*a1 + y[1]*a12

    term1 = (bi / b_mix) * (Z - 1)
    term2 = -np.log(Z - B)

    term3 = (A / (2*np.sqrt(2)*B)) * ((2*sum_aij/a_mix) - (bi/b_mix))

    log_term = np.log((Z+(1+np.sqrt(2))*B)/(Z+(1-np.sqrt(2))*B))
    ln_phi = term1 + term2 - term3*log_term

    return np.exp(ln_phi)

def Henry_CO2(T):
    C0, C1, C2, C3 = -6.1384, 42.842, -44.358, 12.786
    val = C0 + C1*(T0/T) + C2*(T0/T)**2 + C3*(T0/T)**3
    return np.exp(val) * 1e6

def poynting(P, T):
    V_inf = 20e-6
    Pref = 1e5
    exponent = V_inf * (P - Pref) / (R*T)
    return np.exp(min(exponent, 2))

def equilibrium_residual(P, T, x1, y1):
    P_pa = P * 1e6
    y = [y1, 1 - y1]

    phi = fugacity_CO2(y, P_pa, T)
    H = Henry_CO2(T)
    Poy = poynting(P_pa, T)

    return y1 * phi * P_pa - x1 * H * Poy

def solve_pressure(T, x1, y1):
    try:
        return brentq(lambda P: equilibrium_residual(P, T, x1, y1), 0.1, 20)
    except:
        return None