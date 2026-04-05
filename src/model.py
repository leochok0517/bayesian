import numpy as np
from scipy.integrate import odeint


def sir_rhs(y, t, beta, gamma):
    S, I, R = y
    N = S + I + R

    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I

    return [dSdt, dIdt, dRdt]


def solve_sir(beta, gamma, S0, I0, R0, t):
    """
    Solve SIR model
    """
    y0 = [S0, I0, R0]
    sol = odeint(sir_rhs, y0, t, args=(beta, gamma))

    S = sol[:, 0]
    I = sol[:, 1]
    R = sol[:, 2]

    return S, I, R