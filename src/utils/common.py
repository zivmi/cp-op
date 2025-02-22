import numpy as np
from scipy.stats import norm

def d1(S, K, r, sigma, tau):
    return (np.log(S / K) + (r + sigma**2 / 2) * tau) / (sigma * np.sqrt(tau))

def d2(S, K, r, sigma, tau):
    return d1(S, K, r, sigma, tau) - sigma * np.sqrt(tau)

def black_scholes_price(S, K, r, sigma, tau, option_type="call"):
    D1 = d1(S, K, r, sigma, tau)
    D2 = d2(S, K, r, sigma, tau)
    if option_type == "call":
        return S * norm.cdf(D1) - K * np.exp(-r * tau) * norm.cdf(D2)
    elif option_type == "put":
        return K * np.exp(-r * tau) * norm.cdf(-D2) - S * norm.cdf(-D1)
        