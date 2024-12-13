from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

@dataclass
class Equity:
    spot: float
    dividend_yield: float
    volatility: float

@dataclass
class EquityOption:
    strike: float
    time_to_maturity: float
    put_call: str

@dataclass
class EquityForward:
    strike: float
    time_to_maturity: float

def bsm_pricer(underlying: Equity, option: EquityOption, rate: float) -> float:
    """
    Price an option using Black Scholes Merton Model with continuous dividends
    """
    S = underlying.spot
    K = option.strike
    T = option.time_to_maturity
    r = rate
    q = underlying.dividend_yield
    sigma = underlying.volatility

    if T <= 0:
        if option.put_call.lower() == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option.put_call.lower() == "call":
        price = S * np.exp(-q*T) * norm.cdf(d1) - K*np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K*np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)

    return price

def bsm_delta(underlying: Equity, option: Equity, rate: float, bump=0.001)->float:
    """
    Numerically approximate the delta (the first derivative of price with respect to the spot). Use a relative bump of 0.001.
    """
    original_spot = underlying.spot
    bump_up_spot = original_spot * (1 + bump)
    bump_down_spot = original_spot * (1 - bump)

    underlying_up = Equity(bump_up_spot, underlying.dividend_yield, underlying.volatility)
    underlying_down = Equity(bump_down_spot, underlying.dividend_yield, underlying.volatility)

    price_up = bsm_pricer(underlying_up, option, rate)
    price_down = bsm_pricer(underlying_down, option, rate)

    delta = (price_up - price_down) / (2 * (bump * original_spot))
    return delta

def bsm_gamma(underlying: Equity, option: EquityOption, rate: float, bump=0.001) -> float:
    """
    Numerically approximate gamma as the second derivative of price with respect to the spot. Use a relative bump of 0.001.
    """
    original_spot = underlying.spot
    bump_up_spot = original_spot * (1 + bump)
    bump_down_spot = original_spot * (1 - bump)

    underlying_up = Equity(bump_up_spot, underlying.dividend_yield, underlying.volatility)
    underlying_down = Equity(bump_down_spot, underlying.dividend_yield, underlying.volatility)

    price_up = bsm_pricer(underlying_up, option, rate)
    price_mid = bsm_pricer(underlying, option, rate)
    price_down = bsm_pricer(underlying_down, option, rate)

    gamma = (price_up + price_down - 2 * price_mid) / ((bump * original_spot)**2)
    return gamma

def fwd_pricer(underlying: Equity, forward: EquityForward, rate: float) -> float:
    S = underlying.spot
    q = underlying.dividend_yield
    r = rate
    K = forward.strike
    T = forward.time_to_maturity

    forward_value = S*np.exp((r - q)*T) - K*np.exp(-r*T)
    return forward_value

equity_example = Equity(4450, 0.0, 1e-9)
option_example = EquityOption(5000, 0.25, "put")
test_price = bsm_pricer(equity_example, option_example, 0.000)
print(test_price)

























