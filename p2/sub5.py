import pandas as pd
import numpy as np
import json
import sys
import warnings
warnings.filterwarnings('ignore')

##### DO NOT MODIFY/DELETE THE BELOW CODE #############################################################################

#######################################################################################################################
# CONSTANTS
#######################################################################################################################

EOD_PRICES   = pd.read_csv("./eod_prices.csv", index_col=0, parse_dates=True).sort_index()

TARGET_ETFS  = [f"Target_ETF_{i}" for i in range(1, 6)]
PROXY_ETFS   = [f"Proxy_ETF_{i}" for i in range(1, 11)]
TSY_FUTURES  = ["TU", "FV", "TY", "US"]
TSY_BONDS    = ["UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"]
CDX          = ["CDX_IG_5Y", "CDX_IG_10Y", "CDX_HY_5Y", "CDX_HY_10Y"]

ALL_PROXY    = PROXY_ETFS + TSY_FUTURES + TSY_BONDS + CDX
RISK_PROXY   = TSY_FUTURES + TSY_BONDS + CDX
PAR_QUOTED   = set(TSY_FUTURES + TSY_BONDS + CDX)

COST_BPS = {**{e: 2.0 for e in PROXY_ETFS},
            **{f: 0.5 for f in TSY_FUTURES},
            **{b: 0.5 for b in TSY_BONDS},
            **{c: 1.0 for c in CDX}}

#######################################################################################################################
# HELPER FUNCTIONS
#######################################################################################################################

def compute_returns(prices):
    return prices.pct_change().dropna()

def predict_returns(weights, proxy_returns):
    return proxy_returns @ weights

def reconstruct_nav(weights, proxy_returns, base_nav):
    pred_returns = predict_returns(weights, proxy_returns)
    return (1 + pred_returns).cumprod() * base_nav

def compute_mape(predicted_nav, actual_nav):
    aligned = pd.concat([predicted_nav, actual_nav], axis=1, join="inner").dropna()
    aligned.columns = ["predicted", "actual"]
    return (abs(aligned["predicted"] - aligned["actual"]) / aligned["actual"]).mean() * 100

def compute_pnl_series(notionals: dict, prices: pd.DataFrame) -> pd.Series:
    if not notionals:
        return pd.Series(0.0, index=prices.index[1:])
    p0 = prices.iloc[0]
    scaled = pd.Series({c: n * (1.0 / p0[c] if c not in PAR_QUOTED else 1.0 / 100.0)
                        for c, n in notionals.items()})
    return prices[scaled.index].diff().iloc[1:].mul(scaled, axis=1).sum(axis=1)

def compute_cost(notionals: dict, prices: pd.DataFrame) -> float:
    p0 = prices.iloc[0]
    return float(sum(
        abs(n) * COST_BPS[i] / 1e4 if i not in PAR_QUOTED
        else (abs(n) / 100.0) * p0[i] * COST_BPS[i] / 1e4
        for i, n in notionals.items()
    ))

def compute_her(pnl_port: pd.Series, pnl_hedge: pd.Series) -> float:
    var_p = pnl_port.var(ddof=0)
    return 0.0 if var_p <= 0 else float(
        1.0 - pnl_port.add(pnl_hedge, fill_value=0.0).var(ddof=0) / var_p)

#######################################################################################################################
# YOUR CODE STARTS HERE
#######################################################################################################################

import pickle
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import TimeSeriesSplit

RETURNS = compute_returns(EOD_PRICES)

# Chronological CV - only past data used to predict future data per fold.
# Standard KFold (cv=5) shuffles blocks of the timeline, leaking future returns
# into training; TimeSeriesSplit forces walk-forward order and matches how the
# hidden test period will actually be evaluated.
TSCV = TimeSeriesSplit(n_splits=5)


def ols_weights(proxy_cols, target_cols):
    X = RETURNS[proxy_cols].values
    Y = RETURNS[target_cols].values
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return pd.DataFrame(W.T, index=target_cols, columns=proxy_cols)


def lasso_weights(proxy_cols, target_cols, positive=False, normalise=False):
    X = RETURNS[proxy_cols].values
    n_p = len(proxy_cols)
    W = np.zeros((len(target_cols), n_p))
    for i, tgt in enumerate(target_cols):
        y = RETURNS[tgt].values
        m = LassoCV(cv=TSCV, fit_intercept=False, positive=positive,
                    max_iter=50000, n_alphas=80, random_state=0).fit(X, y)
        w = m.coef_
        if normalise:
            s = w.sum()
            if s > 0:
                w = w / s
        W[i] = w
    return pd.DataFrame(W, index=target_cols, columns=proxy_cols)


# ------- NAV + Risk submissions ----------------------------------------------
nav_weights  = lasso_weights(ALL_PROXY, TARGET_ETFS,
                             positive=False, normalise=False)
risk_weights = lasso_weights(RISK_PROXY, TARGET_ETFS,
                             positive=True,  normalise=True)
nav_weights.index.name  = "ETF"
risk_weights.index.name = "ETF"

with open("./nav_weights.pkl", "wb") as f:
    pickle.dump(nav_weights, f)
with open("./risk_weights.pkl", "wb") as f:
    pickle.dump(risk_weights, f)


def _get_portfolio():
    g = globals()
    if isinstance(g.get("portfolio"), dict):
        return g["portfolio"]
    for name in ("test_input", "input_json", "_input", "test_case", "TEST_CASE"):
        v = g.get(name)
        if isinstance(v, dict) and isinstance(v.get("portfolio"), dict):
            return v["portfolio"]
        if isinstance(v, str):
            try:
                obj = json.loads(v)
                if isinstance(obj, dict) and isinstance(obj.get("portfolio"), dict):
                    return obj["portfolio"]
            except Exception:
                pass
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw.strip():
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    if isinstance(obj.get("portfolio"), dict):
                        return obj["portfolio"]
                    if obj and all(isinstance(k, str) and k.startswith("Target_ETF")
                                   for k in obj):
                        return obj
    except Exception:
        pass
    return {}


_portfolio_input = _get_portfolio()


# ---------------------------------------------------------------------------
# HEDGE - LassoCV (TimeSeriesSplit) over ALL_PROXY; same as sub_4 but with
# walk-forward CV so the chosen alpha generalises forward in time.
# ---------------------------------------------------------------------------

def _effective_pnl_for_lasso(prices, universe):
    """PnL per 1 mm USD for each proxy: r_j (ETF) or (p0/100)*r_j (par-quoted)."""
    p0 = prices.iloc[0]
    r  = compute_returns(prices)
    cols = {j: ((p0[j] / 100.0) * r[j] if j in PAR_QUOTED else r[j])
            for j in universe}
    return pd.DataFrame(cols).dropna()


EFF_PNL = _effective_pnl_for_lasso(EOD_PRICES, ALL_PROXY)


def build_hedge_minvar(portfolio_input, fallback_index=ALL_PROXY):
    if not portfolio_input:
        out = pd.DataFrame({"Notional": [0.0] * len(fallback_index)},
                           index=fallback_index)
        out.index.name = "Instrument"
        return out

    port_pnl = compute_pnl_series(portfolio_input, EOD_PRICES)
    eff = EFF_PNL.reindex(port_pnl.index).dropna()
    y = -port_pnl.loc[eff.index].values
    X = eff.values

    m = LassoCV(cv=TSCV, fit_intercept=False, max_iter=100000,
                n_alphas=80, random_state=0).fit(X, y)
    h = m.coef_

    out = pd.DataFrame({"Notional": h}, index=eff.columns)
    out.index.name = "Instrument"
    out = out[out["Notional"] != 0.0]
    if out.empty:
        out = pd.DataFrame({"Notional": [0.0] * len(fallback_index)},
                           index=fallback_index)
        out.index.name = "Instrument"
    return out


HEDGE_MOCK = build_hedge_minvar(_portfolio_input)


# ---------------------------------------------------------------------------
# Manual alpha sweep version (kept for later experimentation).
# To enable: comment out the LassoCV block above and uncomment everything
# below, then replace the HEDGE_MOCK assignment with the sweep call.
# ---------------------------------------------------------------------------
# ALPHA_GRID  = np.logspace(-8, -2, 40)
# LAMBDA_TE   = 0.5
# LAMBDA_COST = 1.0
#
# def build_hedge_minvar_sweep(portfolio_input,
#                               alphas=ALPHA_GRID,
#                               lambda_te=LAMBDA_TE,
#                               lambda_cost=LAMBDA_COST,
#                               fallback_index=ALL_PROXY):
#     if not portfolio_input:
#         out = pd.DataFrame({"Notional": [0.0] * len(fallback_index)},
#                            index=fallback_index)
#         out.index.name = "Instrument"
#         return out
#
#     port_pnl = compute_pnl_series(portfolio_input, EOD_PRICES)
#     eff = EFF_PNL.reindex(port_pnl.index).dropna()
#     port_pnl_for_fit = port_pnl.loc[eff.index]
#     X = eff.values
#     y = -port_pnl_for_fit.values
#
#     gross_notional = sum(abs(v) for v in portfolio_input.values())
#     gross_notional = max(gross_notional, 1e-9)
#     port_std = float(port_pnl.std(ddof=0))
#
#     best = None
#     for alpha in alphas:
#         m = Lasso(alpha=alpha, fit_intercept=False,
#                   max_iter=100000, random_state=0).fit(X, y)
#         h = m.coef_
#         notionals = {c: float(v) for c, v in zip(eff.columns, h) if v != 0.0}
#         if not notionals:
#             continue
#
#         pnl_hedge = compute_pnl_series(notionals, EOD_PRICES)
#         her       = compute_her(port_pnl, pnl_hedge)
#         cost      = compute_cost(notionals, EOD_PRICES)
#
#         net_pnl = port_pnl.add(pnl_hedge, fill_value=0.0)
#         te = float(net_pnl.std(ddof=0) / port_std) if port_std > 0 else 1.0
#
#         rel_cost = cost / gross_notional
#         score = her - lambda_te * te - lambda_cost * rel_cost
#
#         if (best is None) or (score > best["score"]):
#             best = {"score": score, "alpha": alpha, "notionals": notionals,
#                     "her": her, "te": te, "cost": cost}
#
#     if best is None:
#         out = pd.DataFrame({"Notional": [0.0] * len(fallback_index)},
#                            index=fallback_index)
#         out.index.name = "Instrument"
#         return out
#
#     print(f"[hedge] picked alpha={best['alpha']:.2e}  HER={best['her']:.4f}  "
#           f"TE={best['te']:.4f}  cost={best['cost']:.4f}  score={best['score']:.4f}",
#           file=sys.stderr)
#
#     out = pd.DataFrame.from_dict(best["notionals"], orient="index",
#                                  columns=["Notional"])
#     out.index.name = "Instrument"
#     return out
#
# HEDGE_MOCK = build_hedge_minvar_sweep(_portfolio_input)

#######################################################################################################################
# YOUR CODE ENDS HERE
#######################################################################################################################

# TASK 1 - NAV ESTIMATION
nav_model = nav_weights

# TASK 2 - RISK ESTIMATION
risk_model = risk_weights

# TASK 3 - HEDGE CREATION
hedging_basket = HEDGE_MOCK

submission = {
  "nav": nav_model.to_csv(),
  "risk": risk_model.to_csv(),
  "hedge": hedging_basket.to_csv() }

final_output = json.dumps(submission)
print(final_output)