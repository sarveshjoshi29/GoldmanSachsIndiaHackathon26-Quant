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

def winsorize_returns(returns, lower_pct=0.01, upper_pct=0.99):
    """
    Caps extreme return outliers at the specified percentiles per instrument.
    """
    lower_bounds = returns.quantile(lower_pct)
    upper_bounds = returns.quantile(upper_pct)
    
    # Clip limits extreme values to the bounds without dropping the dates
    cleaned_returns = returns.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    return cleaned_returns



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
        1.0 - pnl_port.add(pnl_hedge, fill_value=0.0).var(ddof=0) / var_p
    )

#######################################################################################################################
# YOUR CODE STARTS HERE
#######################################################################################################################

import pickle
from sklearn.linear_model import ElasticNetCV, HuberRegressor
from sklearn.model_selection import TimeSeriesSplit

RAW_RETURNS = compute_returns(EOD_PRICES)
RETURNS = winsorize_returns(RAW_RETURNS)

def robust_relaxed_elastic_net(proxy_cols, target_cols, n_splits=5):
    """
    Applies a Robust Relaxed Elastic Net approach:
    1. Uses ElasticNetCV to select a sparse, yet collinearity-aware subset of features.
    2. Uses HuberRegressor to unshrink the weights robustly, ignoring fat-tailed market outliers.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    X = RETURNS[proxy_cols].values
    n_p = len(proxy_cols)
    W = np.zeros((len(target_cols), n_p))

    l1_ratios_to_test = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

    for i, tgt in enumerate(target_cols):
        y = RETURNS[tgt].values

        # Step 1: Feature selection (sweep l1_ratio to trade off sparsity vs stability)
        m = ElasticNetCV(
            l1_ratio=l1_ratios_to_test,
            cv=tscv,
            fit_intercept=False,
            max_iter=100000,
            random_state=42,
        ).fit(X, y)
        support = m.coef_ != 0

        # Step 2: Unshrink weights robustly
        if np.sum(support) > 0:
            X_sel = X[:, support]
            huber = HuberRegressor(fit_intercept=False, epsilon=1.35).fit(X_sel, y)
            W[i, support] = huber.coef_

    return pd.DataFrame(W, index=target_cols, columns=proxy_cols)

# ------- NAV + Risk submissions ---------------------------------------------
nav_weights  = robust_relaxed_elastic_net(ALL_PROXY, TARGET_ETFS)
risk_weights = robust_relaxed_elastic_net(RISK_PROXY, TARGET_ETFS)

nav_weights.index.name  = "ETF"
risk_weights.index.name = "ETF"

with open("./nav_weights.pkl", "wb") as f:
    pickle.dump(nav_weights, f)
with open("./risk_weights.pkl", "wb") as f:
    pickle.dump(risk_weights, f)

# ---------------------------------------------------------------------------
# Test-time `portfolio` extractor
# ---------------------------------------------------------------------------
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
        import sys
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw.strip():
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    if isinstance(obj.get("portfolio"), dict):
                        return obj["portfolio"]
                    if obj and all(isinstance(k, str) and k.startswith("Target_ETF") for k in obj):
                        return obj
    except Exception:
        pass
    return {}

_portfolio_input = _get_portfolio()

# ---------------------------------------------------------------------------
# HEDGE STRATEGY - Performance-Driven Robust Relaxed Elastic Net 
# ---------------------------------------------------------------------------
def effective_pnl_matrix(prices, universe):
    """Per-1-mm-USD daily PnL series for every proxy in `universe`, aligned."""
    p0 = prices.iloc[0]
    r = compute_returns(prices)
    cols = {}
    for j in universe:
        if j in PAR_QUOTED:
            cols[j] = (p0[j] / 100.0) * r[j]
        else:
            cols[j] = r[j]
    return pd.DataFrame(cols).dropna()

EFF_PNL = effective_pnl_matrix(EOD_PRICES, ALL_PROXY)

# IMPORTANT: Reverted to ALL_PROXY to allow Proxy ETFs to maximize HER
def build_hedge_performance_driven(portfolio_input, fallback_index=ALL_PROXY, n_splits=5):
    if not portfolio_input:
        out = pd.DataFrame({"Notional": [0.0] * len(fallback_index)}, index=fallback_index)
        out.index.name = "Instrument"
        return out

    port_pnl = pd.Series(0.0, index=RETURNS.index)
    for tgt, notional in portfolio_input.items():
        if tgt in RETURNS.columns:
            port_pnl = port_pnl.add(notional * RETURNS[tgt], fill_value=0.0)

    # Restored full ALL_PROXY universe
    eff = EFF_PNL[ALL_PROXY].reindex(port_pnl.index).dropna().sort_index()
    y   = -port_pnl.loc[eff.index].values
    X   = eff.values

    l1_ratios_to_test = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]

    # Step 1: Selection via Elastic Net (walk-forward CV)
    # Removed artificial cost scaling matrix (X / costs) to allow pure variance reduction mapping
    tscv = TimeSeriesSplit(n_splits=n_splits)
    m = ElasticNetCV(
        l1_ratio=l1_ratios_to_test,
        cv=tscv,
        fit_intercept=False,
        max_iter=100000,
        random_state=42,
    ).fit(X, y)
    support = m.coef_ != 0

    h = np.zeros(X.shape[1])
    if np.sum(support) > 0:
        # Step 2: Robust Unshrinking
        X_sel = X[:, support]
        huber = HuberRegressor(fit_intercept=False, epsilon=1.35).fit(X_sel, y)
        h[support] = huber.coef_

    out = pd.DataFrame({"Notional": h}, index=eff.columns)
    out.index.name = "Instrument"
    out = out[out["Notional"] != 0.0]

    if out.empty:
        out = pd.DataFrame({"Notional": [0.0] * len(fallback_index)}, index=fallback_index)
        out.index.name = "Instrument"

    return out

HEDGE_MOCK = build_hedge_performance_driven(_portfolio_input)

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
    "hedge": hedging_basket.to_csv(),
}

final_output = json.dumps(submission)
print(final_output)