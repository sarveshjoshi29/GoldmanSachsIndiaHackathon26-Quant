import pandas as pd
import numpy as np
import json
import sys
import warnings
from sklearn.linear_model import ElasticNetCV, RidgeCV, LassoCV

warnings.filterwarnings('ignore')

##### DO NOT MODIFY/DELETE THE BELOW CODE #############################################################################

#######################################################################################################################
# CONSTANTS STARTS
# BELOW ARE "NECESSARY" CONSTANTS
######################################################################################################################=

"""End-Of-Day (EOD ) prices of target ETFs and proxy instruments, you have to use this to build whatever model you wan to."""
EOD_PRICES     = pd.read_csv( "./eod_prices.csv", index_col = 0, parse_dates=True ).sort_index()

# Defn. of Instruments - use these while constructing your output dataframes
TARGET_ETFS  = [f"Target_ETF_{i}" for i in range(1, 6)]

# PROXY INSTRUMENTS AVAILABLE

# 1). ETFs
PROXY_ETFS   = [f"Proxy_ETF_{i}" for i in range(1, 11)]

# 2). TREASURY FUTURES
TSY_FUTURES  = ["TU", "FV", "TY", "US"]

# 3). TREASURY OTR ("one-the-run") BONDS
TSY_BONDS    = ["UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"]

# 4). CREDIT DEFAULT SWAP INDICES
CDX          = ["CDX_IG_5Y", "CDX_IG_10Y", "CDX_HY_5Y", "CDX_HY_10Y"]

"""You can use all proxy instruments for 'NAV' but ONLY 'RISK_PROXY' instruments for building proxy model for 'Risk'."""

# AVAILABE PROXY INSTRUMENTS FOR NAV MODEL ( = ALL PROXY INSTRUMENTS )
ALL_PROXY    = PROXY_ETFS + TSY_FUTURES + TSY_BONDS + CDX

# AVAILABE PROXY INSTRUMENTS FOR RISK MODEL ( = ALL - PROXT ETFs INSTRUMENTS )
RISK_PROXY   = TSY_FUTURES + TSY_BONDS + CDX

# Par-quoted instruments (price per 100); ETFs are NAV per share
# EOD_PRICES has price per 100 par notional for PAR_QUOTED instruments, whereas, for ETFs (target & proxy), it's per share == NAV
PAR_QUOTED   = set(TSY_FUTURES + TSY_BONDS + CDX)

# Transaction cost for instruemnts, this will be applied on your hedging basket
# One-way transaction cost (bps on |market value|)
COST_BPS = {**{e: 2.0 for e in PROXY_ETFS},
            **{f: 0.5 for f in TSY_FUTURES},
            **{b: 0.5 for b in TSY_BONDS},
            **{c: 1.0 for c in CDX}}

#######################################################################################################################
# CONSTANTS END
#######################################################################################################################

##### DO NOT MODIFY/DELETE THE BELOW CODE #############################################################################

#######################################################################################################################
# HELPER FUNCTIONS

# BELOW FUNCTIONS ARE PROVIDED AS BOILER-PLATE CODE
# THESE FUNCTIONS ARE ALSO USED IN EVALUATION, RECOMMENDED TO USE THEM, BUT NOT MANDATORY
#######################################################################################################################

#######################################################################################################################
# HELPER FUNCTIONS START
#######################################################################################################################

def compute_returns(prices):
    """Compute simple daily returns from prices."""
    return prices.pct_change().dropna()

def predict_returns(weights, proxy_returns):
    """Predict target ETF returns from proxy returns and model weights."""
    return proxy_returns @ weights

def reconstruct_nav(weights, proxy_returns, base_nav):
    """Reconstruct NAV level from predicted returns chained off a base NAV."""
    pred_returns = predict_returns(weights, proxy_returns)
    return (1 + pred_returns).cumprod() * base_nav

def compute_mape(predicted_nav, actual_nav):
    """Compute mean absolute percentage error between predicted and actual NAV."""
    aligned = pd.concat([predicted_nav, actual_nav], axis=1, join="inner").dropna()
    aligned.columns = ["predicted", "actual"]
    return (abs(aligned["predicted"] - aligned["actual"]) / aligned["actual"]).mean() * 100
  
def compute_pnl_series(notionals: dict, prices: pd.DataFrame) -> pd.Series:
    """mm USD PnL_t = sum_i scaling_i * (P_i,t - P_i,t-1)."""
    if not notionals:
        return pd.Series(0.0, index=prices.index[1:])
    p0 = prices.iloc[0]
    scaled = pd.Series({c: n * (1.0 / p0[c] if c not in PAR_QUOTED else 1.0 / 100.0)
                        for c, n in notionals.items()})
    return prices[scaled.index].diff().iloc[1:].mul(scaled, axis=1).sum(axis=1)

def compute_cost(notionals: dict, prices: pd.DataFrame) -> float:
    """
    compute transaction cost of setting up the hedging basket
    One-way cost on |market value| at t=0:
      ETF : |MV| * bps / 1e4                       (Notional IS MV)
      Par : |N|/100 * P_0 * bps / 1e4              (par * dirty price / 100)
    """
    p0 = prices.iloc[0]
    return float(sum(
        abs(n) * COST_BPS[i] / 1e4 if i not in PAR_QUOTED
        else (abs(n) / 100.0) * p0[i] * COST_BPS[i] / 1e4
        for i, n in notionals.items()
    ))

def compute_her(pnl_port: pd.Series, pnl_hedge: pd.Series) -> float:
    """compute hedge effectiveness ratio.""" 
    var_p = pnl_port.var(ddof=0)
    return 0.0 if var_p <= 0 else float(
        1.0 - pnl_port.add(pnl_hedge, fill_value=0.0).var(ddof=0) / var_p)

#######################################################################################################################
# HELPER FUNCTIONS END
#######################################################################################################################


#######################################################################################################################
# YOUR CODE STARTS HERE
#######################################################################################################################

RETURNS = compute_returns(EOD_PRICES)

# ---------------------------------------------------------------------------
# Part 1: NAV Weights -> ElasticNetCV
# Fixes OLS overfitting by mixing L1 and L2 penalties automatically.
# ---------------------------------------------------------------------------
def optimize_nav_weights(proxy_cols, target_cols):
    X = RETURNS[proxy_cols].values
    Y = RETURNS[target_cols].values
    df = pd.DataFrame(index=target_cols, columns=proxy_cols)
    
    for i, target in enumerate(target_cols):
        # Removed n_jobs=-1 to prevent HackerRank multiprocessing crash
        model = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 1.0], fit_intercept=False, cv=5)
        model.fit(X, Y[:, i])
        df.loc[target] = model.coef_
    return df

nav_weights = optimize_nav_weights(ALL_PROXY, TARGET_ETFS)


# ---------------------------------------------------------------------------
# Part 2: Risk Weights -> RidgeCV
# Fixes the 'alpha=1e-4' issue by testing a wide logspace to properly penalize.
# ---------------------------------------------------------------------------
def optimize_risk_weights(proxy_cols, target_cols):
    X = RETURNS[proxy_cols].values
    Y = RETURNS[target_cols].values
    df = pd.DataFrame(index=target_cols, columns=proxy_cols)
    
    # Test alphas from 0.01 to 100 to find the actual best regularization
    alphas_to_test = np.logspace(-2, 2, 10)
    
    for i, target in enumerate(target_cols):
        model = RidgeCV(alphas=alphas_to_test, fit_intercept=False, cv=5)
        model.fit(X, Y[:, i])
        df.loc[target] = model.coef_
    return df

risk_weights = optimize_risk_weights(RISK_PROXY, TARGET_ETFS)


# ---------------------------------------------------------------------------
# Robustly extract `portfolio`
# ---------------------------------------------------------------------------
def _get_portfolio():
    try:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
            if raw.strip():
                obj = json.loads(raw)
                if isinstance(obj, dict) and isinstance(obj.get("portfolio"), dict):
                    return obj["portfolio"]
    except Exception:
        pass

    g = globals()
    if isinstance(g.get("portfolio"), dict):
        return g["portfolio"]
    for name in ("test_input", "input_json", "_input"):
        v = g.get(name)
        if isinstance(v, dict) and isinstance(v.get("portfolio"), dict):
            return v["portfolio"]
    return {}

_portfolio_input = _get_portfolio()


# ---------------------------------------------------------------------------
# Part 3: Optimal Hedging -> Lasso Regression on Dollar PnL
# Directly optimizes HER and Tracking Error while punishing transaction costs.
# ---------------------------------------------------------------------------
def build_optimal_hedge(portfolio_input):
    if not portfolio_input:
        df = pd.DataFrame(columns=['Notional'])
        df.index.name = 'Instrument'
        return df

    # 1. Get the historic dollar PnL of our actual starting portfolio
    port_pnl = compute_pnl_series(portfolio_input, EOD_PRICES)
    
    # 2. Get the historic dollar PnL for exactly 1.0 Notional of every proxy
    proxy_pnls = {}
    for p in ALL_PROXY:
        proxy_pnls[p] = compute_pnl_series({p: 1.0}, EOD_PRICES)
        
    X_pnl = pd.DataFrame(proxy_pnls)
    
    # We want our hedge to equal the INVERSE of the portfolio PnL
    y_target = -port_pnl.values 
    
    # 3. LassoCV automatically drops instruments that don't add enough value 
    # Removed n_jobs=-1 to prevent HackerRank multiprocessing crash
    model = LassoCV(fit_intercept=False, cv=5, max_iter=5000)
    
    # If the portfolio is effectively flat, avoid regression errors
    if np.abs(y_target).max() < 1e-6:
        df = pd.DataFrame(columns=['Notional'])
        df.index.name = 'Instrument'
        return df
        
    model.fit(X_pnl.values, y_target)
    
    # 4. Because X is built on 1.0 unit of notional, the coefficients ARE the notionals
    hedging_dict = {}
    for i, proxy in enumerate(ALL_PROXY):
        weight = model.coef_[i]
        if abs(weight) > 1e-2:  # Prune dust
            hedging_dict[proxy] = weight
            
    out = pd.DataFrame.from_dict(hedging_dict, orient='index', columns=['Notional'])
    out.index.name = 'Instrument'
    return out

HEDGE_MOCK = build_optimal_hedge(_portfolio_input)

#######################################################################################################################
# YOUR CODE ENDS HERE
#######################################################################################################################

#######################################################################################################################
# SUBMISSION FORMAT - ASSIGN YOUR OUTPUT PANDAS DATAFRAMES BELOW
# YOU DO NOT NEED TO MODIFY THE DATAFRAMES FOR TASKS WHICH YOU HAVE NOT SOLVED, WE'LL DO PARTIAL GRADING 
# e.g., IF YOU'VE ONLY SOLVED THE 'NAV' TASK, JUST ASSIGN YOUR SOLUTION FOR IT TO 'nav_model' VARIABLE, AND LEAVE THE # # OTHER TWO UNTOUCHED
#######################################################################################################################

# TASK 1 - NAV ESTIMATION
nav_model = nav_weights

# TASK 2 - RISK ESTIMATION
risk_model = risk_weights

# TASK 3 - HEDGE CREATION 
hedging_basket = HEDGE_MOCK

#######################################################################################################################
# SUBMISSION FORMAT - DO NOT MODIFY/DELETE THE BELOW CODE
#######################################################################################################################

submission = {
  "nav": nav_model.to_csv(),
  "risk": risk_model.to_csv(),
  "hedge": hedging_basket.to_csv() }

final_output = json.dumps( submission )
print( final_output )