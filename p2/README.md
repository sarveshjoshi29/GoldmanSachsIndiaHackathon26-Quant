# Problem 2 — ETF Proxy Hedging & NAV Replication

Goldman Sachs India Hackathon 2026, Quant Challenge.

Task: for 5 target ETFs, (1) replicate NAV from a universe of proxy ETFs,
treasury futures, treasury bonds, and CDX; (2) build a risk-only replication
over the risk proxies; (3) construct a hedge basket that offsets a given
portfolio's daily P&L. Transaction costs vary by instrument class (2.0bps
proxy ETFs, 0.5bps rates, 1.0bps CDX).

Score progression (grader score per submission):

| File | Score | What changed |
|---|---|---|
| `sub1.py` | 27.32 | Baseline. Constrained-weights NAV/risk fit via `scipy.optimize.minimize` (small L2 penalty for stability), hedge built as risk-only replication with par-quoted notional rescaling. |
| `sub2.py` | 32.85 | Replaced the custom optimizer with sklearn: `RidgeCV` for risk weights, `LassoCV` on dollar P&L for the hedge (`build_optimal_hedge` — hedge targets the *inverse* of portfolio P&L directly, LassoCV auto-drops weak instruments). |
| `sub3.py` | 58.09 | Big jump. Switched NAV/risk fit to sparse, sign-free `LassoCV` per target (`lasso_weights`) over the full proxy universe — L1 penalty kills irrelevant legs. Hedge basket rebuilt on dense OLS risk-weights instead of the Lasso-on-P&L approach (better hedge effectiveness ratio). |
| `sub4.py` | 67.99 | New hedge strategy: `build_hedge_minvar` — direct min-variance `LassoCV` regression on the full proxy universe's effective P&L matrix, sign-free, sparse by construction (fewer legs → lower transaction cost). NAV/risk weights unchanged from sub3. |
| `sub5.py` | 67.56 | Switched Lasso's CV strategy from plain 5-fold to `TimeSeriesSplit` (walk-forward), avoiding look-ahead in the alpha selection. Added (commented-out) alpha-sweep variant for manual tuning. |
| `sub10.py` | 71.47 (final) | Replaced Lasso with a **Robust Relaxed Elastic Net** (`robust_relaxed_elastic_net`): `ElasticNetCV` for sparse, collinearity-aware feature selection, feeding a performance-driven hedge builder (`build_hedge_performance_driven`) with walk-forward CV. Added return winsorization to limit outlier influence on the fit. |
