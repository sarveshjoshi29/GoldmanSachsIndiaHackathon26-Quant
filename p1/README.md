# Problem 1 — Regime Navigator: Adaptive Portfolio Construction

Goldman Sachs India Hackathon 2026, Quant Challenge.

Task: implement `PortfolioArchitect.allocate()`, called once per rebalance
date (~40 calls/test case). Return a weight vector over 100 assets
(SEC_001..SEC_100), max 50 non-zero positions, `|w_i| <= 0.10`.

Score progression (grader score per submission):

| File | Score | What changed |
|---|---|---|
| `sub1.py` | 23.37 | Baseline. 3-state regime-adaptive portfolio: cross-sectional factor ranks (60d momentum, 5d reversal, 20d realized vol, 20d liquidity proxy, value/quality from fundamentals) + Risk Parity sizing. Regime detected via 3-component `GaussianMixture` on EWMA-smoothed (span=40) indicators, components hand-mapped to risk_on/neutral/risk_off. |
| `sub2.py` | 32.51 | Dropped the GMM in favor of a rule-based regime switch: rolling 60d stress score + drawdown + market-trend checks pick calm vs stressed. Calm = long-only, slower turnover. Stress = adds a limited long/short overlay to cut market beta. Added turnover cap + trade deadband to reduce churn between rebalances. |
| `sub3.py` | 33.78 (`v4`) | Back to GMM-based regime detection, cleaned up. Position sizing switched to inverse-volatility risk parity, explicit `scipy.optimize.minimize` fallback path added for the weight solve. |
| `sub5.py` | 32.94 (`v5.0`) | Architecture change: dropped discrete regime buckets entirely for a **continuous stress spectrum** in `[0, 1]` (GMM + EMA). Every portfolio parameter (risk aversion, turnover penalty, alpha scale, smoothing, net exposure bounds, position cap, deadband) is now `np.interp`'d against this spectrum instead of switched on 3 fixed states. |
| `sub7.py` | 40.27 | Same continuous-spectrum architecture as sub5, re-tuned interpolation breakpoints (risk_aversion, turn_penalty, position cap curves) for better score. |
| `sub9.py` | 43.25 (final) | Final tuning pass on the same architecture: raised inverse-vol exponent (1.0 → 1.3), tightened `pos_cap` in stress regimes (0.2 → 0.005), pushed `risk_aversion`/`turn_penalty` much higher at high stress (150→300 / 30→100) to de-risk harder in tail regimes. |

Supporting analysis (attribution, turnover, exposure/weight transition, regime
plots) is in `../charts/`.
