"""
Regime Navigator v4: Adaptive Portfolio Construction
====================================================
Goldman Sachs India Hackathon 2026  -  Quant Challenge

INSTRUCTIONS
------------
1. Implement the PortfolioArchitect class (the section marked YOUR CODE HERE).
2. Do NOT modify the main() runner at the bottom.
3. Your allocate() is called once per rebalance date (~40 times per test case).
4. Return a numpy array of shape (100,)  -  one weight per asset (SEC_001..SEC_100).
5. Max 50 non-zero positions, |w_i| <= 0.10 per asset.
6. Include a STRATEGY EXPLANATION block (see PROBLEM_STATEMENT.pdf)  -  required.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import argparse
import io
import json
import re
import sys
import numpy as np
import pandas as pd

try:
    from scipy.optimize import minimize as _scipy_minimize
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ======================================================================
# YOUR IMPLEMENTATION
# ======================================================================

"""STRATEGY EXPLANATION
Core approach:
This implementation builds a regime-adaptive, rule-based portfolio using cross-sectional
fundamental and technical factor ranks. Position sizing is risk-parity oriented (inverse
volatility) so that high-volatility names receive smaller weight shares.

Signals used:
- Momentum (60-day trailing) and short-term Reversal (5-day).
- Realized volatility (20-day) and liquidity (20-day average volume proxy).
- Simple fundamental aggregates for Value and Quality computed from the most recent
    fundamentals available per asset.

Regime detection (Rolling Stress Z-Score):
- Rather than retraining a clustering model, the code computes Z-scores of key stress
    indicators (implied volatility, high-yield credit spread, funding stress) relative to
    a recent history (up to 252 rows). The mean Z across available stress series maps to
    regimes using thresholds in code:
    - avg Z > 1.0  -> "crisis"
    - avg Z >= 0.4 -> "risk_off"
    - avg Z <= -0.4 -> "risk_on"
    - otherwise -> "neutral"

Portfolio construction and sizing:
- Cross-sectional features are converted to percentile ranks in [-0.5, 0.5] and
    combined into a regime-weighted composite `score` (the alpha vector).
- A heuristic warm-start picks top/bottom names sized by inverse volatility, then
    feeds a convex optimizer that produces the final weights.

- Utilizes Optimizer (SLSQP convex QP, downside-aware):


Hard post-processing (`_enforce_constraints`):
- Max 50 names, sector cap 30%, short floor -0.30, gross cap 1.50.

Regime knobs (net band / risk_aversion / turn_penalty / alpha_scale):
- Risk-On:  [0.95, 1.10] / 25  / 0.8 / 0.45   (lean into alpha).
- Neutral:  [0.90, 1.05] / 40  / 1.2 / 0.40.
- Risk-Off: [0.85, 0.95] / 70  / 1.8 / 0.35   (shorts on, heavier risk weight).
- Crisis:   [0.85, 0.90] / 120 / 2.5 / 0.30   (max risk aversion, sticky book).

Constraints and execution details:
- Max 50 non-zero positions; per-asset absolute cap 10%; sector absolute cap 30%.
- Short exposure floored to -0.30 (total); gross exposure capped at 1.50.
- Turnover is reduced via the optimizer's L1 penalty, exponential smoothing toward
    previous weights (`smooth_lam` varies by regime), and a 2% trade deadband.
"""


class PortfolioArchitect:
    def __init__(self,
                 prices: pd.DataFrame,
                 fundamentals: pd.DataFrame,
                 indicators: pd.DataFrame):

        self.n_assets = 100
        self.all_assets = sorted(prices['asset_id'].unique())

        # Factor lookback windows
        self._mom_lb = 100
        self._rev_lb = 15
        self._vol_lb = 20
        self._liq_lb = 20

        # Turnover protection state
        self._prev_w = pd.Series(0.0, index=self.all_assets)
        self._prev_date = None

        # Sector fallback so missing fundamentals don't break sector caps
        self._sector_fallback = {}
        if fundamentals is not None and len(fundamentals) > 0 and "sector" in fundamentals.columns:
            latest_f = (fundamentals
                        .sort_values("report_date")
                        .drop_duplicates("asset_id", keep="last"))
            self._sector_fallback = dict(zip(latest_f["asset_id"], latest_f["sector"]))

    def allocate(self,
                 prices_to_date: pd.DataFrame,
                 fundamentals_to_date: pd.DataFrame,
                 indicators_to_date: pd.DataFrame,
                 current_date: str) -> np.ndarray:

        live = self._live_assets(prices_to_date, current_date)
        if not live:
            return np.zeros(self.n_assets)

        # 1. Regime (multi-indicator stress z-score) + smooth stress level
        regime = self._predict_regime(indicators_to_date, current_date)
        stress_level = self._stress_level(indicators_to_date, current_date)

        # 2. Cross-sectional features
        feats = self._compute_price_features(prices_to_date, current_date)
        fund = self._compute_fundamental_scores(fundamentals_to_date)
        sector_map = fund["sector"].copy()

        # 3. Robust percentile ranking [-0.5, 0.5]
        mom = self._cs_rank(feats["momentum"])
        rev = self._cs_rank(feats["reversal"])
        vol = self._cs_rank(feats["volatility"])
        liq = self._cs_rank(feats["liquidity"])
        val = self._cs_rank(fund["value"])
        qual = self._cs_rank(fund["quality"])

        raw_vol = feats["volatility"].clip(lower=0.001)
        inv_vol = 1.0 / raw_vol

        # 4. Regime knobs
        # FIX #4: shorts disabled in risk_off / crisis (caused SAMPLE_B drawdown).
        # FIX #1: pos_cap < 0.10 in non-bull regimes (forces diversification).
        if regime == "risk_on":
            a_mom, a_rev, a_val, a_qual, a_vol, a_liq = 0.97, 0.10, 0.20, 0.20, 0.15, 0.15
            net_lo, net_hi = 0.95, 1.10
            allow_shorts = False
            smooth_lam = 0.25
            risk_aversion = 25.0
            turn_penalty = 0.0    # let alpha rotate freely in bull markets
            alpha_scale = 0.50
            pos_cap = 0.10        # contest max in bull — let alpha breathe
        elif regime == "neutral":
            a_mom, a_rev, a_val, a_qual, a_vol, a_liq = 0.4, 0.2, 0.25, 0.5, 0.50, 0.10
            net_lo, net_hi = 0.90, 1.05
            allow_shorts = False
            smooth_lam = 0.20
            risk_aversion = 40.0
            turn_penalty = 1.5
            alpha_scale = 0.42
            pos_cap = 0.065
        elif regime == "risk_off":
            a_mom, a_rev, a_val, a_qual, a_vol, a_liq = 0.10, 0.50, 0.55, 0.90, 1.20, 0.10
            net_lo, net_hi = 0.85, 0.95
            allow_shorts = False  # was True in v2 — shorts hurt SAMPLE_B
            smooth_lam = 0.15
            risk_aversion = 80.0
            turn_penalty = 2.0
            alpha_scale = 0.30
            pos_cap = 0.04
        else:  # crisis
            a_mom, a_rev, a_val, a_qual, a_vol, a_liq = 0.00, 0.50, 0.70, 1.20, 1.50, 0.20
            net_lo, net_hi = 0.85, 0.90
            allow_shorts = False  # was True in v2
            smooth_lam = 0.10
            risk_aversion = 150.0
            turn_penalty = 3.0
            alpha_scale = 0.25
            pos_cap = 0.035

        # FIX #2: stress-aware net deleveraging.
        # Shrink the *upper* net target toward net_lo as stress rises. Even inside
        # a single regime label, an unusually volatile day cuts gross.
        net_hi_eff = net_hi - (net_hi - net_lo) * float(stress_level)
        net_lo_eff = net_lo  # keep the lower bound at the regime floor

        # 5. Composite alpha
        score = (
            a_mom * mom + a_rev * rev + a_val * val +
            a_qual * qual - a_vol * vol + a_liq * liq
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        live_mask = pd.Series([a in live for a in self.all_assets], index=self.all_assets)
        score = score.where(live_mask, -np.inf)
        inv_vol = inv_vol.where(live_mask, 0.0)

        target = pd.Series(0.0, index=self.all_assets)

        # 6. Heuristic warm-start (only used on the very first rebalance)
        target_net_mid = 0.5 * (net_lo_eff + net_hi_eff)
        if not allow_shorts:
            # Aim for ~ceil(target / pos_cap) names as warm-start so the QP
            # starts near a feasible diversified point.
            k_long = min(50, max(int(np.ceil(target_net_mid / pos_cap)) + 5, 20))
            k_long = min(k_long, int(live_mask.sum()))
            if k_long > 0:
                longs = score.sort_values(ascending=False).head(k_long)
                long_w = inv_vol.loc[longs.index]
                if long_w.sum() > 1e-8:
                    target.loc[longs.index] = (long_w / long_w.sum()) * target_net_mid
        else:
            k_long = min(35, int(live_mask.sum() * 0.6))
            k_short = min(15, int(live_mask.sum() * 0.3))
            if k_long > 0 and k_short > 0:
                longs = score.sort_values(ascending=False).head(k_long)
                shorts = score[score != -np.inf].sort_values(ascending=True).head(k_short)
                long_w = inv_vol.loc[longs.index]
                short_w = inv_vol.loc[shorts.index]
                short_alloc = -0.15
                long_alloc = target_net_mid - short_alloc
                if long_w.sum() > 1e-8:
                    target.loc[longs.index] = (long_w / long_w.sum()) * long_alloc
                if short_w.sum() > 1e-8:
                    target.loc[shorts.index] = (short_w / short_w.sum()) * short_alloc

        target = self._enforce_constraints(target, sector_map, pos_cap=pos_cap)

        # 7. QP
        sigma = self._estimate_covariance(prices_to_date, current_date)
        w_prev_arr = self._prev_w.reindex(self.all_assets).fillna(0.0).values.astype(float)
        if self._prev_date is not None and float(np.abs(w_prev_arr).sum()) > 1e-6:
            warm_start = w_prev_arr.copy()
        else:
            warm_start = target.values.astype(float)

        opt_w = self._optimize_portfolio(
            alpha=(score.values.astype(float) * alpha_scale),
            sigma=sigma,
            w_prev=w_prev_arr,
            warm_start=warm_start,
            net_lo=net_lo_eff,
            net_hi=net_hi_eff,
            allow_shorts=allow_shorts,
            risk_aversion=risk_aversion,
            turn_penalty=turn_penalty,
            live_mask=live_mask.values.astype(bool),
            pos_cap=pos_cap,
        )
        target = pd.Series(opt_w, index=self.all_assets)

        target = self._enforce_constraints(target, sector_map, pos_cap=pos_cap)

        # 8. Exponential smoothing + post-smoothing constraint pass
        if self._prev_date is not None:
            target = (1.0 - smooth_lam) * self._prev_w + smooth_lam * target
            target = target.where(live_mask, 0.0)
            target = self._enforce_constraints(target, sector_map, pos_cap=pos_cap)

        # 9. Minimum-trade deadband (contest filters trades <0.5% anyway, but a
        # tighter book-side filter cuts cost-drag on the smaller end)
        trade_diff = target - self._prev_w
        target = target.where(trade_diff.abs() >= 0.015, self._prev_w)

        target = target.where(live_mask, 0.0)

        self._prev_w = target.copy()
        self._prev_date = current_date

        return target.reindex(self.all_assets).fillna(0.0).values.astype(float)

    # -------------------- REGIME & STRESS --------------------

    # FIX #3: richer multi-indicator stress panel.
    # Each entry: (column, sign). Sign = +1 if higher values mean MORE stress,
    # -1 if higher values mean LESS stress (so we flip the z).
    _STRESS_PANEL = [
        ("impl_vol_index",   +1),
        ("credit_spread_hy", +1),
        ("funding_stress",   +1),
        ("realized_vol_20d", +1),
        ("term_spread",      -1),   # inverted curve (low/neg) -> stress
        ("sentiment_score",  -1),   # bearish sentiment -> stress
        ("macro_surprise",   -1),   # negative surprise -> stress
    ]

    def _stress_zscores(self, indicators_to_date: pd.DataFrame, current_date: str):
        """Returns list of directional z-scores (positive = stress) and the
        averaged stress z. Helpers shared by _predict_regime and _stress_level
        so we don't recompute the panel twice."""
        if indicators_to_date is None or len(indicators_to_date) < 10:
            return [], 0.0
        ind = indicators_to_date[indicators_to_date["date"] <= current_date]
        if len(ind) == 0:
            return [], 0.0
        lookback = ind.tail(252).copy()
        z_scores = []
        for col, sign in self._STRESS_PANEL:
            if col not in lookback.columns:
                continue
            s = pd.to_numeric(lookback[col], errors="coerce").ffill().dropna()
            if len(s) > 5 and s.std() > 0:
                z = (s.iloc[-1] - s.mean()) / s.std()
                z_scores.append(sign * float(z))
        if not z_scores:
            return [], 0.0
        return z_scores, float(np.mean(z_scores))

    def _predict_regime(self, indicators_to_date: pd.DataFrame, current_date: str) -> str:
        _, avg_z = self._stress_zscores(indicators_to_date, current_date)
        if avg_z > 1.0:
            return "crisis"
        elif avg_z >= 0.4:
            return "risk_off"
        elif avg_z <= -0.4:
            return "risk_on"
        else:
            return "neutral"

    def _stress_level(self, indicators_to_date: pd.DataFrame, current_date: str) -> float:
        """Smooth [0, 1] stress dial derived from the same panel.
        Stays at 0 during calm/bull periods; only ramps up once z exceeds 0.3
        (i.e. moderate stress). Hits 1.0 by z = 1.5 (crisis territory)."""
        _, avg_z = self._stress_zscores(indicators_to_date, current_date)
        return float(np.clip((avg_z - 0.3) / 1.2, 0.0, 1.0))

    @staticmethod
    def _cs_rank(x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        if x.isna().all():
            return pd.Series(0.0, index=x.index)
        ranks = x.rank(method="average", pct=True, na_option="keep")
        return (ranks - 0.5).fillna(0.0)

    def _live_assets(self, prices_to_date: pd.DataFrame, current_date: str) -> set:
        latest = prices_to_date[prices_to_date["date"] == current_date]
        if len(latest) == 0:
            return set()
        c = pd.to_numeric(latest["close"], errors="coerce")
        return set(latest.loc[np.isfinite(c), "asset_id"].values)

    def _enforce_constraints(self, w: pd.Series, sector_map: pd.Series,
                              pos_cap: float = 0.10) -> pd.Series:
        """Hard contest-side enforcement. pos_cap may be tighter than the 0.10
        contest cap (we use it to force diversification)."""
        w = w.fillna(0.0)

        # Max 50 positions
        nonz = w.index[w.abs() > 1e-6]
        if len(nonz) > 50:
            keep = w.loc[nonz].abs().sort_values(ascending=False).head(50).index
            w.loc[~w.index.isin(keep)] = 0.0

        # Per-asset cap (clip at min(pos_cap, 0.10) to be safe)
        cap = min(float(pos_cap), 0.10)
        w = w.clip(lower=-cap, upper=cap)

        # Sector absolute cap <= 0.30
        sector_map = sector_map.reindex(w.index).fillna("UNK")
        for _ in range(3):
            sec_sum = w.abs().groupby(sector_map).sum()
            violators = sec_sum[sec_sum > 0.30 + 1e-6]
            if len(violators) == 0:
                break
            for sec, exp in violators.items():
                idx = sector_map[sector_map == sec].index
                scale = 0.30 / float(exp)
                w.loc[idx] *= scale

        # Short floor >= -0.30
        short_sum = w[w < 0].sum()
        if short_sum < -0.30:
            w.loc[w < 0] *= (-0.30 / short_sum)

        # Gross cap <= 1.50
        gross = float(w.abs().sum())
        if gross > 1.50:
            w *= (1.50 / gross)

        return w

    # -------------------- OPTIMIZER --------------------

    def _estimate_covariance(self, prices_to_date: pd.DataFrame, current_date: str) -> np.ndarray:
        dates = sorted([d for d in prices_to_date["date"].unique().tolist() if d <= current_date])
        if len(dates) < 6:
            return np.eye(self.n_assets) * 1e-4
        look = dates[-120:]
        sub = prices_to_date[prices_to_date["date"].isin(look)]
        close = (sub.pivot(index="date", columns="asset_id", values="close")
                 .reindex(columns=self.all_assets).sort_index())
        rets = close.pct_change().dropna(how="all").fillna(0.0).values
        if rets.shape[0] < 5:
            return np.eye(self.n_assets) * 1e-4
        T = rets.shape[0]
        full = np.cov(rets, rowvar=False, ddof=1)
        neg = np.minimum(rets, 0.0)
        semi = (neg.T @ neg) / max(T - 1, 1)
        sigma = 0.4 * full + 0.6 * semi
        diag = np.diag(np.diag(sigma))
        sigma = 0.85 * sigma + 0.15 * diag
        sigma += np.eye(self.n_assets) * 1e-6
        return sigma * 252.0

    def _optimize_portfolio(self,
                            alpha: np.ndarray,
                            sigma: np.ndarray,
                            w_prev: np.ndarray,
                            warm_start: np.ndarray,
                            net_lo: float,
                            net_hi: float,
                            allow_shorts: bool,
                            risk_aversion: float,
                            turn_penalty: float,
                            live_mask: np.ndarray,
                            pos_cap: float) -> np.ndarray:
        n = self.n_assets
        alpha = np.nan_to_num(np.asarray(alpha, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        alpha = np.where(np.isfinite(alpha), alpha, 0.0)
        sigma = np.asarray(sigma, dtype=float)
        w_prev = np.nan_to_num(np.asarray(w_prev, dtype=float))

        cap = float(min(pos_cap, 0.10))
        lo = -cap if allow_shorts else 0.0
        hi = cap
        bounds = [((lo if live_mask[i] else 0.0), (hi if live_mask[i] else 0.0)) for i in range(n)]

        x0 = np.clip(np.nan_to_num(warm_start), lo, hi)
        x0 = np.where(live_mask, x0, 0.0)
        s0 = float(x0.sum())
        if s0 > 1e-8 and not (net_lo <= s0 <= net_hi):
            x0 *= (0.5 * (net_lo + net_hi)) / max(s0, 1e-8)
            x0 = np.clip(x0, lo, hi)

        if not _HAS_SCIPY:
            return x0

        eps = 1e-3

        def _obj(w):
            diff = w - w_prev
            risk = float(w @ sigma @ w)
            ret = float(alpha @ w)
            turn = float(np.sum(np.sqrt(diff * diff + eps * eps)))
            return risk_aversion * risk - ret + turn_penalty * turn

        def _grad(w):
            diff = w - w_prev
            g_risk = 2.0 * (sigma @ w)
            g_turn = diff / np.sqrt(diff * diff + eps * eps)
            return risk_aversion * g_risk - alpha + turn_penalty * g_turn

        cons = [
            {"type": "ineq", "fun": lambda w: float(np.sum(w) - net_lo),
                              "jac": lambda w: np.ones(n)},
            {"type": "ineq", "fun": lambda w: float(net_hi - np.sum(w)),
                              "jac": lambda w: -np.ones(n)},
        ]

        try:
            res = _scipy_minimize(
                _obj, x0, jac=_grad, method="SLSQP",
                bounds=bounds, constraints=cons,
                options={"maxiter": 80, "ftol": 1e-7, "disp": False},
            )
            w_opt = res.x if (res.success and np.all(np.isfinite(res.x))) else x0
            w_opt = np.clip(w_opt, [b[0] for b in bounds], [b[1] for b in bounds])
            s = float(w_opt.sum())
            if s > 1e-8 and not (net_lo - 1e-4 <= s <= net_hi + 1e-4):
                target_s = min(max(s, net_lo), net_hi)
                w_opt *= target_s / s
                w_opt = np.clip(w_opt, [b[0] for b in bounds], [b[1] for b in bounds])
            return w_opt
        except Exception:
            return x0

    def _compute_price_features(self, prices_to_date: pd.DataFrame, current_date: str) -> pd.DataFrame:
        dates = [d for d in prices_to_date["date"].unique().tolist() if d <= current_date]
        if not dates:
            return pd.DataFrame({"momentum": 0.0, "reversal": 0.0, "volatility": 0.0, "liquidity": 0.0}, index=self.all_assets)

        dates = sorted(dates)
        need = max(self._mom_lb, self._rev_lb, self._vol_lb, self._liq_lb) + 1
        use_dates = dates[-need:]

        sub = prices_to_date[prices_to_date["date"].isin(use_dates)]
        close = (sub.pivot(index="date", columns="asset_id", values="close")
                 .reindex(columns=self.all_assets).sort_index())
        rets = close.pct_change()

        feats = pd.DataFrame(index=self.all_assets)
        if len(close.dropna(how="all")) >= 2:
            c = close.dropna(how="all")
            start_idx = max(0, len(c) - (self._mom_lb + 1))
            feats["momentum"] = (c.iloc[-1] / c.iloc[start_idx] - 1.0)
        else:
            feats["momentum"] = 0.0

        r = rets.dropna(how="all")
        if len(r) >= 2:
            start_idx = max(0, len(r) - self._rev_lb)
            cum = (1.0 + r.iloc[start_idx:]).prod(axis=0) - 1.0
            feats["reversal"] = -cum
        else:
            feats["reversal"] = 0.0

        if len(r) > 2:
            feats["volatility"] = r.tail(self._vol_lb).std(axis=0, ddof=0)
        else:
            feats["volatility"] = 0.0

        if "volume" in sub.columns:
            vol = (sub.pivot(index="date", columns="asset_id", values="volume")
                   .reindex(columns=self.all_assets).sort_index())
            avg = vol.tail(self._liq_lb).mean(axis=0)
            feats["liquidity"] = np.log1p(avg)
        else:
            feats["liquidity"] = 0.0

        return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _compute_fundamental_scores(self, fundamentals_to_date: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=self.all_assets)
        if fundamentals_to_date is None or len(fundamentals_to_date) == 0:
            out["value"], out["quality"] = 0.0, 0.0
            out["sector"] = pd.Series([self._sector_fallback.get(a, "UNK") for a in self.all_assets], index=self.all_assets)
            return out

        latest = (fundamentals_to_date
                  .sort_values("report_date")
                  .drop_duplicates("asset_id", keep="last")
                  .set_index("asset_id")).reindex(self.all_assets)

        def col(name: str) -> pd.Series:
            if name not in latest.columns:
                return pd.Series(np.nan, index=latest.index)
            return pd.to_numeric(latest[name], errors="coerce")

        value = (
            self._cs_rank(-col("pe_ratio")) +
            self._cs_rank(-col("pb_ratio")) +
            self._cs_rank(col("dividend_yield")) +
            self._cs_rank(col("free_cash_flow_yield"))
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        quality = (
            self._cs_rank(col("roe")) +
            self._cs_rank(col("revenue_growth")) +
            self._cs_rank(col("earnings_surprise")) +
            self._cs_rank(-col("debt_equity"))
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        out["value"] = value
        out["quality"] = quality
        if "sector" in latest.columns:
            out["sector"] = latest["sector"].fillna("UNK")
        else:
            out["sector"] = pd.Series([self._sector_fallback.get(a, "UNK") for a in self.all_assets], index=self.all_assets)
        return out


# ======================================================================
# RUNNER  -  do not modify below this line                         # @RP
# ======================================================================

_SECTION_RE = re.compile(r"(?m)^===(\w+)===\s*$\n")

def _parse_sections(text: str) -> dict:
    parts = _SECTION_RE.split(text)
    return dict(zip(parts[1::2], (s.rstrip("\n") for s in parts[2::2])))

def _load_from_stdin():
    sections = _parse_sections(sys.stdin.read())
    cfg      = json.loads(sections["CONFIG"])
    prices   = pd.read_csv(io.StringIO(sections["PRICES"]))
    fund     = pd.read_csv(io.StringIO(sections["FUND"]))
    ind      = pd.read_csv(io.StringIO(sections["IND"]))
    return cfg, prices, fund, ind

def _load_from_dir(window_dir: str):
    prices = pd.read_csv(f"{window_dir}/asset_prices.csv")
    fund   = pd.read_csv(f"{window_dir}/asset_fundamentals.csv")
    ind    = pd.read_csv(f"{window_dir}/asset_indicators.csv")
    with open(f"{window_dir}/window_config.json") as f:
        cfg = json.load(f)
    return cfg, prices, fund, ind

def main(window_dir=None):
    if window_dir is not None:
        cfg, prices, fund, ind = _load_from_dir(window_dir)
    elif "--window-dir" in sys.argv:
        ap = argparse.ArgumentParser()
        ap.add_argument("--window-dir", required=True)
        args = ap.parse_args()
        cfg, prices, fund, ind = _load_from_dir(args.window_dir)
    else:
        cfg, prices, fund, ind = _load_from_stdin()

    train_end   = cfg["train_end_date"]
    rebal_dates = cfg["rebalance_dates"]
    all_assets  = cfg.get("asset_columns") or sorted(prices["asset_id"].unique())

    architect = PortfolioArchitect(
        prices[prices['date']        <= train_end].copy(),
        fund  [fund  ['report_date'] <= train_end].copy(),
        ind   [ind   ['date']        <= train_end].copy(),
    )

    print("date," + ",".join(all_assets))

    for date in rebal_dates:
        p = prices[prices['date']        <= date]
        f = fund  [fund  ['report_date'] <= date]
        i = ind   [ind   ['date']        <= date]

        try:
            w = architect.allocate(p, f, i, date)
            w = np.asarray(w, dtype=float)
            if w.ndim != 1 or len(w) != 100:
                w = np.ones(100) / 100
            w = np.where(np.isfinite(w), w, 0.0)
        except Exception as exc:
            print(f"# allocate() raised {exc!r} at {date}  -  using equal weight", file=sys.stderr)
            w = np.ones(100) / 100

        print(date + "," + ",".join(f"{x:.8f}" for x in w))
        sys.stdout.flush()

if __name__ == "__main__":
    main()
