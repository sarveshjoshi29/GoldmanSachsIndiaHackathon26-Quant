"""
Regime Navigator: Adaptive Portfolio Construction
==================================================
Goldman Sachs India Hackathon 2026  -  Quant Challenge

INSTRUCTIONS
------------
1. Implement the PortfolioArchitect class (the section marked YOUR CODE HERE).
2. Do NOT modify the main() runner at the bottom.
3. Your allocate() is called once per rebalance date (~40 times per test case).
4. Return a numpy array of shape (100,)  -  one weight per asset (SEC_001..SEC_100).
5. Max 50 non-zero positions, |w_i| <= 0.10 per asset. See PROBLEM_STATEMENT.pdf for all constraints.
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

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ======================================================================
# YOUR IMPLEMENTATION  -  edit only this class
# ======================================================================

"""STRATEGY EXPLANATION
Core approach:
A 3-state regime-adaptive portfolio that ranks assets by cross-sectional factor scores
and constructs a constrained portfolio each rebalance using Risk Parity sizing.

Signals used:
- Momentum (60-day trailing) and Short-term Reversal (5-day).
- Realized Volatility (20-day) and Liquidity (20-day average volume proxy).
- Simple fundamental aggregates for Value and Quality derived from the latest available fundamentals.

Regime detection:
- A 3-component GaussianMixture (diag covariance) is fit to EWMA-smoothed (span=40) 
  indicator series to emulate HMM persistence. Components are ranked by a hand-coded
  risk vector to map components -> {risk_on, neutral, risk_off}.

Scoring & sizing:
- Cross-sectional features are converted to percentile ranks (range ~ [-0.5, 0.5]) to
  avoid outlier distortion.
- Assets are selected by score (top/bottom buckets). Sizing uses Inverse Volatility 
  (1/vol) to achieve risk parity, ensuring high-beta assets do not dominate drawdowns.

Constraints & execution:
- Max 50 non-zero positions, per-asset cap +/-10%, sector absolute cap 30%, gross cap 1.50.
- Net exposure flexes dynamically: 1.10 (Risk-On), 1.00 (Neutral), 0.85 (Risk-Off).
- Short selling up to -0.30 deployed strictly during Risk-Off to offset beta.
- Exponential smoothing against previous weights reduces turnover. A 1.0% minimum-trade 
  band is enforced to avoid tiny, costly trades.
"""

class RegimeDetection:
    def get_regimes_gmm(self, input_data: np.ndarray, params: dict) -> GaussianMixture:
        model = GaussianMixture()
        model = self.initialise_model(model, params)
        return model.fit(input_data)

    def initialise_model(self, model, params: dict):
        for parameter, value in (params or {}).items():
            setattr(model, parameter, value)
        return model


class PortfolioArchitect:
    def __init__(self,
                 prices: pd.DataFrame,
                 fundamentals: pd.DataFrame,
                 indicators: pd.DataFrame):

        self.n_assets = 100
        self.all_assets = sorted(prices['asset_id'].unique())

        self._mom_lb = 60
        self._rev_lb = 5
        self._vol_lb = 20
        self._liq_lb = 20

        self._prev_w = pd.Series(0.0, index=self.all_assets)
        self._prev_date = None
        self._last_regime = "neutral"

        self._sector_fallback = {}
        if fundamentals is not None and len(fundamentals) > 0 and "sector" in fundamentals.columns:
            latest_f = (fundamentals
                        .sort_values("report_date")
                        .drop_duplicates("asset_id", keep="last"))
            self._sector_fallback = dict(zip(latest_f["asset_id"], latest_f["sector"]))

        self._regime_cols = [
            "funding_stress", "realized_vol_20d", "credit_spread_hy",
            "credit_spread_ig", "sentiment_score", "liquidity_index",
            "macro_surprise", "impl_vol_index", "term_spread"
        ]

        self._regime_scaler = None
        self._regime_model = None
        self._regime_order = None 
        self._regime_used_cols = None

        ind = indicators.copy() if indicators is not None else pd.DataFrame()
        if len(ind) > 0:
            X, used_cols = self._build_indicator_matrix(ind)
            # Threshold lowered to 20 to ensure GMM trains on short hidden test cases
            if X.shape[0] >= 20 and X.shape[1] >= 3:
                self._regime_used_cols = used_cols
                self._fit_regime_model(X, used_cols)

    def allocate(self,
                 prices_to_date: pd.DataFrame,
                 fundamentals_to_date: pd.DataFrame,
                 indicators_to_date: pd.DataFrame,
                 current_date: str) -> np.ndarray:

        live = self._live_assets(prices_to_date, current_date)
        if not live:
            return np.zeros(self.n_assets)

        regime = self._predict_regime(indicators_to_date, current_date)
        self._last_regime = regime

        feats = self._compute_price_features(prices_to_date, current_date)
        fund = self._compute_fundamental_scores(fundamentals_to_date)
        sector_map = fund["sector"].copy()

        # Robust Percentile Ranking
        mom = self._cs_rank(feats["momentum"])
        rev = self._cs_rank(feats["reversal"])
        vol = self._cs_rank(feats["volatility"])
        liq = self._cs_rank(feats["liquidity"])
        val = self._cs_rank(fund["value"])
        qual = self._cs_rank(fund["quality"])

        # Inverse Volatility for Risk Parity sizing
        raw_vol = feats["volatility"].clip(lower=0.001)
        inv_vol = 1.0 / raw_vol

        # Regime-based Factor & Exposure Tuning
        if regime == "risk_on":
            a_mom, a_rev, a_val, a_qual, a_vol, a_liq = 1.10, 0.10, 0.20, 0.20, 0.15, 0.10
            target_net = 1.10
            allow_shorts = False
            smooth_lam = 0.9
        elif regime == "neutral":
            a_mom, a_rev, a_val, a_qual, a_vol, a_liq = 0.55, 0.30, 0.45, 0.53, 0.50, 0.10
            target_net = 1.00
            allow_shorts = False
            smooth_lam = 0.9
        else:  # risk_off
            a_mom, a_rev, a_val, a_qual, a_vol, a_liq = 0.10, 0.50, 0.55, 0.90, 1.20, 0.10
            target_net = 0.85 
            allow_shorts = True
            smooth_lam = 0.7

        score = (
            a_mom * mom + a_rev * rev + a_val * val + 
            a_qual * qual - a_vol * vol + a_liq * liq
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        live_mask = pd.Series([a in live for a in self.all_assets], index=self.all_assets)
        score = score.where(live_mask, -np.inf)
        inv_vol = inv_vol.where(live_mask, 0.0)

        target = pd.Series(0.0, index=self.all_assets)

        if not allow_shorts:
            k_long = min(40, int(live_mask.sum()))
            if k_long > 0:
                longs = score.sort_values(ascending=False).head(k_long)
                # Risk Parity Allocation
                long_w = inv_vol.loc[longs.index]
                if long_w.sum() > 1e-8:
                    target.loc[longs.index] = (long_w / long_w.sum()) * target_net
        else:
            k_long = min(30, int(live_mask.sum() * 0.6))
            k_short = min(15, int(live_mask.sum() * 0.3))
            
            if k_long > 0 and k_short > 0:
                longs = score.sort_values(ascending=False).head(k_long)
                shorts = score[score != -np.inf].sort_values(ascending=True).head(k_short)
                
                long_w = inv_vol.loc[longs.index]
                short_w = inv_vol.loc[shorts.index]
                
                # Maximize short defense to -0.30
                short_alloc = -0.30
                long_alloc = target_net - short_alloc 
                
                if long_w.sum() > 1e-8:
                    target.loc[longs.index] = (long_w / long_w.sum()) * long_alloc
                if short_w.sum() > 1e-8:
                    target.loc[shorts.index] = (short_w / short_w.sum()) * short_alloc

        target = self._enforce_constraints(target, sector_map)

        if self._prev_date is not None:
            target = (1.0 - smooth_lam) * self._prev_w + smooth_lam * target
            target = target.where(live_mask, 0.0)
            target = self._enforce_constraints(target, sector_map)

        # Minimum trade filter
        trade_diff = target - self._prev_w
        target = target.where(trade_diff.abs() >= 0.01, self._prev_w)
        target = target.where(live_mask, 0.0)

        self._prev_w = target.copy()
        self._prev_date = current_date

        return target.reindex(self.all_assets).fillna(0.0).values.astype(float)


    # -------------------- helpers --------------------

    @staticmethod
    def _cs_rank(x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        if x.isna().all():
            return pd.Series(0.0, index=x.index)
        ranks = x.rank(method="average", pct=True, na_option="keep")
        out = ranks - 0.5
        return out.fillna(0.0)

    def _live_assets(self, prices_to_date: pd.DataFrame, current_date: str) -> set:
        latest = prices_to_date[prices_to_date["date"] == current_date]
        if len(latest) == 0:
            return set()
        c = pd.to_numeric(latest["close"], errors="coerce")
        return set(latest.loc[np.isfinite(c), "asset_id"].values)

    def _compute_price_features(self, prices_to_date: pd.DataFrame, current_date: str) -> pd.DataFrame:
        dates = [d for d in prices_to_date["date"].unique().tolist() if d <= current_date]
        if not dates:
            return pd.DataFrame({
                "momentum": 0.0, "reversal": 0.0, "volatility": 0.0, "liquidity": 0.0
            }, index=self.all_assets)
        dates = sorted(dates)
        need = max(self._mom_lb, self._rev_lb, self._vol_lb, self._liq_lb) + 1
        use_dates = dates[-need:]

        sub = prices_to_date[prices_to_date["date"].isin(use_dates)]
        close = (sub.pivot(index="date", columns="asset_id", values="close")
                 .reindex(columns=self.all_assets)
                 .sort_index())
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
                   .reindex(columns=self.all_assets)
                   .sort_index())
            avg = vol.tail(self._liq_lb).mean(axis=0)
            feats["liquidity"] = np.log1p(avg)
        else:
            feats["liquidity"] = 0.0

        feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return feats

    def _compute_fundamental_scores(self, fundamentals_to_date: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=self.all_assets)
        if fundamentals_to_date is None or len(fundamentals_to_date) == 0:
            out["value"] = 0.0
            out["quality"] = 0.0
            out["sector"] = pd.Series([self._sector_fallback.get(a, "UNK") for a in self.all_assets],
                                       index=self.all_assets)
            return out

        latest = (fundamentals_to_date
                  .sort_values("report_date")
                  .drop_duplicates("asset_id", keep="last")
                  .set_index("asset_id"))
        latest = latest.reindex(self.all_assets)

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
            out["sector"] = pd.Series([self._sector_fallback.get(a, "UNK") for a in self.all_assets],
                                       index=self.all_assets)
        return out

    def _build_indicator_matrix(self, indicators_df: pd.DataFrame) -> tuple:
        ind = indicators_df.sort_values("date")
        cols = [c for c in self._regime_cols if c in ind.columns]
        X = ind[cols].copy()
        for c in cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        # Ensure identical span for train and live inference
        X = X.ffill().fillna(0.0).ewm(span=40, adjust=False).mean()
        return X.values, cols

    def _fit_regime_model(self, X: np.ndarray, used_cols: list) -> None:
        self._regime_scaler = StandardScaler()
        Xs = self._regime_scaler.fit_transform(X)

        rd = RegimeDetection()
        params = {
            "n_components": 3,
            "covariance_type": "diag",
            "random_state": 42, 
            "n_init": 5, 
            "max_iter": 4000,
        }
        self._regime_model = rd.get_regimes_gmm(Xs, params)

        means = self._regime_model.means_
        risk_vec = np.zeros(means.shape[1])
        used_cols = list(used_cols)
        w_by_name = {
            "funding_stress": 1.0, "realized_vol_20d": 1.0, "credit_spread_hy": 0.8,
            "credit_spread_ig": 0.4, "impl_vol_index": 0.4, "sentiment_score": -0.4,
            "liquidity_index": -0.6, "macro_surprise": -0.2, "term_spread": -0.1,
        }
        for j, name in enumerate(used_cols):
            risk_vec[j] = w_by_name.get(name, 0.0)
        comp_risk = means @ risk_vec
        self._regime_order = list(np.argsort(comp_risk))

    def _predict_regime(self, indicators_to_date: pd.DataFrame, current_date: str) -> str:
        if self._regime_model is None or indicators_to_date is None or len(indicators_to_date) == 0:
            return "neutral"
        ind = indicators_to_date[indicators_to_date["date"] <= current_date]
        if len(ind) == 0:
            return "neutral"
        ind = ind.sort_values("date")
        cols = list(self._regime_used_cols or [])
        if len(cols) == 0 or any(c not in ind.columns for c in cols):
            return "neutral"
        
        tmp = ind[cols].copy()
        for c in cols:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        tmp = tmp.ffill().fillna(0.0).ewm(span=40, adjust=False).mean()
        
        x = tmp.iloc[-1].values.astype(float)
        xs = self._regime_scaler.transform(x.reshape(1, -1))
        comp = int(self._regime_model.predict(xs)[0])
        
        order = self._regime_order or [0, 1, 2]
        rank = order.index(comp) if comp in order else 1
        
        regime_map = {0: "risk_on", 1: "neutral", 2: "risk_off"}
        return regime_map.get(rank, "neutral")

    def _enforce_constraints(self, w: pd.Series, sector_map: pd.Series) -> pd.Series:
        w = w.fillna(0.0)

        nonz = w.index[w.abs() > 1e-12]
        if len(nonz) > 50:
            keep = w.loc[nonz].abs().sort_values(ascending=False).head(50).index
            w.loc[~w.index.isin(keep)] = 0.0

        w = w.clip(lower=-0.10, upper=0.10)

        sector_map = sector_map.reindex(w.index).fillna("UNK")
        for _ in range(3): 
            sec_sum = w.abs().groupby(sector_map).sum()
            offenders = sec_sum[sec_sum > 0.30 + 1e-6]
            if len(offenders) == 0:
                break
            for sec, exp in offenders.items():
                idx = sector_map[sector_map == sec].index
                scale = 0.30 / float(exp)
                w.loc[idx] *= scale

        gross = float(w.abs().sum())
        if gross > 1.50:
            w *= (1.50 / gross)

        return w


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
        ap.add_argument("--window-dir", required=True,
                        help="Directory containing asset_prices.csv, "
                             "asset_fundamentals.csv, asset_indicators.csv, "
                             "window_config.json")
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
            print(f"# allocate() raised {exc!r} at {date}  -  using equal weight",
                  file=sys.stderr)
            w = np.ones(100) / 100

        print(date + "," + ",".join(f"{x:.8f}" for x in w))
        sys.stdout.flush()

if __name__ == "__main__":
    main()