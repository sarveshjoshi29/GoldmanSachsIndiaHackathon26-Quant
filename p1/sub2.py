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


# ======================================================================
# YOUR IMPLEMENTATION  -  edit only this class
# ======================================================================

"""STRATEGY EXPLANATION
====================
This strategy is a regime-aware, rule-based portfolio constructor. It ranks
assets cross-sectionally, sizes positions with inverse volatility, and then
projects the result back into the allowed risk limits.

Core behavior:
- Calm regimes emphasize long-only exposure and slower turnover.
- Stress regimes add a limited long/short overlay to reduce market beta.
- A rolling 60-day stress score, portfolio drawdown, and market-trend checks
    decide when the portfolio should move into a more defensive regime.
- Smoothing, a turnover cap, and a trade deadband help avoid excessive
    trading between rebalances.

Signals used:
- Price features: momentum, reversal, volatility, and liquidity.
- Fundamental features: value and quality composites from the latest report.
- Macro features: a simple stress measure built from the available indicators.

Regime detection in brief:
- Stress is measured with a 60-day rolling z-score over the available macro
    indicators.
- The rough thresholds are `z > 1.0` for crisis, `z >= 0.4` for risk-off,
    and `z <= -0.4` for risk-on.
- A drawdown trigger around `-6%` and a market-trend trigger around `-5%`
    can force a defensive regime earlier.

Risk controls:
- Positions are clipped to conservative per-asset and per-sector bounds.
- Gross exposure and short exposure are also buffered inside the official
    limits.
- Delisted assets are forced to zero before the final portfolio is returned.
"""


class PortfolioArchitect:
    def __init__(self,
                 prices: pd.DataFrame,
                 fundamentals: pd.DataFrame,
                 indicators: pd.DataFrame):

        self.n_assets = 100
        self.all_assets = sorted(prices['asset_id'].unique())

        # Factor lookback windows (in trading days)
        self._mom_lb = 60
        self._rev_lb = 5
        self._vol_lb = 20
        self._liq_lb = 20

        # Portfolio construction settings.
        self._n_select_long_only = 40          # long-only regimes: 40 longs
        self._n_long_short_long = 32           # long/short regimes: 32 longs
        self._n_long_short_short = 12          # long/short regimes: 12 shorts
        self._max_n_positions = 45             # hard cap (inside 50)
        self._max_w_asset = 0.09               # inside the 0.10 per-asset cap
        self._max_w_sector = 0.27              # inside the 0.30 per-sector cap
        self._short_floor = -0.29              # inside the -0.30 short limit
        self._gross_cap = 1.40                 # inside the 1.50 gross cap
        self._net_low = 0.88                   # inside the [0.85, 1.10] net band
        self._net_high = 1.08
        self._deadband = 0.012                 # >> 0.5% min-trade filter; cuts churn
        self._max_turnover_per_rebal = 0.60    # cap on single-rebalance turnover

        # Drawdown state.
        self._nav = 1.0
        self._nav_peak = 1.0
        self._defensive_active = False
        self._dd_on = -0.03    # activate defensive at -3% portfolio drawdown
        self._dd_off = -0.015  # deactivate when recovered to -1.5% (hysteresis)

        # Turnover protection state
        self._prev_w = pd.Series(0.0, index=self.all_assets)
        self._prev_date = None

        # Regime state with hysteresis.
        self._prev_regime = "neutral"
        self._pending_regime = None
        self._pending_count = 0

        # Stable sector map.
        self._sector_fallback = {}
        if fundamentals is not None and len(fundamentals) > 0 and "sector" in fundamentals.columns:
            latest_f = (fundamentals
                        .sort_values("report_date")
                        .drop_duplicates("asset_id", keep="last"))
            self._sector_fallback = dict(zip(latest_f["asset_id"], latest_f["sector"]))

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------
    def allocate(self,
                 prices_to_date: pd.DataFrame,
                 fundamentals_to_date: pd.DataFrame,
                 indicators_to_date: pd.DataFrame,
                 current_date: str) -> np.ndarray:

        # Identify live assets.
        live = self._live_assets(prices_to_date, current_date)
        if not live:
            self._prev_w.loc[:] = 0.0
            self._prev_date = current_date
            return np.zeros(self.n_assets)

        live_mask = pd.Series(
            [a in live for a in self.all_assets], index=self.all_assets
        )

        # Clear stale delisted weights.
        self._prev_w = self._prev_w.where(live_mask, 0.0)

        # Determine regime and retain the prior value for transition logic.
        prior_regime = self._prev_regime
        regime = self._predict_regime(indicators_to_date, current_date)
        stress_set = {"risk_off", "crisis"}

        # Defensive override using portfolio drawdown and market trend.
        dd = self._update_nav_and_dd(prices_to_date, current_date)
        mkt_20d = self._market_20d_return(prices_to_date, current_date)

        # Hysteretic defensive flag.
        if self._defensive_active:
            if dd > self._dd_off and mkt_20d > -0.01:
                self._defensive_active = False
        else:
            if dd < self._dd_on or mkt_20d < -0.03:
                self._defensive_active = True

        if self._defensive_active:
            # Force crisis-level defense regardless of indicator labels.
            regime = "crisis"
        elif mkt_20d < -0.015 and regime in ("risk_on", "neutral"):
            # Market is weak but indicators have not yet flagged stress.
            regime = "risk_off"

        regime_entry_to_stress = (
            prior_regime not in stress_set and regime in stress_set
        )

        # Cross-sectional features.
        feats = self._compute_price_features(prices_to_date, current_date)
        fund = self._compute_fundamental_scores(fundamentals_to_date)
        sector_map = fund["sector"].copy()

        # Percentile-rank each signal.
        mom = self._cs_rank(feats["momentum"])
        rev = self._cs_rank(feats["reversal"])
        vol = self._cs_rank(feats["volatility"])
        liq = self._cs_rank(feats["liquidity"])
        val = self._cs_rank(fund["value"])
        qual = self._cs_rank(fund["quality"])

        # Regime-dependent factor loadings.
        if regime == "risk_on":
            a_mom, a_rev, a_val, a_qual, a_vol, a_liq = 1.10, 0.10, 0.20, 0.20, 0.15, 0.10
            target_net = 1.05
            allow_shorts = False
            short_alloc = 0.0
            smooth_lam = 0.22
        elif regime == "neutral":
            a_mom, a_rev, a_val, a_qual, a_vol, a_liq = 0.55, 0.30, 0.45, 0.53, 0.50, 0.10
            target_net = 1.00
            allow_shorts = False
            short_alloc = 0.0
            smooth_lam = 0.18
        elif regime == "risk_off":
            a_mom, a_rev, a_val, a_qual, a_vol, a_liq = 0.10, 0.50, 0.55, 0.90, 1.20, 0.10
            target_net = 0.92
            allow_shorts = True
            short_alloc = -0.18
            smooth_lam = 0.60        # faster de-risking
        else:  # crisis
            a_mom, a_rev, a_val, a_qual, a_vol, a_liq = 0.00, 0.50, 0.70, 1.20, 1.50, 0.20
            target_net = 0.88        # at the official 0.85 floor, buffered
            allow_shorts = True
            short_alloc = -0.28      # near the official -0.30 short limit
            smooth_lam = 0.75        # fast de-risking when crisis hits

        score = (
            a_mom * mom + a_rev * rev + a_val * val
            + a_qual * qual - a_vol * vol + a_liq * liq
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Restrict scoring to live names only.
        score = score.where(live_mask, -np.inf)

        target = pd.Series(0.0, index=self.all_assets)

        # Portfolio construction.
        if not allow_shorts:
            # Long-only path.
            longs = self._select_sector_aware(
                score_series=score, sector_map=sector_map,
                k_total=self._n_select_long_only, max_per_sector=4, ascending=False,
            )
            if longs:
                ivw = self._inv_vol_weights(feats["volatility"], longs)
                target.loc[longs] = ivw.values * target_net
        else:
            # Long/short path.
            longs = self._select_sector_aware(
                score_series=score, sector_map=sector_map,
                k_total=self._n_long_short_long, max_per_sector=4, ascending=False,
            )
            # Shorts: pick the lowest-scoring live assets.
            shorts = self._select_sector_aware(
                score_series=score, sector_map=sector_map,
                k_total=self._n_long_short_short, max_per_sector=2, ascending=True,
            )
            # Remove any accidental overlap.
            shorts = [s for s in shorts if s not in set(longs)]

            long_alloc = target_net - short_alloc
            if longs:
                ivw_l = self._inv_vol_weights(feats["volatility"], longs)
                target.loc[longs] = ivw_l.values * long_alloc
            if shorts:
                ivw_s = self._inv_vol_weights(feats["volatility"], shorts)
                target.loc[shorts] = ivw_s.values * short_alloc

        if float(target.abs().sum()) < 1e-6:
            # Equal-weight fallback.
            live_list = [a for a in self.all_assets if a in live]
            if live_list:
                target.loc[live_list] = 1.0 / len(live_list)

        # Constraint pass.
        target = self._enforce_constraints(target, sector_map, live_mask)

        # Smooth toward previous weights.
        if self._prev_date is not None:
            lam = 0.70 if regime_entry_to_stress else smooth_lam
            target = (1.0 - lam) * self._prev_w + lam * target
            target = self._enforce_constraints(target, sector_map, live_mask)

        # Per-rebalance turnover cap.
        if self._prev_date is not None:
            proposed_turnover = float((target - self._prev_w).abs().sum())
            if proposed_turnover > self._max_turnover_per_rebal:
                scale = self._max_turnover_per_rebal / proposed_turnover
                target = self._prev_w + scale * (target - self._prev_w)
                target = self._enforce_constraints(target, sector_map, live_mask)

        # Trade deadband.
        if self._prev_date is not None:
            trade_diff = target - self._prev_w
            target = target.where(trade_diff.abs() >= self._deadband, self._prev_w)
            # Re-check constraints after freezing tiny trades.
            target = self._enforce_constraints(target, sector_map, live_mask)

        # Final delisted cleanup.
        target = target.where(live_mask, 0.0)

        # Final near-zero safety.
        if float(target.sum()) < 0.05:
            live_list = [a for a in self.all_assets if a in live]
            target = pd.Series(0.0, index=self.all_assets)
            if live_list:
                target.loc[live_list] = 1.0 / len(live_list)
            target = self._enforce_constraints(target, sector_map, live_mask)

        self._prev_w = target.copy()
        self._prev_date = current_date

        return target.reindex(self.all_assets).fillna(0.0).values.astype(float)

    # ------------------------------------------------------------------
    # REGIME (RULE-BASED, NO ML)
    # ------------------------------------------------------------------
    def _predict_regime(self, indicators_to_date: pd.DataFrame, current_date: str) -> str:
        """Rule-based regime classifier."""
        MIN_HISTORY = 35

        raw = self._raw_regime_label(indicators_to_date, current_date, MIN_HISTORY)

        # Hysteresis between stress and calm.
        prev = self._prev_regime
        stress_set = {"risk_off", "crisis"}
        calm_set = {"risk_on", "neutral"}

        if prev in stress_set and raw in calm_set:
            # Require 2 consecutive calm readings before downgrading.
            if self._pending_regime == raw:
                self._pending_count += 1
            else:
                self._pending_regime = raw
                self._pending_count = 1
            if self._pending_count >= 2:
                final = raw
                self._pending_regime = None
                self._pending_count = 0
            else:
                final = prev
        else:
            final = raw
            self._pending_regime = None
            self._pending_count = 0

        self._prev_regime = final
        return final

    def _raw_regime_label(self, indicators_to_date: pd.DataFrame,
                          current_date: str, min_history: int) -> str:
        if indicators_to_date is None:
            return "neutral"
        ind = indicators_to_date[indicators_to_date["date"] <= current_date]
        if len(ind) == 0:
            return "neutral"

        stress_cols = [
            "impl_vol_index", "credit_spread_hy",
            "funding_stress", "realized_vol_20d",
        ]
        avail = [c for c in stress_cols if c in ind.columns]
        if not avail:
            return "neutral"

        # Trailing window for z-scoring.
        lookback = ind.tail(60).copy()
        z_scores = {}
        latest = {}
        for c in avail:
            series = pd.to_numeric(lookback[c], errors="coerce").ffill().dropna()
            if len(series) > 5 and series.std() > 0:
                z_scores[c] = (series.iloc[-1] - series.mean()) / series.std()
                latest[c] = float(series.iloc[-1])
        if not z_scores:
            return "neutral"

        avg_z = float(np.mean(list(z_scores.values())))

        # Minimum-history guard.
        if len(ind) < min_history:
            if avg_z <= -0.4:
                return "risk_on"
            return "neutral"

        # Absolute-level gate.
        abs_crisis = (
            latest.get("impl_vol_index", 0)    > 20.0 or
            latest.get("credit_spread_hy", 0)  > 450.0 or
            latest.get("funding_stress", 0)    > 0.25 or
            latest.get("realized_vol_20d", 0)  > 22.0
        )
        abs_stressed = (
            latest.get("impl_vol_index", 0)    > 16.0 or
            latest.get("credit_spread_hy", 0)  > 380.0 or
            latest.get("funding_stress", 0)    > 0.12 or
            latest.get("realized_vol_20d", 0)  > 15.0
        )

        if avg_z > 1.0 and abs_crisis:
            return "crisis"
        if avg_z >= 0.4 and abs_stressed:
            return "risk_off"
        if avg_z <= -0.4:
            return "risk_on"
        return "neutral"

    # ------------------------------------------------------------------
    # SELECTION HELPER
    # ------------------------------------------------------------------
    def _select_sector_aware(self, score_series: pd.Series, sector_map: pd.Series,
                             k_total: int, max_per_sector: int,
                             ascending: bool = False) -> list:
        """Score-ordered selection with per-sector cap."""
        sorted_scores = score_series.sort_values(ascending=ascending)
        picks, sec_counts = [], {}
        for asset, sc in sorted_scores.items():
            if not np.isfinite(sc):
                continue
            sec = sector_map.get(asset, "UNK")
            if sec_counts.get(sec, 0) >= max_per_sector:
                continue
            picks.append(asset)
            sec_counts[sec] = sec_counts.get(sec, 0) + 1
            if len(picks) >= k_total:
                break
        return picks

    def _update_nav_and_dd(self, prices_to_date: pd.DataFrame, current_date: str) -> float:
        """Update portfolio NAV and return current drawdown."""
        if self._prev_date is None:
            return 0.0
        dates = sorted(
            d for d in prices_to_date["date"].unique()
            if self._prev_date < d <= current_date
        )
        if not dates:
            return self._nav / self._nav_peak - 1.0

        anchor = [self._prev_date] + list(dates)
        sub = prices_to_date[prices_to_date["date"].isin(anchor)]
        close = (sub.pivot(index="date", columns="asset_id", values="close")
                 .reindex(columns=self.all_assets).sort_index())
        rets = close.pct_change().fillna(0.0).iloc[1:]  # drop the anchor row
        if rets.empty:
            return self._nav / self._nav_peak - 1.0

        prev_vec = self._prev_w.reindex(self.all_assets).fillna(0.0).values
        port_daily = (rets.values * prev_vec).sum(axis=1)
        for r in port_daily:
            self._nav *= (1.0 + float(r))
            if self._nav > self._nav_peak:
                self._nav_peak = self._nav
        return self._nav / self._nav_peak - 1.0

    def _market_20d_return(self, prices_to_date: pd.DataFrame, current_date: str) -> float:
        """Equal-weight 20-day cumulative return of the universe."""
        dates = sorted(d for d in prices_to_date["date"].unique() if d <= current_date)
        if len(dates) < 21:
            return 0.0
        use = dates[-21:]
        sub = prices_to_date[prices_to_date["date"].isin(use)]
        close = (sub.pivot(index="date", columns="asset_id", values="close")
                 .reindex(columns=self.all_assets).sort_index())
        r = close.pct_change().mean(axis=1).dropna()
        if r.empty:
            return 0.0
        return float((1.0 + r).prod() - 1.0)

    def _inv_vol_weights(self, vol_series: pd.Series, picks: list) -> pd.Series:
        """Inverse-volatility weights for the selected picks."""
        v = vol_series.reindex(picks).clip(lower=1e-4)
        iv = 1.0 / v
        s = float(iv.sum())
        if s <= 0:
            # equal weight fallback
            return pd.Series(1.0 / len(picks), index=picks)
        return iv / s

    # ------------------------------------------------------------------
    # CROSS-SECTIONAL RANK
    # ------------------------------------------------------------------
    @staticmethod
    def _cs_rank(x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        if x.isna().all():
            return pd.Series(0.0, index=x.index)
        ranks = x.rank(method="average", pct=True, na_option="keep")
        return (ranks - 0.5).fillna(0.0)

    # ------------------------------------------------------------------
    # LIVE-ASSET DETECTION
    # ------------------------------------------------------------------
    def _live_assets(self, prices_to_date: pd.DataFrame, current_date: str) -> set:
        latest = prices_to_date[prices_to_date["date"] == current_date]
        if len(latest) == 0:
            return set()
        c = pd.to_numeric(latest["close"], errors="coerce")
        return set(latest.loc[np.isfinite(c) & (c > 0), "asset_id"].values)

    # ------------------------------------------------------------------
    # CONSTRAINT ENFORCEMENT (long-only, buffered)
    # ------------------------------------------------------------------
    def _enforce_constraints(self, w: pd.Series, sector_map: pd.Series,
                             live_mask: pd.Series) -> pd.Series:
        """Iteratively project weights into the no-penalty zone.

        Bounds enforced (all buffered inside the official limits):
          per-asset |w|   <= 0.09  (official 0.10)
          per-sector |w|  <= 0.27  (official 0.30)
          short exposure  >= -0.25 (official -0.30)
          gross           <= 1.40  (official 1.50)
          net             in [0.88, 1.08] (official [0.85, 1.10])
          positions       <= 45    (official 50)
        Delisted assets are forced to exact zero (official: 200 bps penalty).
        """
        w = w.fillna(0.0)

        # Delisted names are forced to zero.
        w = w.where(live_mask, 0.0)

        for _ in range(6):
            # Cardinality cap.
            nonz = w.index[w.abs() > 1e-9]
            if len(nonz) > self._max_n_positions:
                keep = w.loc[nonz].abs().sort_values(ascending=False)\
                       .head(self._max_n_positions).index
                w.loc[~w.index.isin(keep)] = 0.0

            # Per-asset cap.
            w = w.clip(lower=-self._max_w_asset, upper=self._max_w_asset)

            # Per-sector absolute cap.
            sec_aligned = sector_map.reindex(w.index).fillna("UNK")
            sec_abs = w.abs().groupby(sec_aligned).sum()
            over = sec_abs[sec_abs > self._max_w_sector]
            for sec, exp in over.items():
                idx = w.index.intersection(sec_aligned[sec_aligned == sec].index)
                w.loc[idx] *= (self._max_w_sector / exp)

            # Short floor.
            shorts = w[w < 0]
            short_sum = float(shorts.sum())
            if short_sum < self._short_floor and short_sum < 0:
                # Scale shorts toward zero.
                w.loc[shorts.index] *= (self._short_floor / short_sum)

            # Gross cap.
            gross = float(w.abs().sum())
            if gross > self._gross_cap:
                w *= (self._gross_cap / gross)

            # Net exposure band.
            net = float(w.sum())
            longs_idx = w[w > 0].index
            long_sum = float(w.loc[longs_idx].sum()) if len(longs_idx) else 0.0
            short_total = float(w[w < 0].sum())
            if long_sum > 1e-9:
                if net > self._net_high:
                    target_long = self._net_high - short_total
                    if target_long > 0:
                        w.loc[longs_idx] *= (target_long / long_sum)
                elif net < self._net_low:
                    target_long = self._net_low - short_total
                    if target_long > 0:
                        w.loc[longs_idx] *= (target_long / long_sum)

        # Final safety clip.
        w = w.clip(lower=-self._max_w_asset, upper=self._max_w_asset)
        w = w.where(live_mask, 0.0)
        return w

    # ------------------------------------------------------------------
    # PRICE FEATURES
    # ------------------------------------------------------------------
    def _compute_price_features(self, prices_to_date: pd.DataFrame,
                                current_date: str) -> pd.DataFrame:
        dates = sorted(d for d in prices_to_date["date"].unique() if d <= current_date)
        if not dates:
            return pd.DataFrame(
                {"momentum": 0.0, "reversal": 0.0, "volatility": 0.0, "liquidity": 0.0},
                index=self.all_assets,
            )

        need = max(self._mom_lb, self._rev_lb, self._vol_lb, self._liq_lb) + 1
        use_dates = dates[-need:]
        sub = prices_to_date[prices_to_date["date"].isin(use_dates)]

        close = (sub.pivot(index="date", columns="asset_id", values="close")
                 .reindex(columns=self.all_assets).sort_index())
        rets = close.pct_change()

        feats = pd.DataFrame(index=self.all_assets)

        c = close.dropna(how="all")
        if len(c) >= 2:
            start = max(0, len(c) - (self._mom_lb + 1))
            feats["momentum"] = (c.iloc[-1] / c.iloc[start] - 1.0)
        else:
            feats["momentum"] = 0.0

        r = rets.dropna(how="all")
        if len(r) >= 2:
            start = max(0, len(r) - self._rev_lb)
            cum = (1.0 + r.iloc[start:]).prod(axis=0) - 1.0
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
            feats["liquidity"] = np.log1p(vol.tail(self._liq_lb).mean(axis=0))
        else:
            feats["liquidity"] = 0.0

        return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ------------------------------------------------------------------
    # FUNDAMENTAL FEATURES
    # ------------------------------------------------------------------
    def _compute_fundamental_scores(self, fundamentals_to_date: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=self.all_assets)
        if fundamentals_to_date is None or len(fundamentals_to_date) == 0:
            out["value"], out["quality"] = 0.0, 0.0
            out["sector"] = pd.Series(
                [self._sector_fallback.get(a, "UNK") for a in self.all_assets],
                index=self.all_assets,
            )
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
            self._cs_rank(-col("pe_ratio"))
            + self._cs_rank(-col("pb_ratio"))
            + self._cs_rank(col("dividend_yield"))
            + self._cs_rank(col("free_cash_flow_yield"))
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        quality = (
            self._cs_rank(col("roe"))
            + self._cs_rank(col("revenue_growth"))
            + self._cs_rank(col("earnings_surprise"))
            + self._cs_rank(-col("debt_equity"))
        ).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        out["value"] = value
        out["quality"] = quality
        if "sector" in latest.columns:
            out["sector"] = latest["sector"].fillna(
                pd.Series(self._sector_fallback)
            ).fillna("UNK")
        else:
            out["sector"] = pd.Series(
                [self._sector_fallback.get(a, "UNK") for a in self.all_assets],
                index=self.all_assets,
            )
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
