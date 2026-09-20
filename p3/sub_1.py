import sys
import json
import math
import numpy as np
from scipy.optimize import least_squares, brentq
from scipy.stats import norm
from scipy.linalg import solve_banded
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Yield curve built from zero rates via linear interpolation.
# Also pre-computes piecewise forward rates for use in drift calculations.
# ---------------------------------------------------------------------------
class TermStructureCurve:
    def __init__(self, time_points, zero_rates):
        self.t_arr = np.array(time_points)
        self.r_arr = np.array(zero_rates)

        # Flat extrapolation beyond the given tenor range
        self.interpolator = interp1d(
            self.t_arr, self.r_arr,
            kind="linear",
            bounds_error=False,
            fill_value=(self.r_arr[0], self.r_arr[-1]),
        )

        n = len(self.t_arr)
        self.fwd_rates = np.zeros_like(self.t_arr)

        # Simple forward rate: (r2*t2 - r1*t1) / (t2 - t1), with a tiny floor
        for i in range(n - 1):
            t1, t2 = self.t_arr[i], self.t_arr[i + 1]
            r1, r2 = self.r_arr[i], self.r_arr[i + 1]
            self.fwd_rates[i] = (r2 * t2 - r1 * t1) / (t2 - t1) + 1e-9

        if n > 1:
            self.fwd_rates[-1] = self.fwd_rates[-2]   # replicate last bucket

    def fetch_rate(self, t):
        """Spot zero rate at maturity t."""
        return float(self.interpolator(t))

    def get_discount(self, t, T_end):
        """
        Forward discount factor P(t, T_end).
        When t == 0 this collapses to the standard discount factor P(0, T_end).
        Handles both scalar and array inputs.
        """
        if np.isscalar(t) and np.isscalar(T_end):
            df_T = math.exp(-self.fetch_rate(T_end) * T_end)
            if t == 0:
                return df_T
            df_t = math.exp(-self.fetch_rate(t) * t)
            return df_T / df_t

        df_T = np.exp(-self.interpolator(T_end) * T_end)
        df_t = np.exp(-self.interpolator(t) * t)
        return df_T / df_t

    def inst_forward(self, t):
        """
        Instantaneous forward rate f(0, t) via numerical differentiation
        of the zero-rate curve: f = r(t) + t * dr/dt.
        """
        eps = 1e-6
        if isinstance(t, np.ndarray):
            r_up = self.interpolator(t + eps)
            r_dn = self.interpolator(np.maximum(0.0, t - eps))
            dr   = (r_up - r_dn) / (eps + np.minimum(t, eps))
            return self.interpolator(t) + t * dr

        r_up = self.fetch_rate(t + eps)
        r_dn = self.fetch_rate(max(0.0, t - eps))
        dr   = (r_up - r_dn) / (eps + min(t, eps))
        return self.fetch_rate(t) + t * dr


# ---------------------------------------------------------------------------
# Hull-White one-factor short-rate model.
# dr(t) = [theta(t) - kappa * r(t)] dt + sigma(t) dW(t)
# Sigma is piecewise-constant over a user-supplied time grid.
# ---------------------------------------------------------------------------
class HWOneFactor:
    def __init__(self, reversion, vol_array, time_grid, yield_curve):
        self.kappa      = float(reversion)
        self.vols       = np.array(vol_array, dtype=float)
        self.grid_times = np.array(time_grid,  dtype=float)
        self.curve      = yield_curve

    def volatility_at(self, t):
        """Return the piecewise-constant sigma active at time t."""
        idx = np.searchsorted(self.grid_times, t)
        if isinstance(t, np.ndarray):
            idx = np.minimum(idx, len(self.vols) - 1)
            return self.vols[idx]
        return self.vols[min(idx, len(self.vols) - 1)]

    def calc_variance(self, t):
        """
        Var[x(t)] for the Ornstein-Uhlenbeck state variable x(t),
        accumulated piecewise across the vol grid up to time t.
        """
        if t == 0:
            return 0.0

        total, prev_t = 0.0, 0.0
        for i, edge in enumerate(self.grid_times):
            cur = min(t, edge)
            if cur > prev_t:
                dt    = cur - prev_t
                decay = math.exp(-2.0 * self.kappa * dt)
                inc   = self.vols[i] ** 2 * (1.0 - decay) / (2.0 * self.kappa)
                total = total * decay + inc
                prev_t = cur
            if cur == t:
                break
        return total

    def integral_term(self, t):
        """
        Integral of sigma^2 * exp(-kappa*(t-s)) ds from 0 to t,
        required for the drift correction alpha(t).
        """
        if t == 0:
            return 0.0

        accum, prev_t = 0.0, 0.0
        for i, edge in enumerate(self.grid_times):
            cur = min(t, edge)
            if cur > prev_t:
                dt    = cur - prev_t
                decay = math.exp(-self.kappa * dt)
                inc   = self.vols[i] ** 2 * (1.0 - decay) / self.kappa
                accum = accum * decay + inc
                prev_t = cur
            if cur == t:
                break
        return accum

    def drift_alpha(self, t):
        """
        Deterministic shift alpha(t) that fits the model to the initial
        zero curve: alpha = f(0,t) + [integral_term - variance] / kappa.
        """
        return self.curve.inst_forward(t) + (self.integral_term(t) - self.calc_variance(t)) / self.kappa

    def drift_theta(self, t):
        """
        Mean-reversion level theta(t) in the short-rate SDE.
        Obtained by differentiating the drift condition.
        """
        eps   = 1e-6
        f_up  = self.curve.inst_forward(t + eps)
        f_dn  = self.curve.inst_forward(max(0.0, t - eps))
        df_dt = (f_up - f_dn) / (eps + min(t, eps))
        return df_dt + self.kappa * self.curve.inst_forward(t) + self.calc_variance(t)

    def calc_B(self, t, T_end):
        """B(t, T) factor: (1 - exp(-kappa*(T-t))) / kappa."""
        return (1.0 - math.exp(-self.kappa * (T_end - t))) / self.kappa

    def cond_df(self, t, T_end, x_state):
        """
        Conditional discount factor P(t, T | x(t)) in the HW model.
        Analytic formula: P(0,T)/P(0,t) * exp(-B(t,T)*x - 0.5*B^2*Var[x(t)]).
        """
        df_0T  = self.curve.get_discount(0, T_end)
        df_0t  = self.curve.get_discount(0, t)
        B      = self.calc_B(t, T_end)
        var_xt = self.calc_variance(t)
        return (df_0T / df_0t) * np.exp(-B * x_state - 0.5 * B ** 2 * var_xt)


# ---------------------------------------------------------------------------
# Bachelier (normal) swaption pricing utilities.
# ---------------------------------------------------------------------------

def norm_model_premium(fwd, strike, n_vol, t_exp, pv01):
    """Normal (Bachelier) model swaption price for a payer."""
    if n_vol <= 1e-8:
        return pv01 * max(fwd - strike, 0.0)
    sigma_sqrt_t = n_vol * math.sqrt(t_exp)
    d = (fwd - strike) / sigma_sqrt_t
    return pv01 * ((fwd - strike) * norm.cdf(d) + sigma_sqrt_t * norm.pdf(d))


def norm_model_vega(fwd, strike, n_vol, t_exp, pv01):
    """Sensitivity of the Bachelier price to a unit change in normal vol."""
    if n_vol <= 1e-8:
        return 0.0
    sigma_sqrt_t = n_vol * math.sqrt(t_exp)
    d = (fwd - strike) / sigma_sqrt_t
    return pv01 * math.sqrt(t_exp) * norm.pdf(d)


def extract_implied_vol(target_val, fwd, strike, t_exp, pv01, guess_vol):
    """
    Newton-Raphson inversion of the Bachelier formula to recover
    the implied normal vol from a given swaption price.
    Returns a floor of 1e-6 if the target is at or below intrinsic.
    """
    intrinsic = pv01 * max(fwd - strike, 0.0)
    if target_val <= intrinsic:
        return 1e-6

    vol = guess_vol
    for _ in range(20):      # slightly more iterations than before for robustness
        price = norm_model_premium(fwd, strike, vol, t_exp, pv01)
        err   = price - target_val
        if abs(err) < 1e-9:
            break
        vega = norm_model_vega(fwd, strike, vol, t_exp, pv01)
        if vega < 1e-10:
            break
        vol = max(1e-6, vol - err / vega)
    return vol


# ---------------------------------------------------------------------------
# Analytical European swaption pricer under Hull-White.
# Uses the Jamshidian decomposition: the swaption equals a portfolio of
# bond options with the critical short-rate root as the common strike.
# ---------------------------------------------------------------------------
class EuroPricer:
    def __init__(self, hw_dynamics):
        self.hw = hw_dynamics

    def evaluate_payer(self, start_t, schedule, fixed_rate):
        """
        Price a payer swaption expiring at start_t with payment dates in schedule
        and fixed coupon fixed_rate (as a decimal).

        Steps:
          1. Find x* such that the swap NPV = 0 at expiry (Jamshidian root).
          2. Decompose into bond put options and sum weighted values.
        """
        def swap_npv(x):
            """Swap value as a function of the OU state x at expiry."""
            df_last   = self.hw.cond_df(start_t, schedule[-1], x)
            annuity   = sum(
                self.hw.cond_df(start_t, p, x) * (p - (schedule[i - 1] if i > 0 else start_t))
                for i, p in enumerate(schedule)
            )
            return 1.0 - df_last - fixed_rate * annuity

        std_dev = math.sqrt(self.hw.calc_variance(start_t))
        if std_dev < 1e-8:
            return max(swap_npv(0.0), 0.0)

        # Search the Jamshidian root over 9 standard deviations (wider than before)
        lo, hi = -9.0 * std_dev, 9.0 * std_dev
        npv_lo, npv_hi = swap_npv(lo), swap_npv(hi)

        if npv_lo > 0 and npv_hi > 0:
            x_star = -100.0
        elif npv_lo < 0 and npv_hi < 0:
            x_star = 100.0
        elif np.isnan(npv_lo) or np.isnan(npv_hi):
            x_star = 0.0
        else:
            try:
                x_star = brentq(swap_npv, lo, hi, xtol=1e-10)
            except Exception:
                x_star = 0.0

        # Sum bond-put option values (Jamshidian)
        opt_value = 0.0
        df_start  = self.hw.curve.get_discount(0, start_t)
        var_start  = self.hw.calc_variance(start_t)

        for i, p_time in enumerate(schedule):
            period   = p_time - (schedule[i - 1] if i > 0 else start_t)
            cashflow = fixed_rate * period
            if i == len(schedule) - 1:
                cashflow += 1.0   # notional repayment at maturity

            strike_i = self.hw.cond_df(start_t, p_time, x_star)
            df_i     = self.hw.curve.get_discount(0, p_time)
            v_p      = self.hw.calc_B(start_t, p_time) * math.sqrt(var_start)

            if v_p < 1e-8:
                put_opt = max(strike_i * df_start - df_i, 0.0)
            else:
                d1 = (math.log(df_i / (strike_i * df_start)) + 0.5 * v_p ** 2) / v_p
                d2 = d1 - v_p
                put_opt = strike_i * df_start * norm.cdf(-d2) - df_i * norm.cdf(-d1)

            opt_value += cashflow * put_opt

        return opt_value


# ---------------------------------------------------------------------------
# PDE engine for Bermudan swaption valuation via backward induction.
# Spatial grid = OU state x(t); time-stepping = implicit (Crank-Nicolson
# border with full-implicit interior) using a banded linear system.
# ---------------------------------------------------------------------------
def eval_bermudan_pde(hw_model, trade_params):
    final_mat     = trade_params["swap_end"]
    call_schedule = trade_params["exercise_dates"]
    strike_rate   = trade_params["strike"]

    # --- Spatial grid: symmetric around zero, width = 6 std devs at maturity ---
    nodes   = 351                          # more nodes for better accuracy (was 301)
    peak_sd = math.sqrt(hw_model.calc_variance(final_mat))
    bound_x = max(6.0 * peak_sd, 0.01)   # wider boundary than the original 5

    space_grid = np.linspace(-bound_x, bound_x, nodes)
    dx         = space_grid[1] - space_grid[0]

    # --- Time grid: uniform + exercise dates merged in, run backwards ---
    annual_steps = 250                     # finer time resolution (was 200)
    dt_base      = 1.0 / annual_steps
    t_uniform    = np.arange(final_mat, -dt_base / 2, -dt_base)
    t_array      = np.sort(
        np.unique(np.concatenate((t_uniform, call_schedule, [0.0])))
    )[::-1]

    # PDE state = continuation value at each spatial node
    state = np.zeros(nodes)

    for step in range(len(t_array) - 1):
        t_now  = t_array[step]
        t_next = t_array[step + 1]
        dt     = t_now - t_next

        if dt < 1e-8:
            continue

        # Early-exercise check: if we are on a call date, take max of hold / exercise
        if any(abs(t_now - cd) < 1e-8 for cd in call_schedule):
            pay_sched = np.arange(t_now + 1.0, final_mat + 1e-8, 1.0)
            if len(pay_sched) > 0:
                df_last  = hw_model.cond_df(t_now, pay_sched[-1], space_grid)
                annuity  = np.zeros(nodes)
                for j, p in enumerate(pay_sched):
                    tau = p - (pay_sched[j - 1] if j > 0 else t_now)
                    annuity += hw_model.cond_df(t_now, p, space_grid) * tau

                ex_val = 1.0 - df_last - strike_rate * annuity
                state  = np.maximum(state, ex_val)

        # PDE coefficients evaluated at the mid-point of the time step
        t_mid  = 0.5 * (t_now + t_next)
        alpha  = hw_model.drift_alpha(t_mid)
        sigma  = hw_model.volatility_at(t_mid)

        D  = sigma ** 2 / dx ** 2          # diffusion coefficient
        Mu = hw_model.kappa / dx           # drift coefficient (from mean-reversion)

        # Build banded tridiagonal matrix (implicit scheme)
        band = np.zeros((3, nodes))

        # Main diagonal (interior nodes)
        band[1, 1:-1] = 1.0 + dt * (D + space_grid[1:-1] + alpha)
        # Boundary nodes (one-sided drift)
        band[1,  0]   = 1.0 + dt * (-Mu * space_grid[0]  + space_grid[0]  + alpha)
        band[1, -1]   = 1.0 + dt * ( Mu * space_grid[-1] + space_grid[-1] + alpha)

        # Super-diagonal (upper band, interior)
        band[0, 2:]   = -dt * (0.5 * D - 0.5 * Mu * space_grid[1:-1])
        band[0, 1]    =  dt *  Mu * space_grid[0]

        # Sub-diagonal (lower band, interior)
        band[2, :-2]  = -dt * (0.5 * D + 0.5 * Mu * space_grid[1:-1])
        band[2, -2]   = -dt *  Mu * space_grid[-1]

        state = solve_banded((1, 1), band, state)

    # Interpolate PDE grid at x = 0 (the risk-neutral initial state)
    pv_at_zero = np.interp(0.0, space_grid, state)
    return pv_at_zero * trade_params["notional"]


# ---------------------------------------------------------------------------
# Hull-White calibration to market swaption normal vols.
# Optimises mean reversion (kappa) and piecewise vols jointly.
# Regularisation penalises vol-surface roughness between consecutive buckets.
# ---------------------------------------------------------------------------
def fit_hw_model(ts_curve, mkt_vols, config, start_guess=None):
    call_dates  = config["exercise_dates"]
    final_mat   = config["swap_end"]
    time_nodes  = sorted(set(call_dates + [final_mat]))
    n_vols      = len(time_nodes)

    # Default initial guess: kappa ~ 5%, sigma ~ 1.5% (slightly higher than before)
    if start_guess is None:
        guess_arr = np.concatenate(([0.05], np.full(n_vols, 0.015)))
    else:
        guess_arr = np.asarray(start_guess)

    lb = [1e-4] + [1e-4] * n_vols
    ub = [2.0]  + [0.5]  * n_vols   # kappa up to 200%, vol up to 50%

    # Pre-compute market targets (price + vega weight) for each swaption
    targets = []
    for item in mkt_vols:
        exp_t    = float(item["expiry"])
        tnr      = float(item["tenor"])
        mat_t    = exp_t + tnr
        vol_norm = float(item["vol_bps"]) / 10_000.0

        pay_sched = np.arange(exp_t + 1.0, mat_t + 1e-8, 1.0)
        df_start  = ts_curve.get_discount(0, exp_t)
        df_end    = ts_curve.get_discount(0, mat_t)
        pvbp      = sum(
            ts_curve.get_discount(0, p) * (p - (pay_sched[i - 1] if i > 0 else exp_t))
            for i, p in enumerate(pay_sched)
        )
        fwd_rate  = (df_start - df_end) / pvbp if pvbp > 0 else 0.0
        fixed_k   = float(config["strike"])

        tgt_price = norm_model_premium(fwd_rate, fixed_k, vol_norm, exp_t, pvbp)
        tgt_vega  = max(norm_model_vega(fwd_rate, fixed_k, vol_norm, exp_t, pvbp), 1e-6)
        targets.append((exp_t, pay_sched, fixed_k, tgt_price, tgt_vega, vol_norm, fwd_rate, pvbp))

    def loss_func(params):
        kappa_fit = params[0]
        vols_fit  = params[1:]
        model     = HWOneFactor(kappa_fit, vols_fit, time_nodes, ts_curve)
        engine    = EuroPricer(model)

        residuals = []
        for (e_t, p_sch, k_rate, _, _, m_vol, f_rate, a_bp) in targets:
            hw_price = engine.evaluate_payer(e_t, p_sch, k_rate)
            hw_iv    = extract_implied_vol(hw_price, f_rate, k_rate, e_t, a_bp, m_vol)
            residuals.append((hw_iv - m_vol) * 10_000.0)   # residual in bps

        # Smoothness regularisation: penalise vol steps between adjacent buckets
        reg = 0.005   # lighter than original 0.01 to allow more vol structure
        for i in range(1, len(vols_fit)):
            residuals.append(reg * (vols_fit[i] - vols_fit[i - 1]) * 10_000.0)

        return residuals

    result      = least_squares(loss_func, guess_arr, bounds=(lb, ub), xtol=1e-7, ftol=1e-7)
    best_kappa  = result.x[0]
    best_vols   = result.x[1:]
    fitted_hw   = HWOneFactor(best_kappa, best_vols, time_nodes, ts_curve)
    return fitted_hw, result.x


# ---------------------------------------------------------------------------
# Main entry point: read JSON from stdin, price, and compute Greeks.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    raw = sys.stdin.read()
    if not raw.strip():
        sys.exit(0)

    payload    = json.loads(raw)
    zc_inputs  = payload["zero_curve"]
    vol_inputs = payload["swaption_vols"]
    trade_spec = payload["bermudan_spec"]

    t_pts = [node["maturity"] for node in zc_inputs]
    r_pts = [node["rate"]     for node in zc_inputs]

    base_curve  = TermStructureCurve(t_pts, r_pts)
    hw_model, x_opt = fit_hw_model(base_curve, vol_inputs, trade_spec)
    base_price  = eval_bermudan_pde(hw_model, trade_spec)

    # --- Vega: bump each swaption vol by +1 bp and re-price ---
    vega_out = []
    for i, vpt in enumerate(vol_inputs):
        bumped = json.loads(json.dumps(vol_inputs))
        bumped[i]["vol_bps"] += 1.0

        hw_bumped, _ = fit_hw_model(base_curve, bumped, trade_spec, start_guess=x_opt)
        bumped_price  = eval_bermudan_pde(hw_bumped, trade_spec)

        exp_t, tnr = vpt["expiry"], vpt["tenor"]
        vega_out.append({
            "expiry": int(exp_t) if exp_t == int(exp_t) else exp_t,
            "tenor":  int(tnr)   if tnr   == int(tnr)   else tnr,
            "vega_dollars_per_bp": round(bumped_price - base_price, 4),
        })

    # --- Delta: bump each zero rate by +1 bp and re-price ---
    delta_out = []
    for i, node in enumerate(zc_inputs):
        bumped_r        = list(r_pts)
        bumped_r[i]    += 0.0001
        bumped_curve    = TermStructureCurve(t_pts, bumped_r)

        hw_bumped, _ = fit_hw_model(bumped_curve, vol_inputs, trade_spec, start_guess=x_opt)
        bumped_price  = eval_bermudan_pde(hw_bumped, trade_spec)

        delta_out.append({
            "maturity": node["maturity"],
            "delta_dollars_per_bp": round(bumped_price - base_price, 4),
        })

    # --- Curve snapshot: discount factors and forward rates ---
    curve_snap = [
        {
            "maturity":        t,
            "discount_factor": round(float(math.exp(-r_pts[i] * t)), 4),
            "forward_rate":    round(float(base_curve.fwd_rates[i]), 4),
        }
        for i, t in enumerate(t_pts)
    ]

    # --- Calibrated HW parameters ---
    hw_params = {"mean_reversion": round(hw_model.kappa, 6)}
    for i, v in enumerate(hw_model.vols):
        hw_params[f"sigma_{i + 1}"] = round(v, 6)

    output = {
        "curve": curve_snap,
        "calibration": {
            "model":      "Hull-White",
            "parameters": hw_params,
        },
        "price_dollars": round(base_price, 2),
        "vega":  vega_out,
        "delta": delta_out,
    }

    print(json.dumps(output, indent=2))