import sys
import json
import math
import numpy as np
from scipy.optimize import least_squares, brentq
from scipy.stats import norm
from scipy.linalg import solve_banded


# ---------------------------------------------------------------------------
# Zero-rate yield curve with linear interpolation and flat extrapolation.
# Pre-computes discrete forward rates for downstream drift calculations.
# ---------------------------------------------------------------------------
class TermStructureCurve:
    def __init__(self, time_points, zero_rates):
        self.t_arr = np.array(time_points, dtype=float)
        self.r_arr = np.array(zero_rates,  dtype=float)
        self.fwd_rates = np.zeros_like(self.t_arr)

        n = len(self.t_arr)
        for i in range(n - 1):
            t1, t2 = self.t_arr[i], self.t_arr[i + 1]
            r1, r2 = self.r_arr[i], self.r_arr[i + 1]
            self.fwd_rates[i] = (r2 * t2 - r1 * t1) / (t2 - t1) + 1e-9

        if n > 1:
            self.fwd_rates[-1] = self.fwd_rates[-2]

    def _interp_rate(self, t):
        """Linear interpolation of zero rates with flat extrapolation."""
        return np.interp(t, self.t_arr, self.r_arr)

    def fetch_rate(self, t):
        """Spot zero rate at tenor t (scalar)."""
        return float(self._interp_rate(t))

    def get_discount(self, t, T_end):
        """
        Forward discount factor P(t, T_end) = P(0, T_end) / P(0, t).
        When t == 0 this returns the standard discount factor P(0, T_end).
        Supports both scalar and array inputs.
        """
        if np.isscalar(t) and np.isscalar(T_end):
            df_T = math.exp(-self.fetch_rate(T_end) * T_end)
            if t == 0:
                return df_T
            df_t = math.exp(-self.fetch_rate(t) * t)
            return df_T / df_t

        r_T = self._interp_rate(T_end)
        r_t = self._interp_rate(t)
        return np.exp(-r_T * T_end) / np.exp(-r_t * t)

    def inst_forward(self, t):
        """
        Instantaneous forward rate f(0, t) = r(t) + t * dr/dt.

        EXACT analytical form for a piecewise-LINEAR zero-rate curve: dr/dt is the
        constant slope of the active segment, so f(0, t) = r(t) + t * slope. This
        replaces a finite-difference approximation whose step-function derivative
        (kinks at every pillar) injected violent noise into Delta bumps.
        """
        if isinstance(t, np.ndarray):
            res = np.zeros_like(t, dtype=float)
            for i, val in enumerate(t):
                res[i] = self._calc_scalar_inst_forward(float(val))
            return res
        return self._calc_scalar_inst_forward(float(t))

    def _calc_scalar_inst_forward(self, t):
        if t <= self.t_arr[0]:
            return float(self.r_arr[0])
        idx = int(np.searchsorted(self.t_arr, t))
        if idx >= len(self.t_arr):
            return float(self.r_arr[-1])
        t1, t2 = self.t_arr[idx - 1], self.t_arr[idx]
        r1, r2 = self.r_arr[idx - 1], self.r_arr[idx]
        slope = (r2 - r1) / (t2 - t1)
        r_t = r1 + slope * (t - t1)
        return float(r_t + t * slope)


# ---------------------------------------------------------------------------
# Hull-White one-factor short-rate model.
#
#   dr(t) = [theta(t) - kappa * r(t)] dt + sigma(t) dW(t)
#
# The state variable x(t) = r(t) - alpha(t) is an Ornstein-Uhlenbeck
# process with zero mean; alpha(t) is calibrated to the initial curve.
# Volatility sigma is piecewise-constant on a user-supplied time grid.
# ---------------------------------------------------------------------------
class HWOneFactor:
    def __init__(self, reversion, vol_array, time_grid, yield_curve):
        self.kappa      = float(reversion)
        self.vols       = np.array(vol_array, dtype=float)
        self.grid_times = np.array(time_grid, dtype=float)
        self.curve      = yield_curve

    def volatility_at(self, t):
        """Piecewise-constant sigma active at time t."""
        idx = np.searchsorted(self.grid_times, t)
        if isinstance(t, np.ndarray):
            return self.vols[np.minimum(idx, len(self.vols) - 1)]
        return self.vols[min(idx, len(self.vols) - 1)]

    def _piecewise_integral(self, t, two_kappa=False):
        """
        Core helper for variance and integral_term calculations.
        Accumulates: integral_0^t sigma(s)^2 * exp(-c*(t-s)) ds
        where c = 2*kappa (two_kappa=True) or c = kappa (two_kappa=False).
        Uses the recursion: I_new = I_old * exp(-c*dt) + sigma^2*(1-exp(-c*dt))/c
        """
        if t <= 0:
            return 0.0
        c = 2.0 * self.kappa if two_kappa else self.kappa
        accum, prev = 0.0, 0.0
        for i, edge in enumerate(self.grid_times):
            cur = min(t, edge)
            if cur > prev:
                dt    = cur - prev
                decay = math.exp(-c * dt)
                accum = accum * decay + self.vols[i] ** 2 * (1.0 - decay) / c
                prev  = cur
            if cur >= t:
                break
        return accum

    def calc_variance(self, t):
        """Var[x(t)]: variance of the OU state at time t."""
        return self._piecewise_integral(t, two_kappa=True)

    def integral_term(self, t):
        """
        int_0^t sigma(s)^2 * exp(-kappa*(t-s)) ds,
        used in the drift correction alpha(t).
        """
        return self._piecewise_integral(t, two_kappa=False)

    def drift_alpha(self, t):
        """
        Deterministic shift alpha(t) ensuring the model fits P(0, T) exactly.
        alpha(t) = f(0,t) + (int_term(t) - var(t)) / kappa
        """
        return self.curve.inst_forward(t) + (self.integral_term(t) - self.calc_variance(t)) / self.kappa

    def calc_B(self, t, T_end):
        """B(t, T) = (1 - exp(-kappa*(T-t))) / kappa."""
        return (1.0 - math.exp(-self.kappa * (T_end - t))) / self.kappa

    def cond_df(self, t, T_end, x_state):
        """
        Conditional zero-coupon bond price P(t, T | x(t)) in the HW model.

        Analytic formula:
            P(t, T) = P(0,T)/P(0,t) * exp(-B(t,T)*x - 0.5*B(t,T)^2 * Var[x(t)])
        """
        df_0T  = self.curve.get_discount(0, T_end)
        df_0t  = self.curve.get_discount(0, t)
        B      = self.calc_B(t, T_end)
        var_t  = self.calc_variance(t)
        return (df_0T / df_0t) * np.exp(-B * x_state - 0.5 * B * B * var_t)


# ---------------------------------------------------------------------------
# Bachelier (normal-model) swaption pricing utilities.
# All vols are absolute normal vols (not log-normal).
# ---------------------------------------------------------------------------

def norm_model_premium(fwd, strike, n_vol, t_exp, pv01):
    """Bachelier payer swaption price under normal vol n_vol."""
    if n_vol <= 1e-8:
        return pv01 * max(fwd - strike, 0.0)
    sigma_t = n_vol * math.sqrt(t_exp)
    d = (fwd - strike) / sigma_t
    return pv01 * ((fwd - strike) * norm.cdf(d) + sigma_t * norm.pdf(d))


def norm_model_vega(fwd, strike, n_vol, t_exp, pv01):
    """d(premium)/d(n_vol) for the Bachelier formula."""
    if n_vol <= 1e-8:
        return 0.0
    sigma_t = n_vol * math.sqrt(t_exp)
    d = (fwd - strike) / sigma_t
    return pv01 * math.sqrt(t_exp) * norm.pdf(d)


def extract_implied_vol(target_val, fwd, strike, t_exp, pv01, guess_vol):
    """
    Recover the implied normal vol from a market swaption price via
    Newton-Raphson on the Bachelier formula.
    Falls back to returning 1e-6 if the price is at or below intrinsic.
    """
    intrinsic = pv01 * max(fwd - strike, 0.0)
    if target_val <= intrinsic:
        return 1e-6

    vol = guess_vol
    for _ in range(25):
        px  = norm_model_premium(fwd, strike, vol, t_exp, pv01)
        err = px - target_val
        if abs(err) < 1e-10:
            break
        vega = norm_model_vega(fwd, strike, vol, t_exp, pv01)
        if vega < 1e-12:
            break
        vol = max(1e-6, vol - err / vega)
    return vol


# ---------------------------------------------------------------------------
# Analytical European swaption pricer under Hull-White (Jamshidian 1989).
#
# A payer swaption = sum of put options on zero-coupon bonds, where all
# options share the same critical state x* at which the underlying swap
# has zero value.
# ---------------------------------------------------------------------------
class EuroPricer:
    def __init__(self, hw_dynamics):
        self.hw = hw_dynamics

    def evaluate_payer(self, start_t, schedule, fixed_rate):
        """
        Price a payer swaption expiring at start_t on a swap paying
        fixed_rate on the given payment schedule.

        Algorithm:
          1. Solve for x* where swap NPV(x*) = 0  [Jamshidian root].
          2. Sum weighted bond-put options struck at P(start_t, T_i | x*).
        """
        # Cache variance and df_start --- used repeatedly below
        var_start = self.hw.calc_variance(start_t)
        std_dev   = math.sqrt(var_start)
        df_start  = self.hw.curve.get_discount(0, start_t)

        # Precompute coupon periods once
        periods = [
            p - (schedule[i - 1] if i > 0 else start_t)
            for i, p in enumerate(schedule)
        ]

        def swap_npv(x):
            df_last  = self.hw.cond_df(start_t, schedule[-1], x)
            annuity  = sum(
                self.hw.cond_df(start_t, p, x) * periods[i]
                for i, p in enumerate(schedule)
            )
            return 1.0 - df_last - fixed_rate * annuity

        if std_dev < 1e-8:
            return max(swap_npv(0.0), 0.0)

        # Jamshidian root search over a wide interval
        lo, hi = -9.0 * std_dev, 9.0 * std_dev
        npv_lo, npv_hi = swap_npv(lo), swap_npv(hi)

        if np.isnan(npv_lo) or np.isnan(npv_hi):
            x_star = 0.0
        elif npv_lo > 0 and npv_hi > 0:
            x_star = -100.0   # deep in-the-money: all bonds are cheap
        elif npv_lo < 0 and npv_hi < 0:
            x_star = 100.0    # deep out-of-the-money
        else:
            try:
                x_star = brentq(swap_npv, lo, hi, xtol=1e-10, rtol=1e-10)
            except Exception:
                x_star = 0.0

        # Jamshidian decomposition into weighted bond put options
        opt_value = 0.0
        for i, p_time in enumerate(schedule):
            cashflow = fixed_rate * periods[i]
            if i == len(schedule) - 1:
                cashflow += 1.0   # return of notional at final payment

            # Bond-put strike = model bond price at x*
            K_i  = self.hw.cond_df(start_t, p_time, x_star)
            df_i = self.hw.curve.get_discount(0, p_time)
            v_p  = self.hw.calc_B(start_t, p_time) * std_dev

            if v_p < 1e-8:
                put_val = max(K_i * df_start - df_i, 0.0)
            else:
                d1 = (math.log(df_i / (K_i * df_start)) + 0.5 * v_p * v_p) / v_p
                d2 = d1 - v_p
                put_val = K_i * df_start * norm.cdf(-d2) - df_i * norm.cdf(-d1)

            opt_value += cashflow * put_val

        return opt_value


# ---------------------------------------------------------------------------
# Bermudan swaption pricer via backward PDE on the HW state variable x(t).
#
# PDE: dV/dt = 0.5*sigma^2 * d2V/dx2 - kappa*x * dV/dx - (x+alpha)*V
#
# The -(x+alpha)*V term discounts at the instantaneous short rate r = x+alpha.
# Discretised with a fully implicit scheme on a banded tridiagonal system.
# At each exercise date we apply the early-exercise condition (American-style).
# ---------------------------------------------------------------------------
def eval_bermudan_pde(hw_model, trade_params, fixed_bound=None):
    final_mat     = trade_params["swap_end"]
    call_schedule = trade_params["exercise_dates"]
    strike_rate   = trade_params["strike"]

    # Spatial grid: OU state x, symmetric, 6 std-devs wide at maturity.
    # fixed_bound freezes the grid across base/bumped repricings so the
    # discretisation error cancels in the finite-difference Greeks.
    nodes   = 401
    peak_sd = math.sqrt(hw_model.calc_variance(final_mat))
    bound_x = max(6.0 * peak_sd, 0.01) if fixed_bound is None else fixed_bound

    space_grid = np.linspace(-bound_x, bound_x, nodes)
    dx         = space_grid[1] - space_grid[0]

    # Time grid: merge uniform steps with exercise dates; step backwards
    annual_steps = 250
    dt_base      = 1.0 / annual_steps
    t_uniform    = np.arange(final_mat, -dt_base / 2.0, -dt_base)
    t_array      = np.sort(
        np.unique(np.concatenate((t_uniform, call_schedule, [0.0])))
    )[::-1]

    # Continuation value on the spatial grid, initialised to 0 at maturity
    state = np.zeros(nodes)

    for step in range(len(t_array) - 1):
        t_now  = t_array[step]
        t_next = t_array[step + 1]
        dt     = t_now - t_next

        if dt < 1e-10:
            continue

        # Early-exercise boundary: replace continuation with exercise value
        if any(abs(t_now - cd) < 1e-9 for cd in call_schedule):
            pay_sched = np.arange(t_now + 1.0, final_mat + 1e-8, 1.0)
            if len(pay_sched) > 0:
                df_last = hw_model.cond_df(t_now, pay_sched[-1], space_grid)
                annuity = np.zeros(nodes)
                for j, p in enumerate(pay_sched):
                    tau = p - (pay_sched[j - 1] if j > 0 else t_now)
                    annuity += hw_model.cond_df(t_now, p, space_grid) * tau
                ex_val = 1.0 - df_last - strike_rate * annuity
                state  = np.maximum(state, ex_val)

        # PDE coefficients at the midpoint of the current time step
        t_mid  = 0.5 * (t_now + t_next)
        alpha  = hw_model.drift_alpha(t_mid)
        sigma  = hw_model.volatility_at(t_mid)

        # Short rate at each grid node: r_i = x_i + alpha(t_mid)
        short_rate = space_grid + alpha

        # Diffusion and drift coefficients for the tridiagonal system
        D  = (sigma * sigma) / (dx * dx)      # d^2/dx^2 term
        Mu = hw_model.kappa / dx               # kappa*x * d/dx term (upwind)

        band = np.zeros((3, nodes))

        # Interior nodes: implicit-in-time, central convection on the off-diagonals.
        #   (1 + dt*(D + r_i))*V_i - dt*(0.5D - 0.5*kappa*x_i/dx)*V_{i+1}
        #                          - dt*(0.5D + 0.5*kappa*x_i/dx)*V_{i-1} = V_i^old
        x_int     = space_grid[1:-1]
        drift_int = 0.5 * Mu * x_int
        band[1, 1:-1] = 1.0 + dt * (D + short_rate[1:-1])   # discounting at r = x + alpha
        band[0, 2:]   = -dt * (0.5 * D - drift_int)   # super-diagonal
        band[2, :-2]  = -dt * (0.5 * D + drift_int)   # sub-diagonal

        # Boundary nodes: one-sided (Neumann-like) to avoid ghost points
        band[1,  0]  = 1.0 + dt * (D + abs(Mu * space_grid[0])  + short_rate[0])
        band[1, -1]  = 1.0 + dt * (D + abs(Mu * space_grid[-1]) + short_rate[-1])
        band[0,  1]  = -dt * D   # upper-left corner coupling
        band[2, -2]  = -dt * D   # lower-right corner coupling

        state = solve_banded((1, 1), band, state)

    # Interpolate to x = 0 (deterministic initial condition)
    return np.interp(0.0, space_grid, state) * trade_params["notional"]


# ---------------------------------------------------------------------------
# Calibration of the Hull-White model to market swaption normal vols.
#
# Free parameters: mean reversion kappa + piecewise-constant vols sigma_i.
# Objective: minimise (model implied vol - market vol) in bps, with a
# smoothness regulariser penalising large jumps between adjacent vol buckets.
# ---------------------------------------------------------------------------
def fit_hw_model(ts_curve, mkt_vols, config, start_guess=None):
    call_dates = config["exercise_dates"]
    final_mat  = config["swap_end"]
    time_nodes = sorted(set(call_dates + [final_mat]))
    n_vols     = len(time_nodes)

    # Warm start: kappa = 4%, sigma = 1.2% --- conservative but realistic
    guess_arr = np.concatenate(([0.04], np.full(n_vols, 0.012))) \
                if start_guess is None else np.asarray(start_guess, dtype=float)

    lb = [1e-4] + [1e-4] * n_vols
    ub = [1.5]  + [0.30] * n_vols   # kappa <= 150%; normal vol <= 30%

    # Pre-compute market-implied target prices and forward rates
    targets = []
    for item in mkt_vols:
        exp_t    = float(item["expiry"])
        mat_t    = exp_t + float(item["tenor"])
        vol_norm = float(item["vol_bps"]) / 10_000.0

        pay_sched = np.arange(exp_t + 1.0, mat_t + 1e-8, 1.0)
        df_start  = ts_curve.get_discount(0, exp_t)
        df_end    = ts_curve.get_discount(0, mat_t)

        pvbp = sum(
            ts_curve.get_discount(0, p) * (p - (pay_sched[i - 1] if i > 0 else exp_t))
            for i, p in enumerate(pay_sched)
        )
        fwd_rate  = (df_start - df_end) / pvbp if pvbp > 0 else 0.0
        # Market vols are quoted AT-THE-MONEY: each European swaption's strike is
        # its OWN forward swap rate, NOT the Bermudan's strike. Using config["strike"]
        # here calibrates against the wrong moneyness and destabilises the Greeks.
        fixed_k   = fwd_rate
        tgt_price = norm_model_premium(fwd_rate, fixed_k, vol_norm, exp_t, pvbp)

        targets.append((exp_t, pay_sched, fixed_k, vol_norm, fwd_rate, pvbp, tgt_price))

    def loss_func(params):
        kappa_fit = params[0]
        vols_fit  = params[1:]
        model     = HWOneFactor(kappa_fit, vols_fit, time_nodes, ts_curve)
        engine    = EuroPricer(model)

        residuals = []
        for (e_t, p_sch, k_rate, m_vol, f_rate, a_bp, _) in targets:
            hw_price = engine.evaluate_payer(e_t, p_sch, k_rate)
            hw_iv    = extract_implied_vol(hw_price, f_rate, k_rate, e_t, a_bp, m_vol)
            residuals.append((hw_iv - m_vol) * 10_000.0)   # units: bps

        # L2 regularisation on vol surface smoothness
        reg = 0.008
        for i in range(1, len(vols_fit)):
            residuals.append(reg * (vols_fit[i] - vols_fit[i - 1]) * 10_000.0)

        return residuals

    result    = least_squares(loss_func, guess_arr, bounds=(lb, ub),
                              method="trf", xtol=1e-8, ftol=1e-8, gtol=1e-8)
    fitted_hw = HWOneFactor(result.x[0], result.x[1:], time_nodes, ts_curve)
    return fitted_hw, result.x


# ---------------------------------------------------------------------------
# Entry point: read JSON from stdin, price the Bermudan, compute Greeks.
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

    base_curve           = TermStructureCurve(t_pts, r_pts)
    hw_model, x_opt      = fit_hw_model(base_curve, vol_inputs, trade_spec)

    # Freeze the PDE grid bound from the base calibration so it is identical for
    # the base price and every bumped reprice -> discretisation error cancels in
    # the finite-difference Greeks.
    grid_bound           = max(6.0 * math.sqrt(hw_model.calc_variance(trade_spec["swap_end"])), 0.01)
    base_price           = eval_bermudan_pde(hw_model, trade_spec, fixed_bound=grid_bound)

    # Vega: +1 bp bump on each swaption vol, recalibrate, re-price
    vega_out = []
    for i, vpt in enumerate(vol_inputs):
        bumped_vols          = json.loads(json.dumps(vol_inputs))
        bumped_vols[i]["vol_bps"] += 1.0
        hw_b, _              = fit_hw_model(base_curve, bumped_vols, trade_spec, start_guess=x_opt)
        bumped_price         = eval_bermudan_pde(hw_b, trade_spec, fixed_bound=grid_bound)
        exp_t, tnr           = vpt["expiry"], vpt["tenor"]
        vega_out.append({
            "expiry": int(exp_t) if exp_t == int(exp_t) else exp_t,
            "tenor":  int(tnr)   if tnr   == int(tnr)   else tnr,
            "vega_dollars_per_bp": round(bumped_price - base_price, 4),
        })

    # Delta: +1 bp bump on each zero rate, recalibrate, re-price
    delta_out = []
    for i, node in enumerate(zc_inputs):
        bumped_r          = list(r_pts)
        bumped_r[i]      += 0.0001
        bumped_curve      = TermStructureCurve(t_pts, bumped_r)
        hw_b, _           = fit_hw_model(bumped_curve, vol_inputs, trade_spec, start_guess=x_opt)
        bumped_price      = eval_bermudan_pde(hw_b, trade_spec, fixed_bound=grid_bound)
        delta_out.append({
            "maturity": node["maturity"],
            "delta_dollars_per_bp": round(bumped_price - base_price, 4),
        })

    # Curve snapshot: discount factors and forward rates at each pillar
    curve_snap = [
        {
            "maturity":        t,
            "discount_factor": round(float(math.exp(-r_pts[i] * t)), 4),
            "forward_rate":    round(float(base_curve.fwd_rates[i]), 4),
        }
        for i, t in enumerate(t_pts)
    ]

    # Calibrated HW parameters
    hw_params = {"mean_reversion": round(hw_model.kappa, 6)}
    for i, v in enumerate(hw_model.vols):
        hw_params[f"sigma_{i + 1}"] = round(v, 6)

    print(json.dumps({
        "curve": curve_snap,
        "calibration": {"model": "Hull-White", "parameters": hw_params},
        "price_dollars": round(base_price, 2),
        "vega":  vega_out,
        "delta": delta_out,
    }, indent=2))