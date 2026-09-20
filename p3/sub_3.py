import sys
import json
import math
import numpy as np
from scipy.optimize import least_squares, brentq
from scipy.stats import norm
from scipy.linalg import solve_banded


# ---------------------------------------------------------------------------
# Task 1: Yield Curve Construction
#
# Discount factor:  P(0,T) = exp(-r(T) * T)
# Forward rate (grid points):
#   f(0,Ti) = -[ln P(0,Ti+1) - ln P(0,Ti)] / (Ti+1 - Ti)   for i < n-1
#   f(0,Tn) = -[ln P(0,Tn) - ln P(0,Tn-1)] / (Tn - Tn-1)   backward diff
# For off-grid maturities: linear interpolation of zero rates.
# ---------------------------------------------------------------------------
class TermStructureCurve:
    def __init__(self, time_points, zero_rates):
        self.t_arr = np.array(time_points, dtype=float)
        self.r_arr = np.array(zero_rates,  dtype=float)

        n = len(self.t_arr)
        # Discount factors at grid pillars
        self.df_arr = np.exp(-self.r_arr * self.t_arr)

        # Forward rates using the exact log-ratio formula from the rubric
        self.fwd_rates = np.zeros(n)
        for i in range(n - 1):
            lnP_i  = -self.r_arr[i]   * self.t_arr[i]
            lnP_i1 = -self.r_arr[i+1] * self.t_arr[i+1]
            dt     = self.t_arr[i+1] - self.t_arr[i]
            self.fwd_rates[i] = -(lnP_i1 - lnP_i) / dt
        # Last point: backward difference
        if n > 1:
            lnP_n  = -self.r_arr[-1] * self.t_arr[-1]
            lnP_n1 = -self.r_arr[-2] * self.t_arr[-2]
            dt     = self.t_arr[-1] - self.t_arr[-2]
            self.fwd_rates[-1] = -(lnP_n - lnP_n1) / dt

    def _interp_rate(self, t):
        """Linear interpolation of zero rates with flat extrapolation."""
        return np.interp(t, self.t_arr, self.r_arr)

    def fetch_rate(self, t):
        return float(self._interp_rate(t))

    def get_discount(self, t_start, T_end):
        """
        P(t_start, T_end) = P(0, T_end) / P(0, t_start).
        Scalar or array inputs supported.
        """
        if np.isscalar(t_start) and np.isscalar(T_end):
            r_T = self.fetch_rate(T_end)
            df_T = math.exp(-r_T * T_end)
            if t_start == 0:
                return df_T
            r_t = self.fetch_rate(t_start)
            df_t = math.exp(-r_t * t_start)
            return df_T / df_t
        r_T = self._interp_rate(T_end)
        r_t = self._interp_rate(t_start)
        return np.exp(-r_T * T_end) / np.exp(-r_t * t_start)

    def inst_forward(self, t):
        """
        Instantaneous forward rate f(0, t) via numerical differentiation
        of ln P(0, t) = -r(t)*t  =>  f(0,t) = -d/dt[-r(t)*t] = r(t) + t*dr/dt.
        Uses a small finite difference on the interpolated zero rate.
        """
        eps = 1e-5
        if isinstance(t, np.ndarray):
            r_up  = self._interp_rate(np.minimum(t + eps, self.t_arr[-1]))
            t_up  = np.minimum(t + eps, self.t_arr[-1])
            r_dn  = self._interp_rate(np.maximum(t - eps, 0.0))
            t_dn  = np.maximum(t - eps, 0.0)
            lnP_up = -r_up * t_up
            lnP_dn = -r_dn * t_dn
            return -(lnP_up - lnP_dn) / (t_up - t_dn + 1e-30)
        t_up  = min(t + eps, self.t_arr[-1])
        t_dn  = max(t - eps, 0.0)
        lnP_up = -self.fetch_rate(t_up) * t_up
        lnP_dn = -self.fetch_rate(t_dn) * t_dn
        return -(lnP_up - lnP_dn) / (t_up - t_dn + 1e-30)


# ---------------------------------------------------------------------------
# Task 2: Hull-White One-Factor Model
#
#   dr(t) = [theta(t) - a*r(t)] dt + sigma(t) dW(t)
#
# With the substitution x(t) = r(t) - alpha(t), where alpha(t) is chosen
# to fit the initial yield curve exactly:
#
#   alpha(t) = f(0,t) + sigma^2/(2*a^2) * (1 - exp(-a*t))^2
#
# x(t) is then an OU process: dx = -a*x dt + sigma dW
#
# Key quantities (piecewise-constant sigma on time_grid):
#   B(t,T)    = (1 - exp(-a*(T-t))) / a
#   Var[x(t)] = int_0^t sigma(s)^2 * exp(-2a(t-s)) ds
#   P(t,T|x)  = P(0,T)/P(0,t) * exp(-B(t,T)*x - 0.5*B(t,T)^2*Var[x(t)])
# ---------------------------------------------------------------------------
class HWOneFactor:
    def __init__(self, mean_rev, vol_array, time_grid, yield_curve):
        self.a          = float(mean_rev)
        self.vols       = np.array(vol_array, dtype=float)
        self.grid_times = np.array(time_grid,  dtype=float)
        self.curve      = yield_curve

    def volatility_at(self, t):
        """Piecewise-constant sigma(t)."""
        idx = np.searchsorted(self.grid_times, t)
        if isinstance(t, np.ndarray):
            return self.vols[np.minimum(idx, len(self.vols) - 1)]
        return self.vols[min(idx, len(self.vols) - 1)]

    def _piecewise_exp_integral(self, t, c):
        """
        Computes int_0^t sigma(s)^2 * exp(-c*(t-s)) ds
        exactly for piecewise-constant sigma using the recursion:
            I(t_i) = I(t_{i-1}) * exp(-c*dt) + sigma_i^2 * (1-exp(-c*dt)) / c
        """
        if t <= 0:
            return 0.0
        accum = 0.0
        prev  = 0.0
        for i, edge in enumerate(self.grid_times):
            cur = min(t, edge)
            if cur > prev:
                dt    = cur - prev
                e_cdt = math.exp(-c * dt)
                accum = accum * e_cdt + self.vols[i] ** 2 * (1.0 - e_cdt) / c
                prev  = cur
            if cur >= t:
                break
        return accum

    def calc_variance(self, t):
        """Var[x(t)] = int_0^t sigma(s)^2 * exp(-2a*(t-s)) ds."""
        return self._piecewise_exp_integral(t, 2.0 * self.a)

    def calc_B(self, t, T):
        """B(t,T) = (1 - exp(-a*(T-t))) / a."""
        return (1.0 - math.exp(-self.a * (T - t))) / self.a

    def alpha(self, t):
        """
        Deterministic drift: alpha(t) = f(0,t) + sigma^2/(2a^2)*(1-exp(-a*t))^2
        For piecewise-constant sigma, the sigma^2 term uses the variance
        formula: Var[x(t)] = int_0^t sigma^2*exp(-2a*(t-s)) ds, so:
            alpha(t) = f(0,t) + B(0,t)^2 * a^2 / (2a^2) ... 
        More precisely using the standard HW result:
            alpha(t) = f(0,t) + 0.5 * (sigma(t)/a)^2 * (1-exp(-a*t))^2
        For piecewise sigma we use the exact integral-based formula:
            alpha(t) = f(0,t) + Var[x(t)] * ... 
        The correct general formula is:
            alpha(t) = f(0,t) + (1/(2)) * d/dt [ B(0,t)^2 * Var_integrated ]
        We use the standard form which for piecewise sigma becomes:
            alpha(t) = f(0,t) + int_0^t sigma(s)^2 * B(s,t) * exp(-a*(t-s)) ds
        which equals: integral_term(t) where the kernel is B(s,t)*exp(-a*(t-s))
        = (1-exp(-a*(t-s)))/a * exp(-a*(t-s))
        = [exp(-a*(t-s)) - exp(-2a*(t-s))] / a
        So: alpha(t) = f(0,t) + [int_1(t) - int_2(t)] / a
        where int_1 = int sigma^2 * exp(-a*(t-s)) ds  (c=a)
              int_2 = int sigma^2 * exp(-2a*(t-s)) ds (c=2a) = Var[x(t)]
        """
        f0t  = self.curve.inst_forward(t)
        int1 = self._piecewise_exp_integral(t, self.a)
        int2 = self.calc_variance(t)
        return f0t + (int1 - int2) / self.a

    def cond_df(self, t, T, x_state):
        """
        Conditional bond price P(t,T|x(t)):
            P(t,T) = P(0,T)/P(0,t) * exp(-B(t,T)*x - 0.5*B(t,T)^2 * Var[x(t)])
        """
        df_0T = self.curve.get_discount(0, T)
        df_0t = self.curve.get_discount(0, t)
        B     = self.calc_B(t, T)
        var_t = self.calc_variance(t)
        return (df_0T / df_0t) * np.exp(-B * x_state - 0.5 * B * B * var_t)


# ---------------------------------------------------------------------------
# Bachelier (normal-model) swaption formulas for calibration.
# ---------------------------------------------------------------------------

def bachelier_price(fwd, strike, n_vol, t_exp, annuity):
    """Normal model payer swaption price."""
    if n_vol <= 1e-10:
        return annuity * max(fwd - strike, 0.0)
    sig_t = n_vol * math.sqrt(t_exp)
    d = (fwd - strike) / sig_t
    return annuity * ((fwd - strike) * norm.cdf(d) + sig_t * norm.pdf(d))


def bachelier_vega(fwd, strike, n_vol, t_exp, annuity):
    """d(price)/d(n_vol) for the Bachelier formula."""
    if n_vol <= 1e-10:
        return 0.0
    sig_t = n_vol * math.sqrt(t_exp)
    d = (fwd - strike) / sig_t
    return annuity * math.sqrt(t_exp) * norm.pdf(d)


def implied_normal_vol(mkt_price, fwd, strike, t_exp, annuity, vol0):
    """Newton-Raphson inversion of Bachelier formula."""
    intrinsic = annuity * max(fwd - strike, 0.0)
    if mkt_price <= intrinsic:
        return 1e-6
    vol = vol0
    for _ in range(30):
        px  = bachelier_price(fwd, strike, vol, t_exp, annuity)
        err = px - mkt_price
        if abs(err) < 1e-12:
            break
        vg  = bachelier_vega(fwd, strike, vol, t_exp, annuity)
        if vg < 1e-14:
            break
        vol = max(1e-7, vol - err / vg)
    return vol


# ---------------------------------------------------------------------------
# European swaption pricer via Jamshidian decomposition.
# ---------------------------------------------------------------------------
class EuroPricer:
    def __init__(self, hw):
        self.hw = hw

    def price_payer(self, expiry, pay_dates, fixed_rate):
        """
        Payer swaption expiring at `expiry`, with payments on `pay_dates`.
        Uses Jamshidian's decomposition into bond put options.
        """
        var_e   = self.hw.calc_variance(expiry)
        std_e   = math.sqrt(var_e)
        df_0_e  = self.hw.curve.get_discount(0, expiry)

        # Period fractions
        periods = []
        for i, p in enumerate(pay_dates):
            prev = pay_dates[i - 1] if i > 0 else expiry
            periods.append(p - prev)

        def swap_val(x):
            """Swap NPV as function of OU state x at expiry."""
            total = -self.hw.cond_df(expiry, pay_dates[-1], x)  # -P(e,T_n)
            for i, p in enumerate(pay_dates):
                total += -fixed_rate * periods[i] * self.hw.cond_df(expiry, p, x)
            total += 1.0   # receive floating (=1 at par)
            return total   # positive = payer in-the-money

        if std_e < 1e-10:
            return max(swap_val(0.0), 0.0)

        lo, hi = -8.0 * std_e, 8.0 * std_e
        v_lo, v_hi = swap_val(lo), swap_val(hi)

        if np.isnan(v_lo) or np.isnan(v_hi):
            x_star = 0.0
        elif v_lo > 0 and v_hi > 0:
            x_star = lo - 1.0
        elif v_lo < 0 and v_hi < 0:
            x_star = hi + 1.0
        else:
            try:
                x_star = brentq(swap_val, lo, hi, xtol=1e-12, rtol=1e-10, maxiter=100)
            except Exception:
                x_star = 0.0

        # Bond put options: put on P(e, T_i) struck at K_i = P(e, T_i | x*)
        value = 0.0
        for i, p in enumerate(pay_dates):
            cf    = fixed_rate * periods[i] + (1.0 if i == len(pay_dates) - 1 else 0.0)
            K_i   = self.hw.cond_df(expiry, p, x_star)
            df_0p = self.hw.curve.get_discount(0, p)
            v_p   = self.hw.calc_B(expiry, p) * std_e

            if v_p < 1e-10:
                put = max(K_i * df_0_e - df_0p, 0.0)
            else:
                log_m = math.log(df_0p / (K_i * df_0_e))
                d1    = (log_m + 0.5 * v_p * v_p) / v_p
                d2    = d1 - v_p
                put   = K_i * df_0_e * norm.cdf(-d2) - df_0p * norm.cdf(-d1)

            value += cf * put

        return value


# ---------------------------------------------------------------------------
# Task 3: Bermudan swaption via backward PDE (implicit finite difference).
#
# State variable: x(t) = r(t) - alpha(t), Ornstein-Uhlenbeck process.
# PDE (in backward form):
#   -dV/dt = 0.5*sigma^2 * d2V/dx2 - a*x * dV/dx - (x + alpha(t))*V
#
# Fully implicit scheme; early exercise enforced at each call date.
# ---------------------------------------------------------------------------
def price_bermudan_pde(hw, trade):
    T_end      = trade["swap_end"]
    ex_dates   = sorted(trade["exercise_dates"])
    K          = trade["strike"]
    notional   = trade["notional"]

    # Spatial grid
    N       = 401
    n_sd    = 6.0
    peak_sd = math.sqrt(hw.calc_variance(T_end))
    x_max   = max(n_sd * peak_sd, 0.05)
    x_grid  = np.linspace(-x_max, x_max, N)
    dx      = x_grid[1] - x_grid[0]

    # Time grid: uniform + exercise dates, run backward
    steps_per_yr = 365
    dt0          = 1.0 / steps_per_yr
    t_uniform    = np.arange(0.0, T_end + dt0, dt0)
    t_all        = np.unique(np.concatenate((t_uniform, ex_dates, [0.0, T_end])))
    t_all        = np.sort(t_all)[::-1]  # descending: T_end -> 0

    # Terminal condition: V(T_end, x) = 0 (swap expires worthless at final mat)
    V = np.zeros(N)

    ex_set = set(round(e, 10) for e in ex_dates)

    for k in range(len(t_all) - 1):
        t_now  = t_all[k]
        t_next = t_all[k + 1]
        dt     = t_now - t_next
        if dt < 1e-12:
            continue

        # Early exercise at call dates
        if round(t_now, 10) in ex_set:
            pay_dates = np.arange(t_now + 1.0, T_end + 1e-9, 1.0)
            if len(pay_dates) > 0:
                annuity = np.zeros(N)
                for j, p in enumerate(pay_dates):
                    tau = p - (pay_dates[j-1] if j > 0 else t_now)
                    annuity += hw.cond_df(t_now, p, x_grid) * tau
                bond_T = hw.cond_df(t_now, pay_dates[-1], x_grid)
                ex_val = 1.0 - bond_T - K * annuity
                V = np.maximum(V, ex_val)

        # Coefficients at midpoint
        t_mid   = 0.5 * (t_now + t_next)
        sig     = hw.volatility_at(t_mid)
        alph    = hw.alpha(t_mid)
        r_grid  = x_grid + alph          # short rate r = x + alpha

        sig2    = sig * sig
        D       = sig2 / (2.0 * dx * dx)  # half-diffusion coefficient

        # Central difference for drift: -a*x * dV/dx
        # Upwind coefficient for each node
        drift   = -hw.a * x_grid          # drift velocity = -a*x

        # Build tridiagonal: (I + dt*L)*V_new = V_old
        # L_ii = r_i + 2D,  L_{i,i+1} = -D + drift_i/(2dx), L_{i,i-1} = -D - drift_i/(2dx)
        diag  = np.ones(N)
        upper = np.zeros(N)
        lower = np.zeros(N)

        # Interior nodes
        for sign in [None]:  # just a scope block
            idx = slice(1, N - 1)
            d_i   = drift[idx] / (2.0 * dx)
            diag[idx]    = 1.0 + dt * (2.0 * D + r_grid[idx])
            upper[idx]   = -dt * (D - d_i)    # coefficient of V[i+1]
            lower[idx]   = -dt * (D + d_i)    # coefficient of V[i-1]

        # Boundary: Dirichlet (option worthless far from ATM)
        diag[0]  = 1.0;  upper[0]  = 0.0;  lower[0]  = 0.0;  V[0]  = 0.0
        diag[-1] = 1.0;  upper[-1] = 0.0;  lower[-1] = 0.0;  V[-1] = 0.0

        # Pack into solve_banded format: (upper, diag, lower)
        ab = np.zeros((3, N))
        ab[0, 1:]  = upper[:-1]   # super-diagonal shifted
        ab[1, :]   = diag
        ab[2, :-1] = lower[1:]    # sub-diagonal shifted

        V = solve_banded((1, 1), ab, V)
        V = np.maximum(V, 0.0)    # option value non-negative

    return float(np.interp(0.0, x_grid, V)) * notional


# ---------------------------------------------------------------------------
# Task 2: Calibration of Hull-White to market swaption normal vols.
# Free params: a (mean reversion), sigma_1..sigma_n (piecewise vols).
# ---------------------------------------------------------------------------
def calibrate_hw(curve, swaption_vols, config, x0=None):
    ex_dates   = config["exercise_dates"]
    T_end      = config["swap_end"]
    K          = float(config["strike"])
    time_nodes = sorted(set(ex_dates + [T_end]))
    n          = len(time_nodes)

    # Initial guess
    if x0 is None:
        x0 = np.array([0.03] + [0.008] * n)
    else:
        x0 = np.asarray(x0, dtype=float)

    lb = np.array([1e-4] + [1e-4] * n)
    ub = np.array([2.0]  + [0.25] * n)

    # Pre-compute market quantities (independent of model params)
    mkt = []
    for item in swaption_vols:
        exp_t  = float(item["expiry"])
        mat_t  = exp_t + float(item["tenor"])
        mkt_nv = float(item["vol_bps"]) / 10_000.0

        pay_dates = list(np.arange(exp_t + 1.0, mat_t + 1e-9, 1.0))
        df_e  = curve.get_discount(0, exp_t)
        df_m  = curve.get_discount(0, mat_t)
        ann   = sum(
            curve.get_discount(0, p) * (p - (pay_dates[i-1] if i > 0 else exp_t))
            for i, p in enumerate(pay_dates)
        )
        fwd   = (df_e - df_m) / ann if ann > 1e-12 else 0.0
        tgt   = bachelier_price(fwd, K, mkt_nv, exp_t, ann)
        mkt.append((exp_t, pay_dates, K, mkt_nv, fwd, ann, tgt))

    def residuals(params):
        a_val  = params[0]
        s_vals = params[1:]
        hw     = HWOneFactor(a_val, s_vals, time_nodes, curve)
        ep     = EuroPricer(hw)
        res    = []
        for (exp_t, pay_dates, K_, mkt_nv, fwd, ann, tgt) in mkt:
            hw_px = ep.price_payer(exp_t, pay_dates, K_)
            hw_nv = implied_normal_vol(hw_px, fwd, K_, exp_t, ann, mkt_nv)
            res.append((hw_nv - mkt_nv) * 10_000.0)   # bps residual

        # Mild smoothness regularisation
        lam = 0.005
        for i in range(1, len(s_vals)):
            res.append(lam * (s_vals[i] - s_vals[i-1]) * 10_000.0)
        return res

    sol    = least_squares(residuals, x0, bounds=(lb, ub),
                           method="trf", xtol=1e-9, ftol=1e-9, gtol=1e-9,
                           max_nfev=2000)
    hw_cal = HWOneFactor(sol.x[0], sol.x[1:], time_nodes, curve)
    return hw_cal, sol.x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    raw = sys.stdin.read().strip()
    if not raw:
        sys.exit(0)

    payload    = json.loads(raw)
    zc_data    = payload["zero_curve"]
    vol_data   = payload["swaption_vols"]
    trade      = payload["bermudan_spec"]

    t_pts = [d["maturity"] for d in zc_data]
    r_pts = [d["rate"]     for d in zc_data]

    curve              = TermStructureCurve(t_pts, r_pts)
    hw_cal, x_opt      = calibrate_hw(curve, vol_data, trade)
    base_price         = price_bermudan_pde(hw_cal, trade)

    # Task 1 output: discount factors + forward rates
    curve_out = []
    for i, t in enumerate(t_pts):
        curve_out.append({
            "maturity":        t,
            "discount_factor": round(float(curve.df_arr[i]), 6),
            "forward_rate":    round(float(curve.fwd_rates[i]), 6),
        })

    # Task 2 output: calibrated HW parameters
    hw_params = {"mean_reversion": round(hw_cal.a, 6)}
    for i, v in enumerate(hw_cal.vols):
        hw_params[f"sigma_{i+1}"] = round(float(v), 6)

    # Task 4: Vega -- bump each swaption vol by +1bp, recalibrate, reprice
    vega_out = []
    for i, vpt in enumerate(vol_data):
        bumped           = json.loads(json.dumps(vol_data))
        bumped[i]["vol_bps"] += 1.0
        hw_b, _          = calibrate_hw(curve, bumped, trade, x0=x_opt)
        bp               = price_bermudan_pde(hw_b, trade)
        et, tn           = vpt["expiry"], vpt["tenor"]
        vega_out.append({
            "expiry": int(et) if et == int(et) else et,
            "tenor":  int(tn) if tn == int(tn) else tn,
            "vega_dollars_per_bp": round(bp - base_price, 4),
        })

    # Task 4: Delta -- bump each zero rate by +1bp, recalibrate, reprice
    delta_out = []
    for i, node in enumerate(zc_data):
        r_bumped       = list(r_pts)
        r_bumped[i]   += 0.0001
        c_bumped       = TermStructureCurve(t_pts, r_bumped)
        hw_b, _        = calibrate_hw(c_bumped, vol_data, trade, x0=x_opt)
        bp             = price_bermudan_pde(hw_b, trade)
        delta_out.append({
            "maturity": node["maturity"],
            "delta_dollars_per_bp": round(bp - base_price, 4),
        })

    print(json.dumps({
        "curve":       curve_out,
        "calibration": {"model": "Hull-White", "parameters": hw_params},
        "price_dollars": round(base_price, 2),
        "vega":  vega_out,
        "delta": delta_out,
    }, indent=2))