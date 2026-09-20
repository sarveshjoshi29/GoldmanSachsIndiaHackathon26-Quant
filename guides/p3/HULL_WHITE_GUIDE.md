# A Plain-Language Guide to Stochastic Differential Equations & the Hull-White Model

*Written for the p3 rates challenge: calibrating Hull-White to price a Bermudan swaption.*
*No prior stochastic-calculus background assumed — we build everything from scratch.*

---

## Part 0 — The 30-second summary

Interest rates wiggle randomly over time. To price anything that depends on
future rates (like a swaption), we need a **mathematical model of how the rate
wiggles**. The Hull-White model is one such model. It says:

> The short interest rate drifts toward a moving target, is pulled back when it
> strays (mean reversion), and gets kicked around by random noise each instant.

Written as an equation:

```
dr(t) = [ θ(t) − a·r(t) ] dt  +  σ(t) dW(t)
        └──── drift ────┘       └── randomness ──┘
```

Everything below explains what every symbol means, *why* it's built this way,
and how it lets you price the Bermudan in your task.

---

## Part 1 — What is a Stochastic Differential Equation (SDE)?

### 1.1 Start with an ordinary differential equation (ODE)

You already know equations like:

```
dx/dt = 5        →   "x grows at 5 units per second"
```

This is **deterministic**: if you know where you start, you know *exactly* where
you'll be at every future time. `x(t) = x(0) + 5t`. No surprises.

We often write the same thing as:

```
dx = 5 dt
```

Read this as: *"in a tiny time step `dt`, `x` changes by `5·dt`."* It's just the
ODE rearranged. `dx` = small change in x, `dt` = small change in time.

### 1.2 The problem: the real world is noisy

Interest rates don't move smoothly and predictably. They jiggle. A purely
deterministic equation can never capture a rate that's partly random. We need to
**add a random term** to the equation. That's the entire idea of an SDE:

```
dx  =  (predictable part) dt  +  (random part) × (random noise)
```

- The **predictable part** is called the **drift**. It's the average direction
  things move — same role as the right-hand side of an ODE.
- The **random part** is called the **diffusion** (or volatility). It controls
  *how big* the random jiggles are.
- The **random noise** itself is the new ingredient. We need to define it
  carefully — that's Brownian motion.

### 1.3 Brownian motion `W(t)` — the engine of randomness

**Brownian motion** (also called a *Wiener process*, hence `W`) is the standard
mathematical model of "pure random wiggling." Picture a speck of dust in water
getting bumped by molecules from all directions. Its key properties:

1. **Starts at zero:** `W(0) = 0`.
2. **Random increments:** over a time step `Δt`, the change `ΔW = W(t+Δt) − W(t)`
   is a random draw from a **normal (bell-curve) distribution** with:
   - mean `0` (no preferred direction), and
   - variance `Δt` (so standard deviation `√Δt`).

   In symbols: `ΔW ~ Normal(0, Δt)`, i.e. `ΔW = √Δt · Z` where `Z` is a standard
   normal random number (mean 0, variance 1).
3. **Independent increments:** what happens in one time interval is independent
   of what happened before. No memory.

The crucial, slightly weird fact: because the standard deviation of a step is
`√Δt`, the randomness shrinks *slower* than the drift as `Δt → 0`. Over a tiny
step, the drift contributes ~`Δt` (tiny) but the random part contributes ~`√Δt`
(bigger, since √of a small number is larger than the number). **Over short
horizons, randomness dominates; over long horizons, drift dominates.** This is
why a single day of rates looks like pure noise but a decade shows a trend.

In differential notation we write `dW(t)`: an "infinitesimal random kick" that is
Normal with mean 0 and variance `dt`.

### 1.4 Putting it together — a general SDE

A general one-dimensional SDE looks like:

```
dx(t) = μ(x, t) dt  +  σ(x, t) dW(t)
```

- `μ(x,t)` = **drift** function — the average pull per unit time.
- `σ(x,t)` = **diffusion / volatility** function — the size of the random kicks.
- `dW(t)` = the Brownian random kick.

**How to read it:** "Over the next instant `dt`, `x` moves by `μ dt` (the
predictable nudge) plus `σ` times a random normal kick of size `√dt`."

A simulatable version (this is literally how you'd code it — the
**Euler–Maruyama** scheme):

```python
x_next = x + mu(x, t)*dt + sigma(x, t)*sqrt(dt)*np.random.randn()
```

That one line is the computational heart of every Monte Carlo rate simulation,
including Longstaff-Schwartz in your Task 3.

### 1.5 A word on Itô calculus (just enough)

Ordinary calculus breaks slightly when `dW` is involved, because `dW` is so
jagged that `(dW)²` is NOT negligible — it behaves like `dt`. The rulebook for
doing calculus with `dW` is called **Itô calculus**, and its one rule you must
know is **Itô's lemma**, the stochastic version of the chain rule. The single
fact to remember:

```
(dW)² = dt        (and dt·dW = 0,  (dt)² = 0)
```

You rarely need to derive things by hand for this task — but this is *why*
Hull-White has clean closed-form bond prices: the math works out to Gaussian
(normal) distributions, which integrate nicely. We'll use that.

---

## Part 2 — Modeling interest rates: the "short rate"

### 2.1 What is the short rate `r(t)`?

The **short rate** `r(t)` is the interest rate for borrowing/lending over an
*infinitesimally short* period starting at time `t`. Think of it as the
"instantaneous" risk-free rate at each moment. It's a useful abstraction because:

- If you know how `r(t)` behaves over all future times, you can price *any*
  interest-rate product by discounting cashflows along simulated/averaged paths.
- In particular, the price today of a **zero-coupon bond** paying \$1 at time `T`
  is the expected discount factor:

```
P(0, T) = E[ exp( − ∫₀ᵀ r(s) ds ) ]
```

This expectation `E[...]` is over all the random paths the short rate could take.
**A short-rate model is just a specific SDE for `r(t)`** that makes this
expectation computable.

### 2.2 What we want from a good short-rate model

1. **Mean reversion** — rates don't wander off to infinity; they get pulled back
   toward some long-run level. Real rates behave this way (central banks, economic
   forces). A model without mean reversion (like plain Brownian motion) would let
   rates drift arbitrarily far — unrealistic.
2. **Fits today's yield curve exactly** — whatever rates the market quotes today,
   the model must reproduce them, or every price is off from the start.
3. **Analytical tractability** — we want closed-form formulas for bond prices and
   European options, so calibration and risk are fast (you have a 180-second
   budget and must reprice hundreds of times for Greeks).

Hull-White is the simplest model that delivers all three. That's why it's the
industry workhorse and why the task uses it.

---

## Part 3 — The Hull-White model, term by term

The Hull-White one-factor model:

```
dr(t) = [ θ(t) − a·r(t) ] dt  +  σ(t) dW(t)
```

Let's dissect every piece.

### 3.1 The mean-reversion term: `− a·r(t)`

This is the heart of the model. `a` is the **mean-reversion speed** (a positive
constant you calibrate). Look at the sign:

- If `r(t)` is **high**, then `−a·r(t)` is a **large negative** number → the drift
  pushes the rate **down**.
- If `r(t)` is **low** (or negative), `−a·r(t)` is **positive** → drift pushes the
  rate **up**.

So the rate is always pulled back toward a central level — like a spring. The
bigger `a` is, the **stronger and faster** the pull (rates snap back quickly).
A small `a` means rates wander more freely for longer.

**Intuition for `a`:** the "half-life" of a shock is `ln(2)/a`. If `a = 0.03`,
a shock to rates takes about `0.69/0.03 ≈ 23` years to half-decay — slow, persistent.
If `a = 0.3`, the half-life is ~2.3 years — fast reversion.

### 3.2 The drift target: `θ(t)`

`θ(t)` (theta) is a **time-dependent function**, not a constant. Combined with the
mean reversion, the rate reverts toward the level `θ(t)/a` at each instant.

Here's the clever bit that makes Hull-White special: **`θ(t)` is not free — it is
computed analytically so that the model reproduces today's yield curve exactly.**
You do not calibrate it. There is a known formula:

```
θ(t) = ∂f(0,t)/∂t  +  a·f(0,t)  +  (σ²/2a)·(1 − e^(−2at))
```

where `f(0,t)` is today's **instantaneous forward rate** curve (which you build in
Task 1!). The takeaway: `θ(t)` is a *plug* that bends the model's average path to
pass exactly through every market-quoted rate. This is what guarantees property #2
(fits today's curve). You don't even need to compute `θ(t)` explicitly in practice
— it gets absorbed into the bond-price formula (Section 4).

### 3.3 The volatility term: `σ(t) dW(t)`

`σ(t)` is the **volatility** — how hard the random kicks hit. In your task it's
**piecewise constant**: a different constant value on each interval between
exercise dates (`σ₁` on `[0,T₁]`, `σ₂` on `(T₁,T₂]`, etc.). This is one of the two
things you **calibrate** (along with `a`).

Why piecewise constant? Because it gives you just enough flexibility to match the
**term structure of volatility** — the market quotes different implied vols for
swaptions of different expiries, and a single constant `σ` can't fit them all.
Each `σᵢ` is a knob to match the vols in that time bucket.

- Bigger `σ` → wider spread of possible future rates → more valuable options.
- `σ` is what swaption prices are most sensitive to → that's your **Vega**.

### 3.4 Why "one-factor" and "Gaussian"?

- **One-factor:** there is a single source of randomness — one `dW`. The entire
  yield curve is driven by one random variable `r(t)`. This is a simplification
  (real curves can twist in more than one way), but it's fast and adequate for
  many products, including this Bermudan.
- **Gaussian:** because the equation is *linear* in `r` (the `r` only appears
  multiplied by constants, never squared or under a square root), the solution
  `r(t)` is **normally distributed**. This is the magic property:
  - normal distributions integrate in closed form → closed-form bond prices,
  - it's why **Jamshidian's decomposition** (below) works,
  - downside: rates can go **negative** (the normal distribution has no floor).
    In today's low/negative-rate world this is often considered acceptable or even
    desirable; it's the "Normal/Bachelier" convention your vols are quoted in.

---

## Part 4 — From the SDE to actual prices (why HW is tractable)

### 4.1 The closed-form zero-coupon bond price

Because `r(t)` is Gaussian, the bond-price expectation from Section 2.1 can be
solved exactly. Under Hull-White, the price at time `t` (when the short rate is
`r`) of a \$1 zero-coupon bond maturing at `T` is:

```
P(t, T) = A(t, T) · exp( − B(t, T) · r )
```

where

```
B(t, T) = (1 − e^(−a(T−t))) / a
```

and `A(t,T)` is a deterministic function built from today's discount curve and the
`a, σ` parameters (it contains the `θ` plug, so you never compute `θ` directly).

**Why this matters for you:** this single formula is the workhorse. Every node of
your trinomial tree, or every path of your Monte Carlo, evaluates bond prices with
this. The swap value at exercise is a sum of these bond prices. It's fast and
exact — no nested simulation needed.

### 4.2 European swaptions: Jamshidian's decomposition

A swaption is an option on a **swap**, and a swap is a bundle of cashflows, i.e. a
bundle of zero-coupon bonds (a **coupon bond**). Normally, an option on a bundle
is hard (the bundle's value depends on the joint behaviour of many rates).

**Jamshidian's trick** exploits the one-factor Gaussian structure: because *every*
bond price `P(t,T)` is a **monotonic** function of the single state variable `r`
(higher `r` → lower bond prices, always), there is exactly one critical rate `r*`
at which the swap is exactly at-the-money. This lets you **decompose** the
swaption into a **portfolio of options on individual zero-coupon bonds**, each of
which has a **closed-form** Hull-White price (a Black-76-like formula). Sum them
up → the European swaption price, analytically.

This is the method you use in **Task 2 (calibration)**: you need to price
European swaptions thousands of times while searching for `(a, σ₁..σₙ)` that match
the market vols. Closed-form Jamshidian makes each evaluation microseconds instead
of a simulation.

### 4.3 Bermudan swaptions: why we need a tree or Monte Carlo

A **Bermudan** swaption can be exercised at *several* dates (but only once). At
each exercise date you face a decision:

```
V = max( exercise_value , continuation_value )
```

- `exercise_value` = value of entering the swap right now (closed-form, from the
  bond formula).
- `continuation_value` = value of *keeping* the option alive for the future
  (depends on all later possible decisions).

This is an **optimal stopping problem** — there is no single closed-form answer,
because the decision at each date depends on future decisions. So you solve it
**backward** numerically:

- **Trinomial tree:** discretize `r(t)` onto a recombining lattice (rate can go up,
  stay, or down at each step — the three branches are calibrated to match HW's
  mean and variance). Roll backward from the last date: at each node take
  `max(exercise, continuation)`, discounting expected values back one step. The HW
  Gaussian structure is what makes the tree *recombine* (an up-then-down move lands
  on the same node as down-then-up), keeping it computationally small.
- **Monte Carlo + Longstaff-Schwartz:** simulate many random `r(t)` paths
  (using the Euler step from Section 1.4). The problem with MC and early exercise
  is you don't know the continuation value along a path. Longstaff-Schwartz
  estimates it by **regressing** the realized future payoffs against functions of
  the current state, at each exercise date, working backward. The regression's
  fitted value *is* your estimate of continuation value.

Either way, the **early-exercise premium** (the extra value of a Bermudan over a
European) comes out of this backward induction.

### 4.4 Greeks: bump-and-reprice (Task 4)

- **Vega** = sensitivity to each market vol quote. Bump one vol by +1bp,
  **re-calibrate** `(a, σ)`, re-price the Bermudan, take the difference.
- **Delta** = sensitivity to each zero rate. Bump one zero rate by +1bp,
  rebuild the curve, re-calibrate, re-price, difference.

This is why **speed matters**: every Greek is a full re-calibration + re-pricing.
With many curve points and vol quotes, you re-run the whole stack dozens of times.
Closed-form Jamshidian (calibration) + an efficient tree (pricing) keep you inside
the 180-second budget.

---

## Part 5 — Jamshidian's decomposition, step by step

**The problem.** A payer swaption is the right to enter a swap, and a swap is worth
a **bundle** of cashflows (a coupon bond). An option on a *sum* of things is
normally hard, because the sum's value depends on how all the pieces move together.

**Jamshidian's insight** (works precisely because Hull-White is one-factor):
everything at expiry depends on a single random number — the state `x`. And every
bond price `P(expiry, T | x)` is **monotonic** in `x` (higher `x` → higher rates →
lower bond price). So the swap's value is also monotonic in `x`: there is **exactly
one** critical state `x*` where the swap is precisely at-the-money (worth zero). On
one side of `x*` you exercise, on the other you don't. Because that boundary is a
single point, the option on the bundle **splits into a sum of independent options on
each individual bond**, each struck at that bond's value at `x*` — and each of those
has a closed-form (Black-style) price.

**The recipe (exactly what your `EuroPricer` does):**

1. **Write the swap value at expiry as a function of the state `x`:**
   ```
   swap(x) =  1                          (receive floating)
            − Σᵢ K·τᵢ·P(e,Tᵢ|x)          (pay fixed leg)
            − P(e,Tₙ|x)                   (final notional)
   ```
   which is positive when the payer is in-the-money.
2. **Find the critical state `x*`** where `swap(x*) = 0`. This is a one-dimensional
   root-find (your code uses `brentq` over ±8 standard deviations, with guards for
   the all-in / all-out cases).
3. **Turn each cashflow into a bond option.** At `x*`, each bond price `P(e,Tᵢ|x*)`
   becomes a **strike** `Kᵢ`. The swaption is now a **portfolio of bond put
   options**, one per cashflow, each weighted by its cashflow size
   `cᵢ = K·τᵢ` (+1 at the final date).
4. **Price each bond put in closed form** with a Black-76-style formula using the
   bond's Hull-White volatility `vᵢ = B(e,Tᵢ)·std_e`, then **sum**:
   `price = Σᵢ cᵢ · Putᵢ`.

**Why a put, for a payer?** A payer gains when rates *rise*, i.e. when bond prices
*fall*. An option that pays off when a bond price falls below a strike is a **put**
on that bond. So a payer swaption equals a basket of bond puts. (A receiver would be
a basket of bond calls.)

**Payoff:** an exact European price in microseconds, no simulation — which is what
makes the thousands of repricings in calibration (Task 2) feasible inside the time
limit.

---

## Part 6 — The backward-induction engine (your Bermudan PDE pricer)

A Bermudan can be exercised at several dates, and the value today depends on a whole
chain of future decisions — so you cannot price it forward. You solve it
**backward**, from the last date to today. Your code does this with a **PDE on a
grid**, which is a smooth, deterministic cousin of a tree.

### The state variable
Instead of tracking `r` directly, work with `x(t) = r(t) − α(t)`, where `α(t)` is the
deterministic (curve-fitting) part. `x` is a clean mean-reverting process centered at
0 (an **Ornstein-Uhlenbeck** process), which keeps the grid symmetric and stable. The
short rate is recovered as `r = x + α(t)`.

### The grid
- **Space:** a row of possible `x` values from `−x_max` to `+x_max` (your code: 401
  points, `x_max ≈ 6` standard deviations — wide enough to hold essentially all the
  probability).
- **Time:** small steps (daily, 1/365) from today out to `swap_end`, with the
  exercise dates inserted *exactly* so decisions land on grid times.

### The backward march
1. **Start at the end.** At `swap_end` the option is worth 0 everywhere (terminal
   condition `V = 0`).
2. **Step back one `dt` at a time.** Each step answers: "given the value one step in
   the future, what is it worth now?" — the discounted expected future value,
   accounting for (a) the random spreading of `x` (diffusion ½σ²·Vₓₓ), (b) the
   mean-reverting drift (−a·x·Vₓ), and (c) discounting at the current rate
   (−(x+α)·V). Your code does this with one **implicit finite-difference solve** per
   step — a fast tridiagonal system (`solve_banded`). "Implicit" just means a stable
   scheme that won't blow up even with large steps.
3. **At every exercise date, apply the early-exercise test.** Compute the value of
   exercising now (entering the payer swap at that node's rate):
   ```
   exercise(x) = 1 − P(t, swap_end | x) − K · annuity(x)
   ```
   then take, at every grid point:
   ```
   V = max( continuation(x), exercise(x) )
   ```
   You exercise wherever exercising beats holding — this single `max` is what
   captures the **early-exercise premium**.
4. **Keep marching** until `t = 0`.

### Read off the answer
At `t = 0` today's state is `x = 0`, so the price is `V` interpolated at `x = 0`,
multiplied by the notional.

### Why backward, and how it relates to trees / Longstaff-Schwartz
You must go backward because the exercise decision at each date needs the value of
*continuing*, which depends on all *later* dates. Starting at the end and working
back means the continuation value is already known when you reach each decision — the
essence of backward induction (and of optimal-stopping problems generally). Same
idea, different machinery: a **tree** discretizes into up/down branches;
**Longstaff-Schwartz** uses simulated paths plus regression to estimate continuation
values; your **PDE** discretizes the governing equation on a fixed grid and solves it
directly. For a clean one-factor model like this, the PDE is smooth, stable, and very
accurate.

---

## Part 7 — The whole pipeline in one picture

```
  Task 1                Task 2                    Task 3              Task 4
┌─────────┐   ┌──────────────────────┐   ┌──────────────────┐   ┌──────────┐
│ Zero    │   │ Calibrate a, σ(t):   │   │ Price Bermudan:  │   │ Bump &   │
│ curve → │ → │ choose params so     │ → │ backward induct  │ → │ reprice  │
│ DFs,    │   │ HW European prices   │   │ on tree / LSMC,  │   │ → Delta, │
│ forwards│   │ (Jamshidian) match   │   │ max(exercise,    │   │   Vega   │
│         │   │ market vols (min RMSE)│   │ continuation)    │   │          │
└─────────┘   └──────────────────────┘   └──────────────────┘   └──────────┘
   curve         the SDE gets its            the SDE is            sensitivities
   feeds θ(t)    a and σ pinned down         simulated/lattice'd   of the price
```

- The **curve** (Task 1) feeds `θ(t)`, locking the model to today's market.
- **Calibration** (Task 2) pins down `a` and `σ(t)` — the only free parameters —
  by matching European swaption prices to market vols.
- **Pricing** (Task 3) uses the fully-specified SDE to value the early-exercise
  Bermudan via backward induction.
- **Greeks** (Task 4) re-run the chain under small bumps.

---

## Part 8 — Glossary of every symbol

| Symbol | Name | What it is | Free or fixed? |
|---|---|---|---|
| `r(t)` | short rate | instantaneous risk-free rate at time `t` | the random variable |
| `dr(t)` | change in `r` | how `r` moves over an instant `dt` | — |
| `a` | mean-reversion speed | how strongly `r` is pulled back | **calibrated** |
| `θ(t)` | drift target | bends model to fit today's curve | fixed (from curve) |
| `σ(t)` | volatility | size of random kicks, piecewise-constant | **calibrated** |
| `dW(t)` | Brownian kick | random normal noise, variance `dt` | the randomness |
| `W(t)` | Brownian motion | cumulative random walk | — |
| `P(t,T)` | bond price | value at `t` of \$1 paid at `T` | closed-form |
| `f(0,t)` | forward rate | today's instantaneous forward curve | from Task 1 |
| `B(t,T)`, `A(t,T)` | HW coefficients | building blocks of `P(t,T)` | from `a, σ`, curve |

---

## Part 9 — Common pitfalls & sanity checks

1. **Number of `σ` parameters is dynamic.** It equals the number of breakpoint
   intervals `[0,T₁,...,Tₖ,swap_end]`. Read it from the input every time — never
   hardcode (the warning in the PS is explicit and hidden cases will differ).
2. **Negative rates are allowed** in Hull-White (Gaussian). Don't "fix" them by
   flooring at zero — the Normal/Bachelier vol convention expects them.
3. **`θ(t)` is never calibrated.** If you find yourself fitting `θ`, you've
   misunderstood — it's analytic from the curve. Only `a` and `σ` are free.
4. **Calibration objective is RMSE of vols in bps**, but pricing uses MAPE of the
   dollar price — different metrics, make sure your optimizer minimizes the right one.
5. **Recombining tree:** verify up-then-down lands on down-then-up, or your tree
   blows up exponentially and you'll bust the time limit.
6. **Sanity check the bond formula:** `P(t,t) = 1` and `P(0,T)` should match your
   Task-1 discount factors exactly. If not, your `A(t,T)` is wrong.
7. **Greeks should have sensible signs:** a *payer* swaption gains value when rates
   rise, so positive delta to rates; vega is positive everywhere (more vol = more
   option value).

---

## Part 10 — If you want to go deeper

- **Brigo & Mercurio, *Interest Rate Models — Theory and Practice***: the
  definitive reference; Chapter 3 covers Hull-White and Jamshidian in full.
- **Hull, *Options, Futures, and Other Derivatives***: gentler, has the trinomial
  tree construction step-by-step.
- **Original papers:** Hull & White (1990) for the model; Jamshidian (1989) for
  the decomposition; Longstaff & Schwartz (2001) for the MC regression method.

---

*The one sentence to remember: Hull-White is just an SDE that makes the short rate
spring back toward a curve-fitting target while getting randomly kicked — and
because it's linear (hence Gaussian), it hands you closed-form bond and European
prices, which you then stack into a backward-induction pricer for the Bermudan.*
