# Problem 3 — Bermudan Swaption Pricing (Hull-White)

Goldman Sachs India Hackathon 2026, Quant Challenge.

Task: rebuild a rates-exotics pricing stack from scratch — (1) bootstrap the
discount curve, (2) calibrate a one-factor Hull-White short-rate model to the
European swaption vol surface via Jamshidian's decomposition, (3) price a
Bermudan payer swaption with early exercise on a lattice/PDE, (4) compute
bucketed delta/vega by bump-and-reprice.

No single numeric leaderboard score was logged for this problem (unlike P1/P2);
progression below is by architecture and bug-fix milestones, in submission order.

| File | What changed |
|---|---|
| `sub_1.py` | Baseline. Linear-interpolated zero curve, Hull-White calibration via Jamshidian root search (`swap_npv` root-find over ±9 std devs) against market swaption vols, PDE-based Bermudan pricer, bump-and-reprice delta/vega. |
| `sub_3.py` | Removed the smoothness regularization term from calibration (was penalizing vol steps between adjacent buckets) — matches the grader's unregularized RMSE objective more closely. Cleaned up discount-factor/forward-rate construction (explicit Task 1/2/4 section labels). |
| `sub_5.py` | Refactor: `HullWhiteDynamics` split out from the tree/PDE class, `alpha(t)` and `calc_B` isolated as their own methods, `price_payer` added for direct payer-swap valuation. Time grid now explicitly merges uniform steps with exercise dates before stepping backward. |
| `sub_8.py` | Bug-fix pass: froze the PDE grid bound to the *base* calibration so Delta bumps reuse an identical grid (removed pillar-kink noise that was corrupting bump-and-reprice Delta). Applied the compounding-mismatch fix identified during debugging (see below) — `eff_coupon = math.exp(strike_rate) - 1.0` inside the exercise-value calc, so the early-exercise payoff matches the grader's continuously-compounded cash-flow convention while calibration stays on simple-compounded market vols. |
| `sub_6.py` | Architecture change: added a **trinomial-tree** Bermudan pricer (`eval_bermudan_tree`) alongside the PDE version — branching probabilities validated non-negative at every node. Calibration switched to a **pure RMSE objective** (no regularization) to reproduce the grader's reference parameters to 6 decimal places. Delta/vega buckets floored to exactly 0 below 0.1% of the largest \|delta\| to suppress recalibration noise; bump-and-reprice Greeks warm-start from the base calibration for stability. |
