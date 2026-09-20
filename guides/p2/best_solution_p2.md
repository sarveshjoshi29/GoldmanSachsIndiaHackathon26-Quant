---
marp: true
paginate: false
math: katex
size: 16:9
style: |
  :root {
    --navy: #16284a;
    --navy2: #1f3a66;
    --card: #24426f;
    --gold: #d4af37;
    --ink: #eef2f8;
    --muted: #a9bbd6;
    --good: #4cc38a;
  }
  section {
    background: var(--navy);
    color: var(--ink);
    font-family: "Segoe UI", Helvetica, Arial, sans-serif;
    padding: 40px 56px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
  }
  h1 {
    color: #ffffff;
    font-size: 34px;
    margin: 0 0 2px 0;
    letter-spacing: .3px;
  }
  .sub {
    color: var(--muted);
    font-size: 17px;
    margin: 0 0 18px 0;
    border-bottom: 2px solid var(--gold);
    padding-bottom: 12px;
  }
  .row { display: flex; align-items: stretch; gap: 14px; }
  .col { display: flex; flex-direction: column; gap: 10px; flex: 1; }
  .box {
    background: var(--card);
    border: 1px solid #34568c;
    border-radius: 10px;
    padding: 12px 14px;
    text-align: center;
  }
  .box .t { font-size: 18px; font-weight: 700; color: #fff; }
  .box .d { font-size: 13px; color: var(--muted); margin-top: 3px; }
  .engine {
    background: linear-gradient(180deg, #1f3a66, #15294b);
    border: 2px solid var(--gold);
    border-radius: 12px;
    padding: 14px;
    text-align: center;
  }
  .engine .et { font-size: 18px; font-weight: 800; color: var(--gold); }
  .stage {
    background: #24426f; border: 1px solid #34568c; border-radius: 8px;
    padding: 8px 10px; margin-top: 8px; font-size: 14px; color: var(--ink);
  }
  .stage b { color: #fff; }
  .arrow { align-self: center; color: var(--gold); font-size: 26px; font-weight: 900; }
  .down { text-align:center; color: var(--gold); font-size: 22px; font-weight: 900; margin: 2px 0; }
  .out .t { color: var(--good); }
  .foot {
    margin-top: 16px; display: flex; justify-content: space-between;
    align-items: center; font-size: 14px; color: var(--muted);
  }
  .pill {
    background: var(--gold); color: #16284a; font-weight: 800;
    padding: 5px 12px; border-radius: 20px; font-size: 15px;
  }
  code { background: #0e1c36; color: #ffd970; padding: 1px 6px; border-radius: 5px; font-size: 13px; }
---

# ETF Proxy Models — Problem 2 · Best Solution
<p class="sub">Robust Relaxed Elastic Net · static NAV &amp; Risk models + performance-driven PnL hedge</p>

<div class="row">

<div class="col" style="flex:0.9">
<div class="box"><div class="t">EOD Prices</div><div class="d">2025 daily, 5 target ETFs</div></div>
<div class="down">↓</div>
<div class="box"><div class="t">Daily Returns</div><div class="d">simple <code>pct_change</code></div></div>
<div class="down">↓</div>
<div class="box"><div class="t">Winsorize</div><div class="d">clip 1% / 99% tails</div></div>
</div>

<div class="arrow">→</div>

<div class="col" style="flex:1.25">
<div class="engine">
<div class="et">Robust Relaxed Elastic Net</div>
<div class="stage"><b>1 · Select</b> — ElasticNetCV<br>sparse, walk-forward CV (TimeSeriesSplit)</div>
<div class="stage"><b>2 · Refit</b> — HuberRegressor<br>unshrink + outlier-robust betas</div>
</div>
<div class="down">↓ &nbsp; reused on returns &amp; on dollar-PnL &nbsp; ↓</div>
</div>

<div class="arrow">→</div>

<div class="col out" style="flex:1">
<div class="box"><div class="t">NAV weights</div><div class="d">all proxies → min MAPE</div></div>
<div class="box"><div class="t">Risk weights</div><div class="d">rates + credit → DV01 / CS01</div></div>
<div class="box"><div class="t">Hedge basket</div><div class="d">fit on <code>−portfolio PnL</code> → max HER</div></div>
</div>

</div>

<div class="foot">
<span>Key: drop sum-to-1 constraints · keep raw betas · Proxy ETFs included in hedge</span>
<span class="pill">Score 27 → 71.5</span>
</div>
