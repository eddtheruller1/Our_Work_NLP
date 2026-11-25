"""The Net Optimism Oscillator (I_Net), the market phases and the
Buy / Sell / Hold strategy, with a backtest on the MASI.

Rules, fixed before looking at the results:
- Each headline: P(positive) and P(negative) from our fine-tuned DistilCamemBERT
  ("joy" and "fear").
- Week t: S_t = sum of (P_pos - P_neg) over its headlines, N_t = number of headlines.
- I_Net,t = sum_k w_k S_{t-k} / sum_k w_k N_{t-k}, k = 0..3, w = 4, 3, 2, 1
  (recent news weigh more, every headline counts once).
- Standard signal: Buy if I_Net > +20 %, Sell if I_Net < -20 %, Hold otherwise.
- Centred variant: I_Net minus its own average over the previous 52 weeks, same
  +-20 % thresholds (the press is positive by default; centring removes that bias).
- Position for week t+1 is decided at the close of week t: Buy -> invested in the
  MASI, Sell -> cash, Hold -> keep the previous position.
- Costs: a labelled assumption per one-way trade, tested at 0 %, 0.5 % and 1 %.
Output: results/barometer.json and results/charts/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import econ  # noqa: E402

HERE = Path(__file__).parent
CHARTS = HERE / "results" / "charts"
W = np.array([4, 3, 2, 1], dtype=float)
THRESH = 0.20
COSTS = (0.0, 0.005, 0.01)
BASE_COST = 0.005
BLUE, SLATE = "#2F5EA8", "#5E6B78"
PHASE_COLORS = {"Ajustement efficace": "#3E8E6A", "Sous-réaction": "#D08C2C", "Surréaction baissière": "#8E1B1B", "Consolidation / attentisme": "#9AA3AD"}
PHASE_EN = {"Ajustement efficace": "Efficient adjustment", "Sous-réaction": "Under-reaction", "Surréaction baissière": "Bearish over-reaction",
            "Consolidation / attentisme": "Consolidation / wait-and-see"}


def weekly_series():
    s = pd.read_csv(HERE / "data" / "scored.csv", parse_dates=["session"])
    s["d"] = s["p_pos"] - s["p_neg"]
    wk = s["session"].dt.to_period("W-FRI")
    news = pd.DataFrame({"S": s.groupby(wk)["d"].sum(), "N": s.groupby(wk).size()})
    masi, _ = econ.load_market()
    close = masi.groupby(masi.index.to_period("W-FRI")).last()
    df = pd.DataFrame({"close": close}).join(news, how="left").fillna({"S": 0.0, "N": 0})
    df["ret"] = np.log(df["close"]).diff()                       # weekly log return
    num = sum(W[k] * df["S"].shift(k) for k in range(4))
    den = sum(W[k] * df["N"].shift(k) for k in range(4))
    df["inet"] = num / den.replace(0, np.nan)
    df["inet_c"] = df["inet"] - df["inet"].shift(1).rolling(52, min_periods=26).mean()
    df["r4"] = df["ret"].rolling(4).sum()
    df["r4_sd"] = df["r4"].shift(1).rolling(52, min_periods=26).std()
    return df[df.index >= pd.Period("2020-10-02", "W-FRI")]


def signal(x):
    return np.where(x > THRESH, "Achat", np.where(x < -THRESH, "Vente", "Conservation"))


def phases(sig, r4, r4_sd):
    """Phase of each week, from the textual signal and the MASI's last 4 weeks."""
    out = []
    for s_, r, sd in zip(sig, r4, r4_sd):
        if np.isnan(r) or np.isnan(sd):
            out.append(None)
        elif r < -1.5 * sd and s_ != "Vente":
            out.append("Surréaction baissière")        # price falls far more than the news say
        elif (s_ == "Achat" and r > 0) or (s_ == "Vente" and r < 0):
            out.append("Ajustement efficace")          # price moves the way the news point
        elif s_ in ("Achat", "Vente"):
            out.append("Sous-réaction")                # a clear signal the price has not followed yet
        else:
            out.append("Consolidation / attentisme")   # neutral news, no strong price move
    return out


def backtest(df, sig_col, cost):
    """Weekly positions from the signal at t, applied to the return of t+1."""
    pos, cur = [], 1.0
    for s_ in df[sig_col]:
        cur = 1.0 if s_ == "Achat" else (0.0 if s_ == "Vente" else cur)
        pos.append(cur)
    pos = pd.Series(pos, index=df.index).shift(1).fillna(1.0)   # decided at t, held during t+1
    simple = np.exp(df["ret"]) - 1
    trades = pos.diff().abs().fillna(0)
    strat = pos * simple - cost * trades
    return strat, simple, pos, trades


def stats(r, trades=None, pos=None):
    eq = (1 + r).cumprod()
    years = len(r) / 52
    ann = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(52)
    dd = (eq / eq.cummax() - 1).min()
    out = {"annual_return": round(float(ann), 4), "annual_vol": round(float(vol), 4), "sharpe": round(float(ann / vol), 2) if vol else None,
           "max_drawdown": round(float(dd), 4), "final": round(float(eq.iloc[-1]), 3)}
    if trades is not None:
        out["trades"] = int(trades.sum())
        out["time_invested"] = round(float(pos.mean()), 3)
    return out


def chart_dual(df, col, title, fname, lang):
    d = df.dropna(subset=[col])
    x = d.index.to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=200)
    lo, hi = min(-0.6, d[col].min() - 0.05), max(0.8, d[col].max() + 0.05)
    ax.axhspan(THRESH, hi, color="#3E8E6A", alpha=0.10, lw=0)
    ax.axhspan(lo, -THRESH, color="#8E1B1B", alpha=0.10, lw=0)
    for y in (THRESH, -THRESH):
        ax.axhline(y, color="#8C8C8C", lw=0.8, ls="--")
    ax.axhline(0, color="#BFBFBF", lw=0.6)
    ax.scatter(x, d[col], s=9, color=BLUE, zorder=3, label="I_Net")
    ax.set_ylim(lo, hi)
    ax.set_ylabel("I_Net", color=BLUE, fontsize=11)
    ax.tick_params(axis="y", colors=BLUE)
    ax2 = ax.twinx()
    ax2.scatter(x, d["close"], s=7, color=SLATE, zorder=2, label="MASI")
    ax2.set_ylabel("MASI", color=SLATE, fontsize=11)
    ax2.tick_params(axis="y", colors=SLATE)
    buy, sell = ("Zone d'achat (> +20 %)", "Zone de vente (< −20 %)") if lang == "fr" else ("Buy zone (> +20%)", "Sell zone (< −20%)")
    ax.text(x[3], hi - 0.04, buy, fontsize=9, color="#3E8E6A", va="top")
    ax.text(x[3], lo + 0.04, sell, fontsize=9, color="#8E1B1B", va="bottom")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, ["I_Net", "MASI"], loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=9, markerscale=2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)
    ax.set_title(title, fontsize=12, loc="left", color="#1E1E1E")
    fig.tight_layout()
    fig.savefig(CHARTS / fname)
    plt.close(fig)


def chart_phases(df, col_phase, fname, lang):
    d = df.dropna(subset=[col_phase])
    x = d.index.to_timestamp(how="end")
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=200)
    ax.plot(x, d["close"], color="#BFBFBF", lw=1, zorder=1)
    for ph, c in PHASE_COLORS.items():
        m = d[col_phase] == ph
        lab = (ph if lang == "fr" else PHASE_EN[ph]) + f" ({int(m.sum())})"
        ax.scatter(x[m], d["close"][m], s=14, color=c, zorder=3, label=lab)
    ax.set_ylabel("MASI", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHARTS / fname)
    plt.close(fig)


def main():
    CHARTS.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "Calibri"
    df = weekly_series()
    df["sig"] = signal(df["inet"].fillna(0).values)
    df.loc[df["inet"].isna(), "sig"] = "Conservation"
    df["sig_c"] = signal(df["inet_c"].fillna(0).values)
    df["phase"] = phases(df["sig"], df["r4"], df["r4_sd"])
    df["phase_c"] = phases(df["sig_c"], df["r4"], df["r4_sd"])

    res = {"weeks": int(len(df)), "inet": {"mean": round(float(df["inet"].mean()), 3), "min": round(float(df["inet"].min()), 3), "max": round(float(df["inet"].max()), 3),
                                           "share_above": round(float((df["inet"] > THRESH).mean()), 3), "share_below": round(float((df["inet"] < -THRESH).mean()), 3)},
           "inet_c": {"mean": round(float(df["inet_c"].mean()), 3), "sd": round(float(df["inet_c"].std()), 3),
                      "share_above": round(float((df["inet_c"] > THRESH).mean()), 3), "share_below": round(float((df["inet_c"] < -THRESH).mean()), 3)}}
    for col in ("sig", "sig_c"):
        res[f"{col}_counts"] = {k: int(v) for k, v in df[col].value_counts().items()}
    for col in ("phase", "phase_c"):
        res[f"{col}_counts"] = {k: int(v) for k, v in df[col].value_counts().items()}

    # Backtests on the weeks where the centred signal exists (same period for every rule).
    bt = df[df["inet_c"].notna()].copy()
    res["backtest_period"] = [str(bt.index[0])[11:21], str(bt.index[-1])[11:21]]
    res["backtest_weeks"] = int(len(bt))
    res["strategies"] = {}
    curves = {}
    for name, col in (("fixed", "sig"), ("centred", "sig_c")):
        for c in COSTS:
            strat, simple, pos, trades = backtest(bt, col, c)
            res["strategies"][f"{name}_{c}"] = stats(strat, trades, pos)
            if c == BASE_COST:
                curves[name] = (1 + strat).cumprod()
                # H3: does the strategy beat buy-and-hold? Mean weekly difference, Newey-West t.
                diff = (strat - simple).dropna()
                fit = econ.sm.OLS(diff.values, np.ones(len(diff))).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
                res["strategies"][f"{name}_{c}"]["vs_bh_mean_weekly"] = round(float(diff.mean()), 5)
                same = bool(np.allclose(diff.values, 0))        # a rule that never trades is buy-and-hold
                res["strategies"][f"{name}_{c}"]["vs_bh_t"] = None if same else round(float(fit.tvalues[0]), 2)
                res["strategies"][f"{name}_{c}"]["vs_bh_p"] = None if same else round(float(fit.pvalues[0]), 3)
    simple = np.exp(bt["ret"]) - 1
    res["buy_hold"] = stats(simple.shift(-0).fillna(0))
    curves["bh"] = (1 + simple.fillna(0)).cumprod()
    res["curves"] = [{"week": str(i)[11:21], "bh": round(float(curves["bh"][i]), 4), "fixed": round(float(curves["fixed"][i]), 4),
                      "centred": round(float(curves["centred"][i]), 4)} for i in bt.index]

    # H2: does the oscillator announce next week's return? r(t+1) = a + b I(t) + c r(t), Newey-West.
    res["h2"] = {}
    for col in ("inet", "inet_c"):
        y = df["ret"].shift(-1).rename("y")
        X = pd.concat([df[col], df["ret"]], axis=1)
        fit = econ.nw_ols(y, X)
        res["h2"][col] = {"b": round(float(fit.params[col]), 4), "t": round(float(fit.tvalues[col]), 2), "p": round(float(fit.pvalues[col]), 3), "n": int(fit.nobs)}
    # Average next-week return after each signal (centred), for the strategy slide.
    nxt = df["ret"].shift(-1)
    res["next_ret_by_signal_c"] = {k: round(float((np.exp(nxt[df["sig_c"] == k]) - 1).mean() * 100), 3) for k in ("Achat", "Conservation", "Vente")}
    res["next_ret_by_signal_c_n"] = {k: int((df["sig_c"] == k).sum()) for k in ("Achat", "Conservation", "Vente")}

    for lang in ("fr", "en"):
        t1 = "Oscillateur d'Optimisme Net et MASI (seuils fixes à ±20 %)" if lang == "fr" else "Net Optimism Oscillator and the MASI (fixed ±20% thresholds)"
        t2 = "Oscillateur centré (écart à sa moyenne sur 52 semaines) et MASI" if lang == "fr" else "Centred oscillator (gap to its 52-week average) and the MASI"
        chart_dual(df, "inet", t1, f"inet_fixed_{lang}.png", lang)
        chart_dual(df, "inet_c", t2, f"inet_centred_{lang}.png", lang)
        chart_phases(df, "phase_c", f"phases_{lang}.png", lang)
    (HERE / "results" / "barometer.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "curves"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
