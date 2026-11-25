"""Step 5: from headline scores to weekly series, then the tests.

Everything a forecast uses is known at the end of week t (real time):
z-scores use the 52 previous weeks, the "normal tone" regression and the
market sentiment index are re-estimated each week on past data only.
Output: results/econ_results.json, results/chart_data.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).parent
MARKET = HERE / "data" / "market"
START, SPLIT, ROBUST = "2020-10-01", "2023-04-01", "2023-04-01"
NW_LAGS = 4


# ------------------------------------------------------------------ market
def load_market():
    """MASI daily closes and the daily prices of the shares (data/market/*.csv)."""
    masi = pd.read_csv(MARKET / "masi_daily.csv", parse_dates=["date"]).set_index("date")["close"].sort_index()
    masi = masi[masi.index >= "2019-06-01"]
    px = pd.read_csv(MARKET / "shares_daily.csv", parse_dates=["date"])
    return masi, px[px["date"] >= "2019-06-01"]


def weekly_market(masi, px):
    r = np.log(masi).diff().dropna()
    wk = r.index.to_period("W-FRI")
    w = pd.DataFrame({"rv": np.sqrt((r ** 2).groupby(wk).sum()) * 100,       # realized volatility, % per week
                      "ret": r.groupby(wk).sum() * 100,
                      "days": r.groupby(wk).size()})
    w = w[w["days"] >= 2]
    w["log_rv"] = np.log(w["rv"].clip(lower=0.05))

    close = px.pivot_table(index="date", columns="ticker", values="close").sort_index()
    ma50, hi, lo = close.rolling(50).mean(), close.rolling(250).max(), close.rolling(250).min()
    cwk = close.index.to_period("W-FRI")
    last = close.groupby(cwk).last()
    wret = np.log(last).diff()
    live = last.notna() & wret.notna()
    w["breadth"] = (wret > 0).astype(float).where(live).mean(axis=1)
    w["above_ma50"] = (close > ma50).astype(float).where(close.notna()).groupby(cwk).last().mean(axis=1)
    hl = ((close >= hi).astype(float) - (close <= lo).astype(float)).where(hi.notna())
    w["highs_lows"] = hl.groupby(cwk).mean().mean(axis=1)
    w["momentum"] = w["ret"].rolling(4).sum()
    value = px.pivot_table(index="date", columns="ticker", values="value").sum(axis=1, min_count=30)
    value = value[value > 0]
    w["log_turnover"] = np.log(value.groupby(value.index.to_period("W-FRI")).sum())
    w.loc[w.index < pd.Period("2024-01-05", "W-FRI"), "log_turnover"] = np.nan   # no volume data before 2024
    return w


# ------------------------------------------------------------------ news
def weekly_news():
    s = pd.read_csv(HERE / "data" / "scored.csv", parse_dates=["session"])
    wk = s["session"].dt.to_period("W-FRI")
    return pd.DataFrame({"n": s.groupby(wk).size(), "tone": s.groupby(wk)["tone"].mean(),
                         "neg": s.groupby(wk)["p_neg"].mean()})


def past_z(x: pd.Series, window=52, min_periods=26) -> pd.Series:
    """z-score against the previous 52 weeks only (never the current one)."""
    m = x.shift(1).rolling(window, min_periods=min_periods).mean()
    sd = x.shift(1).rolling(window, min_periods=min_periods).std()
    return (x - m) / sd


def realtime_surprise(df, window=104, min_periods=39):
    """tone_z(t) - normal tone given the market of week t, fitted on past weeks only."""
    X = pd.DataFrame({"ret": df["ret"], "abs_ret": df["ret"].abs(), "ret_lag": df["ret"].shift(1)})
    out = pd.Series(np.nan, index=df.index)
    for i in range(len(df)):
        lo = max(0, i - window)
        past = pd.concat([df["tone_z"].iloc[lo:i], X.iloc[lo:i]], axis=1).dropna()
        if len(past) < min_periods or X.iloc[i].isna().any() or np.isnan(df["tone_z"].iloc[i]):
            continue
        fit = sm.OLS(past["tone_z"], sm.add_constant(past[X.columns])).fit()
        out.iloc[i] = df["tone_z"].iloc[i] - fit.predict(sm.add_constant(X.iloc[[i]], has_constant="add")).iloc[0]
    return out


def realtime_msi(df, cols=("breadth", "above_ma50", "highs_lows", "momentum"), min_periods=52):
    """Market sentiment index: first principal component of the market readings
    (no volatility in it), loadings re-estimated each week on past weeks."""
    cols = list(cols)
    out = pd.Series(np.nan, index=df.index)
    for i in range(min_periods, len(df)):
        past = df[cols].iloc[:i + 1].dropna()
        if len(past) < min_periods or df[cols].iloc[i].isna().any():
            continue
        z = (past - past.iloc[:-1].mean()) / past.iloc[:-1].std()
        vals, vecs = np.linalg.eigh(np.cov(z.iloc[:-1].to_numpy(dtype=float).T))
        load = vecs[:, -1] * np.sign(vecs[0, -1])           # sign: higher breadth = higher index
        out.iloc[i] = float(z.iloc[-1].to_numpy(dtype=float) @ load)
    return out


# ------------------------------------------------------------------ tests
def nw_ols(y, X):
    data = pd.concat([y, X], axis=1).dropna()
    return sm.OLS(data.iloc[:, 0], sm.add_constant(data.iloc[:, 1:])).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})


def summary(fit, keep):
    return {"n": int(fit.nobs), "r2": round(float(fit.rsquared), 3), "r2_adj": round(float(fit.rsquared_adj), 3),
            "coef": {k: {"b": round(float(fit.params[k]), 4), "t": round(float(fit.tvalues[k]), 2),
                         "p": round(float(fit.pvalues[k]), 4)} for k in keep if k in fit.params}}


def oos(df, target, base_cols, extra_cols, start=SPLIT):
    """Expanding-window forecasts from `start`; OOS R2 against the base model and Clark-West test."""
    cols = base_cols + extra_cols
    data = df[[target] + cols].dropna()
    e0, e1, f0s, f1s = [], [], [], []
    for t in data.index[data.index >= pd.Period(start, "W-FRI")]:
        train, row = data[data.index < t], data.loc[[t]]
        if len(train) < 52:
            continue
        m0 = sm.OLS(train[target], sm.add_constant(train[base_cols])).fit()
        m1 = sm.OLS(train[target], sm.add_constant(train[cols])).fit()
        f0 = m0.predict(sm.add_constant(row[base_cols], has_constant="add")).iloc[0]
        f1 = m1.predict(sm.add_constant(row[cols], has_constant="add")).iloc[0]
        y = row[target].iloc[0]
        e0.append(y - f0), e1.append(y - f1), f0s.append(f0), f1s.append(f1)
    e0, e1, f0s, f1s = map(np.array, (e0, e1, f0s, f1s))
    if not len(e0):
        return {"n": 0}
    cw = e0 ** 2 - (e1 ** 2 - (f0s - f1s) ** 2)            # Clark-West (2007) adjusted difference
    fit = sm.OLS(cw, np.ones(len(cw))).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return {"n": int(len(e0)), "mse_base": round(float(np.mean(e0 ** 2)), 4), "mse_model": round(float(np.mean(e1 ** 2)), 4),
            "oos_r2_vs_base": round(float(1 - np.mean(e1 ** 2) / np.mean(e0 ** 2)), 4),
            "clark_west_t": round(float(fit.tvalues[0]), 2), "clark_west_p": round(float(1 - stats.norm.cdf(fit.tvalues[0])), 4)}


def build():
    masi, px = load_market()
    df = weekly_market(masi, px).join(weekly_news(), how="left")
    df["n"] = df["n"].fillna(0)
    df["attention"] = np.log1p(df["n"])
    df["tone_z"] = past_z(df["tone"])
    df["att_z"] = past_z(df["attention"])
    df["har_w"] = df["log_rv"]
    df["har_m"] = df["log_rv"].rolling(4).mean()
    df["har_q"] = df["log_rv"].rolling(13).mean()
    df["y_rv"] = df["log_rv"].shift(-1)                      # next week's (log) volatility
    df["y_ret"] = df["ret"].shift(-1)
    df["y_turn"] = df["log_turnover"].shift(-1)
    df["turn_m"] = df["log_turnover"].rolling(4).mean()
    df["surprise"] = realtime_surprise(df)
    df["surprise_neg"] = df["surprise"].clip(upper=0)
    df["surprise_pos"] = df["surprise"].clip(lower=0)
    df["msi"] = realtime_msi(df)
    df = df[df.index >= pd.Period(START, "W-FRI")]
    return df


def run(df, label=""):
    har = ["har_w", "har_m", "har_q"]
    models = {
        "M0 HAR (passé du marché)": har,
        "M1 HAR + ton": har + ["tone_z"],
        "M2 HAR + attention": har + ["att_z"],
        "M3 HAR + surprise (+ et −)": har + ["surprise_pos", "surprise_neg"],
        "M4 HAR + indice de sentiment de marché": har + ["msi"],
        "M5 HAR + indice de marché + surprise + attention": har + ["msi", "surprise_pos", "surprise_neg", "att_z"],
        "N  Presse seule (ton + attention)": ["tone_z", "att_z"],
    }
    full = df.dropna(subset=["y_rv", "msi", "surprise", "att_z", "tone_z"] + har)
    out = {"sample": [str(full.index.min()), str(full.index.max())], "weeks": int(len(full)), "in_sample": {}, "oos": {}}
    for name, cols in models.items():
        out["in_sample"][name] = summary(nw_ols(full["y_rv"], full[cols]), cols)
    for name, cols in models.items():
        if name.startswith("M0"):
            continue
        base = [] if name.startswith("N") else (har if not name.startswith("M5") else har + ["msi"])
        extra = [c for c in cols if c not in base]
        if not base:
            out["oos"][name] = oos_vs_mean(full, "y_rv", extra)
        else:
            out["oos"][name] = oos(full, "y_rv", base, extra)
    return out


def oos_vs_mean(df, target, cols, start=SPLIT):
    data = df[[target] + cols].dropna()
    e0, e1 = [], []
    for t in data.index[data.index >= pd.Period(start, "W-FRI")]:
        train, row = data[data.index < t], data.loc[[t]]
        if len(train) < 52:
            continue
        m1 = sm.OLS(train[target], sm.add_constant(train[cols])).fit()
        y = row[target].iloc[0]
        e0.append(y - train[target].mean())
        e1.append(y - m1.predict(sm.add_constant(row[cols], has_constant="add")).iloc[0])
    e0, e1 = np.array(e0), np.array(e1)
    if not len(e0):
        return {"n": 0}
    return {"n": int(len(e0)), "oos_r2_vs_mean": round(float(1 - np.mean(e1 ** 2) / np.mean(e0 ** 2)), 4)}


def main():
    df = build()
    res = {"weeks_total": int(len(df)), "headlines_per_week_median": float(df["n"].median()),
           "headlines_per_week_median_by_year": {str(k): float(v) for k, v in df["n"].groupby(df.index.year).median().items()}}

    # Why volatility: direction has no memory, the size of moves does.
    res["acf"] = {"ret": [round(float(df["ret"].autocorr(k)), 3) for k in range(1, 7)],
                  "log_rv": [round(float(df["log_rv"].autocorr(k)), 3) for k in range(1, 7)]}
    worst = df.dropna(subset=["surprise"]).nsmallest(3, "surprise")
    res["worst_surprise_weeks"] = [{"week": str(i), "surprise": round(float(r.surprise), 2), "ret": round(float(r.ret), 2),
                                    "rv_next": round(float(r.y_rv), 2) if pd.notna(r.y_rv) else None} for i, r in worst.iterrows()]

    # A. The mirror: how much of the tone does the week's market explain?
    mirror = nw_ols(df["tone_z"], pd.DataFrame({"ret": df["ret"], "abs_ret": df["ret"].abs(), "ret_lag": df["ret"].shift(1)}))
    res["mirror"] = summary(mirror, ["ret", "abs_ret", "ret_lag"])
    res["lead_lag"] = {}
    for k in (-2, -1, 0, 1, 2):                               # tone(t) against the MASI return of week t+k
        f = nw_ols(df["tone_z"], df["ret"].shift(-k).rename("r"))
        res["lead_lag"][str(k)] = {"b": round(float(f.params["r"]), 3), "t": round(float(f.tvalues["r"]), 2)}

    # B. The direction check: does the tone predict next week's return?
    res["direction"] = {"tone": summary(nw_ols(df["y_ret"], df[["tone_z", "ret"]]), ["tone_z", "ret"]),
                        "surprise": summary(nw_ols(df["y_ret"], df[["surprise", "ret"]]), ["surprise", "ret"])}
    res["direction"]["oos_tone"] = oos(df, "y_ret", ["ret"], ["tone_z"])

    # C. The main test: next week's volatility.
    res["volatility"] = run(df)
    res["volatility_from_2023"] = run(df[df.index >= pd.Period(ROBUST, "W-FRI")])

    # D. Volume (turnover data only from 2024).
    vol = df.dropna(subset=["y_turn", "log_turnover", "turn_m", "att_z", "surprise"])
    res["volume"] = {"weeks": int(len(vol)),
                     "base": summary(nw_ols(vol["y_turn"], vol[["log_turnover", "turn_m"]]), ["log_turnover", "turn_m"]),
                     "news": summary(nw_ols(vol["y_turn"], vol[["log_turnover", "turn_m", "att_z", "surprise_pos", "surprise_neg"]]),
                                     ["att_z", "surprise_pos", "surprise_neg"])}

    # Holm over the main hypotheses (written before the results).
    m5 = res["volatility"]["in_sample"]["M5 HAR + indice de marché + surprise + attention"]["coef"]
    family = {"surprise négative → volatilité": m5["surprise_neg"]["p"], "surprise positive → volatilité": m5["surprise_pos"]["p"],
              "attention → volatilité": m5["att_z"]["p"],
              "prévision hors échantillon (Clark-West)": res["volatility"]["oos"]["M5 HAR + indice de marché + surprise + attention"]["clark_west_p"],
              "ton → rendement suivant": res["direction"]["tone"]["coef"]["tone_z"]["p"]}
    adj = multipletests(list(family.values()), method="holm")[1]
    res["holm"] = {k: {"p": round(float(p), 4), "p_holm": round(float(a), 4)} for (k, p), a in zip(family.items(), adj)}
    T = res["volatility"]["weeks"]
    res["power_min_corr"] = round(2.80 / np.sqrt(T), 3)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "econ_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    chart = df[["rv", "tone_z", "surprise", "att_z", "n", "msi"]].copy()
    chart.index = chart.index.astype(str)
    chart.round(3).reset_index().rename(columns={"index": "week"}).to_json(HERE / "results" / "chart_data.json", orient="records")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
