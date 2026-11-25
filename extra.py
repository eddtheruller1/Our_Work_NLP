"""Descriptive facts for the slides (no test here): a worked volatility example
on a real week, headlines per year, data coverage, and the market sentiment
index loadings over the whole sample (for description only: the tests use the
real-time version in econ.py).
Output: results/extra.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import econ

HERE = Path(__file__).parent


def main():
    masi, px = econ.load_market()
    out = {}
    m = masi[masi.index >= "2020-09-25"]
    r = np.log(m).diff().dropna()
    wk = r.index.to_period("W-FRI")
    rv = np.sqrt((r ** 2).groupby(wk).sum()) * 100
    top = rv[rv.index >= pd.Period("2020-10-02", "W-FRI")].idxmax()
    days = r[wk == top]
    prev_close = m[m.index < days.index[0]].iloc[-1]
    out["example_week"] = {
        "week": str(top), "prev_close": round(float(prev_close), 2),
        "days": [{"date": str(d.date()), "close": round(float(m.loc[d]), 2), "r_pct": round(float(v * 100), 2),
                  "r2": round(float((v * 100) ** 2), 2)} for d, v in days.items()],
        "rv_pct": round(float(rv.loc[top]), 2), "log_rv": round(float(np.log(rv.loc[top])), 2),
        "median_rv_pct": round(float(rv.median()), 2)}

    heads = pd.read_csv(HERE / "data" / "clean_headlines.csv", parse_dates=["session"])
    out["headlines_per_year"] = {str(k): int(v) for k, v in heads.groupby(heads["session"].dt.year).size().items()}
    out["market_share_of_headlines"] = round(float((heads["tag"] == "MARKET").mean()), 3)

    mcsv = pd.read_csv(econ.MARKET / "masi_daily.csv")
    scsv = pd.read_csv(econ.MARKET / "shares_daily.csv")
    out["masi_days"] = int(mcsv["date"].between("2020-10-01", "2026-10-16").sum())
    out["shares"] = int(scsv.loc[scsv["date"] >= "2020-10-01", "ticker"].nunique())
    out["share_rows"] = int(scsv["date"].between("2020-10-01", "2026-10-16").sum())
    out["value_from"] = str(scsv.loc[scsv["value"] > 0, "date"].min())

    w = econ.weekly_market(masi, px)
    cols = ["breadth", "above_ma50", "highs_lows", "momentum"]
    z = w[cols].dropna()
    z = (z - z.mean()) / z.std()
    vals, vecs = np.linalg.eigh(np.cov(z.to_numpy(dtype=float).T))
    load = vecs[:, -1] * np.sign(vecs[0, -1])
    out["msi_loadings_full_sample"] = {c: round(float(v), 2) for c, v in zip(cols, load)}
    out["msi_variance_share"] = round(float(vals[-1] / vals.sum()), 2)

    # Weekly MASI level and volatility, for the time-series chart.
    weekly_close = m.groupby(m.index.to_period("W-FRI")).last()
    w2 = pd.DataFrame({"close": weekly_close, "rv": rv}).dropna()
    w2 = w2[w2.index >= pd.Period("2020-10-02", "W-FRI")]
    out["masi_weekly"] = [{"week": str(i)[11:21], "close": round(float(r.close), 0), "rv": round(float(r.rv), 2)} for i, r in w2.iterrows()]

    # Correlations with next week's (log) volatility, on the weeks used in the tests.
    df = econ.build()
    cols = {"log_rv": "past", "msi": "mood", "tone_z": "tone", "surprise": "surprise", "att_z": "attention"}
    sample = df.dropna(subset=["y_rv", "msi", "surprise", "att_z", "tone_z", "har_q"])
    out["corr_next_vol"] = {v: round(float(sample[k].corr(sample["y_rv"])), 2) for k, v in cols.items()}
    out["corr_n"] = int(len(sample))
    out["scatter_tone"] = [[round(float(a), 2), round(float(b), 2)] for a, b in zip(sample["tone_z"], sample["y_rv"])]
    out["scatter_mood"] = [[round(float(a), 2), round(float(b), 2)] for a, b in zip(sample["msi"], sample["y_rv"])]

    # H2 as one joint test: do the week's return, its size and last week's return explain tone?
    X = pd.DataFrame({"ret": df["ret"], "abs_ret": df["ret"].abs(), "ret_lag": df["ret"].shift(1)})
    fit = econ.nw_ols(df["tone_z"], X)
    ft = fit.f_test("ret = 0, abs_ret = 0, ret_lag = 0")
    out["mirror_joint"] = {"F": round(float(np.squeeze(ft.fvalue)), 2), "p": round(float(ft.pvalue), 3), "n": int(fit.nobs)}

    # Out-of-sample forecasts week by week (same loop as econ.oos), for the forecast chart.
    har = ["har_w", "har_m", "har_q"]
    sets = {"har": har, "mood": har + ["msi"], "mood_press": har + ["msi", "surprise_pos", "surprise_neg", "att_z"]}
    data = sample[["y_rv"] + sets["mood_press"]].dropna()
    rows = []
    for tt in data.index[data.index >= pd.Period(econ.SPLIT, "W-FRI")]:
        train, row = data[data.index < tt], data.loc[[tt]]
        rec = {"week": str(tt + 1)[11:21], "actual": round(float(np.exp(row["y_rv"].iloc[0])), 2)}
        for name_, c in sets.items():
            f = econ.sm.OLS(train["y_rv"], econ.sm.add_constant(train[c])).fit()
            rec[name_] = round(float(np.exp(f.predict(econ.sm.add_constant(row[c], has_constant="add")).iloc[0])), 2)
        rows.append(rec)
    out["oos_series"] = rows
    (HERE / "results" / "extra.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
