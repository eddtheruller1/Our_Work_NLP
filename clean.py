"""Step 2: clean the raw headlines and give each one the session it counts for.

Rules (written before looking at any result):
1. A company search keeps a headline only if the company name is in the title
   (Google also matches the article body, which brings namesakes and noise).
2. A market search keeps a headline only in a Moroccan context: a Moroccan word
   in the title or a Moroccan outlet, and never a foreign exchange alone.
3. Job ads and duplicates (same title once punctuation and case are removed) go.
4. A headline counts at the first trading session strictly after its date:
   most headlines carry a date but no reliable hour, so this never uses news
   the market could not have read yet.
Output: data/clean_headlines.csv and data/funnel.json (how many left each step).
"""
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
MASI_CSV = HERE / "data" / "market" / "masi_daily.csv"

MOROCCAN_WORDS = ("maroc", "marocain", "casablanca", "masi", "bank al-maghrib", "bam", "ammc", "dirham",
                  "rabat", "tanger", "agadir", "marrakech", "royaume", "cdg", "ocp", "onee", "hcp")
MOROCCAN_OUTLETS = ("médias24", "medias24", "le360", "boursenews", "le matin", "l'economiste", "leseco", "la vie éco",
                    "challenge", "h24info", "hespress", "le desk", "telquel", "maroc diplomatique", "lebrief",
                    "finances news", "ecoactu", "aujourd'hui le maroc", "map", "barlamane", "yabiladi", "2m",
                    "snrt", "infomédiaire", "le site info", "lodj", "maroc hebdo", "l'opinion", "libération",
                    "al bayane", "media marketing", "industries du maroc", "lavieeco", "ecoactu.ma", "fnh",
                    "bourse news", "le reporter", "hibapress", "mapexpress", "info médiaire", "lematin")
FOREIGN = ("cac 40", "wall street", "nasdaq", "dow jones", "bourse de paris", "euronext", "bourse de tunis",
           "brvm", "tunindex", "bourse d'alger", "nikkei", "dax", "ftse", "s&p 500", "bourse de bruxelles")
JOB_ADS = ("recrute", "recrutement", "offre d'emploi", "offres d'emploi", "concours", "stage ", "emploi :")
# A headline that only describes the session (for the description, not used as a filter).
SESSION_WORDS = ("hausse", "baisse", "termine", "clôture", "cloture", "séance", "recul", "progresse", "repli",
                 "grimpe", "chute", "%", "vert", "rouge", "ouvre", "finit", "gagne", "perd", "bilan hebdo")


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).lower())
    return re.sub(r"\W+", " ", "".join(c for c in text if not unicodedata.combining(c))).strip()


def names_in(query: str) -> list:
    return [norm(n) for n in re.findall(r'"([^"]+)"', query)]


def keep_company(row) -> bool:
    title = " " + norm(row.title) + " "
    return any(" " + n + " " in title for n in names_in(row.query))


def keep_market(row) -> bool:
    low, src = str(row.title).lower(), str(row.source).lower()
    moroccan = any(w in low for w in MOROCCAN_WORDS) or any(o in src for o in MOROCCAN_OUTLETS)
    foreign_only = any(f in low for f in FOREIGN) and "casablanca" not in low and "masi" not in low
    return moroccan and not foreign_only


def sessions() -> pd.DatetimeIndex:
    """Every MASI trading day (the dates the index closed)."""
    days = pd.read_csv(MASI_CSV)["date"]
    return pd.DatetimeIndex(pd.to_datetime(days).sort_values().unique())


def next_session(dates: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    pos = days.searchsorted(dates.values, side="right")     # strictly after the publication date
    ok = pos < len(days)
    out = pd.Series(pd.NaT, index=dates.index)
    out[ok] = days[pos[ok]]
    return out


def main():
    raw = pd.read_csv(HERE / "data" / "raw_headlines.csv")
    funnel = {"lignes collectées": len(raw)}
    raw = raw.dropna(subset=["title"])
    is_market = raw["tag"] == "MARKET"
    keep = pd.Series(False, index=raw.index)
    keep[is_market] = raw[is_market].apply(keep_market, axis=1)
    keep[~is_market] = raw[~is_market].apply(keep_company, axis=1)
    df = raw[keep].copy()
    funnel["pertinentes (nom ou contexte marocain)"] = len(df)

    df = df[~df["title"].str.lower().apply(lambda t: any(j in t for j in JOB_ADS))]
    funnel["sans offres d'emploi"] = len(df)

    df["date"] = pd.to_datetime(df["published"]).dt.normalize()
    df["key"] = df["title"].apply(norm)
    df = df.sort_values("date").drop_duplicates("key", keep="first")
    funnel["titres uniques"] = len(df)

    df["session"] = next_session(df["date"], sessions())
    df = df.dropna(subset=["session"])
    df = df[(df["session"] >= "2020-10-01") & (df["session"] <= "2026-10-16")]
    funnel["rattachés à une séance (oct. 2020 – oct. 2026)"] = len(df)

    low = df["title"].str.lower()
    df["session_report"] = (df["tag"] == "MARKET") & low.apply(lambda t: any(w in t for w in SESSION_WORDS))
    df["id"] = range(len(df))
    cols = ["id", "tag", "title", "source", "date", "session", "session_report"]
    df[cols].to_csv(HERE / "data" / "clean_headlines.csv", index=False)
    funnel["journaux différents"] = int(df["source"].nunique())
    funnel["part des comptes rendus de séance"] = round(float(df["session_report"].mean()), 3)
    (HERE / "data" / "funnel.json").write_text(json.dumps(funnel, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(funnel, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
