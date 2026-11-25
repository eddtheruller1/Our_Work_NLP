"""Step 1: collect six years of French-language Moroccan headlines from Google News.

Searches the market and the 20 most traded companies, one time window at a
time (Google returns at most ~100 items per search, so the market searches use
monthly windows and the company searches quarterly ones).
Output: data/raw_headlines.csv (one row per search result, duplicates kept).
"""
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

OUT = Path(__file__).parent / "data" / "raw_headlines.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
START, END = date(2020, 10, 1), date(2026, 10, 16)

MARKET = {"MARKET": ['"Bourse de Casablanca"', '"MASI" bourse', 'marché boursier Maroc',
                     '"Bank Al-Maghrib" taux', '"AMMC" Maroc']}
# The 20 most traded shares since 2021, with a name that is not ambiguous.
COMPANIES = {
    "ATW": '"Attijariwafa"', "TGC": '"TGCC"', "MSA": '"Marsa Maroc"', "BCP": '"Banque Populaire" Maroc',
    "IAM": '"Maroc Telecom"', "AKT": '"Akdital"', "ADH": '"Addoha"', "ADI": '"Alliances Darna" OR "Alliances Développement"',
    "MNG": '"Managem"', "GTM": '"SGTM"', "RDS": '"Résidences Dar Saada"', "LHM": '"Holcim Maroc" OR "LafargeHolcim Maroc"',
    "BOA": '"Bank of Africa"', "JET": '"Jet Contractors"', "LBV": '"LabelVie" OR "Label\'Vie"', "TQM": '"Taqa Morocco"',
    "CMG": '"CMGP"', "CFG": '"CFG Bank"', "CMA": '"Ciments du Maroc"', "CSR": '"Cosumar"',
}


def windows(months: int):
    y, m = START.year, START.month
    while date(y, m, 1) < END:
        m2, y2 = m + months, y
        while m2 > 12:
            m2, y2 = m2 - 12, y2 + 1
        yield date(y, m, 1), min(date(y2, m2, 1), END)
        y, m = y2, m2


def search(query: str, after: date, before: date) -> list:
    q = f"{query} after:{after.isoformat()} before:{before.isoformat()}"
    url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(q) + "&hl=fr&gl=MA&ceid=MA:fr"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    out = []
    for it in ET.fromstring(r.content).findall(".//item"):
        source = (it.findtext("source") or "").strip()
        title = (it.findtext("title") or "").strip()
        if source and title.endswith(" - " + source):
            title = title[: -len(source) - 3].strip()
        try:
            when = parsedate_to_datetime(it.findtext("pubDate")).astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        out.append({"title": title, "source": source, "published": when.strftime("%Y-%m-%dT%H:%M:%S")})
    return out


def run_window(job):
    tag, q, a, b = job
    for attempt in range(3):
        try:
            return [{"tag": tag, "query": q, **row} for row in search(q, a, b)]
        except Exception as exc:                  # one failed window must not stop the run
            print("retry", tag, a, exc, file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return []


def main(workers: int = 4):
    jobs = [(tag, q, a, b) for tag, qs in MARKET.items() for q in qs for a, b in windows(1)] +            [(t, q, a, b) for t, q in COMPANIES.items() for a, b in windows(3)]
    OUT.parent.mkdir(exist_ok=True)
    lock, n = threading.Lock(), 0
    with OUT.open("w", newline="", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        w = csv.DictWriter(f, ["tag", "query", "title", "source", "published"])
        w.writeheader()
        for k, rows in enumerate(pool.map(run_window, jobs)):
            with lock:
                w.writerows(rows)
                n += len(rows)
            if k % 50 == 0:
                print(k, "/", len(jobs), "windows,", n, "rows", flush=True)


if __name__ == "__main__":
    main()
