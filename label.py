"""Step 3: the teacher. A large language model labels headlines as
positive / neutral / negative for an investor in Moroccan shares.

- teacher: Gemini 3.5 Flash-Lite labels every clean headline (about 6,400), drawn evenly over the years;
- second opinion: Gemini 2.5 Flash labelled 880 of them before its free daily quota ran out, to measure how far two LLMs agree.
Usage: python label.py teacher | python label.py second
Output: data/labels_<who>.csv (id, label). Safe to rerun: done ids are skipped.
"""
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
from llm import build_chain  # noqa: E402  (small Gemini client, key from .env)

BATCH = 100
LABELS = {"positif": 1, "neutre": 0, "négatif": -1, "negatif": -1}
INSTRUCTION = (
    "Tu es analyste financier à la Bourse de Casablanca. Pour chaque titre de presse numéroté, "
    "dis s'il est une bonne nouvelle, une nouvelle neutre ou une mauvaise nouvelle pour un "
    "investisseur en actions marocaines (la société citée, ou le marché si aucune société n'est citée).\n"
    "Juge les faits (résultats en hausse ou en baisse, contrat gagné, sanction, séance en hausse ou en "
    "baisse...), pas le style. Un titre purement informatif sans effet clair est neutre.\n"
    'Réponds par un objet JSON : {"items": [{"i": <numéro>, "label": "positif" | "neutre" | "négatif"}]}. '
    "Inclure chaque titre, dans l'ordre."
)


def sample(n: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(HERE / "data" / "clean_headlines.csv", parse_dates=["session"])
    df["year"] = df["session"].dt.year
    per_year = n // df["year"].nunique()
    parts = [g.sample(min(len(g), per_year), random_state=seed) for _, g in df.groupby("year")]
    out = pd.concat(parts)
    if len(out) < n:                                    # thin years leave room: fill from the rest
        rest = df.drop(out.index)
        out = pd.concat([out, rest.sample(min(len(rest), n - len(out)), random_state=seed)])
    return out


def label_batch(chain, batch: pd.DataFrame) -> list:
    doc = "\n".join(f"{i}. {t}" for i, t in enumerate(batch["title"]))
    for _ in range(3):
        try:
            raw, _ = chain.generate_json(INSTRUCTION, doc)
            text = raw[raw.find("{"): raw.rfind("}") + 1]
            items = json.loads(text, strict=False)["items"]
            out = []
            for it in items:
                i, lab = int(it["i"]), str(it["label"]).strip().lower()
                if 0 <= i < len(batch) and lab in LABELS:
                    out.append((int(batch["id"].iloc[i]), LABELS[lab]))
            return out
        except Exception as exc:                          # a bad answer: ask again
            print("batch failed:", exc, file=sys.stderr)
    return []


def run(who: str):
    if who == "teacher":
        os.environ["GEMINI_MODEL"] = "gemini-3.5-flash-lite"
        todo, chain = sample(7000, seed=1), build_chain("gemini")
    else:
        teacher = pd.read_csv(HERE / "data" / "labels_teacher.csv")
        todo = pd.read_csv(HERE / "data" / "clean_headlines.csv")
        todo = todo[todo["id"].isin(teacher["id"])].sample(600, random_state=2)
        os.environ["GEMINI_MODEL"] = "gemini-2.5-flash"
        chain = build_chain("gemini")
    path = HERE / "data" / f"labels_{who}.csv"
    done = set(pd.read_csv(path)["id"]) if path.exists() else set()
    todo = todo[~todo["id"].isin(done)]
    batches = [todo.iloc[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(who, len(todo), "headlines in", len(batches), "batches", flush=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f, ThreadPoolExecutor(8 if who == "teacher" else 3) as pool:
        w = csv.writer(f)
        if new:
            w.writerow(["id", "label"])
        for k, rows in enumerate(pool.map(lambda b: label_batch(chain, b), batches)):
            w.writerows(rows)
            f.flush()
            if k % 10 == 0:
                print("batch", k, flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "teacher")
