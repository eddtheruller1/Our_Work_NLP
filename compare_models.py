"""Justify the model: compare our fine-tuned DistilCamemBERT with FinBERT on the
same 941 test headlines.

FinBERT (ProsusAI/finbert) is the reference finance model, but it is English.
It is run twice: on the French headlines as they are, and on the same headlines
translated to English by an LLM (Gemini 3.5 Flash-Lite).
Output: results/model_compare.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "models"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

import train  # noqa: E402  (same split, same metrics)

HERE = Path(__file__).parent
FINBERT = "ProsusAI/finbert"
torch.set_num_threads(os.cpu_count() or 4)


def translate(titles: list) -> list:
    os.environ["GEMINI_MODEL"] = "gemini-3.5-flash-lite"
    from llm import build_chain
    chain = build_chain("gemini")
    out = [None] * len(titles)
    instruction = ('Translate each numbered French financial headline into English, faithfully. '
                   'Return JSON: {"items": [{"i": <number>, "en": "<translation>"}]}. Include every headline.')
    for start in range(0, len(titles), 80):
        batch = titles[start:start + 80]
        doc = "\n".join(f"{i}. {t}" for i, t in enumerate(batch))
        for _ in range(3):
            try:
                raw, _ = chain.generate_json(instruction, doc)
                items = json.loads(raw[raw.find("{"): raw.rfind("}") + 1], strict=False)["items"]
                for it in items:
                    i = int(it["i"])
                    if 0 <= i < len(batch):
                        out[start + i] = str(it["en"])
                break
            except Exception as exc:
                print("translation batch failed:", exc, file=sys.stderr)
        print("translated", start + len(batch), flush=True)
    return [o if o else t for o, t in zip(out, titles)]     # untranslated ones stay French


def finbert(texts: list) -> list:
    tok = AutoTokenizer.from_pretrained(FINBERT)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT).eval()
    labels = model.config.id2label                      # positive / negative / neutral
    to_class = {"positive": 1, "neutral": 0, "negative": -1}
    preds = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], padding=True, truncation=True, max_length=64, return_tensors="pt")
            idx = model(**enc).logits.argmax(-1).tolist()
            preds += [to_class[labels[j].lower()] for j in idx]
    return preds


def main():
    heads, _, _, test = train.data()
    scored = pd.read_csv(HERE / "data" / "scored.csv")
    test = test.merge(scored[["id", "p_neg", "p_neu", "p_pos"]], on="id")
    y = test["label"].tolist()
    student = list(test[["p_neg", "p_neu", "p_pos"]].values.argmax(1) - 1)
    cache = HERE / "data" / "test_translated.csv"
    titles = test["title"].tolist()
    en = pd.read_csv(cache)["en"].tolist() if cache.exists() else list(titles)
    missing = [i for i, (e, t) in enumerate(zip(en, titles)) if e == t]   # untranslated after a failed run
    if missing:
        done = translate([titles[i] for i in missing])
        for i, e in zip(missing, done):
            en[i] = e
        pd.DataFrame({"id": test["id"], "title": titles, "en": en}).to_csv(cache, index=False)
    print("translated:", sum(e != t for e, t in zip(en, titles)), "of", len(titles), flush=True)
    preds = {"FinBERT (titres français)": finbert(test["title"].tolist()),
             "FinBERT (titres traduits en anglais)": finbert(en),
             "DistilCamemBERT ajusté (notre élève)": student}
    res = {name: {"macro_f1": round(train.macro_f1(y, p), 3), "accuracy": round(float(np.mean(np.array(p) == np.array(y))), 3),
                  "ci95": train.bootstrap_ci(y, p)} for name, p in preds.items()}
    res["mcnemar_student_vs_finbert_en"] = train.mcnemar(y, student, preds["FinBERT (titres traduits en anglais)"])
    res["n_test"] = len(y)
    (HERE / "results" / "model_compare.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
