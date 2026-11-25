"""Step 4: the student. Fine-tune DistilCamemBERT on the teacher's labels, compare
it with the zero-shot models, then score every headline.

Data split (fixed seed, stratified by label): 70 % train, 15 % validation
(choose the epoch), 15 % test (touched once, at the end).
Output: results/model_results.json, data/scored.csv (tone = P(pos) - P(neg)).
The fine-tuned model (about 260 MB) is saved in models/, which git ignores.
"""
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).parent / "models"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,  # noqa: E402
                          pipeline)

HERE = Path(__file__).parent
BASE = "cmarkea/distilcamembert-base"
NLI = "cmarkea/distilcamembert-base-nli"
STARS = "cmarkea/distilcamembert-base-sentiment"
SAVE = Path(os.environ["HF_HOME"]) / "mots-tempete-student"
CLASSES = [-1, 0, 1]                      # négatif, neutre, positif
torch.manual_seed(0)
torch.set_num_threads(os.cpu_count() or 4)


def data():
    heads = pd.read_csv(HERE / "data" / "clean_headlines.csv")
    lab = pd.read_csv(HERE / "data" / "labels_teacher.csv").drop_duplicates("id")
    df = heads.merge(lab, on="id")
    train, rest = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=0)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["label"], random_state=0)
    return heads, train, val, test


def macro_f1(y, p):
    return float(f1_score(y, p, labels=CLASSES, average="macro"))


def bootstrap_ci(y, p, n=1000):
    y, p, rng = np.asarray(y), np.asarray(p), np.random.default_rng(0)
    scores = [macro_f1(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return [round(float(np.percentile(scores, 2.5)), 3), round(float(np.percentile(scores, 97.5)), 3)]


def mcnemar(y, a, b):
    """Exact test on the headlines where only one of two models is right."""
    from scipy.stats import binomtest
    y, a, b = map(np.asarray, (y, a, b))
    only_a, only_b = int(((a == y) & (b != y)).sum()), int(((a != y) & (b == y)).sum())
    return {"only_first_right": only_a, "only_second_right": only_b,
            "p": float(binomtest(only_a, only_a + only_b, 0.5).pvalue) if only_a + only_b else 1.0}


def zero_shot_nli(texts):
    clf = pipeline("zero-shot-classification", model=NLI, tokenizer=AutoTokenizer.from_pretrained(NLI, use_fast=False), device=-1)
    names = {"positif": 1, "neutre": 0, "négatif": -1}
    out = clf(list(texts), candidate_labels=list(names), batch_size=32,
              hypothesis_template="Cette nouvelle est {} pour les investisseurs.")
    return [names[o["labels"][0]] for o in out]


def zero_shot_stars(texts):
    clf = pipeline("text-classification", model=STARS, tokenizer=AutoTokenizer.from_pretrained(STARS, use_fast=False), device=-1)
    out = clf(list(texts), batch_size=64, truncation=True)
    stars = [int(o["label"].split()[0]) for o in out]
    return [-1 if s <= 2 else (0 if s == 3 else 1) for s in stars]


def encode(tok, texts):
    return tok(list(texts), padding=True, truncation=True, max_length=48, return_tensors="pt")


def predict_proba(model, tok, texts, bs=128):
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = encode(tok, texts[i:i + bs])
            probs.append(torch.softmax(model(**enc).logits, -1).numpy())
    return np.vstack(probs)


def fine_tune(train, val, epochs=2, lr=5e-5, bs=32, max_train=3000, frozen_layers=3):
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=3)
    # A 4-core laptop: train on 3,000 headlines and freeze the embeddings and the
    # first 3 of 6 layers, so only the upper layers and the classifier learn.
    for p in model.roberta.embeddings.parameters():
        p.requires_grad = False
    for layer in model.roberta.encoder.layer[:frozen_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    train = train.sample(min(len(train), max_train), random_state=0)
    y = torch.tensor(train["label"].values + 1)          # -1,0,1 -> 0,1,2
    texts = train["title"].tolist()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    steps = epochs * ((len(texts) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    history, best = [], (-1, None)
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(texts))
        losses = []
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            enc = encode(tok, [texts[j] for j in idx])
            loss = model(**enc, labels=y[idx]).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
            losses.append(loss.item())
            if len(losses) % 20 == 0:
                print("epoch", ep + 1, "step", len(losses), "loss", round(float(np.mean(losses[-20:])), 3), flush=True)
        p = predict_proba(model, tok, val["title"].tolist()).argmax(1) - 1
        f1 = macro_f1(val["label"], p)
        history.append({"epoch": ep + 1, "train_loss": round(float(np.mean(losses)), 4), "val_f1": round(f1, 4)})
        print(history[-1], flush=True)
        if f1 > best[0]:
            best = (f1, ep + 1)
            model.save_pretrained(SAVE), tok.save_pretrained(SAVE)
    return history, best[1]


def main():
    heads, train, val, test = data()
    res = {"n": {"train": len(train), "validation": len(val), "test": len(test)},
           "label_share": {str(k): round(float(v), 3) for k, v in train["label"].value_counts(normalize=True).items()}}
    y = test["label"].tolist()
    preds = {"Toujours « neutre »": [0] * len(y)}
    t0 = time.time()
    print("zero-shot NLI...", flush=True)
    preds["DistilCamemBERT zero-shot (NLI)"] = zero_shot_nli(test["title"])
    res["nli_seconds_per_1000"] = round((time.time() - t0) / len(y) * 1000, 1)
    print("NLI done", round(time.time() - t0), "s; stars...", flush=True)
    preds["DistilCamemBERT avis (étoiles)"] = zero_shot_stars(test["title"])
    print("stars done; fine-tuning...", flush=True)
    t0 = time.time()
    res["history"], res["best_epoch"] = fine_tune(train, val)
    res["trained_on"] = min(len(train), 3000)
    res["train_minutes"] = round((time.time() - t0) / 60, 1)
    tok, model = AutoTokenizer.from_pretrained(SAVE, use_fast=False), AutoModelForSequenceClassification.from_pretrained(SAVE)
    t0 = time.time()
    proba = predict_proba(model, tok, test["title"].tolist())
    res["student_seconds_per_1000"] = round((time.time() - t0) / len(y) * 1000, 1)
    preds["DistilCamemBERT ajusté (notre élève)"] = list(proba.argmax(1) - 1)
    res["models"] = {name: {"macro_f1": round(macro_f1(y, p), 3), "accuracy": round(float(accuracy_score(y, p)), 3),
                            "ci95": bootstrap_ci(y, p)} for name, p in preds.items()}
    student = preds["DistilCamemBERT ajusté (notre élève)"]
    res["mcnemar_student_vs_nli"] = mcnemar(y, student, preds["DistilCamemBERT zero-shot (NLI)"])
    res["confusion_student"] = confusion_matrix(y, student, labels=CLASSES).tolist()

    all_p = predict_proba(model, tok, heads["title"].tolist())
    heads["p_neg"], heads["p_neu"], heads["p_pos"] = all_p[:, 0], all_p[:, 1], all_p[:, 2]
    heads["tone"] = heads["p_pos"] - heads["p_neg"]
    heads.to_csv(HERE / "data" / "scored.csv", index=False)
    second = HERE / "data" / "labels_second.csv"
    if second.exists():                                   # how far two LLMs agree, and the student with each
        from sklearn.metrics import cohen_kappa_score
        t = pd.read_csv(HERE / "data" / "labels_teacher.csv").drop_duplicates("id")
        s = pd.read_csv(second).drop_duplicates("id").merge(t, on="id", suffixes=("_gemini", "_teacher"))
        s = s.merge(heads[["id", "p_neg", "p_neu", "p_pos"]], on="id")
        s["student"] = s[["p_neg", "p_neu", "p_pos"]].values.argmax(1) - 1
        st = s[s["id"].isin(test["id"])]                 # the student never trained on these
        res["second_opinion_test"] = {"n": len(st), "f1_student_vs_gemini": round(macro_f1(st["label_gemini"], st["student"]), 3),
                                      "f1_teacher_vs_gemini": round(macro_f1(st["label_gemini"], st["label_teacher"]), 3)}
        res["second_opinion"] = {
            "n": len(s),
            "kappa_teacher_gemini": round(float(cohen_kappa_score(s["label_teacher"], s["label_gemini"])), 3),
            "agreement_teacher_gemini": round(float((s["label_teacher"] == s["label_gemini"]).mean()), 3),
            "f1_student_vs_gemini": round(macro_f1(s["label_gemini"], s["student"]), 3)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "model_results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["models"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
