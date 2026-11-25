# Sentiment de la presse financière marocaine et MASI

Can the tone of the Moroccan financial press tell us something about the Casablanca stock market (the MASI index)?

We collect six years of French-language Moroccan financial headlines and train a small French language model, **DistilCamemBERT**, to read their tone. From that tone we build the **Net Optimism Oscillator (I_Net)**, turn it into weekly **Buy / Sell / Hold** signals, label the market's **phases**, and backtest the resulting **strategy** against simply holding the MASI.

## Results in five lines

1. **The model reads tone well.** On 941 headlines it never saw, it scores F1 = 0.76, against 0.62 for FinBERT (on translated headlines) and 0.45 for DistilCamemBERT without training.
2. **The press is optimistic by default.** The raw I_Net is above +20% in 92% of weeks and never below −20%, so fixed thresholds say "buy" almost all the time.
3. **Centred on its own past year**, the oscillator gives rare, clear signals: 15 buy and 8 sell weeks out of 289.
4. **The signals do not forecast next week's return** (p = 0.63), and the strategy does not beat buy-and-hold on return (7.3% against 7.2% a year, p = 0.97).
5. **It lowers risk.** Volatility falls from 14.2% to 12.3% and the worst fall from −30% to −16%: it sold before the 2022 decline. That rests on only 5 trades.

## What is in this repository

| Open | What it is |
|---|---|
| `deck/presentation_EN.pptx`, `deck/presentation_FR.pptx` | The presentation (17 slides), in English and French. |
| `deck/econometrics_EN.pptx` | Appendix: each formula and statistical test, with a plain-language explanation. |
| `results/charts/` | The dual-axis charts (I_Net vs MASI, fixed and centred) and the market-phase chart. |
| `results/*.json` | Every number in the slides, as the code produced it. |
| `data/` | The headlines at each stage: collected, cleaned, labelled, scored. |
| the scripts | The full pipeline, below. |

## The pipeline

```
collect.py          1. collect headlines from Google News, Morocco edition (Oct 2020 – Oct 2026)
clean.py            2. keep relevant headlines, remove duplicates, date each one to the next trading session
label.py            3. a large language model (Gemini) labels each headline: positive, neutral or negative
train.py            4. fine-tune DistilCamemBERT on those labels and score every headline
compare_models.py   5. compare with FinBERT, on French and on translated headlines
barometer.py        6. Net Optimism Oscillator, Buy / Sell / Hold signals, market phases, strategy backtest, charts
econ.py, extra.py   7. side analysis on volatility (one slide in the presentation)
llm.py              a small Gemini client used by label.py and compare_models.py
```

## Data files

| File | Content |
|---|---|
| `data/raw_headlines.csv` | 11,267 rows returned by the 25 searches |
| `data/clean_headlines.csv` | 6,368 headlines after cleaning, each with its trading session |
| `data/labels_teacher.csv` | labels given by Gemini 3.5 Flash-Lite (1 positive, 0 neutral, −1 negative) |
| `data/labels_second.csv` | a second set of labels (Gemini 2.5 Flash) used to check agreement |
| `data/scored.csv` | every headline with the trained model's P(positive), P(neutral), P(negative) and tone |
| `data/test_translated.csv` | the 941 test headlines translated to English, for the FinBERT comparison |
| `data/funnel.json` | how many headlines remain after each cleaning step |

Market prices (MASI daily closes and share prices) are not included. Scripts that need them read `data/market/masi_daily.csv` (date, close) and `data/market/shares_daily.csv` (ticker, date, close, value).

## Running it

```
pip install -r requirements.txt
```

- `label.py` and `compare_models.py` need a Gemini API key: copy `.env.example` to `.env` and fill it in.
- `train.py` needs `torch` (CPU is enough, about 15 minutes). The trained model is saved in `models/`.
- `clean.py`, `barometer.py`, `econ.py` and `extra.py` need the market price files described above.

## Rules fixed before looking at the results

- A headline counts at the first trading session after its date, so no future information is used.
- Every average, threshold and model weight uses past weeks only.
- The 941 test headlines were never used in training.
- Three hypotheses, each tested against H0 ("no effect"), rejected only if p < 5%.

## Limits

Headlines only (not full articles), French-language press only, fewer headlines before 2023, labels given by a language model, and a strategy that trades only 5 times.
