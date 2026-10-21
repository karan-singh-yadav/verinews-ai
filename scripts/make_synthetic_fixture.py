"""Generate a SYNTHETIC toy dataset (Fake.csv / True.csv) for pipeline smoke tests only.

!! These articles are template-generated gibberish with an artificially easy signal.
!! Any metric computed from them says NOTHING about real-world fake news detection.
!! Use the real dataset (README: 'Dataset preparation') for any reported result.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

REAL_POOL = [
    "officials said the committee released a detailed report on budget figures on tuesday",
    "the finance ministry confirmed that quarterly revenue rose by {n} percent compared with last year",
    "lawmakers voted on the proposed infrastructure bill after several hours of scheduled debate",
    "a spokesperson for the agency told reporters that the review would conclude next month",
    "according to the statement the two governments agreed to continue trade negotiations",
    "analysts noted that employment data for the period showed a modest increase of {n} thousand jobs",
    "the court scheduled a hearing for the case and both parties submitted written filings",
    "the central bank kept interest rates unchanged and cited stable inflation expectations",
]
FAKE_POOL = [
    "shocking secret exposed elites are hiding the truth and they do not want you to know",
    "share this before it gets deleted the mainstream media is covering up everything",
    "you will not believe what they just admitted about the hoax number {n} in leaked footage",
    "wake up people the corrupt establishment is lying again and patriots are furious",
    "breaking bombshell proves the whole story was staged and insiders are panicking",
    "they tried to silence this brave whistleblower but the truth is going viral now",
    "unbelievable outrage as radical puppets destroy everything and nobody is talking about it",
    "this one weird trick that the globalists are terrified you will discover today",
]
TOPICS = ["the senate", "the city council", "the national team", "the health department",
          "the transport authority", "the university board"]


def make_article(rng: random.Random, pool: list[str], n_sent: int) -> tuple[str, str]:
    sents = rng.sample(pool, n_sent)
    sents = [s.format(n=rng.randint(1, 99)) for s in sents]
    topic = rng.choice(TOPICS)
    title = f"{topic} update {rng.randint(1000, 9999)}"
    return title, f"{topic} {'. '.join(sents)}. extra detail {rng.randint(1, 10**6)}."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-per-class", type=int, default=400)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, pool in (("Fake.csv", FAKE_POOL), ("True.csv", REAL_POOL)):
        rows = []
        for _ in range(a.n_per_class):
            t, b = make_article(rng, pool, rng.randint(4, 6))
            rows.append({"title": t, "text": b, "subject": "synthetic", "date": "n/a"})
        pd.DataFrame(rows).to_csv(out / name, index=False)
    print(f"Wrote SYNTHETIC fixture to {out} (not for reporting metrics)")


if __name__ == "__main__":
    main()
