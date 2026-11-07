"""Generate SYNTHETIC CNN/DailyMail-style CSV files (train / validation / test) for smoke tests and CI.

Columns: ``id``, ``article``, ``highlights`` (highlights separated by newlines), exactly like the Kaggle
files. Some articles carry a "(CNN) --" prefix or a Daily Mail "PUBLISHED / UPDATED" header so the
cleaning code is exercised. The text is template based and exists only so the code and tests run
without downloading anything. Never report results obtained on this data.

    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

PEOPLE = ["Maria Lopez", "John Carter", "Aisha Khan", "Peter Novak", "Li Wei", "Sara Johansson", "Omar Haddad"]
ORGS = ["the city council", "Northern Rail", "the health ministry", "Acme Motors", "the national football league",
        "the central bank", "Globex Energy"]
CITIES = ["London", "Karachi", "Berlin", "Toronto", "Nairobi", "Sydney", "Madrid"]
ACTIONS = ["build", "close", "fund", "launch", "buy", "cut", "expand"]
THINGS = ["new bus lines", "three hospitals", "solar power plants", "primary schools", "bridges", "train stations"]
REASONS = ["a sharp rise in costs", "months of public pressure", "a safety review", "record passenger numbers",
           "a court ruling"]
ADJS = ["historic", "controversial", "long overdue", "ambitious", "expensive"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
FILLER = [
    "Residents of {city} have waited years for a decision of this kind.",
    "Analysts said the plan would take {n} months to complete.",
    "Critics argued that the money could be spent more wisely.",
    "A similar project in {city} was finished two years ago.",
    "The opposition said it would study the details carefully.",
    "Local business owners welcomed the news but asked for more information.",
    "The weather in {city} was cloudy on the day of the announcement.",
    "Experts expect the first results to be visible next year.",
    "A public meeting will be held in {city} later this month.",
]


def make_row(rng: random.Random, idx: int) -> tuple[str, str, str]:
    p, p2 = rng.sample(PEOPLE, 2)
    org, city = rng.choice(ORGS), rng.choice(CITIES)
    action, thing, reason = rng.choice(ACTIONS), rng.choice(THINGS), rng.choice(REASONS)
    n, day = rng.randint(3, 40), rng.choice(DAYS)
    key = [
        f"{p} said on {day} that {org} will {action} {n} {thing} in {city}.",
        f"The move comes after {reason}, according to officials.",
        f"{p2}, a spokesperson, described the decision as {rng.choice(ADJS)}.",
    ]
    filler = [rng.choice(FILLER).format(city=rng.choice(CITIES), n=rng.randint(6, 30))
              for _ in range(rng.randint(5, 9))]
    article = " ".join(key + filler)
    r = rng.random()
    if r < 0.33:
        article = f"{city.upper()}, England (CNN) -- {article}"
    elif r < 0.66:
        article = (f"By . Daily Mail Reporter . PUBLISHED: . 05:12 EST, {rng.randint(1, 28)} March 2013 . | . "
                   f"UPDATED: . 08:33 EST, {rng.randint(1, 28)} March 2013 . {article}")
    highlights = f"{org.capitalize()} will {action} {n} {thing} in {city}\n{p} said the move follows {reason}"
    return f"{idx:06d}", article, highlights


def write(path: Path, n: int, seed: int) -> None:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "article", "highlights"])
        for i in range(n):
            w.writerow(make_row(rng, i))
    print(f"Wrote {n} synthetic articles -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/sample"))
    args = parser.parse_args()
    write(args.out_dir / "train.csv", 120, seed=1)
    write(args.out_dir / "validation.csv", 30, seed=2)
    write(args.out_dir / "test.csv", 40, seed=3)


if __name__ == "__main__":
    main()
