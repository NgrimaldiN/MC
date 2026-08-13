"""Download EUR/USD reference rates from Frankfurter and save a local CSV."""

from __future__ import annotations

import csv
import json
import pathlib
import urllib.request


URL = "https://api.frankfurter.app/2023-01-01..2024-12-31?from=EUR&to=USD"
OUTPUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "eurusd_frankfurter_2023_2024.csv"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    rows = []
    for date, rate_dict in sorted(payload["rates"].items()):
        rows.append({"date": date, "eur_usd": rate_dict["USD"]})

    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["date", "eur_usd"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
