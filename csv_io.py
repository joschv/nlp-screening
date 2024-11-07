import csv
from os import PathLike


def import_collection(fp: str | PathLike) -> list[dict]:
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows += [r]
    return rows


def export_collection(fp: str | PathLike, data: list[dict]) -> None:
    with open(fp, "w+", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())

        writer.writeheader()
        for row in data:
            writer.writerow(rowdict=row)


if __name__ == '__main__':
    r = import_collection("collection_with_abstracts.csv")
