import csv
from os import PathLike


def import_collection(fp: str | PathLike) -> list[dict]:
    rows = []
    with open(fp, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows += [r]
    return rows


if __name__ == '__main__':
    r = import_collection("collection_with_abstracts.csv")
