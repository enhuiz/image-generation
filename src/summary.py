import argparse
from pathlib import Path

import pandas as pd


def _display(df, col, dup, lower_better=False):
    # When value are the same, we choose a smaller step
    df = df.sort_values("step")
    df = df.sort_values(col, ascending=lower_better)
    df = df.drop_duplicates(dup)
    assert isinstance(df, pd.DataFrame)
    print(df.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs", type=Path)
    parser.add_argument("--filename", default="fid.txt")
    args = parser.parse_args()

    paths = args.log_dir.rglob(f"**/{args.filename}")

    rows = []
    for path in paths:
        try:
            with open(path, "r") as f:
                value = float(f.read())
            rows.append(
                dict(
                    path=path.parents[2],
                    split=path.parents[0].stem,
                    step=int(path.parents[1].stem),
                    value=value,
                )
            )
        except:
            pass

    df = pd.DataFrame(rows)

    name = args.filename.split(".")[0]
    df[name] = df["value"]
    del df["value"]

    for split, grp in df.groupby("split"):
        print(split)
        _display(grp, col=name, dup=["path"], lower_better=True)


if __name__ == "__main__":
    main()
