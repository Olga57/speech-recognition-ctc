from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.metrics import get_wer, get_cer


def read_txt(p: str) -> str:
    with open(p, "r", encoding="utf-8") as f:
        return f.read().strip().lower()


def collect(folder: str):
    d = {}
    for n in os.listdir(folder):
        if n.endswith(".txt"):
            d[os.path.splitext(n)[0]] = os.path.join(folder, n)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", type=str, required=True)
    ap.add_argument("--hyp_dir", type=str, required=True)
    args = ap.parse_args()

    ref = collect(args.ref_dir)
    hyp = collect(args.hyp_dir)

    ids = sorted(set(ref.keys()) & set(hyp.keys()))
    if not ids:
        raise RuntimeError("No common ids between ref_dir and hyp_dir")

    wers = []
    cers = []
    for uid in ids:
        r = read_txt(ref[uid])
        h = read_txt(hyp[uid])
        wers.append(get_wer(r, h))
        cers.append(get_cer(r, h))

    print("Items:", len(ids))
    print("WER:", sum(wers) / len(wers))
    print("CER:", sum(cers) / len(cers))


if __name__ == "__main__":
    main()
