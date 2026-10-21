"""POST HOC checks on the departures results, after the copy audit (not in the plan).

(1) Event-1 placebos of the homogeneous B - F differences-in-differences, (B - F)_conf - (B - F)_team, on buyer surplus,
    joint surplus and the final-bid margin (identical B and F inputs at event 1), beside their events 2-6 values and the
    change (events 2-6 minus event 1) per seed.
(2) The decomposition of the mixed B:F conflict gap at event 1 and at events 2-6: win share and the share of own final
    bids below own cost, B firm and F firm, and B - F; also under team objectives.
(3) H1 minus its event-1 value, per seed.
(4) The primary family on the 18 seeds without 28 and 39: H1-H4, unadjusted p and Holm over four.
Exact two-sided sign-flip p as in followup17_primary.sign_flip_p; seeds counted on values rounded to 1e-9.
Usage: python departures18_extra_checks.py [output file]
"""
import sys
from pathlib import Path

import numpy as np

import departures18_analysis as da
import followup17_primary as fp
import pilot17_analysis as pa

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else pa.RESULTS / "departures18_extra_checks.txt"
E1, LATE = range(1, 2), range(2, 7)


def firm_shares(run, sid, events):
    _, rows, evs = run
    win = [float(rows[k]["winner"] == sid) for k in events]
    below = [float(evs[k]["outcome"]["all_inflation_rates"][sid] < 0) for k in events]
    return win, below


def decomposition(runs, x, y, events, seeds):
    """Per seed, pooled over institutions and positions event by event: (B win, F win, B below, F below)."""
    out = {}
    for s in seeds:
        acc = {"bw": [], "fw": [], "bb": [], "fb": []}
        for inst in pa.INSTS:
            for pos, bs in (("xa", "A"), ("xb", "B")):
                r = runs.get(("persistent", f"{x}_vs_{y}_{pos}", inst, s))
                fs = "B" if bs == "A" else "A"
                bw, bb = firm_shares(r, bs, events)
                fw, fb = firm_shares(r, fs, events)
                acc["bw"] += bw
                acc["fw"] += fw
                acc["bb"] += bb
                acc["fb"] += fb
        out[s] = {k: float(np.mean(v)) for k, v in acc.items()}
    return out


def line(name, pairs, test=False):
    d = [v for _, v in pairs]
    txt = f"  {name:<58} {da.describe(pairs)}"
    if test:
        txt += f"  p {fp.sign_flip_p(d):.6f}"
    return txt


def main():
    runs, problems = pa.load(da.DEP_ROOTS)
    S = da.SEEDS
    lines = [f"POST HOC checks. Runs {len(runs)}; problems {len(problems)}"]
    w = lines.append
    b, f = da.org("B", True), da.org("F", True)
    bt, ft = da.org("B", False), da.org("F", False)

    w("\n(1) Homogeneous DiD (B - F)_conf - (B - F)_team, institutions pooled:")
    for stat in ("buyer", "joint", "margin"):
        e1 = da.per_seed(runs, S, da.INSTS, da.did(runs, stat, E1, b, f, bt, ft))
        late = da.per_seed(runs, S, da.INSTS, da.did(runs, stat, LATE, b, f, bt, ft))
        chg = [(s, v - dict(e1)[s]) for s, v in late]
        w(line(f"{stat:<7} event 1", e1))
        w(line(f"{stat:<7} events 2-6", late))
        w(line(f"{stat:<7} events 2-6 minus event 1", chg, test=True))

    w("\n(2) Mixed B:F decomposition (pooled over institutions and positions event by event):")
    for tag, x, y in (("conflict", b, f), ("team", bt, ft)):
        for wtag, ev in (("event 1", E1), ("events 2-6", LATE)):
            dec = decomposition(runs, x, y, ev, S)
            lv = {k: np.mean([d[k] for d in dec.values()]) for k in ("bw", "fw", "bb", "fb")}
            w(f"  [{tag}, {wtag}] levels: B win {lv['bw']:.4f}  F win {lv['fw']:.4f}  B below {lv['bb']:.4f}  "
              f"F below {lv['fb']:.4f}")
            w(line(f"[{tag}, {wtag}] win share B - F", [(s, d["bw"] - d["fw"]) for s, d in dec.items()]))
            w(line(f"[{tag}, {wtag}] below-cost share B - F", [(s, d["bb"] - d["fb"]) for s, d in dec.items()]))

    w("\n(3) H1 minus its event-1 value, per seed:")
    h1_e1 = dict(da.per_seed(runs, S, da.INSTS, da.mixed(runs, b, f, E1)))
    h1 = da.per_seed(runs, S, da.INSTS, da.mixed(runs, b, f, LATE))
    w(line("H1 (events 2-6) minus H1 at event 1", [(s, v - h1_e1[s]) for s, v in h1], test=True))

    w("\n(4) The primary family on the 18 seeds without 28 and 39 (events 2-6, institutions pooled):")
    keep = [s for s in S if s not in (28, 39)]
    fam = [(n, [(s, v) for s, v in pairs if s in keep]) for n, pairs in da.h_family(runs, LATE)]
    ps = [fp.sign_flip_p([v for _, v in pairs]) for _, pairs in fam]
    for (name, pairs), p, adj in zip(fam, ps, fp.holm(ps)):
        d = np.array([v for _, v in pairs])
        sd = d.std(ddof=1)
        w(f"  {name:<50} n={len(d)}  mean {d.mean():+.3f}  se {sd / np.sqrt(len(d)):.3f}  p {p:.6f}  Holm p {adj:.6f}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
