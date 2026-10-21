"""Confirmatory tests on the frozen code, as fixed in `2026-10-17 confirmatory runs on the frozen code
(pre-registration).md`.

Test M (random environment, mediator): per seed, d = U0 - Ubar in market profit (summary.csv supplier_profit, the
winner's profit, 0 if nothing is awarded), mean over events 2-6; Ubar = (U + U')/2. Exact two-sided sign-flip, 0.05.
Test C, frozen-code part (persistent environment, buyer's agent): per run, the within-event price cut (round-1 price
minus the price of the supplier's last bid in the event), mean over both suppliers and events 2-6; per seed, C1 = O - U
and C2 = O - O_pass. Their p-values are printed here; Holm over C1-C3 is applied when C3 (B - F, departures runs)
exists. Secondary estimates as listed in the pre-registration.

Data: data/runs/confirm17_memory (test M) and data/runs/confirm17_concession (test C), both made at commit 9f1cf25.
Both tests use recorded values only: the supplier_profit, buyer_surplus and joint_surplus columns of summary.csv, and
the recorded bid prices.
Usage: python confirm17_analysis.py [output file] [root;root...]
"""
import sys
from pathlib import Path

import numpy as np

import followup17_primary as fp
import pilot17_analysis as pa

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else pa.RESULTS / "confirm17_results.txt"
ROOTS = ([Path(p) for p in sys.argv[2].split(";")] if len(sys.argv) > 2
         else [pa.RUNS / "confirm17_memory", pa.RUNS / "confirm17_concession"])
LATE, FIRST = range(2, 7), range(1, 2)
T975_19 = 2.093  # the 0.975 quantile of t with 19 degrees of freedom


def describe(pairs):
    d = np.array([v for _, v in pairs], dtype=float)
    if len(d) < 2:
        return f"n={len(d)}"
    sd = float(np.std(d, ddof=1))
    se = sd / np.sqrt(len(d))
    half = (T975_19 if len(d) == 20 else float("nan")) * se
    return (f"n={len(d)}  mean {d.mean():+.3f}  sd {sd:.3f}  se {se:.3f}  95% CI [{d.mean() - half:+.3f}, "
            f"{d.mean() + half:+.3f}]  >0: {int((np.round(d, 9) > 0).sum())}/{len(d)} (=0: {int((np.round(d, 9) == 0).sum())})")


def price_stats(run, events):
    """Mean over both suppliers and the events of: the cut, the opening (round-1) price and the last bid's price."""
    _, rows, evs = run
    cut, op, fi = [], [], []
    for k in events:
        if k not in evs:
            continue
        for sid in ("A", "B"):
            bids = evs[k]["bids"][sid]
            op.append(bids[0]["price"])
            fi.append(bids[-1]["price"])
            cut.append(bids[0]["price"] - bids[-1]["price"])
    return {"cut": float(np.mean(cut)), "open": float(np.mean(op)), "final": float(np.mean(fi))}


def per_seed(runs, env, inst, seeds, value, x, base):
    """Per seed: value(x) - value(base); base 'ubar' is the mean of single_a and single_b. Seeds lacking an arm are
    skipped (and listed by the caller)."""
    out = []
    for s in seeds:
        rx = runs.get((env, x, inst, s))
        if base == "ubar":
            ru, ru2 = runs.get((env, "single_a", inst, s)), runs.get((env, "single_b", inst, s))
            if rx and ru and ru2:
                out.append((s, value(rx) - (value(ru) + value(ru2)) / 2))
        else:
            rb = runs.get((env, base, inst, s))
            if rx and rb:
                out.append((s, value(rx) - value(rb)))
    return out


def main():
    runs, problems = pa.load(ROOTS)
    lines = []
    w = lines.append
    w(f"Confirmatory tests on the frozen code. Roots: {'; '.join(pa.rel(r) for r in ROOTS)}")
    w(f"Runs read {len(runs)}; integrity problems {len(problems)}")
    for p in problems:
        w("  PROBLEM " + p)
    ids = sorted({(r.get("git_commit"), r.get("git_dirty"), r.get("source_sha256")) for r, _, _ in runs.values()},
                 key=str)
    w(f"(commit, dirty, source): {ids}")
    cells = sorted({k[:3] for k in runs})
    for c in cells:
        have = sorted(k[3] for k in runs if k[:3] == c)
        w(f"  {c}: {len(have)} seeds {have[0] if have else ''}-{have[-1] if have else ''}")

    # Test M
    mseeds = sorted({k[3] for k in runs if k[0] == "random" and k[2] == "mediator"})
    w("\nTEST M: random environment, mediator. d = U0 - Ubar, market profit, events 2-6; exact two-sided sign-flip.")
    if mseeds:
        prof = lambda r, ev=LATE: pa.market(r[1], "supplier_profit", ev)  # noqa: E731
        d = per_seed(runs, "random", "mediator", mseeds, prof, "single_norecord", "ubar")
        miss = sorted(set(mseeds) - {s for s, _ in d})
        p = fp.sign_flip_p([v for _, v in d])
        mean = float(np.mean([v for _, v in d]))
        verdict = ("confirmed" if p < 0.05 and mean > 0 else "contradicted" if p < 0.05 and mean < 0
                   else "not confirmed")
        w(f"  U0 - Ubar  {describe(d)}  p {p:.4f}  {verdict}" + (f"  (seeds dropped: {miss})" if miss else ""))
        w(f"      per seed: {', '.join(f's{s} {v:+.2f}' for s, v in d)}")
        w("  Secondary (estimates only):")
        for key in ("supplier_profit", "buyer_surplus", "joint_surplus"):
            for tag, ev in (("events 2-6", LATE), ("event 1", FIRST)):
                f = lambda r, key=key, ev=ev: pa.market(r[1], key, ev)  # noqa: E731
                w(f"    {key:<16} {tag:<10} U0 - Ubar  {describe(per_seed(runs, 'random', 'mediator', mseeds, f, 'single_norecord', 'ubar'))}")
                w(f"    {key:<16} {tag:<10} U' - U     {describe(per_seed(runs, 'random', 'mediator', mseeds, f, 'single_b', 'single_a'))}")

    # Test C, frozen-code part
    cseeds = sorted({k[3] for k in runs if k[0] == "persistent" and k[2] == "buyer_agent"})
    w("\nTEST C (frozen-code part): persistent environment, buyer's agent. The cut = round-1 price minus the last bid's "
      "price, mean over both suppliers and events 2-6. Unadjusted exact two-sided sign-flip p; Holm over C1-C3 is "
      "applied once C3 exists.")
    if cseeds:
        for name, x, base in (("C1 O - U", "organized", "single_a"), ("C2 O - O_pass", "organized", "organized_pass")):
            cut = lambda r: price_stats(r, LATE)["cut"]  # noqa: E731
            d = per_seed(runs, "persistent", "buyer_agent", cseeds, cut, x, base)
            miss = sorted(set(cseeds) - {s for s, _ in d})
            p = fp.sign_flip_p([v for _, v in d])
            w(f"  {name:<14} {describe(d)}  unadjusted p {p:.4f}" + (f"  (seeds dropped: {miss})" if miss else ""))
            w(f"      per seed: {', '.join(f's{s} {v:+.2f}' for s, v in d)}")
        w("  Secondary (estimates only):")
        for tag, ev in (("events 2-6", LATE), ("event 1", FIRST)):
            for name, x, base in (("O - U", "organized", "single_a"), ("O - O_pass", "organized", "organized_pass"),
                                  ("O_pass - U", "organized_pass", "single_a")):
                for stat in ("cut", "open", "final"):
                    f = lambda r, ev=ev, stat=stat: price_stats(r, ev)[stat]  # noqa: E731
                    w(f"    {tag:<10} {name:<11} {stat:<5} {describe(per_seed(runs, 'persistent', 'buyer_agent', cseeds, f, x, base))}")
        w("  Levels by arm, events 2-6 (mean over seeds):")
        for label in ("single_a", "organized", "organized_pass"):
            vals = [price_stats(runs[("persistent", label, "buyer_agent", s)], LATE) for s in cseeds
                    if ("persistent", label, "buyer_agent", s) in runs]
            if vals:
                w(f"    {label:<15} cut {np.mean([v['cut'] for v in vals]):.3f}  open {np.mean([v['open'] for v in vals]):.3f}"
                  f"  final {np.mean([v['final'] for v in vals]):.3f}  (runs {len(vals)})")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
