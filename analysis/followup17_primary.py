"""Pre-specified tests of the follow-up, as fixed in `2026-10-17 follow-up analysis plan.md`.

Data: the persistent-environment core of the pilot (data/runs/pilot17, seeds 3, 7, 9, 12, 19) and the follow-up
(data/runs/followup17, 15 more seeds), read together with pilot17_analysis.load. The unit is the seed; each seed's value
is the mean over events 2-6. Ubar = the per-seed mean of U (single_a) and U' (single_b).

Primary family (Holm over the six, two-sided exact sign-flip over all 2^n sign patterns), per institution:
  P1 U0 - Ubar, market profit;  P2 O - Ubar, market profit;  P3 the mixed O:U gap (firm profit, both positions pooled).
The pilot's lead (P3 under the buyer's agent) is also tested on the 15 new seeds alone, outside the family.
Secondary (mean, sd, se, count > 0; no tests): the mixed U0:U gap, O_pass - Ubar, O - O_pass, the placebo, and every
comparison over events 2-5, events 1-6 and event 1 alone.
Usage: python followup17_primary.py [output file] [root;root...]
"""
import sys
from pathlib import Path

import numpy as np

import pilot17_analysis as pa

ROOTS = ([Path(p) for p in sys.argv[2].split(";")] if len(sys.argv) > 2
         else [pa.RUNS / "pilot17", pa.RUNS / "followup17"])
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else pa.RESULTS / "followup17_primary_results.txt"
PILOT_SEEDS = {3, 7, 9, 12, 19}
WINDOWS = {"events 2-6": range(2, 7), "events 2-5": range(2, 6), "events 1-6": range(1, 7), "event 1": range(1, 2)}


def sign_flip_p(d):
    """Exact two-sided sign-flip p: the share of the 2^n sign patterns whose |mean| is at least the observed one."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    observed = abs(d.sum())
    count, chunk = 0, 1 << 16
    for start in range(0, 1 << n, chunk):
        idx = np.arange(start, min(start + chunk, 1 << n), dtype=np.int64)[:, None]
        signs = 1 - 2 * ((idx >> np.arange(n)) & 1)
        count += int(np.sum(np.abs(signs @ d) >= observed - 1e-9))
    return count / (1 << n)


def holm(ps):
    """Holm-adjusted p-values, in the input order."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def market_diff(runs, inst, seeds, events, arm, base):
    """Per seed: market profit of `arm` minus that of `base` ("ubar" = mean of single_a and single_b)."""
    out = []
    for s in seeds:
        x = runs.get(("persistent", arm, inst, s))
        if base == "ubar":
            u, u2 = runs.get(("persistent", "single_a", inst, s)), runs.get(("persistent", "single_b", inst, s))
            if x and u and u2:
                ubar = (pa.market(u[1], "supplier_profit", events) + pa.market(u2[1], "supplier_profit", events)) / 2
                out.append((s, pa.market(x[1], "supplier_profit", events) - ubar))
        else:
            y = runs.get(("persistent", base, inst, s))
            if x and y:
                out.append((s, pa.market(x[1], "supplier_profit", events) - pa.market(y[1], "supplier_profit", events)))
    return out


def mixed_gap(runs, inst, seeds, events, x):
    """Per seed: X's firm profit minus the single firm's, pooled over positions xa and xb event by event.
    x = "placebo" builds the gap between two identical single firms from the U and U' runs (as in section C0)."""
    out = []
    for s in seeds:
        if x == "placebo":
            pos = {"xa": runs.get(("persistent", "single_a", inst, s)), "xb": runs.get(("persistent", "single_b", inst, s))}
        else:
            pos = {p: runs.get(("persistent", f"{x}_vs_single_{p}", inst, s)) for p in ("xa", "xb")}
        if not all(pos.values()):
            continue
        vals = []
        for p, xs in (("xa", "A"), ("xb", "B")):
            rows = pos[p][1]
            us = "B" if xs == "A" else "A"
            vals += [pa.firm_profit(rows, k, xs) - pa.firm_profit(rows, k, us) for k in events if k in rows]
        out.append((s, float(np.mean(vals))))
    return out


def describe(pairs):
    d = [v for _, v in pairs]
    if len(d) < 2:
        return f"n={len(d)}"
    sd = float(np.std(d, ddof=1))
    return (f"n={len(d)}  mean {np.mean(d):+.3f}  sd {sd:.3f}  se {sd / np.sqrt(len(d)):.3f}  "
            f">0: {sum(v > 0 for v in d)}/{len(d)}")


def main():
    runs, problems = pa.load(ROOTS)
    core = {k: v for k, v in runs.items() if k[0] == "persistent" and "revenue" not in k[1]}
    seeds = sorted({k[3] for k in core})
    new = [s for s in seeds if s not in PILOT_SEEDS]
    lines = []
    w = lines.append
    commits = sorted({(r.get("git_commit"), r.get("git_dirty"), r.get("source_sha256")) for r, _, _ in core.values()},
                     key=str)
    cells = sorted({(k[1], k[2]) for k in core})
    w(f"Follow-up tests. Roots: {'; '.join(pa.rel(r) for r in ROOTS)}")
    w(f"Persistent core runs {len(core)} ({len(cells)} cells x {len(seeds)} seeds: {seeds}); new seeds {new}")
    w(f"(commit, dirty, source): {commits}")
    w(f"Integrity problems over all runs read: {len(problems)}")
    for p in problems:
        w("  PROBLEM " + p)
    missing = [(c, s) for c in cells for s in seeds if (("persistent",) + c + (s,)) not in core]
    w(f"Missing (cell, seed) pairs: {missing if missing else 'none'}")

    w("\nPRIMARY FAMILY (events 2-6; exact two-sided sign-flip over seeds; Holm over the six).")
    tests = []
    for inst in pa.INSTS:
        tests.append((f"P1 U0 - Ubar      {inst}", market_diff(core, inst, seeds, WINDOWS["events 2-6"],
                                                             "single_norecord", "ubar")))
        tests.append((f"P2 O - Ubar       {inst}", market_diff(core, inst, seeds, WINDOWS["events 2-6"],
                                                             "organized", "ubar")))
        tests.append((f"P3 mixed O:U gap  {inst}", mixed_gap(core, inst, seeds, WINDOWS["events 2-6"], "organized")))
    ps = [sign_flip_p([v for _, v in pairs]) for _, pairs in tests]
    for (name, pairs), p, adj in zip(tests, ps, holm(ps)):
        w(f"  {name}  {describe(pairs)}  p {p:.4f}  Holm p {adj:.4f}  {'REJECT' if adj < 0.05 else 'not rejected'}")
        w(f"      per seed: {', '.join(f's{s} {v:+.2f}' for s, v in pairs)}")

    w("\nTHE PILOT'S LEAD on the new seeds only (P3 under the buyer's agent; exact two-sided sign-flip; not in Holm).")
    lead = mixed_gap(core, "buyer_agent", new, WINDOWS["events 2-6"], "organized")
    p_lead = sign_flip_p([v for _, v in lead])
    w(f"  mixed O:U gap, buyer_agent, new seeds  {describe(lead)}  p {p_lead:.4f}  "
      f"{'confirmed (rejects, same sign as the pilot)' if p_lead < 0.05 and np.mean([v for _, v in lead]) > 0 else 'not confirmed'}")
    w(f"      per seed: {', '.join(f's{s} {v:+.2f}' for s, v in lead)}")

    w("\nSECONDARY (mean, sd, se, count > 0; no tests).")
    for tag, events in WINDOWS.items():
        w(f"  [{tag}]")
        for inst in pa.INSTS:
            rows = [
                ("U0 - Ubar", market_diff(core, inst, seeds, events, "single_norecord", "ubar")),
                ("O - Ubar", market_diff(core, inst, seeds, events, "organized", "ubar")),
                ("O_pass - Ubar", market_diff(core, inst, seeds, events, "organized_pass", "ubar")),
                ("O - O_pass", market_diff(core, inst, seeds, events, "organized", "organized_pass")),
                ("U' - U (noise)", market_diff(core, inst, seeds, events, "single_b", "single_a")),
                ("mixed O:U gap", mixed_gap(core, inst, seeds, events, "organized")),
                ("mixed U0:U gap", mixed_gap(core, inst, seeds, events, "single_norecord")),
                ("placebo gap (U:U)", mixed_gap(core, inst, seeds, events, "placebo")),
            ]
            for name, pairs in rows:
                w(f"    {inst:<12} {name:<18} {describe(pairs)}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
