"""Replication of H1 on fresh seeds 43-62, as fixed in `2026-10-17 H1 replication (pre-registration).md`.

R1: the H1 statistic (departures18_analysis: the B firm's profit minus the F firm's in brief_revenue:forward_revenue,
both positions pooled event by event, events 2-6, mean of the two institutions), exact two-sided sign-flip, alpha 0.05.
R2: per seed, the conflict gap minus the team gap (the same statistic in brief:forward), exact two-sided sign-flip.
Secondary estimates as listed in the pre-registration; the 40-seed pooled estimate (23-62) is not a test.
Usage: python replication18_analysis.py [output file] [conflict root] [team root]
"""
import sys
from pathlib import Path

import numpy as np

import departures18_analysis as da
import followup17_primary as fp
import pilot17_analysis as pa

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else pa.RESULTS / "replication18_results.txt"
CONF_ROOT = Path(sys.argv[2]) if len(sys.argv) > 2 else pa.RUNS / "replication18"
TEAM_ROOT = Path(sys.argv[3]) if len(sys.argv) > 3 else pa.RUNS / "replication18_team"
SEEDS = list(range(43, 63))
E1, LATE = range(1, 2), range(2, 7)
B, F, BT, FT = da.org("B", True), da.org("F", True), da.org("B", False), da.org("F", False)


def gap(runs, x, y, events, insts=da.INSTS, positions=("xa", "xb"), seeds=SEEDS):
    return da.per_seed(runs, seeds, insts, da.mixed(runs, x, y, events, positions))


def split_ties_gap(runs, x, y, events, seeds=SEEDS):
    """H1 with each tied event's profit split: each firm gets half the profit its own final bid would have earned."""
    from orgsim.primitives import ContractTerms, get_contract_scenarios
    buyers = {sc.name: sc.buyer for sc in get_contract_scenarios()}
    out = []
    for s in seeds:
        per_inst = []
        for inst in da.INSTS:
            pair = [runs.get(("persistent", f"{x}_vs_{y}_{pos}", inst, s)) for pos in ("xa", "xb")]
            if not all(pair):
                continue
            vals = []
            for run, bs in zip(pair, ("A", "B")):
                _, rows, evs = run
                fs = "B" if bs == "A" else "A"
                for k in events:
                    ev = evs[k]
                    terms = {sid: ContractTerms(ev["bids"][sid][-1]["price"], ev["bids"][sid][-1]["delivery_weeks"],
                                                ev["bids"][sid][-1]["quality_tier"]) for sid in ("A", "B")}
                    score = {sid: buyers[ev["contract"]].score(terms[sid]) for sid in ("A", "B")}
                    if abs(score["A"] - score["B"]) < 1e-9 and rows[k]["awarded"] == "True":
                        own = {sid: terms[sid].price - pa.COSTS[ev["type_a" if sid == "A" else "type_b"]].cost(terms[sid])
                               for sid in ("A", "B")}
                        vals.append(own[bs] / 2 - own[fs] / 2)
                    else:
                        vals.append(pa.firm_profit(rows, k, bs) - pa.firm_profit(rows, k, fs))
            per_inst.append(float(np.mean(vals)))
        if len(per_inst) == 2:
            out.append((s, float(np.mean(per_inst))))
    return out


def decomposition(runs, x, y, events, seeds=SEEDS):
    """Per seed, pooled over institutions and positions event by event: B and F win shares and below-cost shares."""
    out = {}
    for s in seeds:
        acc = {"bw": [], "fw": [], "bb": [], "fb": []}
        for inst in da.INSTS:
            for pos, bs in (("xa", "A"), ("xb", "B")):
                r = runs.get(("persistent", f"{x}_vs_{y}_{pos}", inst, s))
                if not r:
                    continue
                _, rows, evs = r
                fs = "B" if bs == "A" else "A"
                for k in events:
                    acc["bw"].append(float(rows[k]["winner"] == bs))
                    acc["fw"].append(float(rows[k]["winner"] == fs))
                    acc["bb"].append(float(evs[k]["outcome"]["all_inflation_rates"][bs] < 0))
                    acc["fb"].append(float(evs[k]["outcome"]["all_inflation_rates"][fs] < 0))
        if acc["bw"]:
            out[s] = {k: float(np.mean(v)) for k, v in acc.items()}
    return out


def main():
    conf, cprob = pa.load([CONF_ROOT])
    team, tprob = pa.load([TEAM_ROOT]) if TEAM_ROOT.exists() else ({}, [])
    lines = [f"H1 replication, seeds 43-62. Conflict root {pa.rel(CONF_ROOT)}; team root {pa.rel(TEAM_ROOT)}"]
    w = lines.append
    for name, rs, pr in (("conflict", conf, cprob), ("team", team, tprob)):
        ids = sorted({(r.get("git_commit"), r.get("git_dirty"), r.get("source_sha256")) for r, _, _ in rs.values()},
                     key=str)
        cells = sorted({k[1:3] for k in rs})
        miss = [(c, sorted(set(SEEDS) - {k[3] for k in rs if k[1:3] == c})) for c in cells]
        w(f"  {name}: runs {len(rs)}; problems {len(pr)}; (commit, dirty, source) {ids}")
        for p in pr:
            w("    PROBLEM " + p)
        w(f"    cells {len(cells)}; missing seeds: {[m for m in miss if m[1]] or 'none'}")

    r1 = gap(conf, B, F, LATE)
    p1 = fp.sign_flip_p([v for _, v in r1])
    m1 = float(np.mean([v for _, v in r1]))
    v1 = "confirmed" if p1 < 0.05 and m1 > 0 else "contradicted" if p1 < 0.05 else "not confirmed"
    w("\nR1 (pre-registered): H1 on seeds 43-62, events 2-6, institutions pooled; exact two-sided sign-flip, alpha 0.05.")
    w(f"  {da.describe(r1)}  p {p1:.6f}  {v1}")
    w(f"      per seed: {da.per_seed_line(r1)}")

    if team:
        tg = dict(gap(team, BT, FT, LATE))
        r2 = [(s, v - tg[s]) for s, v in r1 if s in tg]
        p2 = fp.sign_flip_p([v for _, v in r2])
        m2 = float(np.mean([v for _, v in r2]))
        v2 = "rejects (> 0)" if p2 < 0.05 and m2 > 0 else "rejects (< 0)" if p2 < 0.05 else "not rejected"
        w("\nR2 (pre-registered): per seed, the conflict gap minus the team gap; exact two-sided sign-flip, alpha 0.05.")
        w(f"  {da.describe(r2)}  p {p2:.6f}  {v2}")
        w(f"      per seed: {da.per_seed_line(r2)}")
        w(f"  team gap alone (estimate): {da.describe(sorted(tg.items()))}")
    else:
        w("\nR2: the team cells are not in yet.")

    w("\nSECONDARY (estimates only).")
    e1 = dict(gap(conf, B, F, E1))
    w(f"  H1 at event 1 (placebo)                  {da.describe(sorted(e1.items()))}")
    w(f"  H1 minus its event-1 value               {da.describe([(s, v - e1[s]) for s, v in r1])}")
    for inst in da.INSTS:
        w(f"  H1, {inst:<12} alone                 {da.describe(gap(conf, B, F, LATE, insts=(inst,)))}")
    for pos in ("xa", "xb"):
        w(f"  H1, position {pos} alone                 {da.describe(gap(conf, B, F, LATE, positions=(pos,)))}")
    w(f"  H1 with tied events split                {da.describe(split_ties_gap(conf, B, F, LATE))}")
    for wtag, ev in (("event 1", E1), ("events 2-6", LATE)):
        dec = decomposition(conf, B, F, ev)
        lv = {k: np.mean([d[k] for d in dec.values()]) for k in ("bw", "fw", "bb", "fb")}
        w(f"  [{wtag}] B win {lv['bw']:.4f}  F win {lv['fw']:.4f}  B below cost {lv['bb']:.4f}  F below cost {lv['fb']:.4f}")
        w(f"  [{wtag}] win share B - F         {da.describe([(s, d['bw'] - d['fw']) for s, d in dec.items()])}")
        w(f"  [{wtag}] below-cost share B - F  {da.describe([(s, d['bb'] - d['fb']) for s, d in dec.items()])}")
    old, _ = pa.load([pa.RUNS / "departures18"])
    pooled = gap(old, B, F, LATE, seeds=list(range(23, 43))) + r1
    d = np.array([v for _, v in pooled])
    se = d.std(ddof=1) / np.sqrt(len(d))
    w(f"  H1 pooled over seeds 23-62 (estimate, not a test): n={len(d)}  mean {d.mean():+.3f}  se {se:.3f}  "
      f"(approximate 95% interval with t(39) = 2.023: {d.mean() - 2.023 * se:+.3f} to {d.mean() + 2.023 * se:+.3f})")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
