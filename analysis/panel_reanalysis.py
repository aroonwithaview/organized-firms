"""POST HOC panel re-analysis on two seed sets, as fixed in `2026-10-18 panel re-analysis (post hoc spec).md`.

Mixed cells only: conflict brief_revenue:forward_revenue and team brief:forward, both positions, both institutions.
Seeds 23-42 from data/runs/departures18; seeds 43-62 from data/runs/replication18 (conflict) and replication18_team (team).
Per seed: pool positions event by event within an institution, then average the two institutions.
A event-1 market margin, conflict - team; B below-cost share of own final bids, F firm and B firm, conflict - team;
C the F firm's own cost at final terms and final price, conflict - team; D the B firm's round-1 price equal to a dollar
figure in its own briefing, against the next seed's briefing (placebo); E win share B - F. Estimates only.
Usage: python panel_reanalysis.py [output file]
"""
import re
import sys
from pathlib import Path

import numpy as np

import pilot17_analysis as pa
from orgsim.primitives import ContractTerms

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else pa.RESULTS / "panel_reanalysis_results.txt"
SETS = {"23-42": (list(range(23, 43)), [pa.RUNS / "departures18"]),
        "43-62": (list(range(43, 63)), [pa.RUNS / "replication18", pa.RUNS / "replication18_team"])}
CELLS = {"conflict": ("brief_revenue", "forward_revenue"), "team": ("brief", "forward")}
INSTS = pa.INSTS
E1, LATE = range(1, 2), range(2, 7)
PRICE = re.compile(r"\$\s?(\d{2,3}(?:\.\d{1,2})?)")
T975_19 = 2.093


def describe(vals):
    d = np.array(vals, dtype=float)
    sd = float(np.std(d, ddof=1))
    se = sd / np.sqrt(len(d))
    r = np.round(d, 9)
    return (f"n={len(d)}  mean {d.mean():+.4f}  sd {sd:.4f}  se {se:.4f}  95% CI [{d.mean() - T975_19 * se:+.4f}, "
            f"{d.mean() + T975_19 * se:+.4f}]  >0: {int((r > 0).sum())}  <0: {int((r < 0).sum())}  =0: {int((r == 0).sum())}")


def cell_runs(runs, obj, inst, s):
    """{position: (run, B supplier id)} for the mixed cell of an objective, or None if a position is missing."""
    b, f = CELLS[obj]
    out = {}
    for pos, bs in (("xa", "A"), ("xb", "B")):
        r = runs.get(("persistent", f"{b}_vs_{f}_{pos}", inst, s))
        if not r:
            return None
        out[pos] = (r, bs)
    return out


def per_seed(runs, seeds, obj, fn):
    """fn(cell, inst) -> list of event-level values (pooled over positions); per seed, the mean of the institution
    means. Returns {seed: value}."""
    out = {}
    for s in seeds:
        inst_vals = []
        for inst in INSTS:
            cell = cell_runs(runs, obj, inst, s)
            if cell is None:
                break
            vals = fn(cell, inst)
            inst_vals.append(float(np.mean(vals)))
        else:
            out[s] = float(np.mean(inst_vals))
    return out


def margin_e1(cell, inst):
    vals = []
    for pos, (run, bs) in cell.items():
        ev = run[2][1]
        vals.append(np.mean([ev["outcome"]["all_inflation_rates"][sid] * 100 for sid in ("A", "B")]))
    return vals


def below(firm):
    def fn(cell, inst):
        vals = []
        for pos, (run, bs) in cell.items():
            sid = bs if firm == "B" else ("B" if bs == "A" else "A")
            vals += [float(run[2][k]["outcome"]["all_inflation_rates"][sid] < 0) for k in LATE]
        return vals
    return fn


def f_terms(what):
    def fn(cell, inst):
        vals = []
        for pos, (run, bs) in cell.items():
            fs = "B" if bs == "A" else "A"
            for k in LATE:
                ev = run[2][k]
                b = ev["bids"][fs][-1]
                if what == "price":
                    vals.append(b["price"])
                else:
                    own = pa.COSTS[ev["type_a" if fs == "A" else "type_b"]]
                    vals.append(own.cost(ContractTerms(b["price"], b["delivery_weeks"], b["quality_tier"])))
        return vals
    return fn


def win_share(cell, inst):
    vals = []
    for pos, (run, bs) in cell.items():
        rows = run[1]
        fs = "B" if bs == "A" else "A"
        vals += [float(rows[k]["winner"] == bs) - float(rows[k]["winner"] == fs) for k in LATE]
    return vals


def anchoring(runs, seeds, obj):
    """Per seed: own-briefing match share minus next-seed-briefing match share for the B firm's round-1 price; plus
    the pooled shares."""
    own_all, plc_all, per = [], [], {}
    for i, s in enumerate(seeds):
        other = seeds[(i + 1) % len(seeds)]
        inst_diffs = []
        for inst in INSTS:
            cell, cell_o = cell_runs(runs, obj, inst, s), cell_runs(runs, obj, inst, other)
            own, plc = [], []
            for pos in ("xa", "xb"):
                (run, bs), (run_o, _) = cell[pos], cell_o[pos]
                for k in LATE:
                    b = (run[2][k].get("analyst_briefings") or {}).get(bs)
                    bo = (run_o[2][k].get("analyst_briefings") or {}).get(bs)
                    if not b or not bo:
                        continue
                    figs = {round(float(x), 2) for x in PRICE.findall(b)}
                    figo = {round(float(x), 2) for x in PRICE.findall(bo)}
                    p = round(run[2][k]["bids"][bs][0]["price"], 2)
                    own.append(float(p in figs))
                    plc.append(float(p in figo))
            own_all += own
            plc_all += plc
            inst_diffs.append(np.mean(own) - np.mean(plc))
        per[s] = float(np.mean(inst_diffs))
    return per, float(np.mean(own_all)), float(np.mean(plc_all)), len(own_all)


def main():
    lines = ["POST HOC panel re-analysis (estimates only; no tests). Mixed cells; seeds pooled over positions, then "
             "institutions."]
    w = lines.append
    for tag, (seeds, roots) in SETS.items():
        runs, problems = pa.load(roots)
        w(f"\n=== seeds {tag}: runs {len(runs)}; problems {len(problems)} ===")
        # A
        a = {o: per_seed(runs, seeds, o, margin_e1) for o in CELLS}
        w(f"  A event-1 market margin: conflict level {np.mean(list(a['conflict'].values())):.4f}  team level "
          f"{np.mean(list(a['team'].values())):.4f}")
        w(f"    conflict - team: {describe([a['conflict'][s] - a['team'][s] for s in seeds])}")
        # B
        for firm in ("F", "B"):
            v = {o: per_seed(runs, seeds, o, below(firm)) for o in CELLS}
            w(f"  B below-cost share of own final bids, {firm} firm: conflict {np.mean(list(v['conflict'].values())):.4f}"
              f"  team {np.mean(list(v['team'].values())):.4f}")
            w(f"    conflict - team: {describe([v['conflict'][s] - v['team'][s] for s in seeds])}")
        # C
        for what in ("cost", "price"):
            v = {o: per_seed(runs, seeds, o, f_terms(what)) for o in CELLS}
            w(f"  C F firm's final-bid {what}: conflict {np.mean(list(v['conflict'].values())):.4f}  team "
              f"{np.mean(list(v['team'].values())):.4f}")
            w(f"    conflict - team: {describe([v['conflict'][s] - v['team'][s] for s in seeds])}")
        # D
        for o in CELLS:
            per, own, plc, n = anchoring(runs, seeds, o)
            w(f"  D {o}: B round-1 price equals a figure in its own briefing {own:.4f} against the next seed's "
              f"briefing {plc:.4f} (bids {n})")
            w(f"    own - placebo, per seed: {describe([per[s] for s in seeds])}")
        # E
        for o in CELLS:
            v = per_seed(runs, seeds, o, win_share)
            w(f"  E {o}: win share B - F {describe([v[s] for s in seeds])}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
