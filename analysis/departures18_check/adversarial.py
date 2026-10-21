"""Attacks on the claims in both directions: is H1 an artifact; is the H2 null an artifact of pooling, given D8; does
the floor bind."""
import itertools
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BUYERS, CONF, DEP, INSTS, LATE, OUT, SEEDS, buyer_value, ev, firm_profit, fmt, homo_name,
                    load_dir, market_margin, mixed_name, own_cost, sign_flip_p, srow, summarize, supplier_orgs)

lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def rep(label, d, test=False):
    s = summarize(d)
    if test:
        k, n = sign_flip_p(list(d))
        say(f"{label}: {fmt(s, k / n)}")
    else:
        say(f"{label}: {fmt(s)}")
    return s


runs = load_dir(DEP)
conf = load_dir(CONF)
X, Y = "brief_revenue", "forward_revenue"


def gap_cells(seed, insts=INSTS, positions=("xa", "xb"), events=LATE, x=X, y=Y, f=firm_profit):
    per_inst = []
    for inst in insts:
        vals = []
        for pos in positions:
            r = runs[mixed_name(x, y, pos, inst, seed)]
            orgs = supplier_orgs(r["meta"])
            xs = [s for s in "AB" if orgs[s] == x][0]
            ys = [s for s in "AB" if orgs[s] == y][0]
            for e in events:
                rec = ev(r, e)
                vals.append(f(rec, xs) - f(rec, ys))
        per_inst.append(np.mean(vals))
    return float(np.mean(per_inst))


h1 = np.array([gap_cells(s) for s in SEEDS])
say("=== H1 attacks ===")
rep("H1 (reproduced)", h1, True)
# position x institution
for inst in INSTS:
    for pos in ("xa", "xb"):
        rep(f"H1 {inst} {pos}", [gap_cells(s, insts=[inst], positions=(pos,)) for s in SEEDS], True)
# event by event
for e in range(1, 7):
    rep(f"H1 event {e} alone", [gap_cells(s, events=[e]) for s in SEEDS])
# same-type versus different-type seeds
same = {s: ev(runs[homo_name("brief", "mediator", s)], 1)["same_type"] for s in SEEDS}
for flag in (True, False):
    idx = [i for i, s in enumerate(SEEDS) if same[s] == flag]
    say(f"H1 on {'same' if flag else 'different'}-type seeds (n={len(idx)}): mean {h1[idx].mean():+.4f}, "
        f">0 {(np.round(h1[idx], 9) > 0).sum()}/{len(idx)}")
# leave-one-seed-out and dropping the largest seeds
loo = []
for i in range(20):
    d = np.delete(h1, i)
    k, n = sign_flip_p(list(d))
    loo.append((SEEDS[i], d.mean(), k / n))
say("H1 leave-one-seed-out: mean range [%.4f, %.4f]; p range [%.5f, %.5f]; largest p when dropping seed %d" % (
    min(x[1] for x in loo), max(x[1] for x in loo), min(x[2] for x in loo), max(x[2] for x in loo),
    max(loo, key=lambda x: x[2])[0]))
order = np.argsort(-h1)
for k in (1, 2, 3):
    d = np.delete(h1, order[:k])
    kk, n = sign_flip_p(list(d))
    say(f"H1 without the {k} largest seeds {[SEEDS[j] for j in order[:k]]}: mean {d.mean():+.4f}, p {kk / n:.5f} "
        f"(n={len(d)})")
say("H1 per-seed values:", {s: round(v, 3) for s, v in zip(SEEDS, h1)})
# event 1 as a baseline: H1 minus its event-1 placebo
h1e1 = np.array([gap_cells(s, events=[1]) for s in SEEDS])
rep("H1 minus its event-1 value (per seed)", h1 - h1e1, True)

# no bids, walk-aways, ties in the conflict mixed cell
nob = walk = ties = 0
for inst in INSTS:
    for pos in ("xa", "xb"):
        for s in SEEDS:
            r = runs[mixed_name(X, Y, pos, inst, s)]
            for rec in r["events"]:
                if any(len(rec["bids"][sid]) != 3 for sid in "AB"):
                    nob += 1
                if not rec["outcome"]["awarded"]:
                    walk += 1
                sc = {sid: buyer_value(rec["contract"], rec["bids"][sid][-1]["delivery_weeks"],
                                       rec["bids"][sid][-1]["quality_tier"]) - rec["bids"][sid][-1]["price"]
                      for sid in "AB"}
                if abs(sc["A"] - sc["B"]) < 1e-9:
                    ties += 1
                # winner is the highest score (ties to A) and every recorded winner agrees
                w = "A" if sc["A"] >= sc["B"] - 1e-12 else "B"
                assert rec["outcome"]["winner_id"] == w, (r["meta"], rec["event"])
say(f"conflict mixed cell, 480 events: events with a supplier lacking 3 bids {nob}; walk-aways {walk}; exact score ties"
    f" {ties}; recorded winner = highest score in every event (checked)")

# decomposition of H1: win share times profit per win
say("")
say("=== H2 null against the D8 surplus DiD ===")


def best_surplus(contract, ctype):
    best = -1e18
    for q in range(2, 6):
        for d10 in range(10, 81):
            d = d10 / 10.0
            v = buyer_value(contract, d, q) - own_cost(ctype, d, q)
            if v > best:
                best = v
    return best


BW = {(c, t): best_surplus(c, t) for c in BUYERS for t in ("speed", "quality")}


def homo_seed(org, seed, f, insts=INSTS):
    return float(np.mean([np.mean([f(ev(runs[homo_name(org, inst, seed)], e)) for e in LATE]) for inst in insts]))


def winner(rec):
    return rec["outcome"]["winner_id"]


def wtype(rec):
    w = winner(rec)
    return rec["type_a"] if w == "A" else rec["type_b"]


measures = {
    "margin, mean of both suppliers (H2's outcome)": market_margin,
    "winner's margin %": lambda rec: rec["outcome"]["all_inflation_rates"][winner(rec)] * 100,
    "loser's margin %": lambda rec: rec["outcome"]["all_inflation_rates"]["B" if winner(rec) == "A" else "A"] * 100,
    "winner's price": lambda rec: rec["outcome"]["winning_bid"]["price"],
    "winner's cost": lambda rec: rec["outcome"]["winner_true_cost"],
    "buyer value at winner's terms": lambda rec: buyer_value(rec["contract"], rec["outcome"]["winning_bid"]["delivery_weeks"],
                                                             rec["outcome"]["winning_bid"]["quality_tier"]),
    "market profit": lambda rec: rec["outcome"]["supplier_profit"],
    "buyer surplus": lambda rec: rec["outcome"]["buyer_surplus"],
    "joint surplus J": lambda rec: rec["outcome"]["joint_surplus"],
    "allocation loss E - B_w": lambda rec: rec["outcome"]["efficient_surplus"] - BW[(rec["contract"], wtype(rec))],
    "configuration loss B_w - J": lambda rec: BW[(rec["contract"], wtype(rec))] - rec["outcome"]["joint_surplus"],
    "correct-winner share": lambda rec: 1.0 if rec["outcome"]["correct_winner"] else 0.0,
    "winner's delivery weeks": lambda rec: rec["outcome"]["winning_bid"]["delivery_weeks"],
    "winner's quality tier": lambda rec: rec["outcome"]["winning_bid"]["quality_tier"],
}
for lab, f in measures.items():
    lv = {org: np.array([homo_seed(org, s, f) for s in SEEDS]) for org in ("brief", "forward", "brief_revenue",
                                                                          "forward_revenue")}
    did = (lv["brief_revenue"] - lv["forward_revenue"]) - (lv["brief"] - lv["forward"])
    fconf = lv["forward_revenue"] - lv["forward"]
    bconf = lv["brief_revenue"] - lv["brief"]
    s = summarize(did)
    say(f"{lab}: levels B {lv['brief'].mean():.3f}, F {lv['forward'].mean():.3f}, B_conf {lv['brief_revenue'].mean():.3f},"
        f" F_conf {lv['forward_revenue'].mean():.3f} | F_conf-F_team {fconf.mean():+.3f} | B_conf-B_team "
        f"{bconf.mean():+.3f} | DiD {s['mean']:+.3f} (se {s['se']:.3f}, >0 {s['pos']}/20)")
# H2-H4 per institution
for inst in INSTS:
    def m(org, s):
        return homo_seed(org, s, market_margin, insts=[inst])
    h2i = [(m("brief_revenue", s) - m("forward_revenue", s)) - (m("brief", s) - m("forward", s)) for s in SEEDS]
    h4i = [(m("floor_revenue", s) - m("brief_revenue", s)) - (m("floor", s) - m("brief", s)) for s in SEEDS]
    rep(f"H2 {inst} alone", h2i, True)
    rep(f"H4 {inst} alone", h4i, True)
    rep(f"H3 {inst} alone", [gap_cells(s, insts=[inst], x="floor_revenue", y="brief_revenue") for s in SEEDS], True)
for inst in INSTS:
    def bs(org, s):
        return homo_seed(org, s, measures["buyer surplus"], insts=[inst])
    rep(f"D8 buyer-surplus DiD {inst} alone", [(bs("brief_revenue", s) - bs("forward_revenue", s))
                                              - (bs("brief", s) - bs("forward", s)) for s in SEEDS])

say("")
say("=== Does the floor bind? (all floor cells, homogeneous and mixed) ===")
c = Counter()
slack = []
for name, r in runs.items():
    orgs = supplier_orgs(r["meta"])
    for rec in r["events"]:
        for sid in "AB":
            if not orgs[sid].startswith("floor"):
                continue
            ctype = rec["type_a"] if sid == "A" else rec["type_b"]
            for b in rec["bids"][sid]:
                fa = b.get("floor")
                if not fa:
                    continue
                c["audited bids"] += 1
                c["returned"] += bool(fa["reasked"])
                c["raised"] += bool(fa["raised"])
                cost = own_cost(ctype, b["delivery_weeks"], b["quality_tier"])
                sl = b["price"] - cost - fa["minimum_profit"]
                slack.append(sl)
                if sl < -1e-9:
                    c["standing bid below cost + minimum"] += 1
                if fa["minimum_profit"] > 0:
                    c["bids under a positive floor"] += 1
                    if abs(sl) < 0.005:
                        c["priced exactly at cost + minimum (to the cent), positive floor"] += 1
                    if -1e-9 <= sl < 1.0:
                        c["within $1 above cost + minimum, positive floor"] += 1
                    if b["round"] == 3:
                        c["final bids under a positive floor"] += 1
                        if -1e-9 <= sl < 1.0:
                            c["final bids within $1 above cost + minimum, positive floor"] += 1
                if b["price"] < cost - 1e-9:
                    c["audited bids below cost"] += 1
sl = np.array(slack)
say(dict(c))
say(f"slack = price - cost - minimum over audited bids: median {np.median(sl):.2f}, 10th pct {np.percentile(sl, 10):.2f}, "
    f"share < $1 {np.mean(sl < 1):.4f}, share < $2 {np.mean(sl < 2):.4f}")

say("")
say("=== frozen confirm17_concession runs (C1, C2 data) ===")
fl = Counter()
for name, r in conf.items():
    u = r["run"]["usage"]
    fl["runs"] += 1
    fl["complete"] += r["run"]["status"] == "complete"
    fl["sha 46258eff"] += r["run"]["source_sha256"].startswith("46258eff")
    fl["failed/truncated/no_usage"] += u["failed_requests"] + u["truncated"] + u["no_usage"]
    fl["attempts-requests"] += u["http_attempts"] - u["requests"]
    fl["non-200"] += sum(n for k, n in u["http_responses"].items() if k != "200")
    fl["events"] += len(r["events"])
    fl["rule-tagged bids"] += sum("PRICE SET BY RULE" in b["reasoning"] for rec in r["events"] for sid in "AB"
                                  for b in rec["bids"][sid])
    fl["walk-aways"] += sum(not rec["outcome"]["awarded"] for rec in r["events"])
say(dict(fl))
open(os.path.join(OUT, "adversarial_out.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
