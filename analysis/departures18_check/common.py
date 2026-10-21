"""Independent loader and statistics for the departures18 check. Written from the plan and the code of 653e656 only;
no analysis script of the project is read or imported."""
import csv
import json
import math
import os
import re
from fractions import Fraction

import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
DEP = REL + "/data/runs/departures18"
CONF = REL + "/data/runs/confirm17_concession"
OUT = REL + "/analysis/results"  # where adversarial.py writes adversarial_out.txt
os.makedirs(OUT, exist_ok=True)
SEEDS = list(range(23, 43))
INSTS = ["mediator", "buyer_agent"]
HOMO_ORGS = ["brief", "brief_revenue", "forward", "forward_revenue", "floor", "floor_revenue",
             "brief_cap", "brief_cap_revenue", "brief_cost", "brief_cost_revenue"]
MIXED_CELLS = [("brief", "forward"), ("brief_revenue", "forward_revenue"), ("floor_revenue", "brief_revenue")]
LATE = [2, 3, 4, 5, 6]

NAME_RE = re.compile(r"persistent_(.+)_(mediator|buyer_agent)_s(\d+)")
MIX_RE = re.compile(r"(.+)_vs_(.+)_(xa|xb)")

# Buyer preferences (orgsim/primitives.get_contract_scenarios) and cost types (orgsim/sequence).
BUYERS = {"Speed Priority": dict(base=150.0, wd=0.80, wq=0.20, bd=60.0, bq=50.0, dmax=8.0, qmax=5),
          "Quality Priority": dict(base=150.0, wd=0.25, wq=0.75, bd=60.0, bq=50.0, dmax=8.0, qmax=5)}
COSTS = {"speed": dict(c=85.0, gd=0.5, gq=1.8), "quality": dict(c=85.0, gd=1.8, gq=0.4)}


def buyer_value(contract, d, q):
    b = BUYERS[contract]
    return b["base"] + b["wd"] * b["bd"] * (b["dmax"] - d) / b["dmax"] + b["wq"] * b["bq"] * (q / b["qmax"])


def own_cost(ctype, d, q):
    c = COSTS[ctype]
    return c["c"] + c["gd"] * (8.0 - d) ** 2 + c["gq"] * q ** 2


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_name(name):
    m = NAME_RE.fullmatch(name)
    if not m:
        return None
    orgpart, inst, seed = m.group(1), m.group(2), int(m.group(3))
    mm = MIX_RE.fullmatch(orgpart)
    if mm:
        x, y, pos = mm.groups()
        return {"kind": "mixed", "x": x, "y": y, "pos": pos, "inst": inst, "seed": seed}
    return {"kind": "homo", "org": orgpart, "inst": inst, "seed": seed}


def load_dir(root):
    """All run folders under root: {folder_name: {"meta", "run", "summary", "events"}}."""
    runs = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p) or name == "logs":
            continue
        meta = parse_name(name)
        if meta is None:
            raise ValueError("unparsed folder " + name)
        files = set(os.listdir(p))
        rec = {"meta": meta, "files": files, "path": p}
        rec["run"] = json.load(open(os.path.join(p, "run.json"), encoding="utf-8")) if "run.json" in files else None
        rec["summary"] = read_csv(os.path.join(p, "summary.csv")) if "summary.csv" in files else None
        rec["events"] = read_jsonl(os.path.join(p, "events.jsonl")) if "events.jsonl" in files else None
        runs[name] = rec
    return runs


def supplier_orgs(meta):
    """Organization of supplier A and B from the folder name."""
    if meta["kind"] == "homo":
        return {"A": meta["org"], "B": meta["org"]}
    if meta["pos"] == "xa":
        return {"A": meta["x"], "B": meta["y"]}
    return {"A": meta["y"], "B": meta["x"]}


def homo_name(org, inst, seed):
    return f"persistent_{org}_{inst}_s{seed}"


def mixed_name(x, y, pos, inst, seed):
    return f"persistent_{x}_vs_{y}_{pos}_{inst}_s{seed}"


def ev(run, e):
    for rec in run["events"]:
        if rec["event"] == e:
            return rec
    raise KeyError(e)


def srow(run, e):
    for row in run["summary"]:
        if int(row["event"]) == e:
            return row
    raise KeyError(e)


def market_margin(rec):
    r = rec["outcome"]["all_inflation_rates"]
    return (r["A"] * 100 + r["B"] * 100) / 2


def firm_profit(rec, sid):
    return rec["outcome"]["supplier_profit"] if rec["outcome"]["winner_id"] == sid else 0.0


def cut(rec, sid):
    b = rec["bids"][sid]
    return b[0]["price"] - b[-1]["price"]


# ---------------- statistics ----------------
T975_19 = 2.093024054408263  # t quantile, 19 df, 0.975


def summarize(d):
    """Signs are counted on the values rounded to 1e-9, so that a difference that is 0 in exact arithmetic but 1e-16
    in floating point is not counted as > 0; the raw float count is kept as pos_raw."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / math.sqrt(n)
    r = np.round(d, 9)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "pos": int((r > 0).sum()), "neg": int((r < 0).sum()),
            "zero": int((r == 0).sum()), "pos_raw": int((d > 0).sum()),
            "lo": mean - T975_19 * se, "hi": mean + T975_19 * se}


def _all_sums_int(v):
    sums = np.zeros(1, dtype=np.int64)
    for x in v:
        sums = np.concatenate([sums + x, sums - x])
    return sums


def sign_flip_p(d, scale=10 ** 6):
    """Exact two-sided sign-flip p over all 2^n patterns, counted on integers (d rounded to 1/scale)."""
    v = [int(round(x * scale)) for x in d]
    sums = _all_sums_int(v)
    obs = abs(sum(v))
    return int(np.count_nonzero(np.abs(sums) >= obs)), len(sums)


def sign_flip_p_float(d, tol=0.0):
    sums = np.zeros(1, dtype=float)
    for x in d:
        sums = np.concatenate([sums + x, sums - x])
    obs = abs(float(np.sum(d)))
    return int(np.count_nonzero(np.abs(sums) >= obs - tol)), len(sums)


def holm(ps):
    """Holm-adjusted p-values in the input order."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adj[i] = running
    return adj


def fmt(s, p=None):
    out = (f"mean {s['mean']:+.6f}  sd {s['sd']:.6f}  se {s['se']:.6f}  >0 {s['pos']}/{s['n']}  (<0 {s['neg']}, "
           f"=0 {s['zero']}; raw-float >0 {s['pos_raw']})  95% CI [{s['lo']:+.4f}, {s['hi']:+.4f}]")
    if p is not None:
        out += f"  p {p:.6f}"
    return out
