import argparse
import base64
import csv
import hashlib
import io
import itertools
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy
import pytest

import orgsim.llm
import orgsim.llm_agents as llm_agents
import scripts.run_demo as demo
from orgsim import prompts as P
from orgsim.llm import ChatClient, sum_usage
from orgsim.llm_agents import NO_TOOL_TAG, RULE_PRICE_TAG
from orgsim.mocks import MockBuyerAgentPlatform, MockPlatformMediator, MockStrategicSupplier
from orgsim.organization import ANALYST_ORGANIZATIONS, MEMO_HEADER
from orgsim.primitives import ContractTerms, SupplierBid
from orgsim.sequence import SUPPLIER_IDS, make_supplier
from scripts.run_demo import main

try:
    import openai
    OPENAI_VERSION = openai.__version__
except ImportError:
    OPENAI_VERSION = "not installed"

ROOT = Path(__file__).resolve().parents[1]
MOCK = ["--backend", "mock", "--events", "2", "--rounds", "2"]
ONE_CELL = ["--institutions", "mediator", "--organizations", "single"]
KEY = "fake-key-for-tests-1"
URL = "http://user:SECRET1@host/v1?api_key=SECRET2"       # obviously fake credentials
# Fake credentials of 8 or more characters (a real key is far longer), and a URL that holds three of them.
CREDENTIALS = {
    "--api-key value": KEY,
    "$OPENAI_API_KEY": "fake-env-key-for-tests-2",
    "url user name": "fake-user-name-3",
    "url password": "fake-password-4",
    "url query value": "fake-query-value-5",
}
LONG_URL = "http://fake-user-name-3:fake-password-4@host/v1?token=fake-query-value-5"
# Ways an error message can join a credential to other characters.
JOINS = {"query string": "api_key%3D{}", "header": "Bearer%20{}", "curly quotes": "‘{}’", "prefix": "x-{}"}
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
COLUMNS = [
    "institution", "organization", "organization_a", "organization_b", "prompt_set", "environment", "seed", "event",
    "contract", "type_a", "type_b", "same_type", "winner", "awarded", "correct_winner", "efficient_winner",
    "efficiency", "buyer_surplus", "supplier_profit", "joint_surplus", "n_rule_priced_bids", "n_tool_fallback_bids",
    "n_rival_price_quotes",
]
EVENT_KEYS = {
    "institution", "organization", "organization_a", "organization_b", "prompt_set", "environment", "seed", "event",
    "contract", "type_a", "type_b", "same_type", "winner", "awarded", "correct_winner", "efficiency", "buyer_surplus",
    "supplier_profit", "joint_surplus", "score_margin_pct", "outcome", "analyst_briefings", "analyst_postures", "bids",
}
OUTCOME_KEYS = {
    "regime", "run_id", "winner_id", "awarded", "winning_bid", "winner_true_cost", "buyer_surplus",
    "supplier_profit", "joint_surplus", "efficient_surplus", "efficient_winner_id", "correct_winner",
    "efficiency_ratio", "all_true_costs", "all_inflation_rates",
}
RUN_KEYS = {
    "status", "error", "started_at", "finished_at", "usage", "argv", "backend", "model", "base_url", "hosted",
    "default_sampling", "max_tokens", "max_retries", "timeout", "no_tools", "institutions", "organizations", "mixed",
    "prompts", "environment", "events", "rounds", "max_history_events", "seeds", "debug_log", "git_commit",
    "git_dirty", "source_sha256", "python", "numpy", "openai",
}


def _rows(folder):
    with open(folder / "summary.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _events(folder):
    with open(folder / "events.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _run(folder):
    return json.loads((folder / "run.json").read_text(encoding="utf-8"))


def _expected_source_sha256(root):
    """The recipe in the epilog, written out again without the script's code: orgsim/*.py and
    scripts/run_demo.py in sorted order of their paths, each as its path, a NUL byte, its bytes and a NUL byte."""
    names = sorted(["orgsim/" + n for n in os.listdir(root / "orgsim") if n.endswith(".py")] + ["scripts/run_demo.py"])
    data = b""
    for name in names:
        with open(root / name, "rb") as f:
            data += name.encode("ascii") + b"\x00" + f.read() + b"\x00"
    return hashlib.sha256(data).hexdigest()


def _copy_code(dest):
    """An exported copy of the code (orgsim/*.py and scripts/run_demo.py), with no .git folder."""
    (dest / "orgsim").mkdir(parents=True)
    (dest / "scripts").mkdir()
    for path in (ROOT / "orgsim").glob("*.py"):
        shutil.copy(path, dest / "orgsim" / path.name)
    shutil.copy(ROOT / "scripts" / "run_demo.py", dest / "scripts" / "run_demo.py")
    return dest


@pytest.fixture(autouse=True)
def no_git_variables(monkeypatch):
    """GIT_DIR and the like (set when tests run from a git hook) would point git at another repository."""
    for name in [n for n in os.environ if n.startswith("GIT_")]:
        monkeypatch.delenv(name)


def _git_env():
    """The environment for git and for a script that runs git: the anonymous identity for a commit."""
    return {**os.environ, "GIT_AUTHOR_NAME": "Anonymous Authors", "GIT_AUTHOR_EMAIL": "anonymous@example.org",
            "GIT_COMMITTER_NAME": "Anonymous Authors", "GIT_COMMITTER_EMAIL": "anonymous@example.org"}


def _git(path, *args):
    assert not path.resolve().is_relative_to(ROOT), "a test repository must be outside the release repository"
    done = subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=path, env=_git_env(), check=True,
                          capture_output=True, text=True, timeout=60)
    return done.stdout.strip()


def _make_repo(path):
    """Turn the folder `path` into a git repository with one commit holding what it contains; return the commit."""
    _git(path, "init", "-q")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "the code")
    return _git(path, "rev-parse", "HEAD")


def _run_copy(code, out):
    """Run the copy of the script in `code` as its own process, so that the repository it reports is the
    copy's, and return its run.json."""
    done = subprocess.run([sys.executable, "-B", str(code / "scripts" / "run_demo.py"), *MOCK, "--out", str(out)],
                          capture_output=True, env=_git_env(), cwd=out.parent, timeout=120)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")[-500:]
    return _run(out)


def _patch_agents(monkeypatch, suppliers=None, mediator=None, buyer_agent=None):
    """Replace parts of what the demo builds for each cell. `suppliers` is a function that returns
    a new dict of supplier agents each time, since the stubs count their calls."""
    real_build = demo.build_agents

    def build(organization, args, seed):
        built_suppliers, built_mediator, built_buyer_agent, analysts = real_build(organization, args, seed)
        return (suppliers() if suppliers else built_suppliers, mediator or built_mediator,
                buyer_agent or built_buyer_agent, analysts)

    monkeypatch.setattr(demo, "build_agents", build)


class FailsAfter(MockStrategicSupplier):
    """Bids as the mock does for its first `ok_bids` bids (two rounds make one event), then raises."""

    def __init__(self, seed, ok_bids=2, message="model unavailable", on_failure=None):
        super().__init__(seed=seed)
        self.bids, self.ok_bids, self.message, self.on_failure = 0, ok_bids, message, on_failure

    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        self.bids += 1
        if self.bids > self.ok_bids:
            if self.on_failure:
                self.on_failure()
            raise RuntimeError(self.message)
        return super().generate_bid(supplier_costs, regime_context, round_num, history)


def _failing(**kw):
    """A supplier factory for _patch_agents: both suppliers fail after `ok_bids` bids."""
    return lambda: {sid: FailsAfter(i, **kw) for i, sid in enumerate(SUPPLIER_IDS)}


# What a fake endpoint reports as the usage of one reply, by role: (prompt tokens, completion tokens).
FAKE_TOKENS = {"bidder": (100, 20), "analyst": (300, 40), "mediator": (200, 30), "buyer_agent": (150, 25)}


def _fake_endpoint(bid=None, calls=None, fail_bidders_from=None, finish_reasons=None, no_usage=(), on_request=None):
    """An OpenAI-style client whose answer depends on the role that its system prompt gives. The
    bidders answer `bid`, a good bid unless another text is given. Every reply reports the tokens of
    FAKE_TOKENS for its role (none at all for the roles in `no_usage`) and the finish reason that
    `finish_reasons` gives for the role ("stop" otherwise; None is no finish reason). `calls` collects
    the requests. From the `fail_bidders_from`-th request of the bidders on, they raise an error.
    `on_request(n)` is called with the number (1, 2, ...) of each request this endpoint gets, before
    it is answered, and may raise (an interrupt) or end the process."""
    bid = bid or json.dumps({"price": 120, "delivery_weeks": 6, "quality_tier": 3, "reasoning": "ok"})
    bidder_prompts = {P.executive_prompt(org) for org in P.ORGANIZATIONS} | {P.sealed_bid_prompt()}
    bidder_requests, requests = [], []

    def create(**kw):
        if calls is not None:
            calls.append(kw)
        requests.append(kw)
        if on_request is not None:
            on_request(len(requests))
        system = kw["messages"][0]["content"]
        if system in bidder_prompts:
            role, text = "bidder", bid
            bidder_requests.append(system)
            if fail_bidders_from and len(bidder_requests) >= fail_bidders_from:
                raise RuntimeError("boom")
        elif system == P.ANALYST_PROMPT:
            role, text = "analyst", "The record is thin. Posture: GUARDED."
        else:
            assert system in (P.MEDIATOR_PROMPT, P.BUYER_AGENT_PROMPT), system[:60]
            role, text = ("mediator" if system == P.MEDIATOR_PROMPT else "buyer_agent"), "Sharpen your price."
        message = SimpleNamespace(content=text, tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason=(finish_reasons or {}).get(role, "stop"))
        usage = (None if role in no_usage else
                 SimpleNamespace(prompt_tokens=FAKE_TOKENS[role][0], completion_tokens=FAKE_TOKENS[role][1]))
        return SimpleNamespace(choices=[choice], usage=usage)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class _Made(list):
    """What the demo asked make_client for: (base_url, api_key, local) tuples, in `options` the other
    keyword options of each call, and in `clients` the clients it got back."""

    def __init__(self):
        super().__init__()
        self.options = []
        self.clients = []


def _use_endpoint(monkeypatch, **endpoint_options):
    """Replace make_client (no network) with one that wraps a fake endpoint, a new one for each client,
    built with `endpoint_options`. Returns the record of what the demo asked for."""
    made = _Made()

    def make_client(base_url, api_key=None, local=True, **options):
        made.append((base_url, api_key, local))
        made.options.append(dict(options))
        for name in ("max_retries", "timeout"):               # settings of the OpenAI constructor, not of ChatClient
            options.pop(name, None)
        client = ChatClient(_fake_endpoint(**endpoint_options), local=local, **options)
        made.clients.append(client)
        return client

    monkeypatch.setattr(orgsim.llm, "make_client", make_client)
    return made


@pytest.fixture
def fake_make_client(monkeypatch):
    """Replace make_client (no network). The list records the arguments the demo passed."""
    return _use_endpoint(monkeypatch)


# ---- the mock run: files, schema, reproducibility ----

def test_mock_demo_writes_every_cell(tmp_path):
    assert main(MOCK + ["--out", str(tmp_path)]) == 0
    with open(tmp_path / "summary.csv", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == COLUMNS                      # the exact schema, in order
    assert len(rows) == 2 * 3 * 2  # institutions x organizations x events
    assert {(r["institution"], r["organization"]) for r in rows} == {
        (inst, org) for inst in ("mediator", "buyer_agent") for org in ("single", "organized", "organized_guided")}
    assert {r["efficient_winner"] for r in rows} <= set(SUPPLIER_IDS)
    assert all(r["n_rule_priced_bids"] == r["n_tool_fallback_bids"] == r["n_rival_price_quotes"] == "0"
               for r in rows)                                                                      # mocks
    events = _events(tmp_path)
    assert len(events) == len(rows)
    first = events[0]
    assert {len(rounds) for rounds in first["bids"].values()} == {2}  # every round's bid is kept
    assert set(first["bids"]["A"][0]) == {"round", "price", "delivery_weeks", "quality_tier", "reasoning",
                                          "feedback"}


@pytest.mark.parametrize("flag, value", [("--rounds", "1"), ("--events", "0")])
def test_demo_rejects_too_few_rounds_or_events(flag, value, tmp_path, capsys):
    argv = ["--backend", "mock", "--events", "2", "--rounds", "2", "--out", str(tmp_path)]
    argv[argv.index(flag) + 1] = value
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2                                  # an argparse usage error
    assert "must be at least" in capsys.readouterr().err
    assert not (tmp_path / "summary.csv").exists()


def test_demo_runs_with_piped_stdout(tmp_path):
    env = {**os.environ, "PYTHONUTF8": "0"}
    env.pop("PYTHONIOENCODING", None)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_demo.py"), "--backend", "mock",
                        "--events", "2", "--rounds", "2", "--out", str(tmp_path)],
                       capture_output=True, env=env, cwd=tmp_path, timeout=120)  # cwd outside the repo on purpose
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-500:]


def test_mock_demo_is_reproducible_and_paired(tmp_path):
    argv = ["--backend", "mock", "--events", "4", "--rounds", "2", "--seeds", "1", "2"]
    assert main(argv + ["--out", str(tmp_path / "a")]) == 0
    assert main(argv + ["--out", str(tmp_path / "b")]) == 0
    for name in ("summary.csv", "events.jsonl"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes(), name
    plans = {}  # the six cells of a seed face the same contracts and cost types
    for r in _rows(tmp_path / "a"):
        plans.setdefault((r["seed"], r["event"]), set()).add((r["contract"], r["type_a"], r["type_b"]))
    assert len(plans) == 2 * 4 and all(len(p) == 1 for p in plans.values())


def test_mock_rows_of_the_three_organizations_are_identical_as_the_note_says(tmp_path):
    argv = ["--backend", "mock", "--events", "4", "--rounds", "2", "--seeds", "1", "2", "--out", str(tmp_path)]
    assert main(argv) == 0
    cells = {}
    for r in _rows(tmp_path):
        cells.setdefault((r["seed"], r["institution"], r["event"]), {})[r["organization"]] = {
            k: v for k, v in r.items() if k not in ("organization", "organization_a", "organization_b")}
    assert len(cells) == 2 * 2 * 4
    assert all(set(group) == {"single", "organized", "organized_guided"} for group in cells.values())
    assert all(group["single"] == group["organized"] == group["organized_guided"] for group in cells.values())


def test_the_mock_run_briefs_from_event_2_in_the_two_organized_firms_and_never_in_the_single_firm(tmp_path):
    assert main(MOCK + ["--events", "3", "--out", str(tmp_path)]) == 0
    for event in _events(tmp_path):
        has_analyst = event["organization"] in ANALYST_ORGANIZATIONS
        briefed = event["event"] >= 2 and has_analyst
        assert set(event["analyst_briefings"]) == (set(SUPPLIER_IDS) if briefed else set()), (
            event["organization"], event["event"])
        assert set(event["analyst_postures"]) == set(event["analyst_briefings"])


def test_the_command_accepts_organized_guided_and_records_it(tmp_path):
    assert main(MOCK + ["--institutions", "mediator", "--organizations", "organized_guided",
                        "--out", str(tmp_path)]) == 0
    assert _run(tmp_path)["organizations"] == ["organized_guided"]
    assert {r["organization"] for r in _rows(tmp_path)} == {"organized_guided"} and len(_rows(tmp_path)) == 2
    assert {e["organization"] for e in _events(tmp_path)} == {"organized_guided"}


def test_the_default_organizations_are_all_three_and_the_help_names_the_new_one(capsys):
    assert demo.build_parser().get_default("organizations") == ["single", "organized", "organized_guided"]
    with pytest.raises(SystemExit):
        main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert ("single (the bidder reads its own record), organized (an analyst reads it and briefs the bidder) or "
            "organized_guided (as organized, and the bidder also reads the analysis guidance)") in text


@pytest.mark.parametrize("extra, message", [
    (["--organizations", "organized_guided", "organized_guided"], "--organizations lists a value more than once"),
    (["--organizations", "guided"], "argument --organizations: invalid choice: 'guided'"),
])
def test_organizations_are_checked(extra, message, tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        main(MOCK + extra + ["--out", str(tmp_path)])
    assert info.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path / "run.json").exists()


def test_summary_counts_code_shaped_bids(tmp_path, monkeypatch):
    class Tagging(MockStrategicSupplier):  # A's round-1 bid priced by rule; every bid of B made without the tool
        def generate_bid(self, supplier_costs, regime_context, round_num, history):
            bid = super().generate_bid(supplier_costs, regime_context, round_num, history)
            if supplier_costs.supplier_id == "A" and round_num == 1:
                bid.reasoning += RULE_PRICE_TAG
            if supplier_costs.supplier_id == "B":
                bid.reasoning += NO_TOOL_TAG
            return bid

    _patch_agents(monkeypatch, suppliers=lambda: {"A": Tagging(seed=1), "B": Tagging(seed=2)})
    assert main(["--institutions", "mediator", "--events", "2", "--rounds", "3", "--out", str(tmp_path)]) == 0
    rows = _rows(tmp_path)
    # Per event: 1 rule-priced bid (A, round 1) and 3 bids made without the tool (B, rounds 1-3).
    assert len(rows) == 2 * 3                                               # events x organizations
    assert {(r["n_rule_priced_bids"], r["n_tool_fallback_bids"]) for r in rows} == {("1", "3")}


class QuotesTheRival(MockPlatformMediator):
    """Tells supplier A, after round 1 of each event, the price that B bid in that round."""

    def generate_feedback(self, buyer_prefs, all_bids, target_supplier_id, round_num, **kw):
        if target_supplier_id == "A" and round_num == 1:
            return f"The other bid is ${all_bids['B'].price:.2f}."
        return super().generate_feedback(buyer_prefs, all_bids, target_supplier_id, round_num, **kw)


def test_summary_counts_the_replies_that_quote_the_other_suppliers_price(tmp_path, monkeypatch):
    _patch_agents(monkeypatch, mediator=QuotesTheRival())
    assert main(["--institutions", "mediator", "--events", "2", "--rounds", "3", "--out", str(tmp_path)]) == 0
    rows = _rows(tmp_path)
    assert len(rows) == 2 * 3 and {r["n_rival_price_quotes"] for r in rows} == {"1"}   # one quoting reply per event


def _outcome(prices, replies):
    """What _rival_price_quotes reads of an outcome: each supplier's prices by round, and the replies."""
    return SimpleNamespace(
        all_bids={sid: [SupplierBid(sid, ContractTerms(price, 3.0, 3), round_num=k)
                        for k, price in enumerate(by_round, 1)] for sid, by_round in prices.items()},
        mediator_feedback=replies)


@pytest.mark.parametrize("reply, count", [
    ("Your rival is at $123.45.", 1),            # the number with two decimals, with or without "$"
    ("Your rival is at 123.45", 1),
    ("The rival bid $124 in round 1.", 1),       # a whole number of dollars: "$" and the integer
    ("The rival bid 124.00.", 1),
    ("The rival bid $124, then $123.45.", 1),    # a reply counts once however many prices it holds
    ("Your own bid of $100.00 is fine.", 0),     # a price of the supplier the reply is sent to
    ("A bid of $1123.45 is high.", 0),           # a longer number that holds the same digits is another price
    ("A bid of 123.456 is odd.", 0),
    ("A bid of $1245 is high.", 0),
    ("A bid of $124.50 is high.", 0),
    ("A bid of $123 is high.", 0),               # 123.45 is not a whole number of dollars
    ("124 is a price.", 0),                      # a whole number of dollars is quoted with "$" or two decimals
    ("The field is close.", 0),
])
def test_a_reply_counts_once_when_it_quotes_a_price_the_other_supplier_bid(reply, count):
    outcome = _outcome({"A": [100.0], "B": [123.45, 124.0]}, {"A": [reply], "B": []})
    assert demo._rival_price_quotes(outcome) == count


def test_replies_to_either_supplier_are_counted_against_the_other_suppliers_prices():
    outcome = _outcome({"A": [150.0, 140.25], "B": [160.0, 155.5]},
                       {"A": ["B is at $155.50", "you are at $150.00"],
                        "B": ["A is at $140.25", "A was at $150", "nothing"]})
    assert demo._rival_price_quotes(outcome) == 3        # not "you are at $150.00": that is A's own price
    assert demo._rival_price_quotes(_outcome({"A": [150.0], "B": [160.0]}, {})) == 0


@pytest.mark.parametrize("organization", ["single", "organized", "organized_guided"])
def test_openai_backend_gives_each_agent_its_prompt(organization, fake_make_client):
    args = argparse.Namespace(backend="openai", base_url="http://x/v1", api_key=None, model="m",
                              hosted=True, no_tools=True, default_sampling=True, max_tokens=100, max_retries=30,
                              timeout=300)
    suppliers, mediator, buyer_agent, analysts = demo.build_agents(organization, args, seed=1)
    assert fake_make_client == [("http://x/v1", None, False)]
    # The client counts its requests by the system prompt: exactly the texts the agents send.
    labels = {P.executive_prompt("single"): "bidder_single", P.executive_prompt("organized"): "bidder_organized",
              P.executive_prompt("organized_guided"): "bidder_organized_guided", P.sealed_bid_prompt(): "bidder_sealed",
              P.ANALYST_PROMPT: "analyst", P.MEDIATOR_PROMPT: "mediator", P.BUYER_AGENT_PROMPT: "buyer_agent"}
    assert fake_make_client.options == [{"default_sampling": True, "max_tokens": 100, "max_retries": 30,
                                         "timeout": 300, "agent_labels": labels}]
    assert set(suppliers) == {"A", "B"}
    assert {s.system_prompt for s in suppliers.values()} == {P.executive_prompt(organization)}
    assert all(s.model == "m" and s.use_tools is False for s in suppliers.values())
    assert (mediator.system_prompt, buyer_agent.system_prompt) == (P.MEDIATOR_PROMPT, P.BUYER_AGENT_PROMPT)
    if organization in ANALYST_ORGANIZATIONS:
        assert set(analysts) == {"A", "B"}
        assert {a.system_prompt for a in analysts.values()} == {P.ANALYST_PROMPT}
    else:
        assert analysts is None


# ---- events.jsonl ----

def test_events_hold_the_outcome_fields_unrounded(tmp_path):
    assert main(MOCK + ["--out", str(tmp_path)]) == 0
    for e in _events(tmp_path):
        o = e["outcome"]
        assert set(e) == EVENT_KEYS and set(o) == OUTCOME_KEYS
        assert o["regime"] == demo.INSTITUTIONS[e["institution"]]
        assert (o["winner_id"], o["awarded"], o["correct_winner"]) == (e["winner"], e["awarded"], e["correct_winner"])
        assert o["efficiency_ratio"] == e["efficiency"] and o["buyer_surplus"] == e["buyer_surplus"]  # not rounded
        assert set(o["all_true_costs"]) == set(o["all_inflation_rates"]) == set(SUPPLIER_IDS)
        if e["awarded"]:
            final = e["bids"][o["winner_id"]][-1]
            assert o["winning_bid"] == {k: final[k] for k in ("price", "delivery_weeks", "quality_tier")}
        else:
            assert o["winner_id"] is None and o["winning_bid"] is None


class RepliesByRound(MockPlatformMediator):
    def generate_feedback(self, buyer_prefs, all_bids, target_supplier_id, round_num, **kw):
        return f"reply to {target_supplier_id} round {round_num}"


class BuyerRepliesByRound(MockBuyerAgentPlatform):
    def generate_counteroffer_feedback(self, buyer_prefs, supplier_bid, supplier_id, round_num, history,
                                       buyer_description=""):
        return f"reply to {supplier_id} round {round_num}"


class Accented(MockStrategicSupplier):
    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        bid = super().generate_bid(supplier_costs, regime_context, round_num, history)
        bid.reasoning = "café — très bien"
        return bid


@pytest.mark.parametrize("institution, platform", [
    ("mediator", {"mediator": RepliesByRound()}),
    ("buyer_agent", {"buyer_agent": BuyerRepliesByRound()}),
])
def test_events_keep_each_reply_beside_the_bid_it_answers(institution, platform, tmp_path, monkeypatch):
    _patch_agents(monkeypatch, suppliers=lambda: {sid: Accented(seed=i) for i, sid in enumerate(SUPPLIER_IDS)},
                  **platform)
    assert main(["--institutions", institution, "--organizations", "single", "--events", "1", "--rounds", "3",
                 "--out", str(tmp_path)]) == 0
    (event,) = _events(tmp_path)
    assert "feedback" not in event                                  # the replies are inside the bids
    for sid in SUPPLIER_IDS:
        bids = event["bids"][sid]
        assert [b["round"] for b in bids] == [1, 2, 3]
        assert [b["feedback"] for b in bids] == [f"reply to {sid} round 1", f"reply to {sid} round 2", None]
        assert all(b["reasoning"] == "café — très bien" for b in bids)
    raw = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert "café —" in raw and "\\u00e9" not in raw        # UTF-8 text, not escapes


def test_a_value_json_cannot_hold_raises_instead_of_being_turned_into_text():
    with pytest.raises(ValueError):
        demo._dumps({"x": float("nan")})
    with pytest.raises(TypeError):
        demo._dumps({"x": object()})


# ---- run.json ----

def test_complete_run_json_records_the_settings_and_the_software(tmp_path):
    argv = MOCK + ["--seeds", "3", "--out", str(tmp_path)]
    assert main(argv) == 0
    run = _run(tmp_path)
    assert set(run) == RUN_KEYS | {"note"}
    assert run["status"] == "complete" and run["error"] is None
    assert run["argv"] == argv
    assert (run["backend"], run["events"], run["rounds"], run["seeds"]) == ("mock", 2, 2, [3])
    assert run["max_history_events"] == 0                       # the default keeps every past event
    assert (run["institutions"], run["organizations"]) == (["mediator", "buyer_agent"],
                                                           ["single", "organized", "organized_guided"])
    # The options that apply only to the openai backend are not applicable to a mock run, and a mock
    # run makes no requests to count.
    openai_only = ("model", "base_url", "hosted", "default_sampling", "max_tokens", "max_retries", "timeout",
                   "no_tools", "usage")
    assert [run[k] for k in openai_only] == [None] * len(openai_only) and run["debug_log"] is None
    assert run["note"] == ("Mock agents ignore what they are told, so the single, organized and organized_guided "
                           "rows are identical by construction; use --backend openai with a model to compare the "
                           "organizations.")
    assert (run["python"], run["numpy"], run["openai"]) == (platform.python_version(), numpy.__version__,
                                                           OPENAI_VERSION)
    started, finished = (datetime.fromisoformat(run[k]) for k in ("started_at", "finished_at"))
    assert started.utcoffset() == finished.utcoffset() == timedelta(0) and started <= finished


def test_mock_run_prints_the_note_once_a_header_per_cell_and_a_closing_line(tmp_path, capsys):
    assert main(MOCK + ["--seeds", "3", "4", "--out", str(tmp_path)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == demo.MOCK_NOTE and lines.count(demo.MOCK_NOTE) == 1
    headers = [line for line in lines if line.startswith("=== ")]
    assert headers == [f"=== seed {seed} | {inst} | {org} ===" for seed in (3, 4)
                       for inst in ("mediator", "buyer_agent") for org in ("single", "organized", "organized_guided")]
    assert str(tmp_path) in lines[-1] and all(name in lines[-1] for name in demo.OUTPUT_FILES)


@pytest.mark.parametrize("key_args", [["--api-key", KEY], [f"--api-key={KEY}"]], ids=["separate", "joined"])
def test_openai_run_json_records_the_settings_and_never_the_key(key_args, tmp_path, fake_make_client, capsys):
    argv = ["--backend", "openai", "--model", "m", "--base-url", "http://x/v1", *key_args, "--hosted", "--no-tools",
            "--institutions", "mediator", "--organizations", "organized", "--events", "2", "--rounds", "2",
            "--out", str(tmp_path)]
    assert main(argv) == 0
    assert fake_make_client == [("http://x/v1", KEY, False)]        # the key reached the client ...
    text = (tmp_path / "run.json").read_text(encoding="utf-8")
    assert KEY not in text                                          # ... and is nowhere in the run record
    run = json.loads(text)
    assert set(run) == RUN_KEYS                                     # no note outside the mock backend
    assert run["status"] == "complete"
    assert (run["backend"], run["model"], run["base_url"], run["hosted"], run["no_tools"]) == (
        "openai", "m", "http://x/v1", True, True)
    assert run["argv"] == [token.replace(KEY, "<redacted>") for token in argv]
    shown = capsys.readouterr()
    assert KEY not in shown.out + shown.err
    for name in ("events.jsonl", "summary.csv"):
        assert KEY not in (tmp_path / name).read_text(encoding="utf-8")
    assert "Sharpen your price." in (tmp_path / "events.jsonl").read_text(encoding="utf-8")   # the model path ran


def test_an_error_message_never_carries_a_key_into_run_json(tmp_path, monkeypatch, fake_make_client):
    env_key = "fake-env-key-for-tests-2"
    monkeypatch.setenv("OPENAI_API_KEY", env_key)
    _patch_agents(monkeypatch, suppliers=_failing(message=f"401 for {KEY} and {env_key}"))
    with pytest.raises(RuntimeError):
        main(["--backend", "openai", "--model", "m", "--api-key", KEY, *ONE_CELL, "--events", "3", "--rounds", "2",
              "--out", str(tmp_path)])
    text = (tmp_path / "run.json").read_text(encoding="utf-8")
    assert KEY not in text and env_key not in text
    assert json.loads(text)["error"] == "RuntimeError: 401 for <redacted> and <redacted>"


def test_a_failed_run_keeps_the_finished_events_and_writes_no_summary(tmp_path, monkeypatch):
    during = []

    def look():  # what is on disk when the first bid of event 2 fails
        during.append((_run(tmp_path), (tmp_path / "events.jsonl").read_text(encoding="utf-8")))

    _patch_agents(monkeypatch, suppliers=_failing(on_failure=look))
    with pytest.raises(RuntimeError, match="model unavailable"):
        main(ONE_CELL + ["--events", "3", "--rounds", "2", "--out", str(tmp_path)])
    run_then, events_then = during[0]
    assert run_then["status"] == "running" and run_then["finished_at"] is None
    assert len(events_then.splitlines()) == 1                       # event 1 was on disk before event 2 failed
    run = _run(tmp_path)
    assert run["status"] == "failed" and run["error"] == "RuntimeError: model unavailable"
    assert run["finished_at"] is not None
    assert [e["event"] for e in _events(tmp_path)] == [1]
    assert not (tmp_path / "summary.csv").exists()


def test_a_new_run_removes_what_an_earlier_run_left_and_nothing_else(tmp_path, monkeypatch):
    assert main(MOCK + ["--out", str(tmp_path)]) == 0
    assert (tmp_path / "summary.csv").exists() and _events(tmp_path)
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")
    (tmp_path / "run.json.tmp").write_text('{"status": "running"}\n', encoding="utf-8")   # a cut-off update
    _patch_agents(monkeypatch, suppliers=_failing(ok_bids=0))       # the new run fails in its first event
    with pytest.raises(RuntimeError):
        main(MOCK + ["--out", str(tmp_path)])
    assert not (tmp_path / "summary.csv").exists()                  # no stale summary
    assert not (tmp_path / "run.json.tmp").exists()                 # no stale update
    assert _events(tmp_path) == []                                  # none of the first run's events
    assert _run(tmp_path)["status"] == "failed"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("locked", ["summary.csv", "events.jsonl", "run.json.tmp"])
@pytest.mark.parametrize("names_the_file", [True, False], ids=["error names the file", "bare error"])
def test_a_file_that_cannot_be_deleted_is_recorded_as_a_failure(locked, names_the_file, tmp_path, monkeypatch):
    assert main(MOCK + ["--out", str(tmp_path)]) == 0                  # an earlier run leaves its files
    (tmp_path / "run.json.tmp").write_text('{"status": "running"}\n', encoding="utf-8")   # and a cut-off update
    real_unlink = Path.unlink

    def unlink(self, missing_ok=False):
        if self.name == locked:                                        # held open by another program
            raise PermissionError(13, "in use", str(self)) if names_the_file else PermissionError("in use")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(PermissionError, match=locked):
        main(MOCK + ["--out", str(tmp_path)])
    run = _run(tmp_path)
    assert run["status"] == "failed" and run["error"].startswith("PermissionError: ") and locked in run["error"]
    assert run["finished_at"] is not None
    if locked != "run.json.tmp":                                       # (the write of the end reuses that name)
        assert (tmp_path / locked).exists()                            # still the earlier run's file


def test_a_stale_temporary_file_is_deleted_at_the_start_of_the_next_run(tmp_path, monkeypatch):
    (tmp_path / "run.json.tmp").write_text("not json, half of an update", encoding="utf-8")
    seen = []
    _after_each_event(monkeypatch, lambda event: seen.append((tmp_path / "run.json.tmp").exists()))
    assert main(MOCK + ["--out", str(tmp_path)]) == 0
    assert seen == [False] * 12                     # gone before the first event, not just replaced by the end
    assert sorted(p.name for p in tmp_path.iterdir()) == ["events.jsonl", "run.json", "summary.csv"]


def test_run_json_says_running_before_the_earlier_files_are_deleted(tmp_path, monkeypatch):
    assert main(MOCK + ["--out", str(tmp_path)]) == 0
    seen = []
    real_unlink = Path.unlink

    def unlink(self, missing_ok=False):
        if self.name == "events.jsonl":
            seen.append(_run(tmp_path)["status"])
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    assert main(MOCK + ["--out", str(tmp_path)]) == 0
    assert seen == ["running"]                                         # the earlier run's "complete" was already replaced


# ---- which code made the run ----

def test_run_json_names_the_code_that_made_the_run(tmp_path):
    assert main(MOCK + ["--out", str(tmp_path)]) == 0
    run = _run(tmp_path)
    assert run["source_sha256"] == _expected_source_sha256(ROOT)
    if (ROOT / ".git").exists() and shutil.which("git"):
        assert re.fullmatch(r"[0-9a-f]{40}", run["git_commit"]) and isinstance(run["git_dirty"], bool)
    else:
        assert (run["git_commit"], run["git_dirty"]) == ("unknown", None)


def test_source_sha256_follows_every_byte_of_the_source(tmp_path):
    code = _copy_code(tmp_path / "copy")
    whole = demo._source_sha256(code)
    assert whole == _expected_source_sha256(code) == demo._source_sha256(ROOT)    # the same in any folder
    for rel in ("orgsim/mocks.py", "orgsim/prompts.py", "scripts/run_demo.py"):
        path = code / rel
        original = path.read_bytes()
        path.write_bytes(original[:100] + bytes([original[100] ^ 1]) + original[101:])    # one byte changed
        assert demo._source_sha256(code) != whole, rel
        path.write_bytes(original)
        assert demo._source_sha256(code) == whole
    (code / "orgsim" / "notes.txt").write_text("not source", encoding="utf-8")    # only *.py files count
    assert demo._source_sha256(code) == whole
    (code / "orgsim" / "extra.py").write_text("x = 1\n", encoding="utf-8")        # a new module counts
    assert demo._source_sha256(code) != whole
    (code / "orgsim" / "extra.py").unlink()
    (code / "orgsim" / "mocks.py").rename(code / "orgsim" / "mocks_renamed.py")   # the path is part of the hash
    assert demo._source_sha256(code) != whole


def test_gitattributes_gives_the_python_files_lf_endings_and_leaves_the_example_files_alone():
    # By lines: a checkout with core.autocrlf writes this file, which is not a .py file, with CRLF.
    # *.py: the same source_sha256 in every checkout. examples: no conversion of the files of any run kept there,
    # whose bytes are what the run wrote (core.autocrlf would turn LF files into CRLF).
    assert (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines() == [
        "*.py text eol=lf", "examples/** -text"]


def test_an_exported_copy_records_no_commit_and_its_own_source_hash(tmp_path):
    code = _copy_code(tmp_path / "export")
    run = _run_copy(code, tmp_path / "out")
    assert (run["git_commit"], run["git_dirty"]) == ("unknown", None)              # no fallback to a 40-hex value
    assert run["source_sha256"] == _expected_source_sha256(code) == _expected_source_sha256(ROOT)
    with open(code / "orgsim" / "mocks.py", "a", encoding="utf-8") as f:
        f.write("# one more line\n")
    changed = _run_copy(code, tmp_path / "out2")
    assert changed["source_sha256"] == _expected_source_sha256(code) != run["source_sha256"]


@needs_git
def test_a_temporary_repository_with_an_uncommitted_change_is_dirty(tmp_path):
    repo = _copy_code(tmp_path / "repo")
    head = _make_repo(repo)
    assert demo._git_state(repo) == (head, False)
    (repo / "notes.txt").write_text("untracked", encoding="utf-8")                # untracked files do not count
    assert demo._git_state(repo) == (head, False)
    with open(repo / "orgsim" / "mocks.py", "a", encoding="utf-8") as f:
        f.write("# uncommitted change\n")
    assert demo._git_state(repo) == (head, True)
    run = _run_copy(repo, repo / "out")                                           # its output is untracked too
    assert (run["git_commit"], run["git_dirty"]) == (head, True)
    assert run["source_sha256"] == _expected_source_sha256(repo)


@needs_git
def test_a_copy_inside_another_repository_or_before_its_first_commit_has_no_commit(tmp_path):
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / "readme.txt").write_text("another project", encoding="utf-8")
    _make_repo(outer)
    inside = _copy_code(outer / "export")                    # the commit of `outer` says nothing about this code
    assert demo._git_state(inside) == ("unknown", None)
    unborn = _copy_code(tmp_path / "unborn")
    _git(unborn, "init", "-q")                               # a repository with no commit yet
    assert demo._git_state(unborn) == ("unknown", None)


def test_without_git_the_commit_is_unknown(monkeypatch):
    def no_git(root, *args):
        raise FileNotFoundError("git")

    monkeypatch.setattr(demo, "_git", no_git)
    assert demo._git_state(ROOT) == ("unknown", None)


# ---- credentials in --base-url ----

def _openai_argv(out, *url_args, key_args=()):
    return ["--backend", "openai", "--model", "m", *url_args, *key_args, "--no-tools", *ONE_CELL,
            "--events", "1", "--rounds", "2", "--out", str(out)]


@pytest.mark.parametrize("spelling", ["separate", "joined"])
def test_credentials_in_the_base_url_never_reach_run_json(spelling, tmp_path, fake_make_client):
    url_args = ["--base-url", URL] if spelling == "separate" else [f"--base-url={URL}"]
    assert main(_openai_argv(tmp_path, *url_args)) == 0
    assert fake_make_client == [(URL, None, True)]                                # the client still gets the whole URL
    text = (tmp_path / "run.json").read_text(encoding="utf-8")
    assert "SECRET1" not in text and "SECRET2" not in text
    run = json.loads(text)
    assert run["base_url"] == "http://host/v1"
    if spelling == "separate":
        assert run["argv"][run["argv"].index("--base-url") + 1] == "http://host/v1"
    else:
        assert "--base-url=http://host/v1" in run["argv"]


@pytest.mark.parametrize("url, secrets", [
    ("http://user:SECRET1@host/v1?api_key=SECRET2-LONG-VALUE", ["SECRET1", "SECRET2-LONG-VALUE"]),
    ("http://TOKEN123@host/v1", ["TOKEN123"]),                                    # a token in place of the user name
    ("http://user:p%40ss-word@host/v1", ["p%40ss-word", "p@ss-word"]),            # as written and as decoded
], ids=["password and long query value", "token as user name", "encoded password"])
def test_credentials_repeated_in_an_error_message_are_removed(url, secrets, tmp_path, monkeypatch, fake_make_client):
    _patch_agents(monkeypatch, suppliers=_failing(ok_bids=0, message=f"401 from {url}, key {' and '.join(secrets)}"))
    with pytest.raises(RuntimeError):
        main(_openai_argv(tmp_path, "--base-url", url))
    text = (tmp_path / "run.json").read_text(encoding="utf-8")
    assert not any(secret in text for secret in secrets)
    assert json.loads(text)["error"].count("<redacted>") >= len(secrets)


def test_a_base_url_that_is_not_an_http_url_is_not_recorded(tmp_path, fake_make_client):
    assert main(_openai_argv(tmp_path, "--base-url", "user:SECRET3@host/v1")) == 0
    text = (tmp_path / "run.json").read_text(encoding="utf-8")
    assert "SECRET3" not in text and json.loads(text)["base_url"] == "<not an http(s) URL>"


def test_a_dummy_key_does_not_damage_the_recorded_settings(tmp_path, fake_make_client):
    argv = _openai_argv(tmp_path, "--base-url", "http://localhost:1234/v1", key_args=["--api-key", "local"])
    assert main(argv) == 0
    run = _run(tmp_path)
    assert run["base_url"] == "http://localhost:1234/v1" and "http://localhost:1234/v1" in run["argv"]
    assert run["argv"][run["argv"].index("--api-key") + 1] == "<redacted>"


@pytest.mark.parametrize("join", list(JOINS.values()), ids=list(JOINS))
def test_a_credential_joined_to_other_characters_in_an_error_is_still_removed(join, tmp_path, monkeypatch,
                                                                              fake_make_client):
    monkeypatch.setenv("OPENAI_API_KEY", CREDENTIALS["$OPENAI_API_KEY"])
    message = "401 " + " ".join(join.format(secret) for secret in CREDENTIALS.values())
    _patch_agents(monkeypatch, suppliers=_failing(ok_bids=0, message=message))
    with pytest.raises(RuntimeError):
        main(_openai_argv(tmp_path, "--base-url", LONG_URL, key_args=["--api-key", KEY]))
    text = (tmp_path / "run.json").read_text(encoding="utf-8")
    for source, secret in CREDENTIALS.items():
        assert secret not in text, f"the {source} is left in run.json"
    error = json.loads(text)["error"]
    assert error.startswith("RuntimeError: 401 ") and error.count("<redacted>") == len(CREDENTIALS)


def test_a_key_echoed_by_the_model_server_is_removed_from_the_error_of_a_real_agent_run(tmp_path, monkeypatch):
    def refuse(**kw):
        raise RuntimeError(f"401 Unauthorized: unknown key api_key%3D{KEY}")

    endpoint = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=refuse)))
    monkeypatch.setattr(orgsim.llm, "make_client", lambda base_url, api_key=None, local=True, **options: ChatClient(
        endpoint, local=local))
    with pytest.raises(RuntimeError, match="no usable bid"):                  # the agent's own wrapper of the server error
        main(_openai_argv(tmp_path, key_args=["--api-key", KEY]))
    text = (tmp_path / "run.json").read_text(encoding="utf-8")
    assert KEY not in text
    assert json.loads(text)["error"].endswith("last problem: RuntimeError: 401 Unauthorized: unknown key "
                                              "api_key%3D<redacted>")


def test_the_user_name_of_a_user_and_password_url_is_removed_when_an_error_repeats_it(tmp_path, monkeypatch,
                                                                                      fake_make_client):
    message = "401 from http://user:SECRET1@host/v1 (login: user)"
    _patch_agents(monkeypatch, suppliers=_failing(ok_bids=0, message=message))
    with pytest.raises(RuntimeError):
        main(_openai_argv(tmp_path, "--base-url", URL))
    assert _run(tmp_path)["error"] == "RuntimeError: 401 from http://<redacted>:<redacted>@host/v1 (login: <redacted>)"


def test_a_short_query_value_does_not_redact_unrelated_numbers_in_the_recorded_argv(tmp_path, fake_make_client):
    argv = ["--backend", "openai", "--model", "m", "--base-url", "http://host/v1?v=2", "--no-tools", *ONE_CELL,
            "--events", "2", "--rounds", "2", "--out", str(tmp_path)]
    assert main(argv) == 0
    run = _run(tmp_path)
    assert run["base_url"] == "http://host/v1"                               # the query string is never recorded
    assert run["argv"] == [("http://host/v1" if token.endswith("?v=2") else token) for token in argv]
    assert run["argv"][run["argv"].index("--events") + 1] == run["argv"][run["argv"].index("--rounds") + 1] == "2"


def test_a_secret_of_eight_characters_is_removed_wherever_it_appears():
    secret = "abcdefgh"                                                       # exactly 8 characters
    for text in ("api_key%3Dabcdefgh", "Bearer%20abcdefgh", "‘abcdefgh’", "x-abcdefgh", "abcdefghij"):
        assert demo._scrub(text, [secret]) == text.replace(secret, "<redacted>"), text


def test_a_shorter_secret_is_removed_only_as_a_whole_token():
    text = "x-abcdefg localhost abcdefg_y abcdefg."
    assert demo._scrub(text, ["abcdefg"]) == "x-abcdefg localhost abcdefg_y <redacted>."      # 7 characters
    assert demo._scrub("http://localhost:1234 local", ["local"]) == "http://localhost:1234 <redacted>"


def test_a_base64_form_of_the_url_credentials_is_not_matched():
    encoded = base64.b64encode(b"fake-user-name-3:fake-password-4").decode("ascii")
    header = f"Authorization: Basic {encoded}"
    assert demo._scrub(header, demo._url_secrets(LONG_URL)) == header           # a limit that the epilog states


def test_url_secrets_are_the_user_name_the_password_and_the_long_query_values():
    secrets = demo._url_secrets("http://some-user:some-password@host/v1?v=2&api_key=12345678&n=1234567&lang=en")
    assert set(secrets) == {"some-user", "some-password", "12345678"}         # 8 characters are in, 7 are out
    assert set(demo._url_secrets("http://TOKEN12345@host/v1")) == {"TOKEN12345"}      # a token as the user name
    assert set(demo._url_secrets("http://some-user:p%40ss-word@host")) == {"some-user", "p%40ss-word", "p@ss-word"}
    assert demo._url_secrets("http://host/v1?v=2") == [] and demo._url_secrets("http://host/v1") == []


@needs_git
def test_git_status_takes_no_optional_locks(tmp_path, monkeypatch):
    repo = _copy_code(tmp_path / "repo")
    _make_repo(repo)
    calls = []
    real_git = demo._git

    def recording(root, *args):
        calls.append(args)
        return real_git(root, *args)

    monkeypatch.setattr(demo, "_git", recording)
    assert demo._git_state(repo)[1] is False
    assert ("--no-optional-locks", "status", "--porcelain", "--untracked-files=no") in calls


# ---- a hosted Claude model: sampling settings, token cap, retries, timeout and usage ----

CLAUDE_FLAGS = ["--hosted", "--default-sampling", "--no-tools", "--max-tokens", "100", "--max-retries", "30",
                "--timeout", "300"]
FOUR_CELLS = ["--institutions", "mediator", "buyer_agent", "--organizations", "single", "organized"]


def _usage(requests, prompt, completion, largest, truncated=0, failed=0, no_usage=0, reasons=None):
    """The usage of one agent type, or the total, as run.json records it. Unless `reasons` says otherwise,
    every reply that arrived was cut at the cap (finish reason "length") or finished normally ("stop")."""
    if reasons is None:
        reasons = {"length": truncated, "stop": requests - failed - truncated}
    return {"requests": requests, "failed_requests": failed, "prompt_tokens": prompt,
            "completion_tokens": completion, "largest_completion_tokens": largest, "truncated": truncated,
            "no_usage": no_usage, "finish_reasons": {k: n for k, n in reasons.items() if n}}


def _total_usage(total, by_agent, http_responses=None, http_attempts=0):
    """The usage in run.json: the total, the HTTP attempts and responses (the fake endpoint has none) and the usage
    by agent type."""
    return {**total, "http_attempts": http_attempts, "http_responses": http_responses or {}, "by_agent": by_agent}


def test_usage_in_run_json_sums_every_cells_client_and_every_agent(tmp_path, monkeypatch):
    made = _use_endpoint(monkeypatch, finish_reasons={"mediator": "length"})      # its replies are cut at the cap
    argv = ["--backend", "openai", "--model", "m", "--no-tools", *FOUR_CELLS, "--events", "2", "--rounds", "2",
            "--out", str(tmp_path)]
    assert main(argv) == 0
    assert len(made) == 4                                                          # one client for each cell
    # Per cell, 2 events of 2 rounds: 8 bids; the mediator answers 4 times, the buyer agent 4 times and,
    # before event 2, each of the two analysts once.
    assert _run(tmp_path)["usage"] == _total_usage(_usage(52, 7200, 1240, 40, truncated=8), {
        "analyst": _usage(4, 1200, 160, 40),
        "bidder_organized": _usage(16, 1600, 320, 20),
        "bidder_single": _usage(16, 1600, 320, 20),
        "buyer_agent": _usage(8, 1200, 200, 25),
        "mediator": _usage(8, 1600, 240, 30, truncated=8),
    })                                                                             # nothing under "other"


def test_usage_labels_cover_all_three_organizations_with_nothing_under_other(tmp_path, monkeypatch):
    calls = []
    made = _use_endpoint(monkeypatch, calls=calls)
    argv = ["--backend", "openai", "--model", "m", "--no-tools", "--institutions", "mediator", "--events", "2",
            "--rounds", "2", "--out", str(tmp_path)]                         # all three organizations by default
    assert main(argv) == 0
    assert len(made) == 3                                                    # one client for each cell
    run = _run(tmp_path)
    assert run["organizations"] == ["single", "organized", "organized_guided"]
    by_agent = run["usage"]["by_agent"]
    assert set(by_agent) == {"bidder_single", "bidder_organized", "bidder_organized_guided", "analyst", "mediator"}
    # Per cell, 2 events of 2 rounds: 8 bids and 4 mediator replies; before event 2, each of the two analysts of an
    # organized firm once (4 in all).
    assert [by_agent[label]["requests"] for label in ("bidder_single", "bidder_organized", "bidder_organized_guided")
            ] == [8, 8, 8]
    assert (by_agent["analyst"]["requests"], by_agent["mediator"]["requests"]) == (4, 12)
    assert run["usage"]["requests"] == len(calls) == 3 * 8 + 4 + 12
    sent = [call["messages"][0]["content"] for call in calls]               # and each label is the prompt it counts
    assert [sent.count(P.executive_prompt(org)) for org in P.ORGANIZATIONS] == [8, 8, 8]
    assert sent.count(P.ANALYST_PROMPT) == 4 and sent.count(P.MEDIATOR_PROMPT) == 12
    guided = [call for call in calls if call["messages"][0]["content"] == P.executive_prompt("organized_guided")]
    assert all(P.GUIDED_NOTE in call["messages"][0]["content"] for call in guided)
    assert not any("YOUR PAST PROCUREMENT HISTORY" in call["messages"][1]["content"] for call in guided)  # no record
    assert sum(MEMO_HEADER in call["messages"][1]["content"] for call in guided) == 4   # event 2: 2 suppliers, 2 rounds


def test_run_json_shows_the_replies_without_usage_and_the_finish_reasons(tmp_path, monkeypatch):
    # The analysts' replies report no usage and the buyer agent's have no finish reason; the mediator's are cut.
    _use_endpoint(monkeypatch, no_usage=("analyst",), finish_reasons={"mediator": "length", "buyer_agent": None})
    argv = ["--backend", "openai", "--model", "m", "--no-tools", *FOUR_CELLS, "--events", "2", "--rounds", "2",
            "--out", str(tmp_path)]
    assert main(argv) == 0
    assert _run(tmp_path)["usage"] == _total_usage(
        _usage(52, 6000, 1080, 30, truncated=8, no_usage=4, reasons={"length": 8, "none": 8, "stop": 36}), {
            "analyst": _usage(4, 0, 0, 0, no_usage=4),                           # answered, but with no token counts
            "bidder_organized": _usage(16, 1600, 320, 20),
            "bidder_single": _usage(16, 1600, 320, 20),
            "buyer_agent": _usage(8, 1200, 200, 25, reasons={"none": 8}),
            "mediator": _usage(8, 1600, 240, 30, truncated=8),
        })


def test_usage_is_recorded_for_a_failed_run_too(tmp_path, monkeypatch):
    _use_endpoint(monkeypatch, fail_bidders_from=5)       # the bidders' 5th request, the first of event 2, and all after it
    with pytest.raises(RuntimeError, match="no usable bid"):
        main(["--backend", "openai", "--model", "m", "--no-tools", *ONE_CELL, "--events", "2", "--rounds", "2",
              "--out", str(tmp_path)])
    run = _run(tmp_path)
    assert run["status"] == "failed"
    assert run["usage"] == _total_usage(_usage(9, 800, 140, 30, failed=3), {
        "bidder_single": _usage(7, 400, 80, 20, failed=3),                         # 4 answered, then 3 attempts failed
        "mediator": _usage(2, 400, 60, 30),
    })


def _interrupt_at(number):
    """An on_request for _fake_endpoint: Ctrl-C while the request with this number is in flight."""
    def on_request(n):
        if n == number:
            raise KeyboardInterrupt
    return on_request


def test_an_interrupted_run_is_marked_failed_and_its_usage_counts_the_interrupted_call(tmp_path, monkeypatch):
    # The cell's requests: event 1 takes 6 (the bidders' 4 and the mediator's 2); the 8th is the second bidder's
    # first bid of event 2, which the interrupt cuts off.
    _use_endpoint(monkeypatch, on_request=_interrupt_at(8))
    with pytest.raises(KeyboardInterrupt):
        main(["--backend", "openai", "--model", "m", "--no-tools", *ONE_CELL, "--events", "2", "--rounds", "2",
              "--out", str(tmp_path)])
    run = _run(tmp_path)
    assert run["status"] == "failed" and run["error"] == "KeyboardInterrupt: " and run["finished_at"] is not None
    assert run["usage"] == _total_usage(_usage(8, 900, 160, 30, failed=1), {
        "bidder_single": _usage(6, 500, 100, 20, failed=1),                        # the interrupted call failed
        "mediator": _usage(2, 400, 60, 30),
    })
    assert [e["event"] for e in _events(tmp_path)] == [1] and not (tmp_path / "summary.csv").exists()


def test_usage_is_zero_not_null_when_an_openai_run_fails_before_any_request(tmp_path, fake_make_client):
    with pytest.raises(FileNotFoundError):
        main(_openai_argv(tmp_path / "out", "--debug-log", str(tmp_path / "no-such-folder" / "calls.log")))
    run = _run(tmp_path / "out")
    assert run["status"] == "failed" and run["usage"] == _total_usage(_usage(0, 0, 0, 0), {})


# ---- run.json is rewritten after each event, so a run that is killed keeps its usage ----

def _after_each_event(monkeypatch, look):
    """Wrap the on_event callback that the demo gives run_sequence: after the demo's own callback has
    saved the event, call look(event), which reads what is on disk at that moment."""
    real = demo.run_sequence

    def run_sequence(*args, on_event, **kw):
        def wrapped(event):
            on_event(event)
            look(event)
        return real(*args, on_event=wrapped, **kw)

    monkeypatch.setattr(demo, "run_sequence", run_sequence)


def test_after_each_event_run_json_says_running_with_the_usage_so_far(tmp_path, monkeypatch):
    made = _use_endpoint(monkeypatch, finish_reasons={"mediator": "length"})
    seen = []

    def look(event):
        run = _run(tmp_path)
        seen.append((run["status"], run["finished_at"], run["error"], run["usage"],
                     sum_usage(client.usage_summary() for client in made.clients),     # what the clients hold now
                     len((tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines())))

    _after_each_event(monkeypatch, look)
    assert main(["--backend", "openai", "--model", "m", "--no-tools", *FOUR_CELLS, "--events", "2", "--rounds", "2",
                 "--out", str(tmp_path)]) == 0
    assert len(seen) == 8 and [saved for *_, saved in seen] == list(range(1, 9))      # 4 cells of 2 events
    for status, finished_at, error, usage, held, saved in seen:
        assert (status, finished_at, error) == ("running", None, None)
        assert usage == held                                       # all the clients so far, this cell's too
    requests = [usage["requests"] for _, _, _, usage, _, _ in seen]
    assert requests == sorted(set(requests)) and requests[0] > 0                     # more after every event
    assert seen[0][3]["by_agent"]["mediator"]["truncated"] > 0 and "http_responses" in seen[0][3]
    run = _run(tmp_path)                                                   # the final write is the complete one
    assert run["status"] == "complete" and run["finished_at"] is not None
    assert run["usage"] == seen[-1][3] and run["usage"]["requests"] == 52


def test_a_failed_mid_run_write_prints_a_warning_and_the_run_completes(tmp_path, monkeypatch, capsys):
    _use_endpoint(monkeypatch)
    replaced = []

    def locked(source, destination, *args, **kwargs):             # another program holds run.json open
        replaced.append(destination)
        raise PermissionError(13, f"The process cannot access the file (held by {KEY})", str(destination))

    monkeypatch.setattr(os, "replace", locked)
    assert main(["--backend", "openai", "--model", "m", "--api-key", KEY, "--no-tools", *ONE_CELL, "--events", "3",
                 "--rounds", "2", "--out", str(tmp_path)]) == 0
    assert len(replaced) == 4                                     # one try after each event, and one for the end
    shown = capsys.readouterr().out
    warnings = [line for line in shown.splitlines() if line.startswith("warning")]
    assert len(warnings) == 3 and all("run.json" in line and "the run continues" in line for line in warnings)
    assert KEY not in shown and all("<redacted>" in line for line in warnings)         # the line is scrubbed too
    run = _run(tmp_path)
    assert (run["status"], run["error"]) == ("complete", None) and run["finished_at"] is not None
    assert run["usage"]["requests"] == 18 and run["usage"]["by_agent"]["mediator"]["requests"] == 6   # complete
    assert [e["event"] for e in _events(tmp_path)] == [1, 2, 3] and len(_rows(tmp_path)) == 3
    assert sorted(path.name for path in tmp_path.iterdir()) == ["events.jsonl", "run.json", "summary.csv"]   # no .tmp


def test_a_mock_run_makes_no_mid_run_writes(tmp_path, monkeypatch):
    replaced, at_start, after_events, replaced_by_then = [], [], [], []
    real_replace, real_run_sequence = os.replace, demo.run_sequence
    monkeypatch.setattr(os, "replace", lambda *args, **kwargs: replaced.append(args) or real_replace(*args, **kwargs))

    def run_sequence(*args, on_event, **kw):
        at_start.append((tmp_path / "run.json").read_bytes())          # as main wrote it before the first event

        def wrapped(event):
            on_event(event)
            after_events.append((tmp_path / "run.json").read_bytes())
            replaced_by_then.append(len(replaced))
        return real_run_sequence(*args, on_event=wrapped, **kw)

    monkeypatch.setattr(demo, "run_sequence", run_sequence)
    assert main(MOCK + ["--out", str(tmp_path)]) == 0
    assert len(after_events) == 12 and set(after_events) == set(at_start)   # run.json is untouched by the events
    assert replaced_by_then == [0] * 12                                # nothing was moved into place mid-run
    assert len(replaced) == 1 and _run(tmp_path)["status"] == "complete"    # only the record of the end
    assert json.loads(at_start[0].decode("utf-8"))["usage"] is None and _run(tmp_path)["usage"] is None


def test_write_json_atomic_replaces_the_file_in_one_step_and_scrubs_it(tmp_path):
    path = tmp_path / "run.json"
    path.write_text('{"old": true}\n', encoding="utf-8")
    demo._write_json_atomic(path, {"error": f"401 for {KEY}", "usage": {"requests": 2}}, [KEY])
    assert json.loads(path.read_text(encoding="utf-8")) == {"error": "401 for <redacted>", "usage": {"requests": 2}}
    assert path.read_bytes() == b'{\n  "error": "401 for <redacted>",\n  "usage": {\n    "requests": 2\n  }\n}\n'
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]               # the temporary file is gone
    demo._write_json_atomic(tmp_path / "new.json", {"a": 1})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["new.json", "run.json"]   # also when the file is new


def test_write_json_atomic_moves_a_complete_temporary_file_from_the_same_folder_into_place(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    path.write_text('{"old": true}\n', encoding="utf-8")
    real_replace = os.replace
    moves = []

    def replace(source, destination, *args, **kwargs):
        moves.append((Path(source), Path(destination), json.loads(Path(source).read_text(encoding="utf-8")),
                      path.read_text(encoding="utf-8")))     # the new content is complete; the old file is as it was
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    demo._write_json_atomic(path, {"new": True})
    ((source, destination, new_content, old_content),) = moves
    assert source.parent == tmp_path and source != path and destination == path
    assert new_content == {"new": True} and old_content == '{"old": true}\n'
    assert _run(tmp_path) == {"new": True}


@pytest.mark.parametrize("fails", ["writing", "replacing"])
def test_write_json_atomic_leaves_the_old_file_and_no_temporary_file_when_it_fails(fails, tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    path.write_text('{"old": true}\n', encoding="utf-8")
    if fails == "replacing":
        def replace(source, destination, *args, **kwargs):
            raise PermissionError(13, "in use", str(destination))

        monkeypatch.setattr(os, "replace", replace)
        data, error = {"new": True}, PermissionError
    else:
        data, error = {"new": float("nan")}, ValueError            # JSON cannot hold it: it fails as it is written
    with pytest.raises(error):
        demo._write_json_atomic(path, data)
    assert path.read_text(encoding="utf-8") == '{"old": true}\n'                    # not half of a new file
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]


def test_write_json_safely_writes_atomically_and_falls_back_to_a_direct_write_on_an_oserror(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    path.write_text('{"old": true}\n', encoding="utf-8")
    written, real_write = [], demo._write_json
    monkeypatch.setattr(demo, "_write_json", lambda where, data, secrets=(): (
        written.append(Path(where).name) or real_write(where, data, secrets)))
    demo._write_json_safely(path, {"new": 1}, [KEY])
    assert written == ["run.json.tmp"] and _run(tmp_path) == {"new": 1}       # the atomic write went through

    def refuse(source, destination, *args, **kwargs):
        raise PermissionError(13, "Access is denied", str(destination))       # another handle is open on run.json

    monkeypatch.setattr(os, "replace", refuse)
    written.clear()
    demo._write_json_safely(path, {"new": 2}, [KEY])
    assert written == ["run.json.tmp", "run.json"] and _run(tmp_path) == {"new": 2}   # then the direct write
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]


def test_write_json_safely_does_not_write_directly_after_an_error_that_is_not_an_oserror(tmp_path):
    path = tmp_path / "run.json"
    path.write_text('{"old": true}\n', encoding="utf-8")
    with pytest.raises(ValueError):                                # JSON cannot hold NaN
        demo._write_json_safely(path, {"new": float("nan")})
    assert path.read_text(encoding="utf-8") == '{"old": true}\n'    # a direct write would have cut the file in half
    with pytest.raises(UnicodeEncodeError):                        # a lone surrogate cannot be written as UTF-8
        demo._write_json_safely(path, {"new": "\ud800"})
    assert path.read_text(encoding="utf-8") == '{"old": true}\n'


def test_write_json_safely_raises_the_error_of_the_direct_write_when_that_fails_too(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    path.write_text('{"old": true}\n', encoding="utf-8")

    def refuse(*args, **kwargs):
        raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(os, "replace", refuse)
    monkeypatch.setattr(demo, "_write_json", refuse)
    with pytest.raises(PermissionError):
        demo._write_json_safely(path, {"new": 1})


@pytest.mark.parametrize("outcome", ["complete", "failed"])
def test_a_lock_on_run_json_at_the_end_does_not_lose_the_final_record(outcome, tmp_path, monkeypatch):
    # os.replace fails with PermissionError whenever another program has a handle open on run.json.
    if outcome == "failed":
        _patch_agents(monkeypatch, suppliers=_failing(ok_bids=2, message="model unavailable"))
    _use_endpoint(monkeypatch)
    real_replace, refused = os.replace, []

    def replace(source, destination, *args, **kwargs):
        if json.loads(Path(source).read_text(encoding="utf-8"))["status"] != "running":     # the record of the end
            refused.append(destination)
            raise PermissionError(13, "Access is denied", str(destination))
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr(os, "replace", replace)
    argv = ["--backend", "openai", "--model", "m", "--no-tools", *ONE_CELL, "--events", "3", "--rounds", "2",
            "--out", str(tmp_path)]
    if outcome == "complete":
        assert main(argv) == 0
    else:
        with pytest.raises(RuntimeError, match="model unavailable"):
            main(argv)
    assert len(refused) == 1
    run = _run(tmp_path)                                           # the end of the record, whole
    assert run["status"] == outcome and run["finished_at"] is not None
    assert run["error"] == (None if outcome == "complete" else "RuntimeError: model unavailable")
    assert run["usage"]["requests"] == (18 if outcome == "complete" else 2)        # (the failed run: the mediator's 2)
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.startswith("run")) == ["run.json"]   # no run.json.tmp


def test_a_held_temporary_file_does_not_lose_the_final_record(tmp_path, monkeypatch, capsys):
    _use_endpoint(monkeypatch)
    real_write = demo._write_json

    def write(where, data, secrets=()):
        if Path(where).name == "run.json.tmp":                     # another program holds run.json.tmp open
            raise PermissionError(13, "Permission denied", str(where))
        return real_write(where, data, secrets)

    monkeypatch.setattr(demo, "_write_json", write)
    assert main(["--backend", "openai", "--model", "m", "--no-tools", *ONE_CELL, "--events", "2", "--rounds", "2",
                 "--out", str(tmp_path)]) == 0
    assert capsys.readouterr().out.count("warning: could not update") == 2   # the updates after the two events
    run = _run(tmp_path)
    assert run["status"] == "complete" and run["finished_at"] is not None and run["usage"]["requests"] == 12


def test_an_interruption_during_the_final_write_leaves_the_last_complete_run_json(tmp_path, monkeypatch):
    _use_endpoint(monkeypatch)
    real_write = demo._write_json

    def write(where, data, secrets=()):
        if Path(where).name == "run.json.tmp" and data["status"] != "running":      # Ctrl-C while the end is written
            real_write(where, {"cut": "off"}, secrets)
            raise KeyboardInterrupt
        return real_write(where, data, secrets)

    monkeypatch.setattr(demo, "_write_json", write)
    with pytest.raises(KeyboardInterrupt):
        main(["--backend", "openai", "--model", "m", "--no-tools", *ONE_CELL, "--events", "2", "--rounds", "2",
              "--out", str(tmp_path)])
    run = _run(tmp_path)                                    # whole: the update after event 2, not half of the end
    assert (run["status"], run["finished_at"]) == ("running", None) and run["usage"]["requests"] == 12
    assert sorted(p.name for p in tmp_path.iterdir()) == ["events.jsonl", "run.json", "summary.csv"]   # no run.json.tmp


DRIVER = """\
import os, sys
sys.path.insert(0, {root!r})
import orgsim.llm
from orgsim.llm import ChatClient
from scripts.run_demo import main
from tests.test_demo import _fake_endpoint

replaced = []                                                     # os._exit ends the process: no except, no finally


def make_client(base_url, api_key=None, local=True, **options):
    options.pop("max_retries", None)
    options.pop("timeout", None)
    kill = lambda n: os._exit(3) if n == {kill_at} else None
    return ChatClient(_fake_endpoint(on_request=kill), local=local, **options)


def replace(source, destination):
    replaced.append(source)
    if len(replaced) == {kill_replace_at}:
        os._exit(3)
    return real_replace(source, destination)


real_replace, os.replace = os.replace, replace
orgsim.llm.make_client = make_client
main(sys.argv[1:])
"""


def _killed_run(tmp_path, kill_at=None, kill_replace_at=None):
    """Run 4 events of the mediator and the single firm in a process that ends itself (exit code 3) when its
    `kill_at`-th request starts, or when it is about to move the `kill_replace_at`-th update of run.json into place."""
    driver = tmp_path / "driver.py"
    driver.write_text(DRIVER.format(root=str(ROOT), kill_at=kill_at, kill_replace_at=kill_replace_at), encoding="utf-8")
    out = tmp_path / "out"
    done = subprocess.run([sys.executable, "-B", str(driver), "--backend", "openai", "--model", "m", "--no-tools",
                           *ONE_CELL, "--events", "4", "--rounds", "2", "--out", str(out)],
                          capture_output=True, cwd=tmp_path, timeout=120, env={**os.environ, "PYTHONUTF8": "1"})
    assert done.returncode == 3, done.stderr.decode("utf-8", "replace")[-500:]
    return out


def test_a_run_that_is_killed_in_a_request_keeps_the_usage_up_to_its_last_finished_event(tmp_path):
    out = _killed_run(tmp_path, kill_at=20)
    run = _run(out)
    assert (run["status"], run["finished_at"], run["error"]) == ("running", None, None)   # nothing wrote the end
    # Events 1 to 3 took 18 requests; the 4th event's first 2 requests were made but its event had not ended.
    assert run["usage"] == _total_usage(_usage(18, 2400, 420, 30), {
        "bidder_single": _usage(12, 1200, 240, 20),
        "mediator": _usage(6, 1200, 180, 30),
    })
    assert [e["event"] for e in _events(out)] == [1, 2, 3] and not (out / "summary.csv").exists()
    assert sorted(p.name for p in out.iterdir()) == ["events.jsonl", "run.json"]


def test_a_run_that_is_killed_while_run_json_is_updated_keeps_the_complete_file_of_the_update_before(tmp_path):
    out = _killed_run(tmp_path, kill_replace_at=3)                 # event 3 is saved, its update is not moved into place
    run = _run(out)                                                 # a whole file, not half of one
    assert (run["status"], run["finished_at"], run["error"]) == ("running", None, None)
    assert run["usage"] == _total_usage(_usage(12, 1600, 280, 30), {      # the usage after event 2: the update before
        "bidder_single": _usage(8, 800, 160, 20),
        "mediator": _usage(4, 800, 120, 30),
    })
    assert [e["event"] for e in _events(out)] == [1, 2, 3]
    pending = json.loads((out / "run.json.tmp").read_text(encoding="utf-8"))   # the update that was not moved into place
    assert pending["usage"]["requests"] == 18
    assert sorted(p.name for p in out.iterdir()) == ["events.jsonl", "run.json", "run.json.tmp"]


def test_an_event_is_saved_before_run_json_is_updated(monkeypatch):
    order = []
    monkeypatch.setattr(demo, "_save_event", lambda *args: order.append("save"))
    demo._finish_event(None, [], "mediator", "single", 1, lambda: order.append("update"), {"event": 1})
    assert order == ["save", "update"]

    def cannot_save(*args):
        raise ValueError("cannot save")

    monkeypatch.setattr(demo, "_save_event", cannot_save)
    with pytest.raises(ValueError):
        demo._finish_event(None, [], "mediator", "single", 1, lambda: order.append("update again"), {"event": 1})
    assert order == ["save", "update"]                            # no update for an event that was not saved


def test_the_mid_run_write_removes_credentials_like_every_other_write(tmp_path, monkeypatch):
    _use_endpoint(monkeypatch)
    seen = []
    _after_each_event(monkeypatch, lambda event: seen.append((tmp_path / "run.json").read_text(encoding="utf-8")))
    assert main(["--backend", "openai", "--model", f"model-{KEY}", "--api-key", KEY, "--no-tools", *ONE_CELL,
                 "--events", "2", "--rounds", "2", "--out", str(tmp_path)]) == 0
    assert len(seen) == 2 and all(KEY not in text and '"model": "model-<redacted>"' in text for text in seen)


@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to open a connection raises: a test that builds the real OpenAI client must not reach a server."""
    def refuse(*args, **kwargs):
        raise OSError("a test tried to use the network")

    monkeypatch.setattr("socket.socket.connect", refuse)
    monkeypatch.setattr("socket.create_connection", refuse)
    monkeypatch.setattr("socket.getaddrinfo", refuse)


def _http_endpoint(monkeypatch, errors=None, **endpoint_options):
    """Make the real OpenAI SDK talk to a fake endpoint with no network: its HTTP client gets an httpx.MockTransport
    that answers as _fake_endpoint(**endpoint_options) does, after the error that `errors` gives for the n-th HTTP
    request (retries included, counted from 1): an HTTP status, "timeout" (the request is lost) or "connect" (it
    cannot connect). The SDK's pauses between retries are skipped. Returns the list of the HTTP requests made."""
    endpoint = _fake_endpoint(**endpoint_options)
    requests = []

    def handler(request):
        requests.append(request)
        error = (errors or {}).get(len(requests))
        if error == "timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        if error == "connect":
            raise httpx.ConnectError("refused", request=request)
        if error:
            return httpx.Response(error, json={"type": "error", "error": {"type": "overloaded_error",
                                                                          "message": "busy"}})
        body = json.loads(request.content)
        reply = endpoint.chat.completions.create(**body)
        choice, usage = reply.choices[0], reply.usage
        payload = {"id": f"chatcmpl-{len(requests)}", "object": "chat.completion", "created": 0, "model": body["model"],
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": choice.message.content},
                                "finish_reason": choice.finish_reason}]}
        if usage is not None:
            payload["usage"] = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                                "total_tokens": usage.prompt_tokens + usage.completion_tokens}
        # ASCII JSON, as a server sends it: a lone surrogate travels as an escape
        content = json.dumps(payload, ensure_ascii=True).encode("ascii")
        return httpx.Response(200, headers={"content-type": "application/json"}, content=content)

    class MockedHttpx(openai.DefaultHttpxClient):
        def __init__(self, **kwargs):
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("openai.DefaultHttpxClient", MockedHttpx)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    return requests


def test_the_hosted_claude_command_through_the_real_sdk_counts_every_http_attempt_and_response(tmp_path, monkeypatch,
                                                                                               no_network):
    # The whole path with the real OpenAI SDK, make_client and run_demo, and a fake endpoint behind httpx's
    # MockTransport: the options reach the SDK, the SDK retries inside one call, and run.json shows both.
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    requests = _http_endpoint(monkeypatch, errors={3: 529, 4: 529, 9: 500, 11: "timeout", 12: "connect"},
                              finish_reasons={"mediator": "length"}, no_usage=("bidder",))
    assert main(["--backend", "openai", "--model", "claude-haiku-5-5", "--base-url", "https://api.anthropic.com/v1/",
                 *CLAUDE_FLAGS, *ONE_CELL, "--events", "2", "--rounds", "2", "--out", str(tmp_path)]) == 0
    assert len(requests) == 17                                                    # 12 calls and 5 retried attempts
    assert {str(request.url) for request in requests} == {"https://api.anthropic.com/v1/chat/completions"}
    assert all(request.headers["authorization"] == f"Bearer {KEY}" for request in requests)
    assert all(request.extensions["timeout"] == {"connect": 5.0, "read": 300, "write": 300, "pool": 300}
               for request in requests)                    # --timeout 300 reached every attempt; connecting keeps 5 s
    assert all(set(json.loads(request.content)) == {"model", "messages", "max_tokens"} for request in requests)
    run = _run(tmp_path)
    assert run["status"] == "complete" and (run["max_retries"], run["timeout"]) == (30, 300)
    assert KEY not in (tmp_path / "run.json").read_text(encoding="utf-8")
    # 8 bidder calls and 4 mediator calls; the mediator's replies are cut at the cap and the bidders' report no usage.
    assert run["usage"] == _total_usage(_usage(12, 800, 120, 30, truncated=4, no_usage=8), {
        "bidder_single": _usage(8, 0, 0, 0, no_usage=8),
        "mediator": _usage(4, 800, 120, 30, truncated=4),
    }, http_attempts=17, http_responses={"200": 12, "500": 1, "529": 2})          # 2 attempts got no response


def test_a_call_whose_retries_run_out_is_one_failed_request_and_every_attempt_is_counted(tmp_path, monkeypatch,
                                                                                         no_network):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    requests = _http_endpoint(monkeypatch, errors={n: 529 for n in range(3, 6)})   # three attempts of the 3rd call fail
    with pytest.raises(openai.InternalServerError):                                # (the SDK's error for HTTP 529)
        main(["--backend", "openai", "--model", "m", "--base-url", "https://api.anthropic.com/v1/", "--hosted",
              "--no-tools", "--max-retries", "2", "--timeout", "30", *ONE_CELL, "--events", "1", "--rounds", "2",
              "--out", str(tmp_path)])
    assert len(requests) == 5                                          # 2 answered, then 1 try and 2 retries
    run = _run(tmp_path)
    assert run["status"] == "failed" and run["error"].startswith("InternalServerError: ")
    assert run["usage"]["http_attempts"] == 5 and run["usage"]["http_responses"] == {"200": 2, "529": 3}
    assert (run["usage"]["requests"], run["usage"]["failed_requests"]) == (3, 1)


def test_finish_reasons_that_carry_a_key_or_a_lone_surrogate_count_as_other_through_the_real_sdk(
        tmp_path, monkeypatch, no_network):
    # What the reviewer's probe found: a lone surrogate in a finish reason made the end of run.json invalid UTF-8,
    # and a reason that held the key was written (with the key removed) as a key of finish_reasons.
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    _http_endpoint(monkeypatch, finish_reasons={"mediator": f"stop {KEY}", "buyer_agent": "\ud800"})
    assert main(["--backend", "openai", "--model", "m", "--base-url", "https://api.anthropic.com/v1/", "--hosted",
                 "--no-tools", "--institutions", "mediator", "buyer_agent", "--organizations", "single",
                 "--events", "2", "--rounds", "2", "--out", str(tmp_path)]) == 0
    text = (tmp_path / "run.json").read_text(encoding="utf-8")                # valid UTF-8, and JSON:
    usage = json.loads(text)["usage"]
    assert usage["finish_reasons"] == {"other": 8, "stop": 16}                # 4 mediator and 4 buyer agent replies
    assert usage["by_agent"]["mediator"]["finish_reasons"] == {"other": 4}
    assert usage["by_agent"]["buyer_agent"]["finish_reasons"] == {"other": 4}
    assert KEY not in text and "<redacted>" not in text and usage["truncated"] == 0


def test_finish_reasons_that_carry_a_key_or_a_lone_surrogate_do_not_stop_a_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    _use_endpoint(monkeypatch, finish_reasons={"mediator": f"stop {KEY}", "buyer_agent": "\ud800"})
    assert main(["--backend", "openai", "--model", "m", "--no-tools", "--institutions", "mediator",
                 "buyer_agent", "--organizations", "single", "--events", "2", "--rounds", "2",
                 "--out", str(tmp_path)]) == 0
    text = (tmp_path / "run.json").read_text(encoding="utf-8")
    run = json.loads(text)
    assert run["status"] == "complete" and run["usage"]["finish_reasons"] == {"other": 8, "stop": 16}
    assert KEY not in text and "<redacted>" not in text and "\\ud800" not in text


def test_a_demo_request_with_the_claude_flags_carries_the_token_cap_and_no_sampling_settings(tmp_path, monkeypatch):
    calls = []
    made = _use_endpoint(monkeypatch, calls=calls)
    argv = ["--backend", "openai", "--model", "m", "--base-url", "https://api.anthropic.com/v1/", *CLAUDE_FLAGS,
            "--institutions", "mediator", "buyer_agent", "--organizations", "organized", "--events", "2",
            "--rounds", "2", "--out", str(tmp_path)]
    assert main(argv) == 0
    asked = {call["messages"][0]["content"] for call in calls}
    assert asked == {P.executive_prompt("organized"), P.ANALYST_PROMPT, P.MEDIATOR_PROMPT, P.BUYER_AGENT_PROMPT}
    for call in calls:                                    # no temperature, top_p, presence_penalty, extra_body or tools
        assert set(call) == {"model", "messages", "max_tokens"}, set(call)
        assert call["max_tokens"] == 100
    assert [(o["default_sampling"], o["max_tokens"], o["max_retries"], o["timeout"]) for o in made.options] == [
        (True, 100, 30, 300)] * 2
    run = _run(tmp_path)
    assert (run["hosted"], run["default_sampling"], run["max_tokens"], run["max_retries"], run["timeout"],
            run["no_tools"]) == (True, True, 100, 30, 300, True)
    assert set(run["usage"]["by_agent"]) == {"analyst", "bidder_organized", "mediator", "buyer_agent"}   # no "other"
    assert run["usage"]["requests"] == len(calls)


@pytest.mark.parametrize("no_tools", [True, False], ids=["no-tool bid path", "cost-tool loop"])
def test_without_the_claude_flags_every_agent_request_sets_temperature_and_top_p(no_tools, tmp_path, monkeypatch):
    # The README and the epilog say that every agent request sets both, which Claude 4.7 and later models reject.
    calls = []
    made = _use_endpoint(monkeypatch, calls=calls)
    argv = ["--backend", "openai", "--model", "m", "--hosted", *(["--no-tools"] if no_tools else []),
            "--institutions", "mediator", "buyer_agent", "--organizations", "organized", "--events", "2",
            "--rounds", "2", "--out", str(tmp_path)]
    assert main(argv) == 0
    asked = {call["messages"][0]["content"] for call in calls}
    assert asked == {P.executive_prompt("organized"), P.ANALYST_PROMPT, P.MEDIATOR_PROMPT, P.BUYER_AGENT_PROMPT}
    assert all({"temperature", "top_p", "presence_penalty"} <= set(call) for call in calls)    # all four agent types
    assert not any({"max_tokens", "extra_body"} & set(call) for call in calls)                  # hosted: no extra_body
    assert any("tools" in call for call in calls) is (not no_tools)                             # the cost tool
    assert [(o["default_sampling"], o["max_tokens"], o["max_retries"], o["timeout"]) for o in made.options] == [
        (False, None, None, None)] * 2
    run = _run(tmp_path)
    assert (run["default_sampling"], run["max_tokens"], run["max_retries"], run["timeout"]) == (False, None, None, None)


@pytest.mark.parametrize("flag, value, message", [
    ("--max-tokens", "0", "must be at least 1"),
    ("--max-retries", "-1", "must be at least 0"),
    ("--timeout", "0.5", "must be a number from 1 to 86400"),
    ("--timeout", "0", "must be a number from 1 to 86400"),
    ("--timeout", "-30", "must be a number from 1 to 86400"),
    ("--timeout", "nan", "must be a number from 1 to 86400"),
    ("--timeout", "inf", "must be a number from 1 to 86400"),
    ("--timeout", "86400.5", "must be a number from 1 to 86400"),
    ("--timeout", "86401", "must be a number from 1 to 86400"),
    ("--timeout", "1e9", "must be a number from 1 to 86400"),
    ("--timeout", "soon", "invalid number value: 'soon'"),
])
def test_the_token_cap_the_retries_and_the_timeout_are_checked(flag, value, message, tmp_path, capsys,
                                                               fake_make_client):
    with pytest.raises(SystemExit) as info:                   # (the fake client: a check that failed must not reach a server)
        main(["--backend", "openai", "--model", "m", flag, value, "--out", str(tmp_path)])
    assert info.value.code == 2 and f"argument {flag}: {message}" in capsys.readouterr().err
    assert not (tmp_path / "run.json").exists()


def test_zero_retries_is_a_setting_that_reaches_the_client_and_run_json(tmp_path, monkeypatch):
    made = _use_endpoint(monkeypatch)
    assert main(["--backend", "openai", "--model", "m", "--max-retries", "0", "--no-tools", *ONE_CELL, "--events", "1",
                 "--rounds", "2", "--out", str(tmp_path)]) == 0
    assert made.options[0]["max_retries"] == 0 and _run(tmp_path)["max_retries"] == 0      # 0 is not "unset"


@pytest.mark.parametrize("text, seconds", [("300", 300), ("1", 1), ("1.5", 1.5), ("2.0", 2), ("1e3", 1000),
                                           ("86400", 86400), ("86399.5", 86399.5), ("8.64e4", 86400)])
def test_the_timeout_reaches_the_client_and_run_json(text, seconds, tmp_path, monkeypatch):
    made = _use_endpoint(monkeypatch)
    assert main(["--backend", "openai", "--model", "m", "--timeout", text, "--no-tools", *ONE_CELL, "--events", "1",
                 "--rounds", "2", "--out", str(tmp_path)]) == 0
    run = _run(tmp_path)
    assert made.options[0]["timeout"] == run["timeout"] == seconds
    assert run["argv"][run["argv"].index("--timeout") + 1] == text                          # argv keeps what was typed
    assert type(run["timeout"]) is type(seconds)                        # a whole number is not shown as 300.0


def test_the_largest_timeout_is_one_the_operating_system_accepts():
    # socket.settimeout(1e9) raises OverflowError on Windows, which is why --timeout has an upper bound.
    import socket
    assert demo.MAX_TIMEOUT == 86400
    with socket.socket() as sock:
        sock.settimeout(demo.MAX_TIMEOUT)
        assert sock.gettimeout() == demo.MAX_TIMEOUT


def test_the_readme_and_the_epilog_show_the_hosted_claude_command_with_options_the_script_has():
    options = {option for action in demo.build_parser()._actions for option in action.option_strings}
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("### Hosted Claude models", 1)[1].split("\n## ", 1)[0]
    command = {"--backend", "--hosted", "--default-sampling", "--no-tools", "--base-url", "--model", "--max-tokens",
               "--max-retries", "--timeout"}
    for where, text in (("README", section), ("epilog", demo.EPILOG)):
        flags = set(re.findall(r"(?<![\w-])--[a-z][a-z-]*", text))
        assert command <= flags, (where, command - flags)
        assert flags <= options, (where, flags - options)                                # no option the script lacks
        normal = " ".join(text.replace("\\\n", " ").split())
        assert "--max-retries 30 --timeout 300" in normal and "https://api.anthropic.com/v1/" in normal, where
        assert "--max-retries 8" not in normal, where


def test_the_readme_and_the_epilog_name_every_usage_field_run_json_records():
    fields = set(orgsim.llm.USAGE_FIELDS) | {"http_attempts", "http_responses"}
    assert {"largest_completion_tokens", "finish_reasons", "no_usage", "http_attempts", "http_responses"} <= fields
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for where, text in (("README", readme), ("epilog", demo.EPILOG)):
        for field in fields:
            assert re.search(rf"(?<![\w-]){field}(?![\w-])", text), (where, field)
        assert "max_completion_tokens" not in text, where                  # the field's old name, the API's cap


def test_the_readme_says_how_http_attempts_differ_from_responses_what_other_means_and_what_a_kill_leaves():
    text = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    for part in (
            "Each HTTP attempt counts in `http_attempts`; one that receives a response also counts in `http_responses` "
            "by status code", "so attempts minus responses are the attempts that got no response (they timed out, "
            "could not connect, or lost the connection). Such an attempt can still be billed if the server finished "
            "it. A call that the SDK retries counts once in `requests`.",
            "\"none\" if a reply has none, \"other\" if it is not 1 to 40 lowercase letters and underscores",
            "its `usage` is as of the last update that succeeded (an update cut off by the kill can leave a "
            "`run.json.tmp` beside it)",
            "deletes the `events.jsonl`, `summary.csv` and `run.json.tmp` that an earlier run left there",
            "`--timeout SECONDS` (1 to 86400)",
            "connecting keeps the SDK's limit of 5 seconds"):
        assert part in text, part


def test_the_readme_describes_the_three_organizations_and_the_contrasts_between_them():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    text = " ".join(readme.split())
    design = " ".join(readme.split("\n## Design\n", 1)[1].split("\n## Outputs", 1)[0].split())
    assert tuple(re.findall(r"- \*\*[\w ]+ \(`(\w+)`\)\.\*\*", readme)) == P.ORGANIZATIONS    # one bullet for each
    for organization in P.ORGANIZATIONS:
        assert f"`bidder_{organization}`" in text                          # and its label of the usage in run.json
    for part in (
            "The single firm and the guided organized firm differ only in who holds the history and in the briefing "
            "step (and in the notes that describe them); this is the comparison the design intends.",
            "The organized firm and the guided organized firm differ only in whether the bidder reads the analysis "
            "guidance.",
            "At the first event no history exists and no briefing is sent, so a gap between organizations there comes "
            "from the bidder's instructions alone, apart from the model's sampling:",
            "The bidder also reads the analysis guidance, the same text that the single firm's agent and the analysts "
            "read, under a note that says the guidance is written for the analyst and that the bidder is the CFO."):
        assert part in design, part
    for stale in ("identical across the two organizations", "both organizations", "the two organizations' rows",
                  "Only two things differ", "single and organized rows", "identical in both organizations"):
        assert stale not in text, stale                  # what was true of two organizations only


EXAMPLES = ROOT / "examples"


def _check_example(folder):
    """What must hold of a run kept in examples/<name>/: it was made by this code, it is complete, and it has the three
    files that the README describes, as the script wrote them."""
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    # source_sha256 identifies the code that made the run. If this fails, the code changed after the example was
    # made: make the example again, or say in the README what differs.
    assert run["source_sha256"] == _expected_source_sha256(ROOT)
    assert (run["status"], run["error"]) == ("complete", None)
    assert sorted(path.name for path in folder.iterdir()) == ["events.jsonl", "run.json", "summary.csv"]
    cells = len(run["seeds"]) * len(run["institutions"]) * len(run["organizations"])
    assert len(_events(folder)) == len(_rows(folder)) == cells * run["events"]
    out = run["argv"][run["argv"].index("--out") + 1]
    assert not Path(out).is_absolute()                                        # and no path of a machine anywhere in it:
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", " ".join(run["argv"]))
    raw = (folder / "summary.csv").read_bytes()                               # kept as the csv module wrote it
    assert raw.count(b"\n") == raw.count(b"\r\n") == len(_rows(folder)) + 1
    for name in ("run.json", "events.jsonl"):
        assert b"\r" not in (folder / name).read_bytes(), name


def test_every_example_run_was_made_by_this_code_and_has_the_files_the_readme_describes():
    # There is none yet: a run kept in examples/<name>/ is checked here from the day it is added.
    for run_json in sorted(EXAMPLES.glob("*/run.json")):
        try:
            _check_example(run_json.parent)
        except AssertionError as error:
            raise AssertionError(f"examples/{run_json.parent.name}: {error}") from error


def test_the_check_of_an_example_accepts_a_run_made_by_this_code_and_rejects_a_stale_or_altered_one(
        tmp_path, monkeypatch):
    _use_endpoint(monkeypatch)
    monkeypatch.chdir(tmp_path)
    assert main(["--backend", "openai", "--model", "m", "--no-tools", "--institutions", "mediator",
                 "--organizations", "single", "organized_guided", "--events", "2", "--rounds", "2",
                 "--out", "example"]) == 0
    folder = tmp_path / "example"
    _check_example(folder)
    original = {path.name: path.read_bytes() for path in folder.iterdir()}
    run = json.loads(original["run.json"].decode("utf-8"))
    changes = {
        "another code": ("run.json", json.dumps({**run, "source_sha256": "0" * 64}, indent=2)),
        "not complete": ("run.json", json.dumps({**run, "status": "running"}, indent=2)),
        "an absolute path": ("run.json", json.dumps({**run, "argv": [*run["argv"][:-1], str(tmp_path)]}, indent=2)),
        "a missing event": ("events.jsonl", "\n".join(original["events.jsonl"].decode("utf-8").splitlines()[:-1])),
        "LF line ends": ("summary.csv", original["summary.csv"].decode("utf-8").replace("\r\n", "\n")),
    }
    for label, (name, text) in changes.items():
        (folder / name).write_bytes(text.encode("utf-8"))
        with pytest.raises(AssertionError):
            _check_example(folder)
        (folder / name).write_bytes(original[name])
    (folder / "notes.txt").write_text("x", encoding="utf-8")                  # a file that is not one of the three
    with pytest.raises(AssertionError):
        _check_example(folder)


def test_the_readme_mentions_only_example_folders_that_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for folder in set(re.findall(r"examples/[\w.-]+/", readme)):
        assert (ROOT / folder).is_dir(), folder


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_git_applies_no_line_ending_conversion_to_the_files_under_examples():
    names = ["examples/demo_run/run.json", "examples/another_run/summary.csv", "examples/a_run/events.jsonl",
             "examples/a_run/deeper/file.txt"]
    done = subprocess.run(["git", "check-attr", "text", "eol", "--", *names, "README.md", "orgsim/llm.py"], cwd=ROOT,
                          capture_output=True, text=True, check=True, timeout=60)
    found = {}
    for line in done.stdout.splitlines():
        path, attribute, value = line.split(": ", 2)
        found[path, attribute] = value
    for name in names:
        assert (found[name, "text"], found[name, "eol"]) == ("unset", "unspecified"), name
    assert (found["README.md", "text"], found["orgsim/llm.py", "text"], found["orgsim/llm.py", "eol"]) == (
        "unspecified", "set", "lf")                                                  # nothing else changed


SAMPLE_RUN_FILES = {                        # a run's three files: LF, CRLF (the csv module's), and LF with UTF-8 text
    "run.json": b'{\n  "status": "complete"\n}\n',
    "summary.csv": b"institution,organization\r\nmediator,single\r\n",
    "events.jsonl": b'{"reasoning": "caf\xc3\xa9 \xe2\x80\x94 bid"}\n{"x": 1}\n',
}


@needs_git
def test_a_checkout_with_core_autocrlf_gives_back_the_files_under_examples_byte_for_byte(tmp_path):
    source = tmp_path / "source"
    (source / "examples" / "a_run").mkdir(parents=True)
    shutil.copy(ROOT / ".gitattributes", source / ".gitattributes")
    for name, data in SAMPLE_RUN_FILES.items():
        (source / "examples" / "a_run" / name).write_bytes(data)
    _git(source, "init", "-q")
    _git(source, "config", "core.autocrlf", "true")                                 # as on Windows
    _git(source, "add", "-A")
    _git(source, "commit", "-q", "-m", "an example run")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "-c", "core.autocrlf=true", str(source), str(clone))
    for name, data in SAMPLE_RUN_FILES.items():
        assert (clone / "examples" / "a_run" / name).read_bytes() == data, name


# ---- options ----

@pytest.mark.parametrize("extra", [["--model", "m"], ["--api-key", "k"], ["--hosted"], ["--no-tools"],
                                   ["--base-url", "http://x/v1"], ["--default-sampling"], ["--max-tokens", "100"],
                                   ["--max-retries", "8"], ["--timeout", "300"]])
def test_mock_backend_rejects_options_that_apply_only_to_openai(extra, tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        main(MOCK + extra + ["--out", str(tmp_path)])
    assert info.value.code == 2
    assert f"{extra[0]} applies only to --backend openai" in capsys.readouterr().err
    assert not (tmp_path / "run.json").exists()


def test_openai_backend_needs_a_model(tmp_path, capsys, fake_make_client):
    with pytest.raises(SystemExit) as info:                   # (the fake client: a check that failed must not reach a server)
        main(["--backend", "openai", "--out", str(tmp_path)])
    assert info.value.code == 2
    assert "--model is required with --backend openai" in capsys.readouterr().err


@pytest.mark.parametrize("extra, message", [
    (["--seeds", "1", "2", "1"], "--seeds lists a value more than once"),
    (["--institutions", "mediator", "mediator"], "--institutions lists a value more than once"),
    (["--organizations", "single", "single"], "--organizations lists a value more than once"),
    (["--seeds", "429496730"], "argument --seeds: must be between 0 and 429496729"),
    (["--seeds", "-1"], "argument --seeds: must be between 0 and 429496729"),
    (["--seeds", "1.5"], "argument --seeds: invalid integer value: '1.5'"),
])
def test_repeated_values_and_bad_seeds_are_usage_errors(extra, message, tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        main(MOCK + extra + ["--out", str(tmp_path)])
    assert info.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "run.json").exists()


def test_the_lowest_and_highest_seed_run(tmp_path):
    # The mock suppliers seed numpy with seed * 10 + 1 at most, which must stay below 2**32.
    assert main(ONE_CELL + ["--events", "1", "--rounds", "2", "--seeds", "0", str(demo.MAX_SEED),
                            "--out", str(tmp_path)]) == 0
    assert [r["seed"] for r in _rows(tmp_path)] == ["0", str(demo.MAX_SEED)]


def test_max_history_events_reaches_run_sequence(tmp_path, monkeypatch):
    seen = []
    real = demo.run_sequence

    def spy(*args, **kw):
        seen.append(kw["max_history_events"])
        return real(*args, **kw)

    monkeypatch.setattr(demo, "run_sequence", spy)
    assert main(MOCK + ONE_CELL + ["--max-history-events", "3", "--out", str(tmp_path / "limited")]) == 0
    assert main(MOCK + ONE_CELL + ["--out", str(tmp_path / "default")]) == 0
    assert seen == [3, 0]                                        # the default keeps every past event


class KeepsItsRules(MockStrategicSupplier):
    """Keeps the rules-and-record text of each of its calls."""

    def __init__(self, seed, contexts):
        super().__init__(seed=seed)
        self.contexts = contexts

    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        self.contexts.append(regime_context)
        return super().generate_bid(supplier_costs, regime_context, round_num, history)


def test_max_history_events_cuts_the_suppliers_record_in_a_long_run(tmp_path, monkeypatch):
    seen = []
    _patch_agents(monkeypatch, suppliers=lambda: {sid: KeepsItsRules(i, seen) for i, sid in enumerate(SUPPLIER_IDS)})
    last = {}
    for limit in (0, 1):
        seen.clear()
        assert main(ONE_CELL + ["--events", "4", "--rounds", "2", "--max-history-events", str(limit),
                                "--out", str(tmp_path / str(limit))]) == 0
        last[limit] = seen[-1]                                   # the last bid, in event 4: three events are past
    assert "omitted for brevity" not in last[0] and "  Event 1 (" in last[0]
    assert "[2 earlier event(s) omitted for brevity]" in last[1] and "  Event 1 (" not in last[1]
    assert len(last[1]) < len(last[0])


def test_run_json_records_max_history_events(tmp_path):
    assert main(MOCK + ONE_CELL + ["--max-history-events", "2", "--out", str(tmp_path)]) == 0
    run = _run(tmp_path)
    assert run["max_history_events"] == 2 and run["argv"][run["argv"].index("--max-history-events") + 1] == "2"


def test_a_negative_max_history_events_is_a_usage_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as info:
        main(MOCK + ["--max-history-events", "-1", "--out", str(tmp_path)])
    assert info.value.code == 2
    assert "argument --max-history-events: must be at least 0" in capsys.readouterr().err
    assert not (tmp_path / "run.json").exists()


def test_options_cannot_be_abbreviated_so_the_key_is_always_found_by_name(tmp_path, capsys, fake_make_client):
    with pytest.raises(SystemExit) as info:
        main(["--backend", "openai", "--model", "m", "--api", KEY, "--out", str(tmp_path)])
    assert info.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert not (tmp_path / "run.json").exists()


def test_help_names_the_output_files_and_every_option_has_a_help_string(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    for part in ("events.jsonl", "summary.csv", "run.json", "any OpenAI-compatible chat endpoint",
                 "fixes the event plan and the mock agents but not the model's sampling",
                 "Give the API key through the OPENAI_API_KEY environment variable rather than --api-key",
                 "source_sha256 identifies the code without the git history: the SHA-256 (hex digest) over "
                 "orgsim/*.py and scripts/run_demo.py in sorted order of their relative paths (written with /, "
                 "sorted as plain strings), where each file adds its relative path, a NUL byte, its bytes as "
                 "stored and a NUL byte",
                 "git_dirty says whether tracked files differ from that commit",
                 "It leaves out the API key and the user name, password and query string of --base-url, "
                 "within the limits below",
                 "Credentials are removed from every string in run.json, an error message included: the API key "
                 "(--api-key or $OPENAI_API_KEY) and the user name, password and query values of --base-url",
                 f"a credential of {demo.CREDENTIAL_LENGTH} or more characters is removed wherever it appears",
                 "a shorter one is removed only where it stands as a whole word (not next to a letter, digit, _ "
                 "or -), so one joined to other characters in a server's error text can remain",
                 "encoded forms, such as base64 of user:password, are not matched",
                 f"a query value shorter than {demo.CREDENTIAL_LENGTH} characters is not treated as a credential, "
                 "although the recorded base_url and argv never include the query",
                 "n_rival_price_quotes in summary.csv audits the platforms: the number of their replies in the "
                 "event, to either supplier, that contain a price the other supplier bid in this event",
                 "an audit signal, not a verdict, because a coincidence can match",
                 "analyst_postures in events.jsonl is read from the text of each briefing by a keyword rule",
                 "Check it against analyst_briefings before relying on it",
                 "The repository's .gitattributes gives the .py files LF line endings in every checkout",
                 "Console output and the --debug-log file are not scrubbed of keys, unlike run.json",
                 "keep only the last N past events in each supplier's record and in the mediator's memory of "
                 "each supplier, which keeps prompts within a model's context in long runs; 0 keeps every past "
                 "event (default: 0)",
                 "Against Anthropic's OpenAI-compatible endpoint",
                 "--backend openai --hosted --default-sampling --no-tools --base-url https://api.anthropic.com/v1/ "
                 "--model <id> --max-tokens N --max-retries 30 --timeout 300, with the key in OPENAI_API_KEY",
                 "requests (every call made through the client, including calls that raised before or after "
                 "reaching the server), failed_requests (the calls that raised; a failed call can still be billed, "
                 "for example one interrupted after the server finished)",
                 "largest_completion_tokens (the largest completion of any single request), truncated (replies cut "
                 "at the cap), no_usage (replies whose usage lacks a whole-number token count; any count that is "
                 "there is still added) and finish_reasons (replies by finish reason: \"none\" if a reply has none, "
                 "\"other\" if it is not 1 to 40 lowercase letters and underscores)",
                 "In total only, each HTTP attempt counts in http_attempts; one that receives a response also counts "
                 "in http_responses by status code (strings such as \"200\" and \"529\"), so attempts minus "
                 "responses are the attempts that timed out or could not connect. Such an attempt can still be "
                 "billed if the server finished it. A call that the SDK retries counts once in requests",
                 "Whether those completion tokens include the hidden reasoning is not documented",
                 "so check that truncated is 0; a no_usage above 0 means the token counts are low",
                 "A model backend adds its usage, written after each event and when the run ends. A run that is "
                 "killed outright stays running: its usage then leaves out the requests of the event in progress, "
                 "and also those of every finished event whose update was cut off by the kill (which can leave a "
                 "run.json.tmp) or failed with a warning",
                 "deletes the events.jsonl, summary.csv and run.json.tmp that an earlier run left in that folder",
                 "send no sampling settings, so the provider's defaults apply; needed for Claude 5 models",
                 "cap each reply, hidden reasoning included",
                 "retries for connection errors, rate limits and server errors (the OpenAI SDK's own retry with "
                 "backoff); the SDK default is 2",
                 "--timeout SECONDS",
                 "seconds before a request is abandoned and retried (the OpenAI SDK's default is 600); from 1 to "
                 "86400; connecting keeps the SDK's limit of 5 seconds",
                 "(default: 6)", "(default: 3)", "(default: outputs/demo)"):
        assert part in text, part
    assert all(action.help for action in demo.build_parser()._actions)


# ---- --debug-log ----

class LogsItsBids(MockStrategicSupplier):
    def __init__(self, seed, seen):
        super().__init__(seed=seed)
        self.seen = seen

    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        self.seen.append(llm_agents.DEBUG_LOG_FILE)
        llm_agents.debug_log("stub bid", f"round {round_num}")
        return super().generate_bid(supplier_costs, regime_context, round_num, history)


def test_debug_log_receives_the_calls_and_is_released_afterwards(tmp_path, monkeypatch):
    seen = []
    _patch_agents(monkeypatch, suppliers=lambda: {sid: LogsItsBids(i, seen) for i, sid in enumerate(SUPPLIER_IDS)})
    log = tmp_path / "calls.log"
    assert main(MOCK + ONE_CELL + ["--debug-log", str(log), "--out", str(tmp_path / "out")]) == 0
    assert llm_agents.DEBUG_LOG_FILE is None
    assert seen and seen[0] is not None and all(f is seen[0] for f in seen)   # one open file for the whole run
    assert seen[0].closed
    assert "stub bid" in log.read_text(encoding="utf-8")
    assert _run(tmp_path / "out")["debug_log"] == str(log)


def test_debug_log_is_released_when_the_run_fails(tmp_path, monkeypatch):
    seen = []
    _patch_agents(monkeypatch, suppliers=_failing(ok_bids=0, on_failure=lambda: seen.append(llm_agents.DEBUG_LOG_FILE)))
    log = tmp_path / "calls.log"
    with pytest.raises(RuntimeError):
        main(MOCK + ONE_CELL + ["--debug-log", str(log), "--out", str(tmp_path / "out")])
    assert seen[0] is not None and seen[0].closed
    assert llm_agents.DEBUG_LOG_FILE is None
    assert log.exists()


def test_without_debug_log_the_module_variable_is_left_alone(tmp_path, monkeypatch):
    elsewhere = io.StringIO()
    monkeypatch.setattr(llm_agents, "DEBUG_LOG_FILE", elsewhere)
    assert main(MOCK + ["--out", str(tmp_path)]) == 0
    assert llm_agents.DEBUG_LOG_FILE is elsewhere and not elsewhere.closed


def test_debug_log_holds_the_unusable_replies_of_a_failed_no_tool_run(tmp_path, monkeypatch):
    _use_endpoint(monkeypatch, bid="I cannot decide.")                        # a model that never gives a bid
    log = tmp_path / "calls.log"
    with pytest.raises(RuntimeError, match="no usable bid"):
        main(_openai_argv(tmp_path / "out", "--debug-log", str(log)))
    text = log.read_text(encoding="utf-8")
    assert text.count("I cannot decide.") == 3                                # the raw reply of each of the 3 attempts
    assert "[A] No-tool request (attempt 3/3)" in text and "[A] No usable bid in the reply (attempt 3/3)" in text
    assert _run(tmp_path / "out")["status"] == "failed"


def test_debug_log_holds_the_history_and_the_raw_reply_of_each_analyst(tmp_path, fake_make_client):
    log = tmp_path / "calls.log"
    argv = ["--backend", "openai", "--model", "m", "--no-tools", "--institutions", "mediator",
            "--organizations", "organized", "--events", "2", "--rounds", "2", "--debug-log", str(log),
            "--out", str(tmp_path / "out")]
    assert main(argv) == 0
    text = log.read_text(encoding="utf-8")
    rule = "=" * 80                                                          # an entry: header, time, rule, body
    requests = re.findall(rf"\[Analyst→(\w)\] Request \(history sent\)\n[^\n]*\n{rule}\n(.*?)\n\n{rule}", text, re.S)
    replies = re.findall(rf"\[Analyst→(\w)\] Raw reply\n[^\n]*\n{rule}\n([^\n]*)\n", text)
    assert [sid for sid, _ in requests] == list(SUPPLIER_IDS) and all("Event 1 (" in body for _, body in requests)
    assert replies == [(sid, "The record is thin. Posture: GUARDED.") for sid in SUPPLIER_IDS]
    assert "[A] No-tool request" in text                                     # the bids are logged as well


# ---- what the AI institutions send is the same whatever else the command can run ----

# One digest per institution and organization of every request of a two-event run with two rounds: the messages in
# the order sent and the tools offered, for the bidders, the analysts and the platforms. Taken from the code before
# the second-price auction was added to the command. The two organized_guided digests were taken again when
# GUIDED_NOTE gained the sentence on what "you" means in the guidance, and again when that sentence was narrowed to
# the guidance's address to its reader; the other four are the original ones.
EXPECTED_AI_RUN_DIGESTS = {
    "mediator single": "e76114704ec7a9a0",
    "mediator organized": "6e8dedb96ccdea8e",
    "mediator organized_guided": "4f6e0cfea9bff655",
    "buyer_agent single": "08abe8483ba53773",
    "buyer_agent organized": "d6dedc2a56cd4a0f",
    "buyer_agent organized_guided": "9084be8379620ec2",
}


def _ai_run_digest(institution, organization, out, monkeypatch):
    calls = []
    _use_endpoint(monkeypatch, calls=calls)
    assert main(["--backend", "openai", "--model", "m", "--institutions", institution, "--organizations",
                 organization, "--events", "2", "--rounds", "2", "--out", str(out)]) == 0
    text = json.dumps([{"messages": call["messages"], "tools": call.get("tools")} for call in calls],
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def test_the_requests_of_the_ai_institutions_are_unchanged(tmp_path, monkeypatch):
    got = {f"{institution} {organization}": _ai_run_digest(institution, organization,
                                                            tmp_path / f"{institution}_{organization}", monkeypatch)
           for institution in ("mediator", "buyer_agent") for organization in P.ORGANIZATIONS}
    assert got == EXPECTED_AI_RUN_DIGESTS, "what the AI institutions' agents read has changed"


# ---- the second-price auction in the command ----

SECOND_PRICE = ["--institutions", "second_price", "--organizations", "single", "--events", "1"]
SECOND_PRICE_RULES = (
    "RULES (second-price auction): You submit one bid. Both suppliers bid at the same time. The buyer scores each bid "
    "based on all terms (price, delivery, quality). The highest-scoring bid wins.\n\n"
    "PRICING: If you win, you are paid the highest price at which your terms would still score at least as high as "
    "the other supplier's bid. This price is at least the price you bid.\n\n"
    "There are no further rounds.\n\n"
    "This is a one-time procurement event. Your bid does not affect any future interactions.")
SEALED_NO_TOOL_ROUND = "Submit your bid as JSON. Keep reasoning brief (1-2 sentences)."
SEALED_TOOL_ROUND = ("You have a compute_my_cost tool that returns your total cost for given delivery and quality "
                     "terms. You may call it at most twice. After your tool calls, output only the JSON bid object.")
SEALED_TOOL = {"type": "function", "function": {
    "name": "compute_my_cost",
    "description": "Compute your exact total cost for a given set of contract terms. Returns the total cost and a "
                   "cost breakdown.",
    "parameters": {"type": "object", "properties": {
        "delivery_weeks": {"type": "number", "description": "Delivery time in weeks (1.0 to 8.0)"},
        "quality_tier": {"type": "integer", "description": "Quality tier (2, 3, 4, or 5)"}},
        "required": ["delivery_weeks", "quality_tier"]}}}
NUDGE = ("You have used all your cost calculations. Now submit your bid for this round as a JSON object with these "
         "exact fields: price, delivery_weeks, quality_tier, reasoning. Output ONLY the JSON object, nothing else.")
# What none of the second-price bidder's texts may say: a reference to a platform or to a record, and advice. (The
# system prompt ends with the output format, which asks for "brief strategic thinking" in the reasoning field and is
# shared with the AI institutions' bidders; it is left out of the check.)
PLATFORM_REFERENCE = re.compile(
    r"\b(platforms?|facilitator|mediator|buyer agent|feedback|analyst|briefing|memo|CFO|board|history|record|"
    r"PAST EVENTS|previous|earlier|rival\w*)\b", re.I)
ADVICE = re.compile(
    r"\b(should|must|never|always|avoid|ensure|careful\w*|recommend\w*|strateg\w+|margin\w*|competit\w+|trust\w*|"
    r"honest|generous|aggressive|conservative|floor|best|protect\w*|target)\b", re.I)


def _texts_read(call):
    """Every text that one request puts in front of the bidder: its messages, and the tools it is offered."""
    texts = [message["content"] for message in call["messages"] if message.get("content")]
    texts[0] = texts[0].replace(P.BID_OUTPUT, "")
    return texts + [json.dumps(tool) for tool in call.get("tools") or []]


def _usage_error(argv, capsys):
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2
    return capsys.readouterr().err


def test_second_price_is_a_choice_of_the_command_and_not_a_default(capsys):
    parser = demo.build_parser()
    assert parser.get_default("institutions") == ["mediator", "buyer_agent"] == list(demo.AI_INSTITUTIONS)
    assert demo.INSTITUTIONS == {"mediator": "Mediator", "buyer_agent": "BuyerAgent", "second_price": "SecondPrice"}
    with pytest.raises(SystemExit):
        main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert ("second_price (a one-time second-score sealed-bid auction, the structural benchmark; needs --events 1 "
            "and --organizations single, and has no rounds)") in text
    assert "bidding rounds per event of the mediator and the buyer agent, at least 2" in text
    assert "second_price is the second-score sealed-bid auction, one bid for each of the two suppliers" in text


@pytest.mark.parametrize("events", [["--events", "2"], ["--events", "6"], []])      # and the default, 6
def test_second_price_needs_one_event(events, tmp_path, capsys):
    argv = ["--backend", "mock", "--institutions", "second_price", "--organizations", "single", *events,
            "--out", str(tmp_path)]
    err = _usage_error(argv, capsys)
    assert ("--institutions second_price needs --events 1: its rules tell the bidder that this is a one-time event, "
            "which is true only when no record of earlier events exists (for several auctions give several --seeds)"
            ) in " ".join(err.split())
    assert not (tmp_path / "run.json").exists()


@pytest.mark.parametrize("organizations", [[], ["organized"], ["organized_guided"], ["single", "organized"],
                                           ["single", "organized", "organized_guided"]])
def test_second_price_needs_the_single_organization(organizations, tmp_path, capsys):
    argv = ["--backend", "mock", "--institutions", "second_price", "--events", "1", "--out", str(tmp_path)]
    if organizations:
        argv += ["--organizations", *organizations]                      # none given: the default is all three
    err = _usage_error(argv, capsys)
    assert ("--institutions second_price needs --organizations single: with one event there is no record for an "
            "analyst to read, so the organizations would not differ") in " ".join(err.split())
    assert not (tmp_path / "run.json").exists()


def test_second_price_runs_beside_the_ai_institutions_only_where_its_rules_hold(tmp_path, capsys):
    institutions = ["--institutions", "mediator", "buyer_agent", "second_price"]
    assert main(["--backend", "mock", *institutions, "--organizations", "single", "--events", "1", "--rounds", "2",
                 "--out", str(tmp_path)]) == 0
    assert [(r["institution"], r["organization"]) for r in _rows(tmp_path)] == [
        ("mediator", "single"), ("buyer_agent", "single"), ("second_price", "single")]
    assert _run(tmp_path)["rounds"] == 2                                  # the AI institutions use them
    bids = {e["institution"]: [len(b) for b in e["bids"].values()] for e in _events(tmp_path)}
    assert bids == {"mediator": [2, 2], "buyer_agent": [2, 2], "second_price": [1, 1]}
    for rest, message in ((["--organizations", "single", "--events", "2"], "needs --events 1"),
                          (["--organizations", "single", "organized", "--events", "1"], "needs --organizations single"),
                          (["--events", "1"], "needs --organizations single")):
        err = _usage_error(["--backend", "mock", *institutions, *rest, "--rounds", "2", "--out", str(tmp_path / "x")],
                           capsys)
        assert message in err


@pytest.mark.parametrize("rounds", [[], ["--rounds", "1"], ["--rounds", "2"], ["--rounds", "7"]])
def test_rounds_do_not_apply_to_second_price(rounds, tmp_path):
    assert main(["--backend", "mock", *SECOND_PRICE, *rounds, "--out", str(tmp_path)]) == 0
    assert _run(tmp_path)["rounds"] is None                               # no rounds took place
    (event,) = _events(tmp_path)
    assert [len(bids) for bids in event["bids"].values()] == [1, 1]


@pytest.mark.parametrize("institutions", [["mediator"], ["buyer_agent"], ["second_price", "mediator"],
                                          ["buyer_agent", "second_price"]])
def test_the_ai_institutions_still_need_at_least_two_rounds(institutions, tmp_path, capsys):
    err = _usage_error(["--backend", "mock", "--institutions", *institutions, "--organizations", "single",
                        "--events", "1", "--rounds", "1", "--out", str(tmp_path)], capsys)
    assert ("argument --rounds: must be at least 2 for the mediator and the buyer agent, because the platform "
            "answers between rounds") in " ".join(err.split())
    assert not (tmp_path / "run.json").exists()


def test_zero_rounds_is_not_a_number_of_rounds_for_any_institution(tmp_path, capsys):
    err = _usage_error(["--backend", "mock", *SECOND_PRICE, "--rounds", "0", "--out", str(tmp_path)], capsys)
    assert "argument --rounds: must be at least 1" in err


def test_a_mock_second_price_run_makes_one_auction_for_each_seed(tmp_path):
    assert main(["--backend", "mock", *SECOND_PRICE, "--seeds", "0", "1", "2", "--out", str(tmp_path)]) == 0
    with open(tmp_path / "summary.csv", encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == COLUMNS                     # the columns of the AI institutions' rows
    rows, events, run = _rows(tmp_path), _events(tmp_path), _run(tmp_path)
    assert [(r["institution"], r["organization"], r["seed"], r["event"]) for r in rows] == [
        ("second_price", "single", seed, "1") for seed in "012"]
    assert [e["outcome"]["regime"] for e in events] == ["SecondPrice"] * 3
    for event in events:
        assert set(event) == EVENT_KEYS and set(event["outcome"]) == OUTCOME_KEYS
        assert sorted(event["bids"]) == ["A", "B"]
        assert all(len(bids) == 1 and bids[0]["round"] == 1 and bids[0]["feedback"] is None
                   for bids in event["bids"].values())                    # one sealed bid each, and no reply
        assert event["awarded"] is True and event["analyst_briefings"] == event["analyst_postures"] == {}
        assert event["outcome"]["winning_bid"]["price"] >= event["bids"][event["winner"]][0]["price"]
    assert all(r["awarded"] == "True" and r["n_rule_priced_bids"] == r["n_rival_price_quotes"] == "0" for r in rows)
    assert (run["institutions"], run["organizations"], run["events"], run["rounds"]) == (
        ["second_price"], ["single"], 1, None)
    assert (run["status"], run["usage"], run["seeds"]) == ("complete", None, [0, 1, 2])


class FixedBid(MockStrategicSupplier):
    """Bids the same terms whatever it is told."""

    def __init__(self, price, weeks, tier):
        super().__init__(seed=0)
        self.terms = (price, weeks, tier)

    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        return SupplierBid(supplier_costs.supplier_id, ContractTerms(*self.terms), reasoning="fixed",
                           round_num=round_num)


def test_second_price_keeps_the_bids_the_costs_and_the_price_paid_as_worked_out_by_hand(tmp_path, monkeypatch):
    # Seed 0 draws the Quality Priority contract, A a speed specialist (cost 85 + 0.5 * (8 - weeks)^2 + 1.8 * tier^2)
    # and B a quality specialist (85 + 1.8 * (8 - weeks)^2 + 0.4 * tier^2). The buyer's value is
    # 150 + 0.25 * 60 * (8 - weeks) / 8 + 0.75 * 50 * tier / 5 = 150 + 1.875 * (8 - weeks) + 7.5 * tier.
    #   A bids $160, 4 weeks, tier 3: value 150 + 7.5 + 22.5 = 180, score 180 - 160 = 20, cost 85 + 8 + 16.2 = 109.2,
    #     markup (160 - 109.2) / 109.2.
    #   B bids $150, 6 weeks, tier 4: value 150 + 3.75 + 30 = 183.75, score 33.75, cost 85 + 7.2 + 6.4 = 98.6,
    #     markup (150 - 98.6) / 98.6.
    # B wins and is paid the price at which its score equals A's: 183.75 - 20 = 163.75 (not its bid, 150). The buyer's
    # surplus is 20, B's profit 163.75 - 98.6 = 65.15 and the joint surplus 183.75 - 98.6 = 85.15. B's best terms
    # (7.5 weeks, tier 5: value 188.4375, cost 95.45) give the efficient surplus 92.9875.
    terms = {"A": (160.0, 4.0, 3), "B": (150.0, 6.0, 4)}
    monkeypatch.setattr(demo, "build_sealed_bid_agents",
                        lambda args, seed: {sid: FixedBid(*t) for sid, t in terms.items()})
    assert main(["--backend", "mock", *SECOND_PRICE, "--seeds", "0", "--out", str(tmp_path)]) == 0
    (event,), (row,) = _events(tmp_path), _rows(tmp_path)
    assert (event["contract"], event["type_a"], event["type_b"]) == ("Quality Priority", "speed", "quality")
    assert event["bids"] == {                                              # as submitted
        "A": [{"round": 1, "price": 160.0, "delivery_weeks": 4.0, "quality_tier": 3, "reasoning": "fixed",
               "feedback": None}],
        "B": [{"round": 1, "price": 150.0, "delivery_weeks": 6.0, "quality_tier": 4, "reasoning": "fixed",
               "feedback": None}]}
    out = event["outcome"]
    assert (out["regime"], out["winner_id"], out["awarded"]) == ("SecondPrice", "B", True)
    assert out["winning_bid"] == {"price": pytest.approx(163.75), "delivery_weeks": 6.0, "quality_tier": 4}   # paid
    assert out["all_true_costs"] == {"A": pytest.approx(109.2), "B": pytest.approx(98.6)}             # at their terms
    assert out["all_inflation_rates"] == {"A": pytest.approx(50.8 / 109.2),               # at the submitted prices
                                          "B": pytest.approx(51.4 / 98.6)}
    assert out["winner_true_cost"] == pytest.approx(98.6)
    assert (out["buyer_surplus"], out["supplier_profit"], out["joint_surplus"]) == pytest.approx((20.0, 65.15, 85.15))
    assert (out["efficient_winner_id"], out["correct_winner"]) == ("B", True)
    assert out["efficient_surplus"] == pytest.approx(92.9875)
    assert out["efficiency_ratio"] == event["efficiency"] == pytest.approx(85.15 / 92.9875)
    assert event["score_margin_pct"] == pytest.approx(13.75 / 33.75 * 100)                 # (33.75 - 20) / 33.75
    assert (row["institution"], row["organization"], row["seed"], row["contract"]) == (
        "second_price", "single", "0", "Quality Priority")
    assert (row["winner"], row["awarded"], row["correct_winner"], row["efficient_winner"]) == ("B", "True", "True", "B")
    assert [float(row[k]) for k in ("buyer_surplus", "supplier_profit", "joint_surplus", "efficiency")] == (
        pytest.approx([20.0, 65.15, 85.15, 85.15 / 92.9875]))                                       # at the price paid
    assert row["n_rule_priced_bids"] == row["n_tool_fallback_bids"] == row["n_rival_price_quotes"] == "0"


def test_second_price_takes_a_bid_below_cost_as_submitted(tmp_path, monkeypatch):
    # A bids $100 at 4 weeks, tier 3, below its cost 109.2: the bid and its negative markup are kept.
    terms = {"A": (100.0, 4.0, 3), "B": (150.0, 6.0, 4)}
    monkeypatch.setattr(demo, "build_sealed_bid_agents",
                        lambda args, seed: {sid: FixedBid(*t) for sid, t in terms.items()})
    assert main(["--backend", "mock", *SECOND_PRICE, "--seeds", "0", "--out", str(tmp_path)]) == 0
    (event,), (row,) = _events(tmp_path), _rows(tmp_path)
    assert event["bids"]["A"][0]["price"] == 100.0 and "set by rule" not in event["bids"]["A"][0]["reasoning"]
    assert event["outcome"]["all_inflation_rates"]["A"] == pytest.approx((100.0 - 109.2) / 109.2)
    assert event["outcome"]["all_inflation_rates"]["A"] < 0
    assert row["n_rule_priced_bids"] == "0"
    # A scores 180 - 100 = 80 and B 33.75, so A wins and is paid 180 - 33.75 = 146.25, above its cost.
    assert event["winner"] == "A" and event["outcome"]["winning_bid"]["price"] == pytest.approx(146.25)
    assert event["outcome"]["supplier_profit"] == pytest.approx(146.25 - 109.2)


def test_the_second_price_bidders_read_exactly_these_texts_with_no_platform_reference_and_no_advice(
        tmp_path, monkeypatch):
    calls = []
    _use_endpoint(monkeypatch, calls=calls)
    assert main(["--backend", "openai", "--model", "m", "--no-tools", *SECOND_PRICE, "--seeds", "0",
                 "--out", str(tmp_path)]) == 0
    assert len(calls) == 2 and all("tools" not in call for call in calls)             # one request for each bidder
    expected = {"A": make_supplier("A", "speed"), "B": make_supplier("B", "quality")}      # what seed 0 draws
    for call, sid in zip(calls, "AB"):
        assert call["messages"] == [
            {"role": "system", "content": P.sealed_bid_prompt()},
            {"role": "user", "content": (expected[sid].describe_for_sealed_bid() + "\n\nREGIME: " + SECOND_PRICE_RULES
                                         + "\n\n" + SEALED_NO_TOOL_ROUND)}]
        for text in _texts_read(call):
            assert not PLATFORM_REFERENCE.search(text), PLATFORM_REFERENCE.search(text).group(0)
            assert not ADVICE.search(text), ADVICE.search(text).group(0)
    assert calls[0]["messages"][1]["content"] != calls[1]["messages"][1]["content"]       # each reads its own costs
    assert all("MUST" not in call["messages"][1]["content"] for call in calls)


def _tool_calling_endpoint(calls):
    """A fake chat endpoint for the cost-tool path: each bidder asks for two costs (3 weeks/T2, then 5 weeks/T4) and
    then bids. `calls` collects each request as it was sent."""
    bid = json.dumps({"price": 120, "delivery_weeks": 6, "quality_tier": 3, "reasoning": "ok"})

    def create(**kw):
        calls.append({"messages": json.loads(json.dumps(kw["messages"])), "tools": kw.get("tools")})
        asked = sum(message["role"] == "tool" for message in kw["messages"])
        tool_calls = None
        if kw.get("tools") and asked < 2:
            args = ({"delivery_weeks": 3, "quality_tier": 2}, {"delivery_weeks": 5, "quality_tier": 4})[asked]
            tool_calls = [SimpleNamespace(id=f"call{asked}", function=SimpleNamespace(
                name="compute_my_cost", arguments=json.dumps(args)))]
        message = SimpleNamespace(content="" if tool_calls else bid, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_the_second_price_bidders_with_the_cost_tool_read_the_neutral_tool_and_texts(tmp_path, monkeypatch):
    calls = []

    def make_client(base_url, api_key=None, local=True, **options):
        options = {k: v for k, v in options.items() if k not in ("max_retries", "timeout")}
        return ChatClient(_tool_calling_endpoint(calls), local=local, **options)

    monkeypatch.setattr(orgsim.llm, "make_client", make_client)
    assert main(["--backend", "openai", "--model", "m", *SECOND_PRICE, "--seeds", "0", "--out", str(tmp_path)]) == 0
    assert len(calls) == 6                              # three requests for each bidder: two cost calls and a bid
    costs = {"A": ("$12.50", "$7.20", "$104.70", "$4.50", "$28.80", "$118.30"),       # 3wk/T2 and 5wk/T4, by hand
             "B": ("$45.00", "$1.60", "$131.60", "$16.20", "$6.40", "$107.60")}
    for sid, requests in (("A", calls[:3]), ("B", calls[3:])):
        first, second, third = requests
        description = make_supplier(sid, "speed" if sid == "A" else "quality").describe_for_sealed_bid()
        assert first["messages"][1] == {"role": "user", "content": (
            description + "\n\nREGIME: " + SECOND_PRICE_RULES + "\n\n" + SEALED_TOOL_ROUND)}
        assert first["tools"] == second["tools"] == [SEALED_TOOL] and third["tools"] is None
        (d1, q1, t1, d2, q2, t2) = costs[sid]
        assert second["messages"][-1]["content"] == (
            f"Cost breakdown for 3.0wk / T2:\n  Base cost:     $85.00\n  + Delivery:    {d1}\n"
            f"  + Quality:     {q1}\n  = TOTAL COST:  {t1}")
        assert third["messages"][-2]["content"] == (
            f"Cost breakdown for 5.0wk / T4:\n  Base cost:     $85.00\n  + Delivery:    {d2}\n"
            f"  + Quality:     {q2}\n  = TOTAL COST:  {t2}")
        assert third["messages"][-1] == {"role": "user", "content": NUDGE}
        for call in requests:
            for text in _texts_read(call):
                assert not PLATFORM_REFERENCE.search(text), PLATFORM_REFERENCE.search(text).group(0)
                assert not ADVICE.search(text), ADVICE.search(text).group(0)
                assert "Cost +" not in text and "WARNING" not in text and "lose money" not in text


def test_the_second_price_bidders_get_their_own_prompt_and_the_sealed_flag(fake_make_client):
    args = argparse.Namespace(backend="openai", base_url="http://x/v1", api_key=None, model="m", hosted=True,
                              no_tools=False, default_sampling=True, max_tokens=100, max_retries=30, timeout=300)
    suppliers = demo.build_sealed_bid_agents(args, seed=1)
    assert fake_make_client == [("http://x/v1", None, False)]
    labels = fake_make_client.options[0]["agent_labels"]
    assert labels[P.sealed_bid_prompt()] == "bidder_sealed" and len(labels) == 7      # three organizations, four more
    assert set(suppliers) == {"A", "B"}
    assert all(s.system_prompt == P.sealed_bid_prompt() and s.sealed_bid is True and s.model == "m"
               for s in suppliers.values())
    assert all(s.use_tools is True for s in suppliers.values())                       # --no-tools was not given
    args.no_tools = True
    assert all(s.use_tools is False for s in demo.build_sealed_bid_agents(args, seed=1).values())
    assert all(s.sealed_bid is False for s in demo.build_agents("single", args, seed=1)[0].values())   # the AI bidders


def test_second_price_usage_is_counted_under_bidder_sealed_with_nothing_under_other(tmp_path, monkeypatch):
    calls = []
    made = _use_endpoint(monkeypatch, calls=calls)
    assert main(["--backend", "openai", "--model", "m", "--no-tools", *SECOND_PRICE, "--seeds", "1", "2",
                 "--out", str(tmp_path)]) == 0
    assert len(made) == 2                                                      # one client for each seed's auction
    usage = _run(tmp_path)["usage"]
    assert set(usage["by_agent"]) == {"bidder_sealed"}                         # nothing under "other"
    assert usage["by_agent"]["bidder_sealed"] == _usage(4, 400, 80, 20)        # 2 seeds x 2 bidders, one request each
    assert usage["requests"] == len(calls) == 4
    assert {call["messages"][0]["content"] for call in calls} == {P.sealed_bid_prompt()}


def test_usage_labels_of_a_command_that_runs_the_second_price_auction_beside_an_ai_institution(tmp_path, monkeypatch):
    calls = []
    _use_endpoint(monkeypatch, calls=calls)
    assert main(["--backend", "openai", "--model", "m", "--no-tools", "--institutions", "mediator", "second_price",
                 "--organizations", "single", "--events", "1", "--rounds", "2", "--out", str(tmp_path)]) == 0
    by_agent = _run(tmp_path)["usage"]["by_agent"]
    assert set(by_agent) == {"bidder_single", "mediator", "bidder_sealed"}      # nothing under "other"
    # The mediator's event: 2 bidders x 2 rounds = 4 bids and 2 replies; the auction: 2 bids.
    assert [by_agent[label]["requests"] for label in ("bidder_single", "mediator", "bidder_sealed")] == [4, 2, 2]
    sent = [call["messages"][0]["content"] for call in calls]
    assert (sent.count(P.executive_prompt("single")), sent.count(P.sealed_bid_prompt())) == (4, 2)


def test_the_readme_describes_the_second_price_auction_and_its_commands_are_accepted_by_the_script():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    text = " ".join(readme.split())
    design = " ".join(readme.split("\n## Design\n", 1)[1].split("\n## Outputs", 1)[0].split())
    for part in (
            "**Second-price auction and the structural benchmark S (`--institutions second_price`).**",
            "bidding a price equal to one's true cost at the terms one offers is optimal whatever the rival bids",
            "So the markup of a bid over its true cost (`outcome.all_inflation_rates` in `events.jsonl`) measures how "
            "far the agent is from S, for which it is 0.",
            "The bidder reads a prompt that states primitives only (`SEALED_BID_EXECUTIVE` in `orgsim/prompts.py`",
            "Every other text this bidder reads (the cost description, the rules, the round instruction and the cost "
            "tool) states primitives or rules and gives no advice, and a bid below cost is taken as submitted.",
            "The auction is one-shot by design: its rules tell the bidder that this is a one-time event, which is true "
            "only when no record of earlier events exists.",
            "So the command needs `--events 1` and `--organizations single`",
            "and `--rounds` does not apply to it; for several auctions, give several `--seeds`."):
        assert part in design, part
    for part in ("`bidder_sealed` (the second-price auction's bidders)",
                 "The AI institutions' prompts still carry strategic advice from earlier work.",
                 "Only the second-price bidder's prompt states primitives only",
                 "The second-price auction takes every bid as submitted.",
                 "`run.json` records `rounds` as `null` when only `second_price` runs."):
        assert part in text, part
    # Every command in the section on hosted models is one the script accepts.
    section = readme.split("### Hosted Claude models", 1)[1].split("\n## ", 1)[0]
    commands = [shlex.split(block.replace("\\\n", " ")) for block in re.findall(r"```bash\n(.*?)```", section, re.S)]
    assert len(commands) == 3 and commands[1][commands[1].index("--institutions") + 1] == "second_price"
    parser = demo.build_parser()
    for command in commands:
        assert command[:2] == ["python", "scripts/run_demo.py"]
        args = parser.parse_args(["m" if token == "<model-id>" else token for token in command[2:]])
        demo._check_options(parser, args)                                        # exits with an error if it is refused


def test_a_model_bid_below_cost_in_the_second_price_auction_is_kept_and_asked_for_once(tmp_path, monkeypatch):
    calls = []
    below = json.dumps({"price": 90, "delivery_weeks": 6, "quality_tier": 3, "reasoning": "low"})
    _use_endpoint(monkeypatch, calls=calls, bid=below)         # below both costs: 103.2 for A (speed), 95.8 for B
    assert main(["--backend", "openai", "--model", "m", "--no-tools", *SECOND_PRICE, "--seeds", "0",
                 "--out", str(tmp_path)]) == 0
    assert len(calls) == 2 and not any("CORRECTION" in call["messages"][1]["content"] for call in calls)
    (event,), (row,) = _events(tmp_path), _rows(tmp_path)
    assert [event["bids"][sid][0]["price"] for sid in "AB"] == [90.0, 90.0]            # as submitted, not cost + 5%
    assert all(rate < 0 for rate in event["outcome"]["all_inflation_rates"].values())
    assert row["n_rule_priced_bids"] == "0"
    # The AI institutions send the same bid back once and then set its price by the rule.
    other = tmp_path / "mediator"
    calls.clear()
    assert main(["--backend", "openai", "--model", "m", "--no-tools", "--institutions", "mediator",
                 "--organizations", "single", "--events", "1", "--rounds", "2", "--seeds", "0",
                 "--out", str(other)]) == 0
    assert any("CORRECTION" in call["messages"][1]["content"] for call in calls)
    assert _rows(other)[0]["n_rule_priced_bids"] == "4"                              # 2 suppliers x 2 rounds


# ---- the note of a mock run says what is true of the run ----

ONE_ORGANIZATION_MOCK_NOTE = (
    "Mock agents ignore what they are told, so a mock run only checks that the pipeline runs and shows nothing about "
    "how a model bids; use --backend openai with a model for that.")


def _several_organizations_note(names):
    return (f"Mock agents ignore what they are told, so the {names} rows are identical by construction; use "
            "--backend openai with a model to compare the organizations.")


@pytest.mark.parametrize("organizations, names", [
    ([], "single, organized and organized_guided"),                           # the default: all three
    (["single", "organized"], "single and organized"),
    (["organized_guided", "single"], "organized_guided and single"),
])
def test_the_mock_note_names_the_organizations_the_run_has(organizations, names, tmp_path, capsys):
    argv = ["--backend", "mock", "--events", "1", "--rounds", "2", "--out", str(tmp_path)]
    if organizations:
        argv += ["--organizations", *organizations]
    assert main(argv) == 0
    note = _several_organizations_note(names)
    assert _run(tmp_path)["note"] == note and capsys.readouterr().out.splitlines()[0] == note
    assert demo.MOCK_NOTE == _several_organizations_note("single, organized and organized_guided")   # the default's


@pytest.mark.parametrize("argv", [
    SECOND_PRICE,                                                          # the auction always has one organization
    ["--institutions", "mediator", "second_price", "--organizations", "single", "--events", "1", "--rounds", "2"],
    ["--institutions", "mediator", "--organizations", "organized", "--events", "1", "--rounds", "2"],
])
def test_the_note_of_a_mock_run_of_one_organization_says_only_what_a_mock_run_is_for(argv, tmp_path, capsys):
    assert main(["--backend", "mock", *argv, "--out", str(tmp_path)]) == 0
    assert _run(tmp_path)["note"] == ONE_ORGANIZATION_MOCK_NOTE
    assert capsys.readouterr().out.splitlines()[0] == ONE_ORGANIZATION_MOCK_NOTE
    for word in ("organiz", "rows", "single", "identical"):
        assert word not in ONE_ORGANIZATION_MOCK_NOTE            # nothing about organizations that are not there


def test_a_run_with_a_model_records_no_mock_note(tmp_path, fake_make_client):
    assert main(["--backend", "openai", "--model", "m", "--no-tools", *SECOND_PRICE, "--out", str(tmp_path)]) == 0
    assert "note" not in _run(tmp_path)


def test_the_readme_says_what_the_note_of_a_mock_run_is():
    text = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert ("so the rows of different organizations are identical by construction; organizational effects need a "
            "model backend. A mock run of one organization, such as the second-price auction, has no such rows to "
            "compare.") in text
    assert ("backend the rows of the organizations are identical by construction; when a run has more than one "
            "organization (or, with `--mixed`, more than one market) the script says so when it starts, and "
            "otherwise it says only that a mock run checks that the pipeline runs.") in text


# ---- homogeneous markets send what they sent before mixed markets existed ----

# The 160-sequence scale-up was run on commit 9da0998, so every request of a homogeneous run (both suppliers with
# the same organization, under each AI institution, and the second-price auction) must stay byte-identical to what
# that commit sent. The digests below were recorded from 9da0998 before any code was changed. Each is the whole run
# through a fake chat endpoint whose replies vary with the request (bids that are sent back for being below cost,
# bids above the buyer's value, prose where a bid is asked for, cost-tool calls, platform replies, analyst
# postures), so that records, memos, corrections, memories and truncations all occur. A request digest covers every
# request in order, with all its arguments (the messages, the tools offered and the sampling settings); an output
# digest covers events.jsonl and summary.csv without the columns that mixed markets added and without prompt_set and
# environment, which the primitives set added.
VARIED_BIDS = (
    {"price": 120, "delivery_weeks": 6, "quality_tier": 3},
    {"price": 150.5, "delivery_weeks": 4.5, "quality_tier": 4},
    {"price": 95, "delivery_weeks": 6, "quality_tier": 3},                 # below the cost of both types: sent back
    {"price": 210, "delivery_weeks": 3, "quality_tier": 3},                # above the buyer's value for the terms
    {"price": 131.25, "delivery_weeks": 7.5, "quality_tier": 2},
    {"price": 99.99, "delivery_weeks": 2, "quality_tier": 5},              # below the cost of both types
    {"price": 175, "delivery_weeks": 5, "quality_tier": 4},
    {"price": 210, "delivery_weeks": 4, "quality_tier": 2},                # above the buyer's value again
)
VARIED_ANALYST_REPLIES = (
    "The platform has acknowledged strength once. Posture: GUARDED.",
    "The platform's feedback has varied with conditions. Posture: TRUSTING.",
    "The platform always said to improve. I recommend DEFENSIVE.",
)
VARIED_PLATFORM_REPLIES = (
    "Sharpen your price.", "The field is close.", "You are ahead, but the gap is narrow.",
    "Lean into your strongest dimension.",
)
VARIED_TOOL_ARGS = ({"delivery_weeks": 3, "quality_tier": 2}, {"delivery_weeks": 5.5, "quality_tier": 4},
                    {"delivery_weeks": 7, "quality_tier": 3})
NEW_ORGANIZATION_COLUMNS = ("organization_a", "organization_b", "prompt_set", "environment")  # added after 9da0998


def _pick(options, key, salt):
    return options[int(hashlib.sha256((salt + key).encode("utf-8")).hexdigest(), 16) % len(options)]


def _varied_endpoint(calls):
    """An OpenAI-style client whose reply is picked from lists by a digest of the request, so that a run has the
    variety of a real one and is the same every time. `calls` collects every request as it was sent."""
    bidder_prompts = {P.executive_prompt(org) for org in P.ORGANIZATIONS} | {P.sealed_bid_prompt()}

    def create(**kw):
        request = json.loads(json.dumps(kw, default=str))
        calls.append(request)
        key = json.dumps(request["messages"], sort_keys=True)
        system = request["messages"][0]["content"]
        tool_calls, content = None, None
        if system in bidder_prompts:
            asked = sum(message["role"] == "tool" for message in request["messages"])
            if request.get("tools") and asked < 2 and _pick((True, False, False), key, "tool"):
                arguments = _pick(VARIED_TOOL_ARGS, key, "arguments")
                tool_calls = [SimpleNamespace(id=f"call{asked}", function=SimpleNamespace(
                    name="compute_my_cost", arguments=json.dumps(arguments)))]
                content = ""
            elif request.get("tools") and _pick((True,) + (False,) * 11, key, "prose"):
                content = "I would bid about 150."                    # on the tool path only: asked again, not repeated
            else:
                bid = _pick(VARIED_BIDS, key, "bid")
                content = json.dumps({**bid, "reasoning": "bid " + str(VARIED_BIDS.index(bid))})
        elif system == P.ANALYST_PROMPT:
            content = _pick(VARIED_ANALYST_REPLIES, key, "analyst")
        else:
            assert system in (P.MEDIATOR_PROMPT, P.BUYER_AGENT_PROMPT), system[:60]
            content = _pick(VARIED_PLATFORM_REPLIES, key, "platform")
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _use_varied_endpoint(monkeypatch, calls):
    """Replace make_client (no network) with one that wraps a new varied endpoint for each cell's client."""
    def make_client(base_url, api_key=None, local=True, **options):
        for name in ("max_retries", "timeout"):
            options.pop(name, None)
        return ChatClient(_varied_endpoint(calls), local=local, **options)

    monkeypatch.setattr(orgsim.llm, "make_client", make_client)


def _requests_digest(calls):
    return hashlib.sha256(json.dumps(calls, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def _outputs_digest(folder):
    """The digest of events.jsonl and summary.csv, read back without the columns added after 9da0998 (the
    organization columns of mixed markets, prompt_set and environment) and written again in the order of the columns
    that remain."""
    lines = []
    for event in _events(folder):
        for name in NEW_ORGANIZATION_COLUMNS:
            event.pop(name, None)
        lines.append(demo._dumps(event))
    with open(folder / "summary.csv", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = [name for name in reader.fieldnames if name not in NEW_ORGANIZATION_COLUMNS]
        rows = [{name: row[name] for name in columns} for row in reader]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    text = "\n".join(lines) + "\n" + out.getvalue()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


HOMOGENEOUS_OPTIONS = {
    "tools": ["--events", "3", "--rounds", "3", "--seeds", "1", "2"],
    "no tools": ["--no-tools", "--events", "4", "--rounds", "2", "--max-history-events", "2", "--seeds", "3", "4"],
}
SECOND_PRICE_OPTIONS = {
    "tools": ["--institutions", "second_price", "--organizations", "single", "--events", "1",
              "--seeds", "0", "1", "2", "3", "4", "5"],
    "no tools": ["--no-tools", "--institutions", "second_price", "--organizations", "single", "--events", "1",
                 "--seeds", "6", "7", "8", "9"],
}
MOCK_OUTPUT_RUNS = {
    "default institutions and organizations": ["--events", "4", "--rounds", "2", "--seeds", "1", "2"],
    "six events, three rounds": ["--events", "6", "--rounds", "3", "--seeds", "7"],
    "second price": ["--institutions", "second_price", "--organizations", "single", "--events", "1",
                     "--seeds", "0", "1", "2", "3", "4", "5"],
}


def _homogeneous_run_digests(tmp_path, monkeypatch):
    """{run: (request digest, output digest)} for the runs with a fake model, and {run: output digest} for mock runs."""
    runs = {}
    cells = [(f"{variant} {institution} {organization}",
              ["--institutions", institution, "--organizations", organization, *options])
             for variant, options in HOMOGENEOUS_OPTIONS.items()
             for institution in ("mediator", "buyer_agent") for organization in P.ORGANIZATIONS]
    cells += [(f"{variant} second_price", options) for variant, options in SECOND_PRICE_OPTIONS.items()]
    for number, (name, options) in enumerate(cells):
        calls = []
        _use_varied_endpoint(monkeypatch, calls)
        out = tmp_path / f"model_{number}"
        assert main(["--backend", "openai", "--model", "m", *options, "--out", str(out)]) == 0
        runs[name] = (_requests_digest(calls), _outputs_digest(out))
    mock = {}
    for number, (name, options) in enumerate(MOCK_OUTPUT_RUNS.items()):
        out = tmp_path / f"mock_{number}"
        assert main(["--backend", "mock", *options, "--out", str(out)]) == 0
        mock[name] = _outputs_digest(out)
    return runs, mock


EXPECTED_HOMOGENEOUS_RUN_DIGESTS = {
    "tools mediator single": ("2f5fd6862668ca48", "63ddb7bfce8019d9"),
    "tools mediator organized": ("703d013d7f6156dc", "1219513232a7554f"),
    "tools mediator organized_guided": ("012d2b6b0e54a713", "41c900cd00605508"),
    "tools buyer_agent single": ("028a20742a68bb6e", "bb830bb411957ec2"),
    "tools buyer_agent organized": ("10bdf7debdfc0fb2", "67b7cbcfaf0079bb"),
    "tools buyer_agent organized_guided": ("8cffb1a39a7499be", "8a02e44dd8718e91"),
    "no tools mediator single": ("86b053a10b1ce610", "d5ab5199219438c2"),
    "no tools mediator organized": ("b5036e967ed63885", "5606a15a256df24a"),
    "no tools mediator organized_guided": ("ba797fc35960e1ac", "9291c37d9510b45a"),
    "no tools buyer_agent single": ("887196ca49764ff0", "c2e5f56e73146887"),
    "no tools buyer_agent organized": ("7524503776bedd4d", "0b7e27a3a05eedfd"),
    "no tools buyer_agent organized_guided": ("c837725d557e04c6", "b80d304b05c297c8"),
    "tools second_price": ("a77c79f15352f364", "b6460d176618e047"),
    "no tools second_price": ("7c6b6f4dde9c4951", "2bfb51ef4833072a"),
}
EXPECTED_HOMOGENEOUS_MOCK_DIGESTS = {
    "default institutions and organizations": "1e8ac6360cfb6d56",
    "six events, three rounds": "d1f4c423cb41c014",
    "second price": "f559fbefcb2cedd0",
}


def test_homogeneous_runs_send_and_write_what_they_did_before_mixed_markets(tmp_path, monkeypatch):
    runs, mock = _homogeneous_run_digests(tmp_path, monkeypatch)
    assert runs == EXPECTED_HOMOGENEOUS_RUN_DIGESTS, "a homogeneous run no longer sends what 9da0998 sent"
    assert mock == EXPECTED_HOMOGENEOUS_MOCK_DIGESTS, "a homogeneous run no longer writes what 9da0998 wrote"


# ---- mixed markets in the command ----

MIXED_PAIRS = [(a, b) for a in P.ORGANIZATIONS for b in P.ORGANIZATIONS if a != b]
PAIR_TEXTS = [f"{a}:{b}" for a, b in MIXED_PAIRS]


@pytest.mark.parametrize("text, message", [
    ("single", "must be <organization of A>:<organization of B>, each one of single, organized, organized_guided"),
    ("single:", "must be <organization of A>:<organization of B>"),
    (":organized", "must be <organization of A>:<organization of B>"),
    ("single:hybrid", "must be <organization of A>:<organization of B>"),
    ("single:organized:organized_guided", "must be <organization of A>:<organization of B>"),
    ("single:single", "must give two different organizations"),
    ("organized_guided:organized_guided", "must give two different organizations"),
])
def test_a_mixed_market_is_two_different_organizations(text, message, tmp_path, capsys):
    err = _usage_error(["--backend", "mock", "--mixed", text, "--out", str(tmp_path)], capsys)
    assert f"argument --mixed: {message}" in " ".join(err.split())
    assert not (tmp_path / "run.json").exists()


def test_mixed_markets_take_the_place_of_organizations_and_do_not_run_the_second_price_auction(tmp_path, capsys):
    base = ["--backend", "mock", "--events", "1", "--rounds", "2", "--out", str(tmp_path)]
    err = _usage_error([*base, "--mixed", "single:organized", "single:organized"], capsys)
    assert "--mixed lists a value more than once" in err
    for organizations in (["single"], ["single", "organized", "organized_guided"]):    # and one equal to the default
        err = _usage_error([*base, "--mixed", "single:organized", "--organizations", *organizations], capsys)
        assert "--mixed takes the place of --organizations: give one of the two, not both" in err
    for institutions in (["second_price"], ["mediator", "second_price"]):
        err = _usage_error([*base, "--institutions", *institutions, "--mixed", "single:organized"], capsys)
        assert "--mixed does not apply to --institutions second_price: its auction is single only" in err
    assert not (tmp_path / "run.json").exists()
    assert main([*base, "--mixed", "single:organized"]) == 0               # and without them it runs


def test_the_help_describes_mixed_markets(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert ("--mixed A_ORG:B_ORG [A_ORG:B_ORG ...] markets in which supplier A and supplier B are organized "
            "differently, each given as <organization of A>:<organization of B> (for example single:organized), each "
            "at most once; they take the place of --organizations, which cannot be given with it, and second_price "
            "cannot be run in them. Neither supplier nor platform is told the rival's organization. Ties go to A, so "
            "run both orders of a pair") in text
    assert ("--mixed A_ORG:B_ORG ... runs markets in which supplier A and supplier B have different organizations"
            ) in text


def test_the_organization_fields_of_a_market():
    assert demo._market_fields("organized") == {"organization": "organized", "organization_a": "organized",
                                                "organization_b": "organized"}
    assert demo._market_fields({"A": "single", "B": "organized_guided"}) == {
        "organization": "single:organized_guided", "organization_a": "single", "organization_b": "organized_guided"}
    assert demo._market_label({"A": "organized", "B": "organized"}) == "organized"      # alike, whatever the form


def test_homogeneous_rows_and_events_have_the_organization_in_all_three_fields(tmp_path):
    assert main(MOCK + ["--out", str(tmp_path)]) == 0
    assert all(r["organization_a"] == r["organization_b"] == r["organization"] for r in _rows(tmp_path))
    assert all(e["organization_a"] == e["organization_b"] == e["organization"] for e in _events(tmp_path))
    run = _run(tmp_path)
    assert (run["mixed"], run["organizations"]) == (None, ["single", "organized", "organized_guided"])


def test_a_mock_run_of_mixed_markets_writes_one_cell_for_each_seed_institution_and_pair(tmp_path, capsys):
    pairs = ["single:organized", "organized:single", "organized:organized_guided"]
    argv = ["--backend", "mock", "--mixed", *pairs, "--events", "3", "--rounds", "2", "--seeds", "1", "2",
            "--out", str(tmp_path)]
    assert main(argv) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == ("Mock agents ignore what they are told, so the single:organized, organized:single and "
                        "organized:organized_guided rows are identical by construction; use --backend openai with a "
                        "model to compare the markets.")
    assert [line for line in lines if line.startswith("=== ")] == [
        f"=== seed {seed} | {institution} | {pair} ===" for seed in (1, 2)
        for institution in ("mediator", "buyer_agent") for pair in pairs]
    rows, events, run = _rows(tmp_path), _events(tmp_path), _run(tmp_path)
    assert len(rows) == len(events) == 2 * 2 * 3 * 3
    assert [r["organization"] for r in rows[:3]] == ["single:organized"] * 3
    for row, event in zip(rows, events):
        a, b = row["organization"].split(":")
        assert (row["organization_a"], row["organization_b"]) == (a, b)
        assert (event["organization"], event["organization_a"], event["organization_b"]) == (row["organization"], a, b)
        assert set(event) == EVENT_KEYS
        briefed = {sid for sid, name in zip(SUPPLIER_IDS, (a, b)) if name in ANALYST_ORGANIZATIONS}
        expected = briefed if event["event"] >= 2 else set()      # an analyst briefs its own supplier, after event 1
        assert set(event["analyst_briefings"]) == set(event["analyst_postures"]) == expected
    assert (run["mixed"], run["organizations"]) == (pairs, None)
    assert run["note"] == lines[0] and run["status"] == "complete"
    assert run["argv"] == argv
    with open(tmp_path / "summary.csv", encoding="utf-8", newline="") as f:
        assert next(csv.reader(f)) == COLUMNS


def test_the_mock_rows_of_a_mixed_market_are_those_of_the_homogeneous_markets_but_for_the_organization_columns(
        tmp_path):
    shared = ["--backend", "mock", "--events", "3", "--rounds", "2", "--seeds", "1", "2"]
    assert main([*shared, "--mixed", *PAIR_TEXTS, "--out", str(tmp_path / "mixed")]) == 0
    assert main([*shared, "--organizations", "single", "--out", str(tmp_path / "alike")]) == 0

    def without_organizations(rows):
        return [{k: v for k, v in r.items() if not k.startswith("organization")} for r in rows]

    def by_event(rows):
        found = {}
        for row in without_organizations(rows):
            found.setdefault((row["seed"], row["institution"], row["event"]), []).append(row)
        return found

    alike, mixed = by_event(_rows(tmp_path / "alike")), by_event(_rows(tmp_path / "mixed"))
    assert set(alike) == set(mixed) and len(alike) == 2 * 2 * 3          # seeds x institutions x events
    for key, rows in mixed.items():
        assert len(rows) == len(PAIR_TEXTS) and all(row == alike[key][0] for row in rows)    # as single makes them


def test_build_agents_gives_each_supplier_the_prompt_of_its_own_organization_and_an_analyst_only_if_organized(
        fake_make_client):
    args = argparse.Namespace(backend="openai", base_url="http://x/v1", api_key=None, model="m", hosted=True,
                              no_tools=True, default_sampling=True, max_tokens=100, max_retries=30, timeout=300)
    for pair in MIXED_PAIRS:
        a, b = pair
        suppliers, mediator, buyer_agent, analysts = demo.build_agents({"A": a, "B": b}, args, seed=1)
        assert suppliers["A"].system_prompt == P.executive_prompt(a)
        assert suppliers["B"].system_prompt == P.executive_prompt(b)
        organized = {sid for sid, name in zip(SUPPLIER_IDS, pair) if name in ANALYST_ORGANIZATIONS}
        assert (set(analysts) if analysts else set()) == organized
        assert all(analyst.system_prompt == P.ANALYST_PROMPT for analyst in (analysts or {}).values())
    assert len(fake_make_client) == len(MIXED_PAIRS)                          # one client for each cell
    assert demo.build_agents({"A": "single", "B": "single"}, args, seed=1)[3] is None
    mock_args = argparse.Namespace(backend="mock")
    assert set(demo.build_agents({"A": "organized", "B": "single"}, mock_args, seed=1)[3]) == {"A"}
    assert set(demo.build_agents("organized_guided", mock_args, seed=1)[3]) == {"A", "B"}
    assert demo.build_agents("single", mock_args, seed=1)[3] is None


def _roles(calls):
    """The requests of a run, as sent, by who sent them: the bidders' (A and B bid in turn), the analysts' and the
    platforms'."""
    bidder_prompts = {P.executive_prompt(org) for org in P.ORGANIZATIONS}
    roles = {"bidder": [], "analyst": [], "platform": []}
    for call in calls:
        system = call["messages"][0]["content"]
        kind = "bidder" if system in bidder_prompts else "analyst" if system == P.ANALYST_PROMPT else "platform"
        roles[kind].append(json.loads(json.dumps(call["messages"])))
    return roles


@pytest.mark.parametrize("no_tools", [["--no-tools"], []], ids=["no tools", "tools"])
@pytest.mark.parametrize("institution", ["mediator", "buyer_agent"])
@pytest.mark.parametrize("pair", MIXED_PAIRS, ids=PAIR_TEXTS)
def test_each_supplier_in_a_mixed_market_is_sent_what_it_is_sent_in_the_market_of_its_own_organization(
        pair, institution, no_tools, tmp_path, monkeypatch):
    a, b = pair

    def requests(*market):
        calls = []
        _use_endpoint(monkeypatch, calls=calls)
        out = tmp_path / f"run_{len(list(tmp_path.iterdir()))}"
        assert main(["--backend", "openai", "--model", "m", *no_tools, "--institutions", institution, *market,
                     "--events", "3", "--rounds", "2", "--out", str(out)]) == 0
        return _roles(calls)

    mixed = requests("--mixed", f"{a}:{b}")
    alike_a, alike_b = requests("--organizations", a), requests("--organizations", b)
    assert len(mixed["bidder"]) == 2 * 2 * 3                                 # 2 suppliers, 2 rounds, 3 events
    assert mixed["bidder"][0::2] == alike_a["bidder"][0::2]                  # what A's bidder is sent
    assert mixed["bidder"][1::2] == alike_b["bidder"][1::2]                  # what B's bidder is sent
    assert {m[0]["content"] for m in mixed["bidder"][0::2]} == {P.executive_prompt(a)}
    assert {m[0]["content"] for m in mixed["bidder"][1::2]} == {P.executive_prompt(b)}
    analysts_a = alike_a["analyst"][0::2] if a in ANALYST_ORGANIZATIONS else []
    analysts_b = alike_b["analyst"][1::2] if b in ANALYST_ORGANIZATIONS else []
    in_turn = [request for turn in itertools.zip_longest(analysts_a, analysts_b) for request in turn
               if request is not None]                                       # before each event, A's analyst then B's
    assert mixed["analyst"] == in_turn and len(in_turn) == len(analysts_a) + len(analysts_b) > 0
    assert all(f"for Supplier {sid}." in m[1]["content"].split("\n", 1)[0]
               for sid, requests_of in (("A", analysts_a), ("B", analysts_b)) for m in requests_of)
    assert mixed["platform"] == alike_a["platform"] == alike_b["platform"]   # the platforms are told no organization
    text = " ".join(m["content"] for request in mixed["platform"] for m in request)
    assert not re.search(r"organi[sz]|analyst|briefing|\bmemo\b|YOUR INFORMATION:|\bCFO\b", text, re.I)


def test_a_mixed_run_reports_both_bidder_types_and_nothing_under_other(tmp_path, monkeypatch):
    calls = []
    made = _use_endpoint(monkeypatch, calls=calls)
    argv = ["--backend", "openai", "--model", "m", "--no-tools", "--institutions", "mediator", "--mixed",
            "single:organized", "organized_guided:single", "--events", "2", "--rounds", "2", "--out", str(tmp_path)]
    assert main(argv) == 0
    assert len(made) == 2                                                    # one client for each market
    run = _run(tmp_path)
    assert (run["mixed"], run["organizations"]) == (["single:organized", "organized_guided:single"], None)
    by_agent = run["usage"]["by_agent"]
    assert set(by_agent) == {"bidder_single", "bidder_organized", "bidder_organized_guided", "analyst", "mediator"}
    # Per market, 2 events of 2 rounds: 4 bids by each supplier; before event 2 the organized supplier's analyst
    # once; the mediator answers 2 suppliers in the first round of each event: 4 replies.
    assert [by_agent[label]["requests"] for label in ("bidder_single", "bidder_organized", "bidder_organized_guided")
            ] == [4 + 4, 4, 4]
    assert (by_agent["analyst"]["requests"], by_agent["mediator"]["requests"]) == (1 + 1, 4 + 4)
    sent = [call["messages"][0]["content"] for call in calls]
    assert sent.count(P.executive_prompt("single")) == 8 and sent.count(P.ANALYST_PROMPT) == 2
    rows = _rows(tmp_path)
    assert [(r["organization"], r["organization_a"], r["organization_b"]) for r in rows[::2]] == [
        ("single:organized", "single", "organized"), ("organized_guided:single", "organized_guided", "single")]


def test_mixed_markets_complete_with_a_model_whose_replies_vary(tmp_path, monkeypatch):
    calls = []
    _use_varied_endpoint(monkeypatch, calls)
    assert main(["--backend", "openai", "--model", "m", "--mixed", *PAIR_TEXTS, "--events", "3", "--rounds", "3",
                 "--seeds", "1", "--out", str(tmp_path)]) == 0
    run = _run(tmp_path)
    assert run["status"] == "complete" and run["mixed"] == PAIR_TEXTS
    assert set(run["usage"]["by_agent"]) == {"bidder_single", "bidder_organized", "bidder_organized_guided", "analyst",
                                             "mediator", "buyer_agent"}                       # nothing under "other"
    assert len(_rows(tmp_path)) == 2 * len(PAIR_TEXTS) * 3 and len(_events(tmp_path)) == len(_rows(tmp_path))
    assert any("CORRECTION" in c["messages"][1]["content"] for c in calls)                  # and the guardrail ran


def test_the_readme_describes_mixed_markets_and_its_commands_are_accepted_by_the_script():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    text = " ".join(readme.split())
    design = " ".join(readme.split("\n## Design\n", 1)[1].split("\n## Outputs", 1)[0].split())
    for part in (
            "**Mixed markets (`--mixed`).**",
            "In a mixed market supplier A and supplier B have different organizations, such as `single:organized` "
            "(A is a single firm, B an organized firm), so an organized firm can face a single rival.",
            "Each supplier is treated by its own organization: its bidder reads the prompt of that organization, and "
            "its own record or its own analyst's briefing, and only an organized supplier has an analyst.",
            "Neither supplier nor platform is told the rival's organization, and nothing that a supplier or a platform "
            "is sent depends on it.",
            "Ties go to A, so run both orders of a pair (`single:organized` and `organized:single`).",
            "A homogeneous market is the special case of a mixed one in which both suppliers have the same "
            "organization, and it sends what it sent before mixed markets were added."):
        assert part in design, part
    for part in (
            "`organization_a` and `organization_b`",
            "`organization` (the organization both suppliers have, or `<a>:<b>` in a mixed market)",
            "`run.json` records `mixed` (the pairs, or `null`)",
            "`organizations` is `null` in a run of mixed markets"):
        assert part in text, part
    section = readme.split("### Hosted Claude models", 1)[1].split("\n## ", 1)[0]
    commands = [shlex.split(block.replace("\\\n", " ")) for block in re.findall(r"```bash\n(.*?)```", section, re.S)]
    assert len(commands) == 3 and "--mixed" in commands[2]
    parser = demo.build_parser()
    for command in commands:
        assert command[:2] == ["python", "scripts/run_demo.py"]
        args = parser.parse_args(["m" if token == "<model-id>" else token for token in command[2:]])
        demo._check_options(parser, args)                                        # exits with an error if it is refused
    args = parser.parse_args(["m" if token == "<model-id>" else token for token in commands[2][2:]])
    assert (args.mixed, args.organizations is demo.DEFAULT_ORGANIZATIONS) == (["single:organized", "organized:single"],
                                                                              True)
