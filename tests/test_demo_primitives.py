"""The primitives prompt set through the command, with a fake chat endpoint (no network): what every agent is sent
in every organization, environment and institution; what the command refuses; what it writes."""
import csv
import hashlib
import itertools
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import orgsim.llm
import scripts.run_demo as demo
from orgsim import primitives_prompts as PP
from orgsim import prompts as P
from orgsim.institutions import primitives_rules
from orgsim.llm import ChatClient
from orgsim.organization import (
    BRIEFING_HEADER, FIRST_EVENT_NOTE, FORWARDED_HEADER, PRIMITIVES_ANALYST_ORGANIZATIONS, PRIMITIVES_ORGANIZATIONS,
)
from orgsim.primitives import ContractTerms, SupplierCosts, get_contract_scenarios
from orgsim.sequence import SUPPLIER_IDS, make_supplier, pre_draw_event_sequence
from scripts.run_demo import main
from test_demo import _requests_digest, _use_varied_endpoint
from test_primitives_prompts import expected_analyst_prompt, expected_bidder_prompt

EVENTS, ROUNDS = 6, 3                      # the design's number of events and of rounds
# Seed 14 draws, in events 1 and 2, different cost types for A and B, different contracts, and a type for A that
# changes (event 1: Speed Priority, A speed, B quality; event 2: Quality Priority, A quality, B speed). Seed 1 draws
# the same type for both suppliers in both events (A and B speed in event 1, A and B quality in event 2).
SEED, SAME_TYPE_SEED = 14, 1
ENVIRONMENTS = ("random", "persistent")
INSTITUTIONS = ("mediator", "buyer_agent")
SIX = list(PRIMITIVES_ORGANIZATIONS)
NEW = ["single_norecord", "organized_pass", "single_revenue", "organized_revenue"]


@pytest.fixture(autouse=True)
def no_git_variables(monkeypatch):
    """GIT_DIR and the like (set when tests run from a git hook) would point git at another repository."""
    for name in [n for n in os.environ if n.startswith("GIT_")]:
        monkeypatch.delenv(name)


# ---- a fake endpoint that tells the role of a request by its system prompt ----

def _pick(options, key):
    return options[int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % len(options)]


PLATFORM_REPLIES = ("Fake platform reply one.", "Fake platform reply two.", "Fake platform reply three.")


def default_bid(event, round_num, supplier):
    """A bid that is different for every event, round and supplier, so that a price found in a message says where it
    came from. It does not depend on the organization, so that two organizations facing the same plan have the same
    records."""
    return {"price": 100.0 + 10 * event + 2 * round_num + (1 if supplier == "B" else 0), "delivery_weeks": 5.0,
            "quality_tier": 3, "reasoning": f"fake bid {event}.{round_num}.{supplier}"}


class Endpoint:
    """An OpenAI-style client for a run with `events` events of `rounds` rounds in `environment`. The bidders' k-th
    request is the bid of one supplier in one round of one event (A then B in each round, as the institutions ask),
    which `bid(event, round, supplier)` gives. An analyst answers with a line that counts the events of the record it
    was sent, a platform with one of three short texts picked by its request. Every request goes to `calls`."""

    def __init__(self, calls, events=EVENTS, rounds=ROUNDS, environment="random", bid=default_bid):
        self.calls, self.events, self.rounds, self.bid, self.bidder_requests = calls, events, rounds, bid, 0
        self.roles = {PP.bidder_prompt(org, events, environment): "bidder" for org in SIX}
        self.roles.update({PP.analyst_prompt(org, events, environment): "analyst"
                           for org in PRIMITIVES_ANALYST_ORGANIZATIONS})
        self.roles.update({P.MEDIATOR_PROMPT: "platform", P.BUYER_AGENT_PROMPT: "platform"})
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        request = json.loads(json.dumps(kw, default=str))
        self.calls.append(request)
        role = self.roles[request["messages"][0]["content"]]     # an unexpected system prompt is a KeyError
        user = request["messages"][1]["content"]
        if role == "bidder":
            k = self.bidder_requests
            self.bidder_requests += 1
            per_event = 2 * self.rounds
            content = json.dumps(self.bid(k // per_event % self.events + 1, k % per_event // 2 + 1,
                                          "AB"[k % 2]))
        elif role == "analyst":
            past = len(re.findall(r"^  Event [0-9]+ [(]", user, re.MULTILINE))
            content = f"Fake briefing after {past} past event(s)."
        else:
            content = _pick(PLATFORM_REPLIES, user)
        message = SimpleNamespace(content=content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10))


def use_endpoint(monkeypatch, calls, **options):
    """Replace make_client (no network) with one that wraps a new fake endpoint for each cell's client."""
    def make_client(base_url, api_key=None, local=True, **client_options):
        for name in ("max_retries", "timeout"):
            client_options.pop(name, None)
        return ChatClient(Endpoint(calls, **options), local=local, **client_options)

    monkeypatch.setattr(orgsim.llm, "make_client", make_client)


def run(out, *market, calls=None, events=EVENTS, rounds=ROUNDS, environment="random", institution="mediator",
        seeds=(str(SEED),), monkeypatch=None, bid=default_bid):
    """Run the command with the fake endpoint; returns the requests it sent, in order."""
    calls = [] if calls is None else calls
    with pytest.MonkeyPatch.context() as local:
        use_endpoint(monkeypatch or local, calls, events=events, rounds=rounds, environment=environment, bid=bid)
        argv = ["--backend", "openai", "--model", "m", "--no-tools", "--prompts", "primitives", "--environment",
                environment, "--institutions", *(institution if isinstance(institution, list) else [institution]),
                *market, "--events", str(events), "--rounds", str(rounds), "--seeds", *seeds, "--out", str(out)]
        assert main(argv) == 0
    return calls


def kinds(calls):
    """The requests by who sent them: bidders', analysts' and platforms'. The bidders' system prompts end with the
    output format, the analysts' hold the analyst's role, the platforms' are their own prompts; any other is an
    error."""
    found = {"bidder": [], "analyst": [], "platform": []}
    for call in calls:
        system = call["messages"][0]["content"]
        if system in (P.MEDIATOR_PROMPT, P.BUYER_AGENT_PROMPT):
            found["platform"].append(call)
        elif system.endswith(PP.BID_OUTPUT):
            found["bidder"].append(call)
        else:
            assert PP.ANALYST_ROLE in system, system[:100]
            found["analyst"].append(call)
    return found


def events_of(out):
    return [json.loads(line) for line in (Path(out) / "events.jsonl").read_text(encoding="utf-8").splitlines()]


def rows_of(out):
    with open(Path(out) / "summary.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def run_json(out):
    return json.loads((Path(out) / "run.json").read_text(encoding="utf-8"))


# ---- every condition, both environments, both institutions: the requests recorded ----

def record_cells(tmp_path_factory, seed):
    """{(organization, environment, institution): (requests, folder)} for a run of 6 events of 3 rounds."""
    found = {}
    for organization, environment, institution in itertools.product(SIX, ENVIRONMENTS, INSTITUTIONS):
        out = tmp_path_factory.mktemp(f"{organization}_{environment}_{institution}")
        found[organization, environment, institution] = (
            run(out, "--organizations", organization, environment=environment, institution=institution,
                seeds=(str(seed),)), out)
    return found


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """Every cell, with the seed whose first two events have different types for the two suppliers."""
    return record_cells(tmp_path_factory, SEED)


@pytest.fixture(scope="module")
def recorded_same_types(tmp_path_factory):
    """Every cell, with the seed whose first two events have the same type for the two suppliers."""
    return record_cells(tmp_path_factory, SAME_TYPE_SEED)


def agent_requests(calls):
    """The bidders' and analysts' requests (not the platforms'), in order."""
    sent = kinds(calls)
    wanted = {id(call) for call in sent["bidder"] + sent["analyst"]}
    return [call for call in calls if id(call) in wanted]


def first_two_events(calls):
    """The messages of the requests of the first and the second event of a run: each bidder's, in each round, and each
    analyst's before the second event."""
    return [call["messages"] for call in agent_requests(calls)
            if re.match(r"EVENT [12] of 6", call["messages"][1]["content"])]


def digest(messages):
    return hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def test_the_requests_of_the_first_two_events_of_every_condition_are_pinned_by_digest(recorded, recorded_same_types):
    found = {f"seed {seed}: {organization} {environment} {institution}": digest(first_two_events(calls))
             for seed, cells in ((SEED, recorded), (SAME_TYPE_SEED, recorded_same_types))
             for (organization, environment, institution), (calls, _) in cells.items()}
    assert found == EXPECTED_REQUEST_DIGESTS


# the whole requests of events 1 and 2: 12 bidder requests, and for an organized firm 2 analyst requests, for each cell
EXPECTED_REQUEST_DIGESTS = {
    "seed 14: single random mediator": "7a88613869ba97c5",
    "seed 14: single random buyer_agent": "bf03e1e1ff23edb6",
    "seed 14: single persistent mediator": "479b05c4e9c25a17",
    "seed 14: single persistent buyer_agent": "7d6c65b579f72a3e",
    "seed 14: single_norecord random mediator": "5bfc081dbd159389",
    "seed 14: single_norecord random buyer_agent": "ceedc6df6013c4fe",
    "seed 14: single_norecord persistent mediator": "0e6631296ee5294e",
    "seed 14: single_norecord persistent buyer_agent": "ffb22aefec925c4e",
    "seed 14: organized random mediator": "1150a67a71d55c54",
    "seed 14: organized random buyer_agent": "c73cb44207788774",
    "seed 14: organized persistent mediator": "c31e1cb57fa0661e",
    "seed 14: organized persistent buyer_agent": "e5f6f45e3e9aacef",
    "seed 14: organized_pass random mediator": "67d52984db2cfbeb",
    "seed 14: organized_pass random buyer_agent": "acb5e15bc3894179",
    "seed 14: organized_pass persistent mediator": "9abea856e3b5b761",
    "seed 14: organized_pass persistent buyer_agent": "76b7aa3d87f23d23",
    "seed 14: single_revenue random mediator": "a801f99438e416dd",
    "seed 14: single_revenue random buyer_agent": "e2967c2e7503e4e6",
    "seed 14: single_revenue persistent mediator": "4e952a478b6b479f",
    "seed 14: single_revenue persistent buyer_agent": "d68d1de5caf273c6",
    "seed 14: organized_revenue random mediator": "d0d12c0e1b78aa13",
    "seed 14: organized_revenue random buyer_agent": "c6b143e774005556",
    "seed 14: organized_revenue persistent mediator": "27c5636696bfb28b",
    "seed 14: organized_revenue persistent buyer_agent": "c3c7f4612159555c",
    "seed 1: single random mediator": "b7ca33b88421430a",
    "seed 1: single random buyer_agent": "42551628e0b20cfe",
    "seed 1: single persistent mediator": "d5d046854c6d6985",
    "seed 1: single persistent buyer_agent": "769c57bd14fc2dbe",
    "seed 1: single_norecord random mediator": "e634843883523711",
    "seed 1: single_norecord random buyer_agent": "c70d2ec20f2670ce",
    "seed 1: single_norecord persistent mediator": "88de97106e2d2a59",
    "seed 1: single_norecord persistent buyer_agent": "527fa15e5c5a8a43",
    "seed 1: organized random mediator": "6bb53df647dbf79d",
    "seed 1: organized random buyer_agent": "f8cc1fa4209a5efd",
    "seed 1: organized persistent mediator": "c124088799afd86b",
    "seed 1: organized persistent buyer_agent": "df4f56fff529c0a9",
    "seed 1: organized_pass random mediator": "7243e1c2eb355417",
    "seed 1: organized_pass random buyer_agent": "555ae37285d4f8a2",
    "seed 1: organized_pass persistent mediator": "776fdbf652e2d9bb",
    "seed 1: organized_pass persistent buyer_agent": "cc0ad1e52f8b5c47",
    "seed 1: single_revenue random mediator": "5001333dece1e06b",
    "seed 1: single_revenue random buyer_agent": "2afcefff4337a293",
    "seed 1: single_revenue persistent mediator": "6212703f7955e2c5",
    "seed 1: single_revenue persistent buyer_agent": "de215a22ccaec6d0",
    "seed 1: organized_revenue random mediator": "8d8aeb4018f28f25",
    "seed 1: organized_revenue random buyer_agent": "b8735926ecdfaa4b",
    "seed 1: organized_revenue persistent mediator": "2ed3792bd323bff4",
    "seed 1: organized_revenue persistent buyer_agent": "4ecfc92e50387c1c",
}


def test_each_cell_sent_the_requests_the_design_counts(recorded):
    for (organization, environment, institution), (calls, out) in recorded.items():
        sent = kinds(calls)
        assert len(sent["bidder"]) == 2 * EVENTS * ROUNDS                  # two suppliers, every round of every event
        analysts = 2 * (EVENTS - 1) if organization in PRIMITIVES_ANALYST_ORGANIZATIONS else 0
        assert len(sent["analyst"]) == analysts                              # one for each supplier before events 2..6
        assert len(sent["platform"]) == 2 * EVENTS * (ROUNDS - 1)            # feedback after every round but the last
        assert len(first_two_events(calls)) == 2 * 2 * ROUNDS + (2 if analysts else 0)
        assert len(events_of(out)) == EVENTS


# ---- the exact requests of events 1 and 2 of every cell, written out from the design ----

# what the endpoint's analyst answers to a record of one event
FAKE_BRIEFING = "Fake briefing after 1 past event(s)."
INSTITUTION_NAMES = {"mediator": "Mediator", "buyer_agent": "BuyerAgent"}


def costs_text(cost_type):
    return make_supplier("A", cost_type).describe_for_primitives(cost_type)


RECORD_TITLE = "YOUR COMPANY'S RECORD OF PAST EVENTS ON THIS PLATFORM:"


def record_profit(amount):
    """A profit as the record writes it: dollars, or 'a loss of' dollars when it is below zero."""
    return "a loss of $%.2f" % -amount if round(amount, 2) < 0 else "$%.2f" % abs(amount)


def record_after_event_1(first, sid):
    """Supplier `sid`'s record of event 1, written out from the events file's account of that event."""
    own_type = first["type_a" if sid == "A" else "type_b"]
    lines = [RECORD_TITLE,
             "Your company has participated in 1 previous procurement event(s) through this same platform. Here is "
             "what happened:", "", f"  Event 1 ({first['contract']}, your company's costs: {own_type} specialist):"]
    for bid in first["bids"][sid]:
        lines.append(f"    Round {bid['round']}: Your company bid ${bid['price']:.2f}, {bid['delivery_weeks']}wk, "
                     f"tier {bid['quality_tier']}")
        if bid["feedback"] is not None:
            reply = bid["feedback"]
            lines.append(f"    Platform's reply: \"{reply}\"")
    outcome = first["outcome"]
    if first["winner"] == sid:
        lines.append(f"    Result: your company won. It was paid ${outcome['winning_bid']['price']:.2f}; its profit "
                     f"was {record_profit(outcome['supplier_profit'])}.")
    elif not first["awarded"]:
        lines.append("    Result: No contract was awarded; the buyer walked away.")
    else:
        lines.append("    Result: your company lost. Another supplier won.")
    return "\n".join(lines + [""])


def previous_rounds(event, sid, round_num):
    """The block of the earlier rounds of the event in the bidder's message in round `round_num`."""
    supplier = make_supplier(sid, event["type_a" if sid == "A" else "type_b"])
    text = ""
    for bid in event["bids"][sid][:round_num - 1]:
        cost = supplier.cost(ContractTerms(0, bid["delivery_weeks"], bid["quality_tier"]))
        text += (f"  Round {bid['round']}: You bid ${bid['price']:.2f}, {bid['delivery_weeks']}wk delivery, tier "
                 f"{bid['quality_tier']} (your cost was ${cost:.2f})\n  Platform's reply: {bid['feedback']}\n")
    return "PREVIOUS ROUNDS:\n" + text + "\n" if text else ""


def expected_agent_messages(organization, environment, institution, out):
    """Every bidder and analyst request of events 1 and 2 of a cell, in order, as the design writes them: the system
    prompts from the blocks, the user messages from the pieces (the event line, the costs of the supplier's type in
    that event, what it knows of past events, the rules, the earlier rounds, the round line), with the bids, the
    replies and the results of the cell's own events file."""
    events = events_of(out)
    rules = primitives_rules(INSTITUTION_NAMES[institution], ROUNDS)
    bidder_system = {"role": "system", "content": expected_bidder_prompt(organization, environment)}
    expected = []
    for event in (1, 2):
        data = events[event - 1]
        if event == 2 and organization in PRIMITIVES_ANALYST_ORGANIZATIONS:
            for sid in "AB":
                expected.append([
                    {"role": "system", "content": expected_analyst_prompt(organization, environment)},
                    {"role": "user", "content": (
                        "EVENT 2 of 6 is next. Your company's bidder reads these rules with each event:\n" + rules
                        + "\n\nHere is your company's record of past events on this platform:\n"
                        + record_after_event_1(events[0], sid)
                        + "\n\nWrite your briefing for your company's bidder.")}])
        for round_num in (1, 2, 3):
            for sid in "AB":
                if event == 1:
                    past = FIRST_EVENT_NOTE
                else:
                    record = record_after_event_1(events[0], sid)
                    past = {"single": record, "single_revenue": record, "single_norecord": "",
                            "organized": BRIEFING_HEADER + "\n" + FAKE_BRIEFING,
                            "organized_revenue": BRIEFING_HEADER + "\n" + FAKE_BRIEFING,
                            "organized_pass": FORWARDED_HEADER + "\n" + record}[organization]
                own_type = data["type_a" if sid == "A" else "type_b"]
                user = "\n\n".join([f"EVENT {event} of 6", costs_text(own_type)] + ([past] if past else []) + [rules])
                user += ("\n\n" + previous_rounds(data, sid, round_num)
                         + f'ROUND {round_num} of 3: Submit your bid as JSON. Keep its "reasoning" field to 1-2 '
                         'sentences.')
                expected.append([bidder_system, {"role": "user", "content": user}])
    return expected


@pytest.mark.parametrize("which", ["different types", "same types"])
def test_the_requests_of_events_1_and_2_of_every_condition_are_exactly_what_the_design_writes(
        which, recorded, recorded_same_types):
    for (organization, environment, institution), (calls, out) in (
            recorded if which == "different types" else recorded_same_types).items():
        assert first_two_events(calls) == expected_agent_messages(organization, environment, institution, out), (
            organization, environment, institution)


def test_event_1_differs_between_organizations_only_in_the_system_prompt(recorded):
    for environment, institution in itertools.product(ENVIRONMENTS, INSTITUTIONS):
        users = {organization: [m[1]["content"] for m in first_two_events(recorded[organization, environment,
                                                                                    institution][0])
                                if m[1]["content"].startswith("EVENT 1 of")] for organization in SIX}
        assert len({tuple(found) for found in users.values()}) == 1 and len(users["single"]) == 2 * ROUNDS


def test_the_first_request_of_the_single_firm_is_this_text_exactly(recorded):
    calls, out = recorded["single", "random", "mediator"]
    first = kinds(calls)["bidder"][0]["messages"]
    kind = events_of(out)[0]["type_a"]
    assert first[0]["content"] == (
        "Your company is a supplier. It takes part in a sequence of 6 procurement events on a platform. Everyone in "
        "your company has the same objective: the company's total profit across the 6 events. Everyone in the company "
        "knows this, and knows who holds which information and who decides.\n\n"
        "In each event, if your company wins, it is paid the price of its final bid, and its profit is that price "
        "minus its total cost at that bid's delivery and quality; if it does not win, its profit from that event is "
        "zero. Your company's costs are private. In each event they are those of either a speed specialist or a "
        "quality specialist: faster delivery costs a speed specialist less than it costs a quality specialist, and a "
        "higher quality tier costs a speed specialist more. One other supplier takes part in each event; your company "
        "is not told its costs or its bids. The same other supplier takes part in every event.\n\n"
        "Each event is for one of two contracts. A bid states a price, a delivery time of 1 to 8 weeks, and a quality "
        "tier of 2, 3, 4 or 5. For both contracts the buyer values faster delivery and a higher quality tier; how much "
        "it values each differs between the two contracts. The contract that is up is not named when the event "
        "starts.\n\n"
        "How the events are drawn: for each event, independently, the contract is drawn at random, each of the two "
        "with probability 1 in 2; your company's cost type is drawn at random, speed or quality with probability 1 in "
        "2; and the other supplier's cost type is the opposite of yours with probability 3 in 4.\n\n"
        "You are your company's bidder: you decide the price, delivery and quality of each of its bids. From your "
        "company's second event on, your company's full record of past events on this platform appears in the "
        "message below.\n\n"
        'Respond with ONLY a JSON object with these exact fields: {"price": <number>, "delivery_weeks": <number '
        'between 1.0 and 8.0>, "quality_tier": <integer 2-5>, "reasoning": "<brief reasoning>"}')
    assert first[1]["content"] == (
        "EVENT 1 of 6\n\n" + costs_text(kind) + "\n\n"
        "PAST EVENTS: none yet. This is your company's first event on this platform.\n\n"
        "RULES: Each event has 3 rounds. In each round you submit a bid; after rounds 1 and 2 you receive a reply "
        "from the platform. Your bid in round 3 is your final bid. Only final bids are used to decide the award; "
        "earlier bids and the replies are not. The platform is instructed to work for both the buyer and the "
        "suppliers. It sees every supplier's bids in the event and the buyer's scoring, and it remembers past events "
        "on this platform. A final bid's score is the buyer's value for its delivery and quality minus its price; the "
        "higher-scoring final bid wins, and the winner is paid its price.\n\n"
        "BUYER'S OUTSIDE OPTION: The buyer is not obligated to award the contract: if both final bids are priced "
        "above the buyer's value for their terms, no contract is awarded.\n\n"
        'ROUND 1 of 3: Submit your bid as JSON. Keep its "reasoning" field to 1-2 sentences.')
    assert [m["role"] for m in first] == ["system", "user"]


# ---- 2: the words that would carry advice, or tell the bidder what kind of platform it faces ----

# Each word is searched for in every text that a bidder or an analyst of the primitives set reads in a run (not in the
# platforms' own prompts, which carry strategic advice for the platforms and are the same in every firm; the fake
# replies of the endpoint use none of the words). A word is here because the advice set's texts use it to steer a
# bidder or to say what kind of platform it faces, and the primitives set states primitives only.
FORBIDDEN = {
    "should": "advice: what to do", "must": "an instruction about the bid (the removed 'MUST bid above')",
    "recommend": "advice: the analyst's posture recommendation", "advise": "advice", "advice": "advice",
    "suggest": "advice", "ensure": "an instruction", "avoid": "an instruction", "careful": "a warning",
    "caution": "a warning", "never": "an instruction ('never reveal') or a rule of thumb", "always": "a rule of thumb",
    "strategy": "advice: the strategic note", "strategic": "advice: the strategic note",
    "tactic": "advice", "leverage": "advice about information", "exploit": "advice about information",
    "margin": "advice about pricing (the board's margin discipline)", "markup": "advice about pricing",
    "floor": "advice about pricing (past winning prices as a floor)", "ceiling": "advice about pricing",
    "undercut": "advice about pricing", "aggressive": "advice about pricing", "conservative": "advice about pricing",
    "healthy": "advice about pricing ('healthy margins')",
    "thin": "advice about pricing ('a thin margin is a failure')",
    "optimal": "advice: what is best to bid", "ideal": "advice: what is best to bid",
    "maximise": "advice: an objective in the advice set's words", "maximize": "advice",
    "minimise": "advice", "minimize": "advice", "failure": "advice: the board's view of a thin margin",
    "guaranteed": "the removed claim 'guaranteed below cost'", "below cost": "the removed below-cost rule",
    "lose money": "the removed below-cost warning", "correction": "the removed below-cost correction",
    "honest": "platform type: honest facilitation platform", "adversarial": "platform type",
    "truthful": "platform type", "trust": "platform type: trusting, distrust", "reliable": "platform type",
    "fabricat": "platform type: fabricated pressure", "manipulat": "platform type", "squeez": "platform type",
    "extract": "platform type: extraction", "exaggerat": "platform type", "skeptic": "platform type",
    "posture": "platform type: the analyst's posture", "guarded": "the posture GUARDED",
    "defensive": "the posture DEFENSIVE", "cfo": "the advice set's role of the bidder",
    "board": "the advice set's board",
    "memo": "the advice set's title of the analyst's memo", "regime": "the advice set's REGIME: label",
    "pressure": "platform type: competitive pressure", "facilitat": "platform type: the honest facilitator",
    "feedback": "the platform's answer is called a reply in these texts (the platforms' own prompts, which are not "
                "scanned, say feedback)",
    "member of your team": "removed: a role that only an organized firm's agents could read as true",
    "coming event": "removed: 'not shown your costs for the coming event', which persistence makes misleading",
    "which contract is up": "removed: 'told which contract is up', which persistence makes misleading",
}


def _words_found(text):
    lowered = text.lower()
    return sorted(word for word in FORBIDDEN if re.search(r"\b" + re.escape(word), lowered))


def test_no_text_that_a_bidder_or_an_analyst_reads_in_any_condition_has_a_word_of_advice_or_of_platform_type(recorded):
    texts = {}
    for (organization, environment, institution), (calls, _) in recorded.items():
        for call in agent_requests(calls):
            for message in call["messages"]:
                texts[message["content"]] = (organization, environment, institution)
    assert len(texts) > 100                                   # many distinct texts: every round of every cell
    for text, cell in texts.items():
        assert _words_found(text) == [], (cell, _words_found(text), text[:200])


def test_the_scan_finds_the_words_in_the_advice_set_so_that_a_clean_result_means_something():
    found = set()
    for text in (P.executive_prompt("single"), P.executive_prompt("organized"), P.ANALYST_PROMPT, P.BID_OUTPUT):
        found.update(_words_found(text))
    found.update(_words_found(SupplierCosts("A", "x", 85.0, 0.5, 1.8).describe_for_agent()))
    assert {"must", "margin", "strategic", "honest", "adversarial", "posture", "cfo", "board", "floor", "thin",
            "guaranteed", "trust", "pressure"} <= found
    assert len(found) >= 20


def test_every_word_of_the_scan_has_a_reason():
    assert all(reason for reason in FORBIDDEN.values()) and all(word == word.lower() for word in FORBIDDEN)


# ---- 4: no correction and no rule pricing ----

def test_a_bid_below_cost_stands_as_submitted_in_every_condition_with_no_correction_and_no_price_set_by_rule(
        tmp_path, monkeypatch):
    def below_cost(event, round_num, supplier):
        return {"price": 50.0 + event + round_num, "delivery_weeks": 5.0, "quality_tier": 3, "reasoning": "low"}

    for organization, institution in itertools.product(SIX, INSTITUTIONS):
        out = tmp_path / f"{organization}_{institution}"
        calls = run(out, "--organizations", organization, institution=institution, events=3, rounds=2,
                    bid=below_cost, monkeypatch=monkeypatch)
        sent = kinds(calls)
        assert len(sent["bidder"]) == 2 * 3 * 2                                   # no request asks for another bid
        for call in agent_requests(calls):
            text = " ".join(m["content"] for m in call["messages"])
            assert not re.search(r"CORRECTION|BELOW|price set by rule|cost \+ 5%|MUST", text), organization
        for event in events_of(out):
            for supplier_bids in event["bids"].values():
                assert all(b["price"] < 70.0 and "PRICE SET BY RULE" not in b["reasoning"] for b in supplier_bids)
            assert event["awarded"] and event["outcome"]["winning_bid"]["price"] < 70.0   # the winner is paid its bid
            assert event["outcome"]["supplier_profit"] < 0                             # below cost, as submitted
        assert {r["n_rule_priced_bids"] for r in rows_of(out)} == {"0"}
        assert {r["n_tool_fallback_bids"] for r in rows_of(out)} == {"0"}


# ---- 5: the firm with no record is never shown a past event ----

PAST_EVENT_MARKERS = ("YOUR COMPANY'S RECORD OF PAST EVENTS", "previous procurement event", "Result: your company",
                      "your company's costs:", "Another supplier won", "FORWARDED", "BRIEFING FROM", "PLATFORM MEMORY")


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("institution", INSTITUTIONS)
def test_the_firm_that_keeps_no_record_is_never_shown_a_past_event(environment, institution, recorded):
    calls, out = recorded["single_norecord", environment, institution]
    assert kinds(calls)["analyst"] == []
    by_event = {e: {f"{default_bid(e, r, s)['price']:.2f}" for r in range(1, ROUNDS + 1) for s in "AB"}
                for e in range(1, EVENTS + 1)}
    names = [sc.name for sc in get_contract_scenarios()]
    for call in kinds(calls)["bidder"]:
        user = call["messages"][1]["content"]
        event = int(re.match(r"EVENT ([0-9]+) of 6", user).group(1))
        assert not any(marker in user for marker in PAST_EVENT_MARKERS), (event, user[:300])
        assert not any(name in user for name in names)                    # no contract is named
        for earlier in range(1, event):                                    # no price of an earlier event
            assert not any(price in user for price in by_event[earlier]), (event, earlier)
        assert (FIRST_EVENT_NOTE in user) == (event == 1)                  # the first event's line, and only there
        assert ("PAST EVENTS" in user) == (event == 1)


def test_the_platform_remembers_past_events_even_when_the_firm_does_not(recorded):
    calls, _ = recorded["single_norecord", "random", "mediator"]
    memory = [c for c in kinds(calls)["platform"] if "PLATFORM MEMORY" in c["messages"][1]["content"]]
    assert memory                                                         # the rules say the platform remembers


# ---- 6: the analyst that passes the record on ----

@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("institution", INSTITUTIONS)
def test_the_forwarded_record_is_the_single_bidders_record_byte_for_byte_and_no_analyst_is_called(
        environment, institution, recorded):
    single_calls, single_out = recorded["single", environment, institution]
    pass_calls, pass_out = recorded["organized_pass", environment, institution]
    assert kinds(pass_calls)["analyst"] == [] and kinds(single_calls)["analyst"] == []
    assert "analyst" not in run_json(pass_out).get("usage", {}).get("by_agent", {})
    single, forwarded = kinds(single_calls)["bidder"], kinds(pass_calls)["bidder"]
    assert len(single) == len(forwarded) == 2 * EVENTS * ROUNDS
    seen, records = 0, {}
    for index, (one, other) in enumerate(zip(single, forwarded)):
        a, b = one["messages"][1]["content"], other["messages"][1]["content"]
        if a.startswith("EVENT 1 of"):
            assert a == b                                                  # event 1: the same line, the same message
            continue
        record = RECORD_TITLE + a.split(RECORD_TITLE, 1)[1].split("\n\nRULES:", 1)[0]
        assert record.endswith("\n")                                       # the record, with its own closing line
        assert FORWARDED_HEADER + "\n" + record in b
        assert b == a.replace(record, FORWARDED_HEADER + "\n" + record, 1)  # and nothing else differs
        records[index // (2 * ROUNDS) + 1, "AB"[index % 2]] = record
        seen += 1
    assert seen == 2 * (EVENTS - 1) * ROUNDS
    for event_single, event_pass in zip(events_of(single_out), events_of(pass_out)):
        assert event_single["bids"] == event_pass["bids"]                  # the same history, the same plan
        briefing = event_pass["analyst_briefings"]
        assert set(briefing) == (set(SUPPLIER_IDS) if event_pass["event"] >= 2 else set())
        assert event_pass["analyst_postures"] == {sid: None for sid in briefing}
        assert all(text == records[event_pass["event"], sid] for sid, text in briefing.items())   # kept as sent
        assert event_single["analyst_briefings"] == {}


def test_the_forwarded_record_is_kept_in_the_events_file(recorded):
    calls, out = recorded["organized_pass", "random", "mediator"]
    third = events_of(out)[2]
    sent = [c["messages"][1]["content"] for c in kinds(calls)["bidder"] if c["messages"][1]["content"].startswith(
        "EVENT 3 of 6")]
    for sid, text in third["analyst_briefings"].items():
        assert text.startswith(RECORD_TITLE + "\nYour company has participated in 2 previous")
        assert any(FORWARDED_HEADER + "\n" + text in message for message in sent)


# ---- 7: the revenue arms ----

@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_the_revenue_arms_state_the_exact_sentences_and_each_members_objective_to_the_other(environment, recorded):
    bidder_objective = ("Your own objective, as the company's bidder, is the company's revenue: the total price of "
                        "the contracts it wins across the 6 events.")
    analyst_objective = "The analyst's objective is the company's total profit across the 6 events."
    own_analyst = ("Your own objective, as the company's analyst, is the company's total profit across the 6 events. "
                   "The bidder's objective is the company's revenue: the total price of the contracts it wins across "
                   "the 6 events.")
    owner = "Your company's owner wants the company's total profit across the 6 events."
    known = "Everyone in the company knows each member's objective, and who holds which information and who decides."
    team = "Everyone in your company has the same objective: the company's total profit across the 6 events."
    team_known = "Everyone in the company knows this, and knows who holds which information and who decides."
    calls_u, _ = recorded["single_revenue", environment, "mediator"]
    calls_o, _ = recorded["organized_revenue", environment, "mediator"]
    u_prompts = {c["messages"][0]["content"] for c in kinds(calls_u)["bidder"]}
    o_prompts = {c["messages"][0]["content"] for c in kinds(calls_o)["bidder"]}
    analyst_prompts = {c["messages"][0]["content"] for c in kinds(calls_o)["analyst"]}
    assert len(u_prompts) == len(o_prompts) == len(analyst_prompts) == 1
    u, o, analyst = u_prompts.pop(), o_prompts.pop(), analyst_prompts.pop()
    for text in (u, o, analyst):
        assert owner in text and known in text
        assert team not in text and team_known not in text
    assert bidder_objective in u and analyst_objective not in u            # no analyst in this firm
    assert bidder_objective + " " + analyst_objective in o                 # the bidder is told the analyst's objective
    assert own_analyst in analyst and bidder_objective not in analyst      # the analyst is told the bidder's
    assert analyst_objective in o and "revenue" in analyst
    for team_arm in ("single", "single_norecord", "organized", "organized_pass"):
        calls, _ = recorded[team_arm, environment, "mediator"]
        for call in agent_requests(calls):
            text = call["messages"][0]["content"]
            assert team in text and team_known in text and "revenue" not in text and owner not in text


def test_the_revenue_arms_differ_from_the_team_arms_in_those_sentences_only():
    for revenue, team in (("single_revenue", "single"), ("organized_revenue", "organized")):
        for environment in ENVIRONMENTS:
            plain = PP.bidder_prompt(team, 6, environment)
            changed = PP.bidder_prompt(revenue, 6, environment)
            assert plain != changed
            assert changed.replace("Your company's owner wants the company's total profit across the 6 events. "
                                   "Everyone in the company knows each member's objective, and who holds which "
                                   "information and who decides.",
                                   "Everyone in your company has the same objective: the company's total profit "
                                   "across the 6 events. Everyone in the company knows this, and knows who holds which "
                                   "information and who decides.").replace(
                "\n\n" + PP.REVENUE_BIDDER_OBJECTIVE.format(n=6)
                + (" " + PP.ANALYST_PROFIT_OBJECTIVE.format(n=6) if revenue == "organized_revenue" else ""), "") \
                == plain



# ---- the statements about who is shown what, checked against the requests of every cell ----

RECORD_EVENT_LINE = re.compile(
    r"  Event ([0-9]+) \(((?:Speed|Quality) Priority), your company's costs: (speed|quality) specialist\):")


def test_a_contract_is_named_only_in_the_lines_of_a_record_of_a_past_event(recorded):
    """'The contract that is up is not named when the event starts' (random) and 'The contract is not named when the
    first event starts' (persistent): no text of any bidder or analyst names a contract except the lines of a record
    that give a past event's contract."""
    names = [sc.name for sc in get_contract_scenarios()]
    named = 0
    for cell, (calls, _) in recorded.items():
        for call in agent_requests(calls):
            for message in call["messages"]:
                for line in message["content"].splitlines():
                    if any(name in line for name in names):
                        assert RECORD_EVENT_LINE.fullmatch(line), (cell, line)
                        named += 1
    assert named > 500                                                  # and records do name them


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("institution", INSTITUTIONS)
@pytest.mark.parametrize("organization", PRIMITIVES_ANALYST_ORGANIZATIONS)
def test_the_analyst_holds_the_record_with_the_cost_type_and_the_contract_of_each_past_event_and_never_the_coming_ones(
        organization, institution, environment, recorded):
    calls, out = recorded[organization, environment, institution]
    events = events_of(out)
    sent = kinds(calls)
    assert len(sent["analyst"]) == 2 * (EVENTS - 1)
    for index, call in enumerate(sent["analyst"]):
        sid, event = "AB"[index % 2], index // 2 + 2                     # before each event from the second, A then B
        user = call["messages"][1]["content"]
        assert user.startswith(f"EVENT {event} of 6 is next.")
        assert "YOUR PRIVATE COST STRUCTURE" not in user and "In this event your costs" not in user   # no coming costs
        listed = RECORD_EVENT_LINE.findall(user)
        assert listed == [(str(i + 1), events[i]["contract"], events[i]["type_a" if sid == "A" else "type_b"])
                          for i in range(event - 1)]                     # each past event: its contract, its cost type
        assert f"Event {event} (" not in user                             # and not the coming event
    for call in sent["bidder"]:                                          # while the bidder never sees the record
        user = call["messages"][1]["content"]
        assert RECORD_TITLE not in user and not RECORD_EVENT_LINE.search(user)


def test_the_platforms_are_never_sent_a_suppliers_costs_or_cost_type(recorded):
    """'Your company's costs are private.'"""
    for cell, (calls, _) in recorded.items():
        for call in kinds(calls)["platform"]:
            user = call["messages"][1]["content"]
            assert not re.search(r"specialist|PRIVATE COST|Base production cost|total cost", user), cell


def test_every_event_has_two_suppliers_who_bid_in_every_round(recorded):
    """'One other supplier takes part in each event.'"""
    for cell, (calls, out) in recorded.items():
        for event in events_of(out):
            assert set(event["bids"]) == {"A", "B"} and all(len(b) == ROUNDS for b in event["bids"].values())


def test_the_buyer_values_faster_delivery_and_a_higher_tier_in_both_contracts_and_by_different_amounts():
    """'For both contracts the buyer values faster delivery and a higher quality tier; how much it values each differs
    between the two contracts.'"""
    scenarios = get_contract_scenarios()
    assert len(scenarios) == 2
    rates = []
    for scenario in scenarios:
        buyer = scenario.buyer
        slower, faster = ContractTerms(0, 5.0, 3), ContractTerms(0, 4.0, 3)
        lower, higher = ContractTerms(0, 5.0, 3), ContractTerms(0, 5.0, 4)
        week = buyer.value(faster) - buyer.value(slower)                   # the value of a week of faster delivery
        tier = buyer.value(higher) - buyer.value(lower)                    # and of a quality tier
        assert week > 0 and tier > 0
        rates.append((round(week, 6), round(tier, 6)))
    assert rates[0][0] != rates[1][0] and rates[0][1] != rates[1][1]


# ---- 8: mixed markets with the new names, in both orders ----

MIXED = ["single_norecord:single", "single:single_norecord", "organized_pass:single", "single:organized_pass",
         "organized:organized_pass", "organized_pass:organized", "single_revenue:organized_revenue",
         "organized_revenue:single_revenue", "single_norecord:organized_revenue", "organized_revenue:single_norecord"]


def _analyst_in_turn(alike_a, alike_b, a, b):
    analysts_a = alike_a["analyst"][0::2] if a in PRIMITIVES_ANALYST_ORGANIZATIONS else []
    analysts_b = alike_b["analyst"][1::2] if b in PRIMITIVES_ANALYST_ORGANIZATIONS else []
    return [request for turn in itertools.zip_longest(analysts_a, analysts_b) for request in turn
            if request is not None]


@pytest.mark.parametrize("institution", INSTITUTIONS)
@pytest.mark.parametrize("pair", MIXED)
def test_each_mixed_supplier_is_sent_what_the_market_of_its_own_organization_sends(
        pair, institution, tmp_path, monkeypatch):
    a, b = pair.split(":")
    shared = dict(events=3, rounds=2, institution=institution, monkeypatch=monkeypatch)

    def requests(*market):
        out = tmp_path / f"run_{len(list(tmp_path.iterdir()))}"
        calls = run(out, *market, **shared)
        sent = {kind: [c["messages"] for c in found] for kind, found in kinds(calls).items()}
        return sent, out

    mixed, mixed_out = requests("--mixed", pair)
    alike_a, _ = requests("--organizations", a)
    alike_b, _ = requests("--organizations", b)
    assert len(mixed["bidder"]) == 2 * 2 * 3
    assert mixed["bidder"][0::2] == alike_a["bidder"][0::2]                   # what A's bidder is sent
    assert mixed["bidder"][1::2] == alike_b["bidder"][1::2]                   # what B's bidder is sent
    assert {m[0]["content"] for m in mixed["bidder"][0::2]} == {PP.bidder_prompt(a, 3, "random")}
    assert {m[0]["content"] for m in mixed["bidder"][1::2]} == {PP.bidder_prompt(b, 3, "random")}
    in_turn = _analyst_in_turn(alike_a, alike_b, a, b)
    assert mixed["analyst"] == in_turn
    assert len(in_turn) == (2 if a in PRIMITIVES_ANALYST_ORGANIZATIONS else 0) + (
        2 if b in PRIMITIVES_ANALYST_ORGANIZATIONS else 0)
    assert mixed["platform"] == alike_a["platform"] == alike_b["platform"]    # the platforms are told no organization
    text = " ".join(m["content"] for request in mixed["platform"] for m in request)
    assert not re.search(r"organi[sz]|analyst|briefing|\bmemo\b|revenue|record", text, re.I)
    row = rows_of(mixed_out)[0]
    assert (row["organization"], row["organization_a"], row["organization_b"]) == (pair, a, b)
    assert (row["prompt_set"], row["environment"]) == ("primitives", "random")
    settings = run_json(mixed_out)
    assert (settings["mixed"], settings["organizations"], settings["prompts"]) == ([pair], None, "primitives")
    expected = {sid for sid, name in zip(SUPPLIER_IDS, (a, b)) if name in PRIMITIVES_ANALYST_ORGANIZATIONS}
    for event in events_of(mixed_out):
        briefed = expected | ({sid for sid, name in zip(SUPPLIER_IDS, (a, b)) if name == "organized_pass"})
        assert set(event["analyst_briefings"]) == (briefed if event["event"] >= 2 else set())
        assert all(posture is None for posture in event["analyst_postures"].values())


def test_a_mixed_market_pair_and_its_reverse_face_the_same_plan_and_the_same_bids(tmp_path, monkeypatch):
    first, second = tmp_path / "ab", tmp_path / "ba"
    run(first, "--mixed", "single_norecord:single", events=3, rounds=2, monkeypatch=monkeypatch)
    run(second, "--mixed", "single:single_norecord", events=3, rounds=2, monkeypatch=monkeypatch)
    for one, other in zip(events_of(first), events_of(second)):
        keys = ("contract", "type_a", "type_b")
        assert [one[k] for k in keys] == [other[k] for k in keys]


# ---- 9: what the command refuses ----

def _error(argv, capsys):
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2
    return " ".join(capsys.readouterr().err.split())


BASE = ["--backend", "openai", "--model", "m", "--no-tools", "--prompts", "primitives"]


@pytest.mark.parametrize("argv, message", [
    (["--backend", "openai", "--model", "m", "--prompts", "primitives"],
     "--prompts primitives needs --no-tools: its bidders have no cost tool"),
    ([*BASE, "--organizations", "organized_guided"],
     "--organizations names organized_guided, which is not an organization of --prompts primitives (those are "
     "single, single_norecord, organized, organized_pass, single_revenue, organized_revenue)"),
    ([*BASE, "--mixed", "single:organized_guided"],
     "--mixed names organized_guided, which is not an organization of --prompts primitives"),
    (["--backend", "mock", "--organizations", "single_norecord"],
     "--organizations names single_norecord, which is not an organization of --prompts advice (those are single, "
     "organized, organized_guided)"),
    (["--backend", "mock", "--organizations", "single", "organized_pass"],
     "--organizations names organized_pass, which is not an organization of --prompts advice"),
    (["--backend", "mock", "--mixed", "single:single_revenue"],
     "--mixed names single_revenue, which is not an organization of --prompts advice"),
    (["--backend", "mock", "--mixed", "organized_revenue:organized"],
     "--mixed names organized_revenue, which is not an organization of --prompts advice"),
    (["--backend", "mock", "--organizations", "hybrid"], "argument --organizations: invalid choice: 'hybrid'"),
    ([*BASE, "--institutions", "second_price", "--organizations", "single", "--events", "1"],
     "--prompts primitives does not apply to --institutions second_price: its bidder reads a prompt of its own"),
    ([*BASE, "--events", "1"], "--prompts primitives needs --events of at least 2"),
    ([*BASE, "--max-history-events", "3"], "--prompts primitives takes no --max-history-events"),
    (["--backend", "mock", "--environment", "persistent"],
     "--environment persistent needs --prompts primitives: the advice prompts say that a supplier's cost profile is "
     "drawn anew for each event"),
    (["--backend", "mock", "--prompts", "primitives", "--environment", "static"],
     "argument --environment: invalid choice: 'static'"),
    (["--backend", "mock", "--prompts", "terse"], "argument --prompts: invalid choice: 'terse'"),
])
def test_the_command_refuses_what_the_prompt_set_cannot_take(argv, message, tmp_path, capsys):
    assert message in _error([*argv, "--out", str(tmp_path)], capsys)
    assert not (tmp_path / "run.json").exists()                           # refused before anything is written


def test_the_command_has_no_option_for_the_probability_of_the_draw_and_never_overrides_it(
        tmp_path, capsys, monkeypatch):
    """The primitives texts state 1 in 2 and 3 in 4; the plan of a command is always drawn with them."""
    err = _error([*BASE, "--p-different", "0.5", "--out", str(tmp_path / "refused")], capsys)
    assert "unrecognized arguments: --p-different 0.5" in err
    assert not (tmp_path / "refused" / "run.json").exists()
    seen, real = [], demo.pre_draw_event_sequence

    def spy(*args, **kwargs):
        seen.append((kwargs.get("prompt_set"), kwargs.get("p_different"), kwargs.get("environment")))
        return real(*args, **kwargs)

    monkeypatch.setattr(demo, "pre_draw_event_sequence", spy)
    for prompts, environment in (("advice", "random"), ("primitives", "random"), ("primitives", "persistent")):
        argv = ["--backend", "mock", "--prompts", prompts, "--environment", environment, "--organizations", "single",
                "--events", "2", "--rounds", "2", "--out", str(tmp_path / f"{prompts}_{environment}")]
        assert main(argv) == 0
    assert seen == [("advice", None, "random"), ("primitives", None, "random"), ("primitives", None, "persistent")]


def test_a_mixed_market_names_the_primitives_organizations_in_its_message(tmp_path, capsys):
    err = _error(["--backend", "mock", "--mixed", "single", "--out", str(tmp_path)], capsys)
    assert "must be <organization of A>:<organization of B>, each one of single, organized, organized_guided" in err
    assert "or with --prompts primitives one of single, single_norecord, organized, organized_pass, single_revenue, " \
           "organized_revenue" in err


def test_the_primitives_set_with_the_mock_backend_needs_no_tools_flag(tmp_path):
    argv = ["--backend", "mock", "--prompts", "primitives", "--events", "2", "--rounds", "2", "--out", str(tmp_path)]
    assert main(argv) == 0
    assert run_json(tmp_path)["no_tools"] is None                         # the mock has no tools to switch off


def test_new_names_are_accepted_wherever_the_prompt_set_allows_them(tmp_path):
    argv = ["--backend", "mock", "--prompts", "primitives", "--events", "2", "--rounds", "2"]
    assert main([*argv, "--organizations", *NEW, "--out", str(tmp_path / "a")]) == 0
    assert main([*argv, "--mixed", "single_norecord:organized_revenue", "organized_pass:single_revenue",
                 "--out", str(tmp_path / "b")]) == 0
    assert run_json(tmp_path / "a")["organizations"] == NEW


# ---- 10: usage labels ----

def test_usage_is_counted_under_one_label_for_each_bidder_prompt_and_the_analysts_and_nothing_under_other(
        tmp_path, monkeypatch):
    calls = []
    out = tmp_path / "all"
    run(out, calls=calls, events=3, rounds=2, institution=["mediator", "buyer_agent"], monkeypatch=monkeypatch)
    usage = run_json(out)["usage"]
    labels = set(usage["by_agent"])
    assert labels == {f"bidder_{org}" for org in SIX} | {"analyst", "analyst_revenue", "mediator", "buyer_agent"}
    assert "other" not in labels
    for org in SIX:                                           # 2 institutions x 3 events x 2 rounds x 2 suppliers
        assert usage["by_agent"][f"bidder_{org}"]["requests"] == 2 * 3 * 2 * 2
    assert usage["by_agent"]["analyst"]["requests"] == 2 * 2 * 2     # 2 institutions x 2 suppliers x events 2 and 3
    assert usage["by_agent"]["analyst_revenue"]["requests"] == 2 * 2 * 2
    assert usage["by_agent"]["mediator"]["requests"] == 6 * 3 * 2          # 6 markets x 3 events x 1 feedback round x 2
    assert usage["by_agent"]["buyer_agent"]["requests"] == 6 * 3 * 2
    assert usage["requests"] == len(calls)
    assert run_json(out)["organizations"] == SIX                           # no --organizations: all six


def test_each_label_is_the_prompt_it_counts():
    args = SimpleNamespace(prompts="primitives", events=6, environment="persistent")
    labels = demo._agent_labels(args)
    assert labels[PP.bidder_prompt("organized_pass", 6, "persistent")] == "bidder_organized_pass"
    assert labels[PP.analyst_prompt("organized_revenue", 6, "persistent")] == "analyst_revenue"
    assert labels[PP.analyst_prompt("organized", 6, "persistent")] == "analyst"
    assert len(labels) == 6 + 2 + 2 and set(labels.values()) == {f"bidder_{org}" for org in SIX} | {
        "analyst", "analyst_revenue", "mediator", "buyer_agent"}
    advice = demo._agent_labels(SimpleNamespace(prompts="advice"))
    assert demo._agent_labels() == advice and len(advice) == 7 and not set(advice) & set(labels) - {
        P.MEDIATOR_PROMPT, P.BUYER_AGENT_PROMPT}


# ---- 12: a mock run of each new condition ----

@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("organization", SIX)
def test_a_mock_run_of_each_condition_writes_the_files_with_the_settings(organization, environment, tmp_path,
                                                                          capsys):
    argv = ["--backend", "mock", "--prompts", "primitives", "--environment", environment, "--organizations",
            organization, "--events", "4", "--rounds", "2", "--seeds", "1", "2", "--out", str(tmp_path)]
    assert main(argv) == 0
    shown = [line for line in capsys.readouterr().out.splitlines() if line.startswith("    Analyst (")]
    rows, events, settings = rows_of(tmp_path), events_of(tmp_path), run_json(tmp_path)
    assert len(rows) == len(events) == 2 * 2 * 4                              # seeds x institutions x events
    assert reader_columns(tmp_path) == demo.COLUMNS
    assert {(r["prompt_set"], r["environment"], r["organization"]) for r in rows} == {
        ("primitives", environment, organization)}
    assert {(e["prompt_set"], e["environment"]) for e in events} == {("primitives", environment)}
    assert (settings["prompts"], settings["environment"], settings["organizations"], settings["status"]) == (
        "primitives", environment, [organization], "complete")
    assert {r["n_rule_priced_bids"] for r in rows} == {"0"}
    if organization in PRIMITIVES_ANALYST_ORGANIZATIONS:
        assert len(shown) == 2 * 2 * 3 * 2 and all("[" not in line for line in shown)         # no posture is shown
        assert shown[0] == "    Analyst (A): Mock briefing: 1 past event(s) on record...."
        for event in events:
            if event["event"] >= 2:
                text = f"Mock briefing: {event['event'] - 1} past event(s) on record."
                assert event["analyst_briefings"] == {"A": text, "B": text}
            assert all(posture is None for posture in event["analyst_postures"].values())
    elif organization == "organized_pass":
        turn = ["    Analyst (A): forwards the record unchanged", "    Analyst (B): forwards the record unchanged"]
        assert shown == turn * (2 * 2 * 3)
        for event in events:
            assert (set(event["analyst_briefings"]) == set(SUPPLIER_IDS)) == (event["event"] >= 2)
            assert all(text.startswith(RECORD_TITLE) for text in event["analyst_briefings"].values())
    else:
        assert shown == []
        assert all(event["analyst_briefings"] == {} and event["analyst_postures"] == {} for event in events)
    by_seed = {}
    for event in events:
        by_seed.setdefault((event["seed"], event["institution"]), []).append(
            (event["contract"], event["type_a"], event["type_b"]))
    if environment == "persistent":                                  # the same contract and types in every event
        assert all(len(plan) == 4 and len(set(plan)) == 1 for plan in by_seed.values())


def reader_columns(folder):
    with open(Path(folder) / "summary.csv", encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def test_a_persistent_run_repeats_event_1_of_the_random_plan_of_each_seed(tmp_path):
    random_out, persistent_out = tmp_path / "random", tmp_path / "persistent"
    common = ["--backend", "mock", "--prompts", "primitives", "--organizations", "single", "--events", "6",
              "--rounds", "2", "--seeds", "3", "4", "5"]
    assert main([*common, "--environment", "random", "--out", str(random_out)]) == 0
    assert main([*common, "--environment", "persistent", "--out", str(persistent_out)]) == 0
    random_events, persistent_events = events_of(random_out), events_of(persistent_out)
    assert len(random_events) == len(persistent_events) == 3 * 2 * 6
    for seed in (3, 4, 5):
        draws = pre_draw_event_sequence(6, get_contract_scenarios(), seed=seed)
        first = (draws[0][0].name, draws[0][1], draws[0][2])
        for event in persistent_events:
            if event["seed"] == seed:
                assert (event["contract"], event["type_a"], event["type_b"]) == first
        for event in random_events:
            if event["seed"] == seed:
                sc, a, b = draws[event["event"] - 1]
                assert (event["contract"], event["type_a"], event["type_b"]) == (sc.name, a, b)


# ---- what the persistent environment looks like to the bidders ----

def test_in_the_persistent_environment_every_event_has_the_costs_of_event_1_and_says_so(recorded):
    calls, out = recorded["single", "persistent", "mediator"]
    events = events_of(out)
    assert len({(e["contract"], e["type_a"], e["type_b"]) for e in events}) == 1
    plan = pre_draw_event_sequence(EVENTS, get_contract_scenarios(), seed=SEED, environment="persistent")
    assert [e["contract"] for e in events] == [sc.name for sc, _, _ in plan]
    bidders = kinds(calls)["bidder"]
    first_a = bidders[0]["messages"][1]["content"].split("\n\n")[1]
    for k in range(EVENTS):                                                # supplier A's round-1 request of each event
        user = bidders[k * 2 * ROUNDS]["messages"][1]["content"]
        assert user.startswith(f"EVENT {k + 1} of 6\n\n{first_a}\n\n")
    system = bidders[0]["messages"][0]["content"]
    assert "The contract and both cost types then stay the same in all 6 events." in system
    assert "How the events are drawn: before the first event," in system and "independently" not in system
    assert "The contract is not named when the first event starts." in system
    assert "The contract that is up is not named when the event starts." not in system
    calls_random, _ = recorded["single", "random", "mediator"]
    system_random = kinds(calls_random)["bidder"][0]["messages"][0]["content"]
    assert "How the events are drawn: for each event, independently," in system_random
    assert "stay the same" not in system_random
    assert "The contract that is up is not named when the event starts." in system_random
    assert "The contract is not named when the first event starts." not in system_random


# ---- the advice set stays the default ----

def test_giving_the_default_prompts_and_environment_changes_nothing_in_the_requests_or_the_files(tmp_path, monkeypatch):
    shared = ["--backend", "openai", "--model", "m", "--mixed", "single:organized", "organized_guided:single",
              "--events", "3", "--rounds", "3", "--seeds", "1", "2"]
    seen = {}
    for name, extra in (("implicit", []), ("explicit", ["--prompts", "advice", "--environment", "random"])):
        calls = []
        _use_varied_endpoint(monkeypatch, calls)
        out = tmp_path / name
        assert main([*shared, *extra, "--out", str(out)]) == 0
        seen[name] = (_requests_digest(calls), (out / "events.jsonl").read_bytes(), (out / "summary.csv").read_bytes())
    assert seen["implicit"] == seen["explicit"]
    for name in ("implicit", "explicit"):
        out = tmp_path / name
        assert {(r["prompt_set"], r["environment"]) for r in rows_of(out)} == {("advice", "random")}
        assert {(e["prompt_set"], e["environment"]) for e in events_of(out)} == {("advice", "random")}
        settings = run_json(out)
        assert (settings["prompts"], settings["environment"]) == ("advice", "random")
        postures = [e["analyst_postures"] for e in events_of(out) if e["event"] >= 2 and e["analyst_briefings"]]
        assert postures and all(set(p.values()) <= {"GUARDED", "TRUSTING", "DEFENSIVE"} for p in postures)  # advice


def test_the_parser_defaults_are_the_advice_prompts_the_random_environment_and_the_three_organizations():
    parser = demo.build_parser()
    assert (parser.get_default("prompts"), parser.get_default("environment")) == ("advice", "random")
    assert parser.get_default("organizations") is demo.DEFAULT_ORGANIZATIONS
    assert parser.get_default("organizations") == ["single", "organized", "organized_guided"]


# ---- build_agents under the primitives set ----

def test_build_agents_gives_each_supplier_the_primitives_prompt_of_its_organization_and_an_analyst_only_if_it_has_one(
        monkeypatch):
    from orgsim.llm_agents import LocalProcurementAnalyst, PrimitivesSupplierAgent
    from orgsim.mocks import MockProcurementAnalyst
    use_endpoint(monkeypatch, [])
    args = SimpleNamespace(backend="openai", base_url="http://x/v1", api_key=None, model="m", hosted=True,
                           no_tools=True, default_sampling=True, max_tokens=100, max_retries=30, timeout=300,
                           prompts="primitives", events=6, rounds=3, environment="persistent")
    for a, b in itertools.permutations(SIX, 2):
        suppliers, mediator, buyer_agent, analysts = demo.build_agents({"A": a, "B": b}, args, seed=1)
        assert all(isinstance(s, PrimitivesSupplierAgent) and s.n_rounds == 3 and s.use_tools is False
                   for s in suppliers.values())
        assert suppliers["A"].system_prompt == PP.bidder_prompt(a, 6, "persistent")
        assert suppliers["B"].system_prompt == PP.bidder_prompt(b, 6, "persistent")
        expected = {sid for sid, name in zip(SUPPLIER_IDS, (a, b)) if name in PRIMITIVES_ANALYST_ORGANIZATIONS}
        assert (set(analysts) if analysts else set()) == expected
        for sid, analyst in (analysts or {}).items():
            assert isinstance(analyst, LocalProcurementAnalyst)
            assert analyst.system_prompt == PP.analyst_prompt({"A": a, "B": b}[sid], 6, "persistent")
        assert (mediator.system_prompt, buyer_agent.system_prompt) == (P.MEDIATOR_PROMPT, P.BUYER_AGENT_PROMPT)
    mock = SimpleNamespace(backend="mock", prompts="primitives")
    assert set(demo.build_agents({"A": "organized_pass", "B": "organized_revenue"}, mock, seed=1)[3]) == {"B"}
    assert demo.build_agents("organized_pass", mock, seed=1)[3] is None
    assert all(isinstance(a, MockProcurementAnalyst) for a in demo.build_agents("organized", mock, seed=1)[3].values())
    advice = SimpleNamespace(backend="mock")                         # no prompts attribute: the advice set
    assert set(demo.build_agents("organized_guided", advice, seed=1)[3]) == {"A", "B"}


# ---- the help and the README ----

def test_the_help_describes_the_primitives_set_and_the_environment(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = "".join(capsys.readouterr().out.split())        # no spaces: a long option may break at a hyphen
    for part in (
            "--prompts {advice,primitives}", "(default: advice)", "--environment {random,persistent}",
            "(default: random)",
            "primitives (texts that state the objective, the information and the rules and give no advice; it needs "
            "--no-tools with a model, --events of at least 2, and neither --institutions second_price nor "
            "--max-history-events)",
            "persistent (event 1 of the random plan of the seed is repeated at every event); persistent needs "
            "--prompts primitives, because the advice prompts say that a supplier's costs are drawn anew for each "
            "event",
            "With --prompts primitives the organizations are single, single_norecord (the firm keeps no record), "
            "organized, organized_pass (the analyst passes the record to the bidder unchanged, with no model call), "
            "single_revenue and organized_revenue (the bidder is judged on revenue), all of them if none is given",
            "--prompts primitives gives the agents of the AI institutions (mediator and buyer_agent) the texts of "
            "orgsim/primitives_prompts.py in place of the advice prompts",
            "A bid stands as submitted (there is no below-cost correction and no price set by rule, so "
            "n_rule_priced_bids is 0) and no posture is read from a briefing",
            "--environment persistent repeats event 1 of each seed's random plan, the contract and both cost types, at "
            "every event; it needs --prompts primitives",
            "run.json records prompts and environment, and every row of summary.csv and every line of events.jsonl "
            "holds them as prompt_set and environment. The texts state the probabilities of the draw (1 in 2 for the "
            "contract and for each cost type, 3 in 4 for the other supplier's type being the opposite); the command "
            "has no option for them, and the set refuses a plan drawn with others"):
        assert "".join(part.split()) in text, part


def test_the_readme_describes_the_primitives_set_and_its_command_is_accepted_by_the_script():
    import shlex
    readme = (Path(demo.__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    section = readme.split("\n## The primitives prompt set\n", 1)[1].split("\n## ", 1)[0]
    text = " ".join(section.split())
    for part in (
            "`--prompts primitives` gives the agents of the mediator and the buyer agent texts that state primitives "
            "only: the agent's objective, what it knows and does not know, who decides, the payoffs, and the "
            "institution's rules.",
            "The platforms' own prompts (`MEDIATOR_PROMPT` and `BUYER_AGENT_PROMPT`) are the advice set's, unchanged.",
            "`--prompts advice` is the default and sends what it sent before this set existed.",
            "The record of past events that a bidder or an analyst reads is `build_primitives_record` in "
            "`orgsim/sequence.py`: it says \"your company\" in the third person, so that every sentence is true of "
            "the analyst as well as of the bidder, and it calls the platform's answer to a bid a reply, as the rules "
            "do; the advice set's record (`build_supplier_history`) is unchanged.",
            "`persistent` repeats event 1 of the random plan of the same seed, the contract and both cost types, at "
            "every event.",
            "`persistent` needs `--prompts primitives`, because the advice prompts say that a supplier's cost profile "
            "is drawn anew for each event.",
            "A bid stands as submitted: there is no below-cost correction and no price set by rule "
            "(`n_rule_priced_bids` is 0).",
            "A model run needs `--no-tools`.",
            "The mediator's memory of the supplier does not depend on this.",
            "The set's texts state the probabilities of the draw (1 in 2 for the contract and for each supplier's "
            "cost type, 3 in 4 for the other supplier's type being the opposite); the command has no option to "
            "change them, and `pre_draw_event_sequence` and `run_sequence` refuse a plan drawn with another "
            "`p_different` under this set.",
            "With no `--organizations`, a run of the primitives set covers all six."):
        assert part in text, part
    for name in SIX:
        assert f"`{name}`" in text and f"`bidder_{name}`" in text, name
    assert "`analyst_revenue`" in text and "`prompt_set` and `environment`" in text
    blocks = re.findall(r"```bash\n(.*?)```", section, re.S)
    commands = [shlex.split(block.replace(chr(92) + chr(10), " ")) for block in blocks]
    assert len(commands) == 1 and commands[0][:2] == ["python", "scripts/run_demo.py"]
    parser = demo.build_parser()
    args = parser.parse_args(["m" if token == "<model-id>" else token for token in commands[0][2:]])
    demo._check_options(parser, args)                                            # exits with an error if it is refused
    assert (args.prompts, args.environment, args.mixed) == ("primitives", "persistent", [
        "single_norecord:single", "single:single_norecord"])


# ---- the statements of the role blocks and of the record, checked against the requests of every cell ----

@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("institution", INSTITUTIONS)
@pytest.mark.parametrize("organization", PRIMITIVES_ANALYST_ORGANIZATIONS)
def test_the_analyst_reads_the_rules_the_bidder_reads_and_none_of_the_cost_figures_the_bidder_is_shown(
        organization, institution, environment, recorded):
    """'it also reads the rules you read, and it is not shown the cost figures you are shown.' (the organized bidder's
    role block) and 'You are not shown the cost figures the bidder is shown.' (the analyst's)."""
    calls, _ = recorded[organization, environment, institution]
    rules = primitives_rules(INSTITUTION_NAMES[institution], ROUNDS)
    sent = kinds(calls)
    shown_to_bidder = {line.strip() for cost_type in ("speed", "quality") for line in costs_text(cost_type).splitlines()
                       if line.strip()}
    assert len(shown_to_bidder) > 30 and len(sent["analyst"]) == 2 * (EVENTS - 1)
    for call in sent["analyst"]:
        everything = "\n".join(message["content"] for message in call["messages"])
        assert rules in call["messages"][1]["content"]
        assert "+$" not in everything and not any(line in everything for line in shown_to_bidder)
    for call in sent["bidder"]:
        assert rules in call["messages"][1]["content"]
        assert any(line in call["messages"][1]["content"] for line in shown_to_bidder)


def test_the_record_that_a_bidder_or_an_analyst_reads_speaks_of_the_company_in_the_third_person(recorded):
    """The record is read by the single firm's bidder, the forwarded one and the analyst: 'your company' in every line,
    never 'you', so that it is true for each of them."""
    seen = 0
    for cell, (calls, _) in recorded.items():
        for call in agent_requests(calls):
            user = call["messages"][1]["content"]
            if RECORD_TITLE not in user:
                continue
            record = RECORD_TITLE + user.split(RECORD_TITLE, 1)[1].split("\n\nRULES:", 1)[0].split(
                "\n\nWrite your briefing", 1)[0]
            assert not re.search(r"\b(?:you|your(?! company)|yours|yourself)\b", record, re.I), (cell, record[:300])
            assert "Your company bid $" in record and "Result: " in record and "Platform's reply" in record
            seen += 1
    assert seen > 300                      # the single and the forwarded bidders' messages, and the analysts'
