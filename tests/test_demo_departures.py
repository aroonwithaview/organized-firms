"""The departures organizations through the command, with a fake chat endpoint (no network): what every agent is sent in
the first two events of every organization, in both institutions and the persistent environment; the identity of the
first events within an objective; the forwarded record; the cost-seeing analyst; the usage labels; what the command
refuses; the mixed markets; a mock run of each."""
import itertools
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import orgsim.llm
import scripts.run_demo as demo
from orgsim import primitives_prompts as PP
from orgsim import prompts as P
from orgsim.institutions import primitives_rules
from orgsim.interfaces import FLOOR_PRICE_TAG
from orgsim.llm import ChatClient
from orgsim.organization import (
    ALL_PRIMITIVES_ANALYST_ORGANIZATIONS, ALL_PRIMITIVES_ORGANIZATIONS, DEPARTURE_ANALYST_ORGANIZATIONS,
    DEPARTURE_BASES, DEPARTURE_ORGANIZATIONS, FIRST_EVENT_NOTE, departure_base,
)
from orgsim.sequence import make_supplier
from scripts.run_demo import main
from test_demo_primitives import (
    FORBIDDEN, INSTITUTION_NAMES, INSTITUTIONS, PLATFORM_REPLIES, _pick, default_bid, digest, events_of,
    previous_rounds, record_after_event_1, rows_of, run_json,
)
from test_departures import (
    COST_HEADER_BIDDER, COST_HEADER_ORIGINAL, HEADER_BRIEF, HEADER_CAP, HEADER_FLOOR, HEADER_FORWARD, NOTICE,
    NOTICE_NOT_SET, RAISED_NOTE, REASK_ANALYST, REASK_BIDDER, cap_reask, expected_analyst_block,
    expected_analyst_prompt, expected_bidder_prompt,
)

EVENTS, ROUNDS = 6, 3
# Seed 23 draws Quality Priority with A a quality specialist and B a speed specialist, and in the persistent environment
# every event is event 1.
SEED = 23
ENVIRONMENT = "persistent"
FAKE_BRIEFING = "Fake briefing after 1 past event(s)."
BASES = DEPARTURE_BASES
ROUND_LINE = 'ROUND {round} of 3: Submit your bid as JSON. Keep its "reasoning" field to 1-2 sentences.'


# ---- a fake endpoint that tells the role of a request by its system prompt, and answers a second request ----

class DepartureEndpoint:
    """An OpenAI-style client for a run of `events` events of `rounds` rounds in `environment`. A bidder's first request
    in a round is the bid of one supplier (A then B in each round, as the institutions ask), which `bid(event, round,
    supplier)` gives; a request that carries earlier turns (more than a system and a user message) is a follow-up to
    the one before, and `resend(event, round, supplier)` gives the answer (the same bid again if none is given). An
    analyst answers with `briefing(event, supplier, attempt, user)` (attempt 1, or 2 for a follow-up), a platform with
    one of three short texts picked by its request. `analysts` lists the suppliers that have an analyst, in the order
    in which the runner calls them. Every request goes to `calls`."""

    def __init__(self, calls, events=EVENTS, rounds=ROUNDS, environment=ENVIRONMENT, analysts=("A", "B"),
                 bid=default_bid, resend=None, briefing=None):
        self.calls, self.analysts = calls, tuple(analysts)
        self.bid, self.resend = bid, resend or bid
        self.briefing = briefing or (
            lambda event, sid, attempt, user: f"Fake briefing after {event - 1} past event(s).")
        self.first_bids = self.first_briefings = 0
        self.roles = {PP.bidder_prompt(org, events, environment): "bidder" for org in ALL_PRIMITIVES_ORGANIZATIONS}
        self.roles.update({PP.analyst_prompt(org, events, environment): "analyst"
                           for org in ALL_PRIMITIVES_ANALYST_ORGANIZATIONS})
        self.roles.update({P.MEDIATOR_PROMPT: "platform", P.BUYER_AGENT_PROMPT: "platform"})
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        request = json.loads(json.dumps(kw, default=str))
        self.calls.append(request)
        messages = request["messages"]
        role = self.roles[messages[0]["content"]]                # an unexpected system prompt is a KeyError
        user, follow_up = messages[1]["content"], len(messages) > 2
        if role == "bidder":
            self.first_bids += 0 if follow_up else 1
            sid = "AB"[(self.first_bids - 1) % 2]
            event, round_num = (int(re.match(r"EVENT ([0-9]+) of", user).group(1)),
                                int(re.search(r"ROUND ([0-9]+) of", user).group(1)))
            content = json.dumps((self.resend if follow_up else self.bid)(event, round_num, sid))
        elif role == "analyst":
            self.first_briefings += 0 if follow_up else 1
            sid = self.analysts[(self.first_briefings - 1) % len(self.analysts)]
            content = self.briefing(int(re.match(r"EVENT ([0-9]+) of", user).group(1)), sid,
                                    2 if follow_up else 1, user)
        else:
            content = _pick(PLATFORM_REPLIES, user)
        message = SimpleNamespace(content=content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10))


def plain_briefing(market):
    """The analysts' answers in the plain scenario: a short briefing, and for the analyst of a floor organization a
    floor line as well: $5.00 for supplier A and none for supplier B."""
    def briefing(event, sid, attempt, user):
        text = f"Fake briefing after {event - 1} past event(s)."
        if departure_base(market[sid]) == "floor":
            text += "\nMINIMUM PROFIT PER CONTRACT: " + ("$5.00" if sid == "A" else "none")
        return text
    return briefing


def plain_notice(sid):
    return NOTICE + ("$5.00" if sid == "A" else "none")


def run_departures(out, *market, calls=None, events=EVENTS, rounds=ROUNDS, environment=ENVIRONMENT,
                   institution="mediator", seeds=(str(SEED),), monkeypatch=None, **endpoint):
    """Run the command with the fake endpoint; returns the requests it sent, in order. `endpoint` gives the options
    of DepartureEndpoint (the suppliers with an analyst and what the fake answers)."""
    calls = [] if calls is None else calls
    with pytest.MonkeyPatch.context() as patch:
        def make_client(base_url, api_key=None, local=True, **client_options):
            for name in ("max_retries", "timeout"):
                client_options.pop(name, None)
            return ChatClient(
                DepartureEndpoint(calls, events=events, rounds=rounds, environment=environment, **endpoint),
                local=local, **client_options)

        (monkeypatch or patch).setattr(orgsim.llm, "make_client", make_client)
        argv = ["--backend", "openai", "--model", "m", "--no-tools", "--prompts", "primitives", "--environment",
                environment, "--institutions", *(institution if isinstance(institution, list) else [institution]),
                *market, "--events", str(events), "--rounds", str(rounds), "--seeds", *seeds, "--out", str(out)]
        assert main(argv) == 0
    return calls


def kinds(calls):
    """The requests by who sent them: bidders', analysts' and platforms'."""
    found = {"bidder": [], "analyst": [], "platform": []}
    for call in calls:
        system = call["messages"][0]["content"]
        if system in (P.MEDIATOR_PROMPT, P.BUYER_AGENT_PROMPT):
            found["platform"].append(call)
        elif system.endswith(PP.BID_OUTPUT):
            found["bidder"].append(call)
        else:
            assert "You are your company's procurement analyst." in system, system[:100]
            found["analyst"].append(call)
    return found


def agent_requests(calls):
    sent = kinds(calls)
    wanted = {id(call) for call in sent["bidder"] + sent["analyst"]}
    return [call for call in calls if id(call) in wanted]


def first_two_events(calls):
    """The messages of the requests of the first and the second event: each bidder's, in each round, and each analyst's
    before the second event."""
    return [call["messages"] for call in agent_requests(calls)
            if re.match(r"EVENT [12] of 6", call["messages"][1]["content"])]


def event_requests(calls, event):
    """The messages of the bidders' requests of one event."""
    return [call["messages"] for call in kinds(calls)["bidder"]
            if call["messages"][1]["content"].startswith(f"EVENT {event} of 6")]


# ---- the exact requests of events 1 and 2, written out from the design ----

def costs_text(cost_type, org, event):
    """The bidder's cost description: its title is the usual one, except for the bidder of brief_cost from event 2."""
    text = make_supplier("A", cost_type).describe_for_primitives(cost_type)
    if event >= 2 and departure_base(org) == "brief_cost":
        assert text.count(COST_HEADER_ORIGINAL) == 1
        text = text.replace(COST_HEADER_ORIGINAL, COST_HEADER_BIDDER)
    return text


def expected_analyst_message(org, sid, events, rules):
    """What the analyst of `org` is sent before event 2."""
    own_type = events[1]["type_a" if sid == "A" else "type_b"]
    shown = ""
    if departure_base(org) == "brief_cost":
        shown = (f"In the coming event your company's costs are those of a {own_type} specialist.\n\n"
                 + expected_analyst_block(own_type) + "\n\n")
    return ("EVENT 2 of 6 is next. Your company's bidder reads these rules with each event:\n" + rules
            + "\n\nHere is your company's record of past events on this platform:\n"
            + record_after_event_1(events[0], sid) + "\n\n" + shown + "Write your briefing for your company's bidder.")


def expected_agent_messages(market, institution, out):
    """Every bidder and analyst request of events 1 and 2 of a cell, in order, as the design writes them. `market`
    maps each supplier to its organization."""
    events = events_of(out)
    rules = primitives_rules(INSTITUTION_NAMES[institution], ROUNDS)
    expected = []
    for event in (1, 2):
        data = events[event - 1]
        if event == 2:
            for sid in "AB":
                if market[sid] in ALL_PRIMITIVES_ANALYST_ORGANIZATIONS:
                    expected.append([
                        {"role": "system", "content": expected_analyst_prompt(market[sid], ENVIRONMENT)},
                        {"role": "user", "content": expected_analyst_message(market[sid], sid, events, rules)}])
        for round_num in (1, 2, 3):
            for sid in "AB":
                org = market[sid]
                own_type = data["type_a" if sid == "A" else "type_b"]
                if event == 1:
                    past = FIRST_EVENT_NOTE
                elif departure_base(org) == "forward":
                    past = HEADER_FORWARD + "\n" + record_after_event_1(events[0], sid)
                elif departure_base(org) == "brief_cap":
                    past = HEADER_CAP + "\n" + FAKE_BRIEFING
                elif departure_base(org) == "floor":
                    past = HEADER_FLOOR + "\n" + FAKE_BRIEFING + "\n\n" + plain_notice(sid)
                else:
                    past = HEADER_BRIEF + "\n" + FAKE_BRIEFING
                user = "\n\n".join([f"EVENT {event} of 6", costs_text(own_type, org, event), past, rules])
                user += "\n\n" + previous_rounds(data, sid, round_num) + ROUND_LINE.format(round=round_num)
                expected.append([{"role": "system", "content": expected_bidder_prompt(org, ENVIRONMENT)},
                                 {"role": "user", "content": user}])
    return expected


def analysts_of(market):
    return tuple(sid for sid in "AB" if market[sid] in ALL_PRIMITIVES_ANALYST_ORGANIZATIONS)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """{(organization, institution): (requests, folder)} for a homogeneous market of every departures organization."""
    found = {}
    for organization, institution in itertools.product(DEPARTURE_ORGANIZATIONS, INSTITUTIONS):
        out = tmp_path_factory.mktemp(f"{organization}_{institution}")
        found[organization, institution] = (
            run_departures(out, "--organizations", organization, institution=institution,
                           analysts=analysts_of({"A": organization, "B": organization}),
                           briefing=plain_briefing({"A": organization, "B": organization})), out)
    return found


def test_each_cell_sent_the_requests_the_design_counts(recorded):
    for (organization, institution), (calls, out) in recorded.items():
        sent = kinds(calls)
        assert len(sent["bidder"]) == 2 * EVENTS * ROUNDS
        analysts = 2 * (EVENTS - 1) if organization in DEPARTURE_ANALYST_ORGANIZATIONS else 0
        assert len(sent["analyst"]) == analysts, (organization, institution)
        assert len(sent["platform"]) == 2 * EVENTS * (ROUNDS - 1)
        assert len(events_of(out)) == EVENTS


@pytest.mark.parametrize("institution", INSTITUTIONS)
@pytest.mark.parametrize("organization", DEPARTURE_ORGANIZATIONS)
def test_the_requests_of_events_1_and_2_are_exactly_what_the_design_writes(organization, institution, recorded):
    calls, out = recorded[organization, institution]
    assert first_two_events(calls) == expected_agent_messages({"A": organization, "B": organization}, institution, out)


def test_the_requests_of_events_1_and_2_of_every_departures_organization_are_pinned_by_digest(recorded):
    found = {f"{organization} {institution}": digest(first_two_events(calls))
             for (organization, institution), (calls, _) in recorded.items()}
    assert found == EXPECTED_DEPARTURE_REQUEST_DIGESTS


EXPECTED_DEPARTURE_REQUEST_DIGESTS = {
    "brief mediator": "5be284d31880a8ff",
    "brief buyer_agent": "d59e5237e12df1ec",
    "forward mediator": "6e7fe8417864faff",
    "forward buyer_agent": "461bcbb5ddcf5c15",
    "floor mediator": "6b6408570008dc05",
    "floor buyer_agent": "97ca5da6537e0833",
    "brief_cap mediator": "097b25df832d0aea",
    "brief_cap buyer_agent": "3df3f26553aef92f",
    "brief_cost mediator": "8b67d8db838f6f85",
    "brief_cost buyer_agent": "c3e74b15bb796fc5",
    "brief_revenue mediator": "24df4eac74b946f5",
    "brief_revenue buyer_agent": "882e299768738d3a",
    "forward_revenue mediator": "2bac24deae09625d",
    "forward_revenue buyer_agent": "8a708ab776165264",
    "floor_revenue mediator": "2e99a91267530cbe",
    "floor_revenue buyer_agent": "7040e23cca84505d",
    "brief_cap_revenue mediator": "741624bc0ec56b20",
    "brief_cap_revenue buyer_agent": "3d814fada1fba8f2",
    "brief_cost_revenue mediator": "afb2918e782e3e5a",
    "brief_cost_revenue buyer_agent": "c1fdf74701c9e656",
}


# ---- the first event is the same in brief, forward and brief_cost, within an objective ----

@pytest.mark.parametrize("suffix", ["", "_revenue"])
@pytest.mark.parametrize("institution", INSTITUTIONS)
def test_the_system_prompt_and_the_requests_of_event_1_are_identical_across_the_organizations_with_the_shared_text(
        suffix, institution, recorded):
    shared = [base for base in BASES if base != "floor"]
    first = {base: event_requests(recorded[base + suffix, institution][0], 1) for base in shared}
    assert len(first["brief"]) == 2 * ROUNDS
    assert all(requests == first["brief"] for requests in first.values())          # system and user, byte for byte
    assert {tuple(m[0]["content"] for m in requests) for requests in first.values()} == {
        (PP.bidder_prompt("brief" + suffix, EVENTS, ENVIRONMENT),) * (2 * ROUNDS)}
    later = {base: event_requests(recorded[base + suffix, institution][0], 2) for base in shared}
    # the message starts at event 2
    assert later["brief"] != later["forward"] and later["brief"] != later["brief_cost"]


@pytest.mark.parametrize("institution", INSTITUTIONS)
def test_the_two_objectives_differ_in_the_system_prompt_and_nowhere_in_the_first_event_user_message(
        institution, recorded):
    team, conflict = (event_requests(recorded["brief" + suffix, institution][0], 1) for suffix in ("", "_revenue"))
    assert [m[1] for m in team] == [m[1] for m in conflict]
    assert [m[0] for m in team] != [m[0] for m in conflict]


# ---- forward: the record, unchanged, and no analyst ----

@pytest.mark.parametrize("institution", INSTITUTIONS)
@pytest.mark.parametrize("organization", ["forward", "forward_revenue"])
def test_the_forwarded_record_is_the_single_bidders_record_byte_for_byte_and_no_analyst_is_called(
        organization, institution, recorded, tmp_path, monkeypatch):
    calls, out = recorded[organization, institution]
    single_org = "single_revenue" if organization.endswith("_revenue") else "single"
    single_calls = run_departures(tmp_path / "single", "--organizations", single_org, institution=institution,
                                  monkeypatch=monkeypatch)
    assert kinds(calls)["analyst"] == [] and kinds(single_calls)["analyst"] == []
    assert "analyst" not in " ".join(run_json(out).get("usage", {}).get("by_agent", {}))
    forwarded, single = kinds(calls)["bidder"], kinds(single_calls)["bidder"]
    assert len(forwarded) == len(single) == 2 * EVENTS * ROUNDS
    seen = 0
    title = "YOUR COMPANY'S RECORD OF PAST EVENTS ON THIS PLATFORM:"
    for one, other in zip(single, forwarded):
        a, b = one["messages"][1]["content"], other["messages"][1]["content"]
        if a.startswith("EVENT 1 of"):
            assert a == b
            continue
        record = title + a.split(title, 1)[1].split("\n\nRULES:", 1)[0]
        assert record.endswith("\n")
        assert HEADER_FORWARD + "\n" + record in b
        assert b == a.replace(record, HEADER_FORWARD + "\n" + record, 1)           # and nothing else differs
        seen += 1
    assert seen == 2 * (EVENTS - 1) * ROUNDS
    for event in events_of(out):
        sent = event["analyst_briefings"]
        assert set(sent) == (set("AB") if event["event"] >= 2 else set()) and event["analyst_postures"] == {
            sid: None for sid in sent}
        assert all(text.startswith(title) for text in sent.values())              # the record, kept as it was sent


# ---- brief_cost: the figures of the analyst are the bidder's, and the bidder's title changes from event 2 ----

@pytest.fixture(scope="module")
def recorded_random(tmp_path_factory):
    """{(organization, institution): (requests, folder)} for brief_cost and its conflict variant in the random
    environment, where the cost types of the events differ (in the persistent environment every event has the types of
    the first)."""
    found = {}
    for organization, institution in itertools.product(("brief_cost", "brief_cost_revenue"), INSTITUTIONS):
        out = tmp_path_factory.mktemp(f"random_{organization}_{institution}")
        market = {"A": organization, "B": organization}
        found[organization, institution] = (
            run_departures(out, "--organizations", organization, institution=institution, environment="random",
                           analysts=analysts_of(market), briefing=plain_briefing(market)), out)
    return found


@pytest.mark.parametrize("institution", INSTITUTIONS)
@pytest.mark.parametrize("organization", ["brief_cost", "brief_cost_revenue"])
def test_the_analysts_cost_figures_are_the_bidders_for_the_same_event_and_type(organization, institution, recorded):
    check_the_analysts_cost_figures(*recorded[organization, institution])


@pytest.mark.parametrize("institution", INSTITUTIONS)
@pytest.mark.parametrize("organization", ["brief_cost", "brief_cost_revenue"])
def test_the_analysts_cost_figures_are_the_bidders_for_the_same_event_and_type_when_the_types_change_between_events(
        organization, institution, recorded_random):
    calls, out = recorded_random[organization, institution]
    events = events_of(out)
    # the types of the events differ, so figures taken from another event cannot pass for the right ones
    assert len({event["type_a"] for event in events}) == len({event["type_b"] for event in events}) == 2
    check_the_analysts_cost_figures(calls, out)


def check_the_analysts_cost_figures(calls, out):
    """The cost block that each analyst of brief_cost is sent is the third-person structure of the cost type of the
    coming event of its supplier, and its figures are those of the structure that the bidder reads in the same event."""
    events = events_of(out)
    sent = kinds(calls)
    assert len(sent["analyst"]) == 2 * (EVENTS - 1)
    numbers = r"[0-9]+(?:[.][0-9]+)?"
    for index, call in enumerate(sent["analyst"]):
        sid, event = "AB"[index % 2], index // 2 + 2
        user = call["messages"][1]["content"]
        own_type = events[event - 1]["type_a" if sid == "A" else "type_b"]
        assert f"In the coming event your company's costs are those of a {own_type} specialist." in user
        block = user[user.index("YOUR COMPANY'S COST STRUCTURE"):].split("\n\nWrite your briefing", 1)[0]
        bidder = [c for c in sent["bidder"] if c["messages"][1]["content"].startswith(f"EVENT {event} of 6")][
            0 if sid == "A" else 1]["messages"][1]["content"]
        shown = bidder[bidder.index(COST_HEADER_BIDDER):].split("\n\nMESSAGE FROM", 1)[0]
        assert re.findall(numbers, block) == re.findall(numbers, shown), (sid, event)
        assert block == expected_analyst_block(own_type)
    for event in range(1, EVENTS + 1):
        for position, messages in enumerate(event_requests(calls, event)):
            user = messages[1]["content"]
            assert (COST_HEADER_BIDDER in user) == (event >= 2) and (COST_HEADER_ORIGINAL in user) == (event == 1)
            own_type = events[event - 1]["type_a" if position % 2 == 0 else "type_b"]
            assert f"In this event your costs are those of a {own_type} specialist." in user


@pytest.mark.parametrize("organization",
                         [name for name in DEPARTURE_ORGANIZATIONS if departure_base(name) != "brief_cost"])
def test_the_title_of_the_cost_structure_does_not_change_in_the_other_organizations(organization, recorded):
    calls, _ = recorded[organization, "mediator"]
    for event in range(1, EVENTS + 1):
        for messages in event_requests(calls, event):
            user = messages[1]["content"]
            assert COST_HEADER_ORIGINAL in user and COST_HEADER_BIDDER not in user


def test_only_the_analyst_of_brief_cost_is_shown_a_cost_structure(recorded):
    for (organization, institution), (calls, _) in recorded.items():
        for call in kinds(calls)["analyst"]:
            text = " ".join(m["content"] for m in call["messages"])
            assert ("COST STRUCTURE" in text) == (departure_base(organization) == "brief_cost"), organization
            assert "+$" in text or departure_base(organization) != "brief_cost"
            assert ("Base production cost" in text) == (departure_base(organization) == "brief_cost")


# ---- usage labels ----

def expected_labels(market, institution):
    labels = {"mediator" if institution == "mediator" else "buyer_agent"}
    for sid in "AB":
        labels.add(f"bidder_{market[sid]}")
        if market[sid] in DEPARTURE_ANALYST_ORGANIZATIONS:
            labels.add(f"analyst_{market[sid]}")
    return labels


@pytest.mark.parametrize("institution", INSTITUTIONS)
@pytest.mark.parametrize("organization", DEPARTURE_ORGANIZATIONS)
def test_usage_is_counted_under_a_label_of_each_organization_and_nothing_under_other(
        organization, institution, recorded):
    calls, out = recorded[organization, institution]
    usage = run_json(out)["usage"]
    market = {"A": organization, "B": organization}
    assert set(usage["by_agent"]) == expected_labels(market, institution)
    assert "other" not in usage["by_agent"]
    assert usage["by_agent"][f"bidder_{organization}"]["requests"] == 2 * EVENTS * ROUNDS
    if organization in DEPARTURE_ANALYST_ORGANIZATIONS:
        assert usage["by_agent"][f"analyst_{organization}"]["requests"] == 2 * (EVENTS - 1)
    assert usage["requests"] == len(calls)


def test_brief_and_forward_are_counted_apart_though_their_bidders_read_the_same_prompt(tmp_path, monkeypatch):
    assert PP.bidder_prompt("brief", EVENTS, ENVIRONMENT) == PP.bidder_prompt("forward", EVENTS, ENVIRONMENT)
    calls = run_departures(tmp_path, "--organizations", "brief", "forward", monkeypatch=monkeypatch)
    usage = run_json(tmp_path)["usage"]["by_agent"]
    assert {"bidder_brief", "bidder_forward", "analyst_brief"} <= set(usage) and "other" not in usage
    assert usage["bidder_brief"]["requests"] == usage["bidder_forward"]["requests"] == 2 * EVENTS * ROUNDS
    assert usage["analyst_brief"]["requests"] == 2 * (EVENTS - 1)
    assert "analyst_forward" not in usage and "analyst" not in usage


def test_an_agent_that_holds_a_labeled_view_of_a_client_is_counted_in_the_client_behind_the_view():
    root = ChatClient(DepartureEndpoint([]))
    labeled = SimpleNamespace(client=root.labeled("bidder_brief"))
    plain = SimpleNamespace(client=root)
    assert list(demo._clients_of(labeled).values()) == [root]
    assert list(demo._clients_of(labeled, plain).values()) == [root]         # one client, not two
    assert demo._clients_of(SimpleNamespace(client=None), SimpleNamespace()) == {}


def test_the_earlier_organizations_are_still_counted_by_their_prompts_and_the_labels_by_prompt_are_unchanged(
        tmp_path, monkeypatch):
    args = SimpleNamespace(prompts="primitives", events=6, environment="persistent")
    labels = demo._agent_labels(args)
    assert len(labels) == 6 + 2 + 2
    assert not any(label.endswith(tuple(DEPARTURE_ORGANIZATIONS)) for label in labels.values())
    run_departures(tmp_path, "--organizations", "organized", "brief", monkeypatch=monkeypatch)
    usage = run_json(tmp_path)["usage"]["by_agent"]
    assert set(usage) == {"bidder_organized", "analyst", "bidder_brief", "analyst_brief", "mediator"}
    assert "other" not in usage
    assert usage["analyst"]["requests"] == usage["analyst_brief"]["requests"] == 2 * (EVENTS - 1)


# ---- what the command accepts and refuses ----

def _error(argv, capsys):
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2
    return " ".join(capsys.readouterr().err.split())


@pytest.mark.parametrize("name", DEPARTURE_ORGANIZATIONS)
def test_the_advice_prompts_refuse_every_departures_organization(name, tmp_path, capsys):
    advice = ["--backend", "mock", "--prompts", "advice", "--out", str(tmp_path)]
    assert (f"--organizations names {name}, which is not an organization of --prompts advice (those are single, "
            "organized, organized_guided)") in _error([*advice, "--organizations", name], capsys)
    assert f"--mixed names {name}, which is not an organization of --prompts advice" in _error(
        [*advice, "--mixed", f"single:{name}"], capsys)
    assert f"--mixed names {name}, which is not an organization of --prompts advice" in _error(
        [*advice, "--mixed", f"{name}:organized"], capsys)
    assert not (tmp_path / "run.json").exists()


def test_the_primitives_set_names_its_departures_organizations_where_it_refuses_another(tmp_path, capsys):
    err = _error(["--backend", "mock", "--prompts", "primitives", "--organizations", "organized_guided", "--out",
                  str(tmp_path)], capsys)
    assert ("--organizations names organized_guided, which is not an organization of --prompts primitives (those are "
            "single, single_norecord, organized, organized_pass, single_revenue, organized_revenue)") in err
    assert f"its departures organizations are {', '.join(DEPARTURE_ORGANIZATIONS)}" in err
    err = _error(["--backend", "mock", "--mixed", "single", "--out", str(tmp_path)], capsys)
    assert ("or with --prompts primitives one of single, single_norecord, organized, organized_pass, single_revenue, "
            f"organized_revenue, {', '.join(DEPARTURE_ORGANIZATIONS)}") in err


def test_a_run_of_the_primitives_set_that_names_no_organization_still_covers_the_six(tmp_path):
    argv = ["--backend", "mock", "--prompts", "primitives", "--events", "2", "--rounds", "2", "--out", str(tmp_path)]
    assert main(argv) == 0
    assert run_json(tmp_path)["organizations"] == ["single", "single_norecord", "organized", "organized_pass",
                                                   "single_revenue", "organized_revenue"]


# ---- mixed markets ----

def mixed_pairs():
    return [f"{a}:{b}" for a, b in itertools.permutations(DEPARTURE_ORGANIZATIONS, 2)]


@pytest.mark.parametrize("pair", mixed_pairs())
def test_a_mock_run_of_every_ordered_pair_of_the_new_organizations_writes_the_files(pair, tmp_path):
    argv = ["--backend", "mock", "--prompts", "primitives", "--environment", ENVIRONMENT, "--mixed", pair, "--events",
            "3", "--rounds", "2", "--seeds", str(SEED), "--out", str(tmp_path)]
    assert main(argv) == 0
    first, second = pair.split(":")
    rows, events, settings = rows_of(tmp_path), events_of(tmp_path), run_json(tmp_path)
    assert len(rows) == len(events) == 2 * 3                                            # institutions x events
    assert {(r["organization"], r["organization_a"], r["organization_b"]) for r in rows} == {(pair, first, second)}
    assert (settings["mixed"], settings["organizations"], settings["status"]) == ([pair], None, "complete")
    for event in events:
        briefed = {sid for sid, name in zip("AB", (first, second)) if name in ALL_PRIMITIVES_ANALYST_ORGANIZATIONS
                   or departure_base(name) == "forward"}
        assert set(event["analyst_briefings"]) == (briefed if event["event"] >= 2 else set())


MIXED_CELLS = ["brief:forward", "forward:brief", "brief_revenue:forward_revenue", "forward_revenue:brief_revenue",
               "floor_revenue:brief_revenue", "brief_revenue:floor_revenue"]


@pytest.mark.parametrize("institution", INSTITUTIONS)
@pytest.mark.parametrize("pair", MIXED_CELLS)
def test_each_mixed_supplier_is_sent_what_the_market_of_its_own_organization_sends(pair, institution, tmp_path,
                                                                                    monkeypatch):
    a, b = pair.split(":")
    market = {"A": a, "B": b}
    mixed = run_departures(tmp_path / "mixed", "--mixed", pair, institution=institution, monkeypatch=monkeypatch,
                           analysts=analysts_of(market), briefing=plain_briefing(market))
    assert first_two_events(mixed) == expected_agent_messages(market, institution, tmp_path / "mixed")
    alike = {name: run_departures(tmp_path / name, "--organizations", name, institution=institution,
                                  monkeypatch=monkeypatch, analysts=analysts_of({"A": name, "B": name}),
                                  briefing=plain_briefing({"A": name, "B": name}))
             for name in (a, b)}
    sent = {kind: [c["messages"] for c in found] for kind, found in kinds(mixed).items()}
    each = {name: {kind: [c["messages"] for c in found] for kind, found in kinds(calls).items()}
            for name, calls in alike.items()}
    assert sent["bidder"][0::2] == each[a]["bidder"][0::2] and sent["bidder"][1::2] == each[b]["bidder"][1::2]
    assert sent["platform"] == each[a]["platform"] == each[b]["platform"]       # the platforms are told no organization
    text = " ".join(m["content"] for request in sent["platform"] for m in request)
    assert not re.search(r"organi[sz]|analyst|briefing|\bmemo\b|revenue|record", text, re.I)
    usage = run_json(tmp_path / "mixed")["usage"]["by_agent"]
    assert set(usage) == expected_labels(market, institution) and "other" not in usage


def test_a_mixed_market_of_a_departures_organization_and_an_earlier_one_works_in_both_orders(tmp_path, monkeypatch):
    for pair in ("brief:organized", "organized:brief", "forward:single", "single:forward_revenue"):
        a, b = pair.split(":")
        market = {"A": a, "B": b}
        calls = run_departures(tmp_path / pair.replace(":", "_"), "--mixed", pair, monkeypatch=monkeypatch,
                               analysts=analysts_of(market))
        assert len(kinds(calls)["bidder"]) == 2 * EVENTS * ROUNDS
        assert {call["messages"][0]["content"] for call in kinds(calls)["bidder"]} == {
            PP.bidder_prompt(a, EVENTS, ENVIRONMENT), PP.bidder_prompt(b, EVENTS, ENVIRONMENT)}


# ---- a mock run of each organization ----

@pytest.mark.parametrize("organization", DEPARTURE_ORGANIZATIONS)
def test_a_mock_run_of_each_departures_organization_writes_the_files_with_the_settings(organization, tmp_path, capsys):
    argv = ["--backend", "mock", "--prompts", "primitives", "--environment", ENVIRONMENT, "--organizations",
            organization, "--events", "4", "--rounds", "2", "--seeds", "1", "2", "--out", str(tmp_path)]
    assert main(argv) == 0
    shown = [line for line in capsys.readouterr().out.splitlines() if line.startswith("    Analyst (")]
    rows, events, settings = rows_of(tmp_path), events_of(tmp_path), run_json(tmp_path)
    assert len(rows) == len(events) == 2 * 2 * 4
    assert {(r["prompt_set"], r["environment"], r["organization"]) for r in rows} == {
        ("primitives", ENVIRONMENT, organization)}
    assert (settings["prompts"], settings["organizations"], settings["status"]) == (
        "primitives", [organization], "complete")
    assert {r["n_rule_priced_bids"] for r in rows} == {"0"}
    assert len(shown) == 2 * 2 * 3 * 2
    if departure_base(organization) == "forward":
        assert set(shown) == {"    Analyst (A): forwards the record unchanged",
                              "    Analyst (B): forwards the record unchanged"}
    else:
        assert all("[" not in line and "Mock briefing" in line for line in shown)       # no posture is shown
    for event in events:
        assert (set(event["analyst_briefings"]) == {"A", "B"}) == (event["event"] >= 2)
        assert all(posture is None for posture in event["analyst_postures"].values())


# ---- the words of advice and of platform type ----

def _words_found(text, words):
    lowered = text.lower()
    return sorted(word for word in words if re.search(r"\b" + re.escape(word), lowered))


# The new texts say "the coming event" truthfully (of the event that the analyst writes the briefing for), so the phrase
# that the scan bans in the earlier organizations' texts, where it began a false sentence, is let through here.
FORBIDDEN_HERE = {word: reason for word, reason in FORBIDDEN.items() if word != "coming event"}


def test_no_text_that_a_departures_bidder_or_analyst_reads_has_a_word_of_advice_or_of_platform_type(recorded):
    texts = {}
    for (organization, institution), (calls, _) in recorded.items():
        for call in agent_requests(calls):
            for message in call["messages"]:
                texts[message["content"]] = (organization, institution)
    assert len(texts) > 100
    assert len(FORBIDDEN_HERE) == len(FORBIDDEN) - 1
    for text, cell in texts.items():
        assert _words_found(text, FORBIDDEN_HERE) == [], (cell, _words_found(text, FORBIDDEN_HERE), text[:200])


# ---- the help and the README ----

def test_the_help_names_the_departures_organizations(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = "".join(capsys.readouterr().out.split())            # no spaces: a long option may break at a hyphen
    for part in (
            "The departures organizations of --prompts primitives (brief, forward, floor, brief_cap, brief_cost, each "
            "also "
            "with the suffix _revenue) are run only when named",
            "The departures organizations of --prompts primitives, run only when they are named in --organizations or "
            "--mixed (they are refused with --prompts advice): in each, the firm's analyst holds the record and sends "
            "the bidder one message before each event from the second.",
            "forward: no model is called, and the message is the full record, the very text that single reads, "
            "forwarded unchanged. floor: as brief, and the analyst may end its briefing with a minimum profit per "
            "contract for the event (\"MINIMUM PROFIT PER CONTRACT: $X\" or \"MINIMUM PROFIT PER CONTRACT: none\"; "
            "asked again once if the line is missing or malformed, and then no floor is set), which the code enforces: "
            "a bid priced below its total cost plus the minimum is returned to the bidder once, and if the bid it "
            "sends "
            "back is also below, its price is raised to the total cost plus the minimum",
            "summary.csv counts the raised bids in n_floor_raised_bids, a column that a run naming a departures "
            "organization has).",
            "In floor the last line that is not only marks is read with some tolerance (any case, markdown marks, an "
            "optional dollar sign), and the bidder reads the briefing without any line that begins with the words "
            "MINIMUM PROFIT PER CONTRACT (in any case, markdown marks aside), then a notice: the minimum, none, or, if "
            "the analyst gave no valid line in two answers, that no minimum is set (floor_status in \"departures\" "
            "tells the cases apart: set, none, malformed).",
            "brief_cap: as brief, with the briefing at most 50 words: an analyst whose briefing is longer is asked "
            "once "
            "more, and a second briefing that is still too long is cut to its first 50 words",
            "Their agents are counted in run.json under bidder_<organization> and analyst_<organization>."):
        assert "".join(part.split()) in text, part


def test_the_readme_describes_the_departures_organizations():
    readme = (Path(demo.__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    section = readme.split("\n## The primitives prompt set\n", 1)[1].split("\n## ", 1)[0]
    text = " ".join(section.split())
    for part in (
            "**Departures organizations** (named in `--organizations` or inside `--mixed`, with `--prompts primitives` "
            "only; they are refused with `--prompts advice`, and a run that names no organization does not include "
            "them).",
            "sends the bidder one message before each event from the second, under a heading that begins `MESSAGE FROM "
            "YOUR COMPANY'S PROCUREMENT ANALYST`.",
            "Each has a conflict variant with the suffix `_revenue` (`brief_revenue`, `forward_revenue`, "
            "`floor_revenue`, `brief_cap_revenue`, `brief_cost_revenue`): the bidder is judged on revenue and is told "
            "that the analyst's objective is the firm's profit (also in `forward_revenue`, where no model plays the "
            "analyst)",
            "`floor`: as `brief`, and the analyst may end its briefing with one line, `MINIMUM PROFIT PER CONTRACT: "
            "$X` "
            "or `MINIMUM PROFIT PER CONTRACT: none`, setting a minimum profit per contract for the event.",
            "The last line that is not only marks (a line such as `---`, `***`, `___` or a code fence is skipped) is "
            "read with some tolerance: the words and `none` may be in any case, white space and markdown marks around "
            "the line or between its parts do not matter (`*`, `_` and backticks at either end of the line, and `#`, "
            "`>`, `-`, `•`, `+`, `–`, `◦` and a list number such as `1.` or `1)` at its start), the words may have "
            "more than one white-space character between them, the dollar sign is optional, and the value may end "
            "with a period, also after closing marks; words after the value, `$1,000`, a negative value and a range "
            "are malformed.",
            "If that line is not a floor line, the analyst is asked once more (the follow-up begins `Write your "
            "briefing again.`), and if it is still not, no floor is set.",
            "The bidder reads the answer without any line that begins, once white space, marks and a list number at "
            "its start are removed, with the words `MINIMUM PROFIT PER CONTRACT` (in any case, with white space "
            "between them), whatever follows and wherever the line is; nothing else is removed, and a line with other "
            "words before the words stays.",
            "Then come a blank line and a notice: `MINIMUM PROFIT PER CONTRACT FOR THIS EVENT, SET BY YOUR COMPANY'S "
            "ANALYST: $X` (to two decimals) or `... : none`, or, when the analyst gave no valid line in two answers, "
            "`NO MINIMUM PROFIT PER CONTRACT IS SET FOR THIS EVENT`.",
            "The heading of the message says that its last line is a notice of the minimum (`MESSAGE FROM YOUR "
            "COMPANY'S PROCUREMENT ANALYST (written by the analyst, except the last line, a notice of the "
            "minimum):`).",
            "In every round of the event, a bid whose price is less than its total cost at its delivery and quality "
            "plus the minimum (strictly) is returned to the bidder once, with a follow-up turn that gives the bid's "
            "price and its total cost with the digits they have beyond the cent (up to nine decimals) and the minimum "
            "to the cent; the platform does not see the returned bid.",
            "If the bid it sends back is also below, its price is raised to its total cost, at the terms of that bid, "
            "plus the minimum, and the bid's reasoning is tagged.",
            "The record and the list of earlier rounds mark a raised price with `(price raised to its total cost plus "
            "the minimum profit your company's analyst set)`, and no note of a price set by rule is ever printed in "
            "the primitives set.",
            "`first` and `resent` hold `price`, `delivery_weeks`, `quality_tier`, `cost` and `reasoning`, the "
            "reasoning as the bidder wrote it",
            "`floor_line` (the last line of the last answer that is not only marks)",
            "`summary.csv` counts the raised bids in `n_floor_raised_bids`, a column that every run naming a "
            "departures "
            "organization has.",
            "The one exception is the bidder of a `floor` organization, whose bid priced below its total cost plus the "
            "minimum profit that its analyst set is returned once and then raised (see `floor`).",
            "`brief_cap`: as `brief`, with the briefing at most 50 words. An analyst whose briefing is longer is asked "
            "once "
            "more, in the same conversation, with a follow-up that says how long the briefing was; a second briefing "
            "that "
            "is still too long is cut to its first 50 words. The follow-up says so: if the second briefing still has "
            "more than 50 words, the bidder reads only its first 50 words.",
            "`events.jsonl` records, under `departures` and for each supplier of `brief_cap`, the original briefing, "
            "whether the analyst was asked again, the second briefing, whether it was cut and the final word count.",
            "The bidders of `brief`, `forward`, `brief_cap` and `brief_cost` read one role text, so their first events "
            "are identical and differ only from the second event on.",
            "`brief` (B): the analyst writes a briefing, and the heading says so.",
            "`forward` (F): no analyst model is called; the message is the full record, the very text that `single` "
            "reads, forwarded unchanged, and the heading says so.",
            "`brief_cost`: as `brief`, and the analyst is also shown the bidder's cost figures for the coming event, "
            "in "
            "the third person; from the second event on, the title of the bidder's cost structure says that its "
            "analyst "
            "is shown it.",
            "The agents of the departures organizations are counted under `bidder_<organization>` and "
            "`analyst_<organization>` (`analyst_brief`, `analyst_brief_cost`, ...), and nothing is counted under "
            "`other`.",
            "With no `--organizations`, a run of the primitives set covers all six."):
        assert part in text, part
    for name in DEPARTURE_ORGANIZATIONS:
        assert name in text
    # the fields of the record of floor in events.jsonl, by name (the events-file tests below pin them)
    for name in ("organization", "raw_briefings", "reasked", "follow_up", "floor_line", "floor_status",
                 "minimum_profit", "bids_returned", "bids_raised", "raised", "first", "resent", "reasoning"):
        assert f"`{name}`" in text, name
    assert "`floor_status` (`set`, `none`, or `malformed` when the analyst gave no valid line in two answers)" in text


# ---- brief_cap: the limit, the follow-up, the cut, and the record ----

def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(1, n + 1))


def over_long(event, sid, attempt, user):
    """Event 2: A writes 60 words, then 20; B writes 55 words, then 70 (7 lines of 10). Event 3: A writes exactly 50
    words and B 51, then 51 again. Every other briefing is short."""
    if event == 2 and sid == "A":
        return words(60, "a") if attempt == 1 else words(20, "x")
    if event == 2 and sid == "B":
        return words(55, "b") if attempt == 1 else "\n".join(words(10, f"y{i}_") for i in range(1, 8))
    if event == 3:
        return words(50, "c") if sid == "A" else words(51, "d")
    return f"Fake briefing after {event - 1} past event(s)."


@pytest.mark.parametrize("institution", INSTITUTIONS)
@pytest.mark.parametrize("organization", ["brief_cap", "brief_cap_revenue"])
def test_a_briefing_over_the_limit_is_asked_for_again_and_a_second_that_is_too_long_is_cut(
        organization, institution, tmp_path):
    calls = run_departures(tmp_path, "--organizations", organization, institution=institution, briefing=over_long)
    analyst = kinds(calls)["analyst"]
    lengths = [len(call["messages"]) for call in analyst]
    assert lengths == [2, 4, 2, 4,                       # event 2: A asked twice, B asked twice
                       2, 2, 4,                          # event 3: A once (50 words), B twice (51, then 51 again)
                       2, 2, 2, 2, 2, 2]                 # events 4 to 6, once each
    a_first, a_again, b_first, b_again = analyst[0:4]
    assert a_again["messages"][:2] == a_first["messages"] and b_again["messages"][:2] == b_first["messages"]
    assert a_again["messages"][2:] == [
        {"role": "assistant", "content": words(60, "a")},
        {"role": "user", "content": cap_reask(60)}]
    assert b_again["messages"][2:] == [
        {"role": "assistant", "content": words(55, "b")},
        {"role": "user", "content": cap_reask(55)}]
    cut = "\n".join(words(10, f"y{i}_") for i in range(1, 6))             # the first 50 of the 70 words
    seen = {sid: [m[1]["content"] for m in event_requests(calls, 2)][0 if sid == "A" else 1] for sid in "AB"}
    assert HEADER_CAP + "\n" + words(20, "x") + "\n\nRULES:" in seen["A"]
    assert HEADER_CAP + "\n" + cut + "\n\nRULES:" in seen["B"] and "y6_1" not in seen["B"]
    events = events_of(tmp_path)
    assert events[0]["departures"] == {}
    assert events[1]["analyst_briefings"] == {"A": words(20, "x"), "B": cut}
    assert events[1]["departures"] == {
        "A": {"organization": organization, "word_limit": 50, "original": words(60, "a"), "original_words": 60,
              "reasked": True,
              "follow_up": cap_reask(60),
              "second": words(20, "x"), "second_words": 20, "cut": False, "final_words": 20},
        "B": {"organization": organization, "word_limit": 50, "original": words(55, "b"), "original_words": 55,
              "reasked": True,
              "follow_up": cap_reask(55),
              "second": "\n".join(words(10, f"y{i}_") for i in range(1, 8)), "second_words": 70, "cut": True,
              "final_words": 50}}
    third = events[2]["departures"]
    assert (third["A"]["reasked"], third["A"]["original_words"], third["A"]["final_words"]) == (False, 50, 50)
    assert (third["B"]["reasked"], third["B"]["original_words"], third["B"]["second_words"], third["B"]["cut"],
            third["B"]["final_words"]) == (True, 51, 51, True, 50)
    assert third["B"]["follow_up"] == cap_reask(51)
    for event in events[3:]:
        assert all(not record["reasked"] and not record["cut"] for record in event["departures"].values())
    usage = run_json(tmp_path)["usage"]["by_agent"]
    assert usage[f"analyst_{organization}"]["requests"] == len(analyst) == 13


def test_a_run_without_brief_cap_or_floor_has_no_departures_record(recorded):
    for (organization, institution), (calls, out) in recorded.items():
        for event in events_of(out):
            assert ("departures" in event) == (departure_base(organization) in ("brief_cap", "floor")), (
                organization, event["event"])


# ---- floor: the line, the notice, the enforcement, the record, the files ----

# The analysts' answers in the scripted scenario, by (event, supplier, attempt). Each event shows one way in which a
# floor line is read, or not.
FLOOR_SCRIPT = {
    (2, "A", 1): "Briefing A2.\nMINIMUM PROFIT PER CONTRACT: $30",
    # markdown and another case: a floor line all the same
    (2, "B", 1): "Briefing B2.\n**Minimum profit per contract:** $10",
    # a numbered list item; none, with a period after it
    (3, "A", 1): "Briefing A3.\n1. MINIMUM PROFIT PER CONTRACT: none.",
    (3, "B", 1): "Briefing B3 without a line.",
    # the floor line in a code fence: the fence is not the floor line, and stays in the briefing
    (3, "B", 2): "Briefing B3 again.\n```\nMINIMUM PROFIT PER CONTRACT: $12.5\n```",
    (4, "A", 1): "Briefing A4, no line.",
    # malformed: the last line is not a floor line; a line that begins with the words of one is not read by the bidder
    (4, "A", 2): "Briefing A4 again.\n  MINIMUM PROFIT PER CONTRACT $9 per contract\nKeep to speed.",
    # an en dash in front of the line, and no space after the colon
    (4, "B", 1): "Briefing B4.\n– MINIMUM PROFIT PER CONTRACT:$7",
    # a rule after the floor line is skipped, and stays in the briefing
    (5, "A", 1): "Briefing A5.\n  MINIMUM PROFIT PER CONTRACT:   none  \n---",
    # malformed: words after the value
    (5, "B", 1): "Briefing B5.\nMINIMUM PROFIT PER CONTRACT: $4 per contract",
    # a mention in a line that does not begin with the words stays; a floor line in a list, another case, extra white
    # space and a space before the colon is removed
    (5, "B", 2): "Briefing B5 again, my MINIMUM PROFIT PER CONTRACT is open.\n- minimum  profit per contract : NONE",
    # an earlier floor line is removed too, and the last one counts, in bold with a period after the closing marks
    (6, "A", 1): "Briefing A6.\nMINIMUM PROFIT PER CONTRACT: $3\n\n**MINIMUM PROFIT PER CONTRACT: $ 0**.\n\n",
    # a line that begins with the words in another case is removed, prose or not; a white circle in front of the last
    (6, "B", 1): ("Briefing B6.\n**Minimum profit per contract:** $2, because costs are high.\n"
                  "◦ MINIMUM PROFIT PER CONTRACT: $0.00"),
}


def scripted_floor_briefing(event, sid, attempt, user):
    return FLOOR_SCRIPT[event, sid, attempt]


def floor_bids(event, round_num, sid):
    """The plain bids, except supplier A's first bid of round 3 of event 2, which is exactly at its total cost plus the
    $30 its analyst set: 85 + 1.8 * (8 - 5)^2 + 0.4 * 3^2 = 104.8 for a quality specialist at 5 weeks and tier 3."""
    if (event, round_num, sid) == (2, 3, "A"):
        return {"price": 134.8, "delivery_weeks": 5.0, "quality_tier": 3, "reasoning": "at the sum"}
    return default_bid(event, round_num, sid)


def floor_resends(event, round_num, sid):
    """What A sends when it is asked again in event 2: a fix in round 1, a bid that is still below, at other terms, in
    round 2."""
    if round_num == 1:
        return {"price": 140.0, "delivery_weeks": 5.0, "quality_tier": 3, "reasoning": "fixed"}
    return {"price": 126.0, "delivery_weeks": 4.0, "quality_tier": 4, "reasoning": "still below"}


@pytest.fixture(scope="module", params=INSTITUTIONS)
def scripted_floor(request, tmp_path_factory):
    out = tmp_path_factory.mktemp(f"floor_{request.param}")
    calls = run_departures(out, "--organizations", "floor", institution=request.param, bid=floor_bids,
                           resend=floor_resends, briefing=scripted_floor_briefing)
    return request.param, calls, out


def bidder_events(calls, event):
    """The bidder requests of one event, in order, with the supplier of each (A and B alternate in each round, and a
    request with earlier turns is the follow-up to the one before it)."""
    found, first = [], 0
    for call in kinds(calls)["bidder"]:
        if len(call["messages"]) == 2:
            first += 1
        if call["messages"][1]["content"].startswith(f"EVENT {event} of 6"):
            found.append(("AB"[(first - 1) % 2], call["messages"]))
    return found


def test_the_analysts_floor_lines_are_read_asked_for_again_and_given_up_on_as_the_design_says(scripted_floor):
    institution, calls, out = scripted_floor
    events = events_of(out)
    analyst = kinds(calls)["analyst"]
    assert [len(call["messages"]) for call in analyst] == [2, 2,           # event 2: both lines are valid
                                                           2, 2, 4,        # event 3: B has no line, and is asked again
                                                           # event 4: A has no line twice, B has a valid one
                                                           2, 4, 2,
                                                           # event 5: B has words after the value, and is asked again
                                                           2, 2, 4,
                                                           2, 2]           # event 6
    again = [call["messages"] for call in analyst if len(call["messages"]) == 4]
    assert all(messages[2]["role"] == "assistant" and messages[3] == {"role": "user", "content": REASK_ANALYST}
               for messages in again)
    assert [messages[2]["content"] for messages in again] == [
        FLOOR_SCRIPT[3, "B", 1], FLOOR_SCRIPT[4, "A", 1], FLOOR_SCRIPT[5, "B", 1]]
    shown = {(e["event"], sid): text for e in events for sid, text in e["analyst_briefings"].items()}
    assert shown == {
        (2, "A"): "Briefing A2.\n\n" + NOTICE + "$30.00", (2, "B"): "Briefing B2.\n\n" + NOTICE + "$10.00",
        (3, "A"): "Briefing A3.\n\n" + NOTICE + "none",
        (3, "B"): "Briefing B3 again.\n```\n```\n\n" + NOTICE + "$12.50",
        (4, "A"): "Briefing A4 again.\nKeep to speed.\n\n" + NOTICE_NOT_SET,
        (4, "B"): "Briefing B4.\n\n" + NOTICE + "$7.00",
        (5, "A"): "Briefing A5.\n---\n\n" + NOTICE + "none",
        (5, "B"): "Briefing B5 again, my MINIMUM PROFIT PER CONTRACT is open.\n\n" + NOTICE + "none",
        (6, "A"): "Briefing A6.\n\n" + NOTICE + "$0.00",
        (6, "B"): "Briefing B6.\n\n" + NOTICE + "$0.00"}
    assert events[0]["departures"] == {}
    minimum = {(e["event"], sid): d["minimum_profit"] for e in events for sid, d in e["departures"].items()}
    assert minimum == {(2, "A"): 30.0, (2, "B"): 10.0, (3, "A"): None, (3, "B"): 12.5, (4, "A"): None, (4, "B"): 7.0,
                       (5, "A"): None, (5, "B"): None, (6, "A"): 0.0, (6, "B"): 0.0}
    record = {(e["event"], sid): (d["reasked"], d["floor_status"])
              for e in events for sid, d in e["departures"].items()}
    assert record[3, "B"] == (True, "set") and record[4, "A"] == (True, "malformed")
    assert record[5, "B"] == (True, "none")
    assert record[2, "A"] == (False, "set") and record[5, "A"] == (False, "none") and record[6, "A"] == (False, "set")
    raw = events[3]["departures"]["A"]
    assert raw["raw_briefings"] == [FLOOR_SCRIPT[4, "A", 1], FLOOR_SCRIPT[4, "A", 2]]
    assert raw["follow_up"] == REASK_ANALYST
    assert raw["floor_line"] == "Keep to speed."                          # the last line of the briefing it gave up on
    assert raw["minimum_profit"] is None and raw["bids_returned"] == raw["bids_raised"] == 0
    # none and no valid line are told apart by the record, and by the notice the bidder reads
    assert events[4]["departures"]["A"]["floor_status"] == "none" and events[3]["departures"]["A"]["floor_status"] == (
        "malformed")
    # a line in markdown and another case is a floor line, kept as written in the record; a period after none is read
    marked = events[1]["departures"]["B"]
    assert (marked["floor_status"], marked["minimum_profit"]) == ("set", 10.0)
    assert marked["raw_briefings"] == [FLOOR_SCRIPT[2, "B", 1]] and marked["floor_line"] == (
        "**Minimum profit per contract:** $10")
    assert events[2]["departures"]["A"]["floor_line"] == "1. MINIMUM PROFIT PER CONTRACT: none." and not (
        events[2]["departures"]["A"]["reasked"])
    # the floor line is the last line that is not only marks: a rule or a code fence after it is not
    assert events[4]["departures"]["A"]["floor_line"] == "MINIMUM PROFIT PER CONTRACT:   none"
    assert events[4]["departures"]["A"]["raw_briefings"] == [FLOOR_SCRIPT[5, "A", 1]]
    assert events[2]["departures"]["B"]["floor_line"] == "MINIMUM PROFIT PER CONTRACT: $12.5"
    assert events[5]["departures"]["A"]["floor_line"] == "**MINIMUM PROFIT PER CONTRACT: $ 0**."
    assert events[3]["departures"]["B"]["floor_line"] == "– MINIMUM PROFIT PER CONTRACT:$7"
    assert events[5]["departures"]["B"]["floor_line"] == "◦ MINIMUM PROFIT PER CONTRACT: $0.00"
    # words after the value make the line malformed: B was asked again in event 5
    assert events[4]["departures"]["B"]["raw_briefings"][0] == FLOOR_SCRIPT[5, "B", 1]
    assert events[4]["departures"]["B"]["reasked"] is True


def test_the_bidder_reads_the_briefing_without_its_floor_line_and_then_the_notice_in_every_round(scripted_floor):
    institution, calls, out = scripted_floor
    for event, sid, expected, notice in (
            (2, "A", "Briefing A2.\n\n" + NOTICE + "$30.00", NOTICE),                           # a valid line
            (2, "B", "Briefing B2.\n\n" + NOTICE + "$10.00", NOTICE),                           # markdown, another case
            (3, "A", "Briefing A3.\n\n" + NOTICE + "none", NOTICE),                            # a list number, a period
            (3, "B", "Briefing B3 again.\n```\n```\n\n" + NOTICE + "$12.50", NOTICE),        # re-asked; the fence stays
            (4, "A", "Briefing A4 again.\nKeep to speed.\n\n" + NOTICE_NOT_SET, NOTICE_NOT_SET),  # no valid line twice
            (4, "B", "Briefing B4.\n\n" + NOTICE + "$7.00", NOTICE),                            # an en dash in front
            (5, "A", "Briefing A5.\n---\n\n" + NOTICE + "none", NOTICE),                        # a rule after it stays
            (5, "B", "Briefing B5 again, my MINIMUM PROFIT PER CONTRACT is open.\n\n" + NOTICE + "none", NOTICE),
            (6, "A", "Briefing A6.\n\n" + NOTICE + "$0.00", NOTICE),                         # an earlier line, a period
            (6, "B", "Briefing B6.\n\n" + NOTICE + "$0.00", NOTICE)):                           # a prose line, a circle
        requests = [messages for who, messages in bidder_events(calls, event) if who == sid and len(messages) == 2]
        assert len(requests) == ROUNDS
        for messages in requests:
            user = messages[1]["content"]
            assert HEADER_FLOOR + "\n" + expected + "\n\nRULES:" in user
            assert user.count(notice) == 1 and user.count("MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST") == 1
            assert (NOTICE in user) == (notice == NOTICE) and (NOTICE_NOT_SET in user) == (notice == NOTICE_NOT_SET)
    # no line that begins with the words of the floor line reaches the bidder, in any case: the part of the message
    # before the notice has none (written out again here, not taken from the code)
    beginning = re.compile(r"[\s*_`#>\-•+–◦]*(?:[0-9]+[.)])?[\s*_`#>\-•+–◦]*minimum\s+profit\s+per\s+contract",
                           re.IGNORECASE)
    for event, sid in ((2, "A"), (2, "B"), (3, "A"), (3, "B"), (4, "A"), (4, "B"), (5, "A"), (5, "B"), (6, "A"),
                       (6, "B")):
        for who, messages in bidder_events(calls, event):
            if who == sid and len(messages) == 2:
                part = messages[1]["content"].split(HEADER_FLOOR + "\n", 1)[1].split("\n\nRULES:")[0]
                written = part.rsplit("\n\n", 1)[0] if "\n\n" in part else ""
                assert not any(beginning.match(line) for line in written.split("\n")), (event, sid)
    assert "MINIMUM PROFIT PER CONTRACT" not in " ".join(
        m["messages"][1]["content"] for m in kinds(calls)["bidder"]
        if m["messages"][1]["content"].startswith("EVENT 1 of"))


def test_a_bid_below_the_cost_plus_the_minimum_is_returned_once_and_a_second_one_below_is_raised(scripted_floor):
    institution, calls, out = scripted_floor
    event2 = bidder_events(calls, 2)
    a_requests = [messages for sid, messages in event2 if sid == "A"]
    b_requests = [messages for sid, messages in event2 if sid == "B"]
    assert [len(m) for m in a_requests] == [2, 4, 2, 4, 2] and [len(m) for m in b_requests] == [2, 2, 2]
    cost_5_3, cost_4_4 = 104.8, 120.2
    follow_up_1 = REASK_BIDDER.replace("$P", "$122.00").replace("$C", "$104.80").replace("$X", "$30.00")
    follow_up_2 = REASK_BIDDER.replace("$P", "$124.00").replace("$C", "$104.80").replace("$X", "$30.00")
    first, again = a_requests[0], a_requests[1]
    assert again[:2] == first and again[2]["role"] == "assistant"
    assert again[3] == {"role": "user", "content": follow_up_1}
    assert json.loads(again[2]["content"])["price"] == 122.0
    assert a_requests[3][3] == {"role": "user", "content": follow_up_2}
    assert json.loads(a_requests[3][2]["content"])["price"] == 124.0
    bids = events_of(out)[1]["bids"]["A"]
    assert [b["price"] for b in bids] == [140.0, pytest.approx(cost_4_4 + 30.0), 134.8]
    assert [(b["delivery_weeks"], b["quality_tier"]) for b in bids] == [(5.0, 3), (4.0, 4), (5.0, 3)]
    assert [FLOOR_PRICE_TAG in b["reasoning"] for b in bids] == [False, True, False]
    assert bids[0]["reasoning"] == "[STATED]: fixed"
    assert bids[1]["reasoning"] == "[STATED]: still below" + FLOOR_PRICE_TAG
    assert bids[0]["floor"] == {
        "minimum_profit": 30.0, "reasked": True, "raised": False, "follow_up": follow_up_1,
        "first": {"price": 122.0, "delivery_weeks": 5.0, "quality_tier": 3, "cost": pytest.approx(cost_5_3),
                  "reasoning": "fake bid 2.1.A"},
        "resent": {"price": 140.0, "delivery_weeks": 5.0, "quality_tier": 3, "cost": pytest.approx(cost_5_3),
                   "reasoning": "fixed"}}
    assert bids[1]["floor"]["raised"] is True and bids[1]["floor"]["resent"]["price"] == 126.0
    assert bids[1]["floor"]["first"]["price"] == 124.0
    assert bids[1]["floor"]["first"]["reasoning"] == "fake bid 2.2.A" and bids[1]["floor"]["resent"]["reasoning"] == (
        "still below")
    # exactly at the sum: passes
    assert bids[2]["floor"] == {"minimum_profit": 30.0, "reasked": False, "raised": False}
    # B has a floor of $10 too
    assert all("floor" in b for b in events_of(out)[1]["bids"]["B"])
    assert [b["floor"]["reasked"] for b in events_of(out)[1]["bids"]["B"]] == [False] * 3
    event_two = events_of(out)[1]["departures"]
    assert event_two["A"]["bids_returned"] == 2 and event_two["A"]["bids_raised"] == 1
    assert event_two["B"]["bids_returned"] == 0 and event_two["B"]["bids_raised"] == 0
    # no bid is returned, and no floor is recorded for it, elsewhere
    for event in (1, 3, 4, 5, 6):
        for bid in sum(events_of(out)[event - 1]["bids"].values(), []):
            assert "floor" not in bid or bid["floor"]["reasked"] is False and bid["floor"]["raised"] is False


def test_the_raised_price_is_in_the_outcome_and_the_summary_counts_it(scripted_floor):
    institution, calls, out = scripted_floor
    rows = rows_of(out)
    assert list(rows[0]) == demo.COLUMNS + ["n_floor_raised_bids"]
    assert [int(row["n_floor_raised_bids"]) for row in rows] == [0, 1, 0, 0, 0, 0]
    assert reader_columns_of(out) == demo.COLUMNS + ["n_floor_raised_bids"]


def reader_columns_of(folder):
    import csv
    with open(Path(folder) / "summary.csv", encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


def test_the_earlier_rounds_and_the_record_mark_the_raised_price_and_no_request_has_the_note_of_a_price_set_by_rule(
        scripted_floor):
    institution, calls, out = scripted_floor
    event2_a = [messages for sid, messages in bidder_events(calls, 2) if sid == "A" and len(messages) == 2]
    # A's request of round 3: the rounds 1 and 2 before it
    block = event2_a[2][1]["content"]
    assert "PREVIOUS ROUNDS:\n" in block
    assert "  Round 1: You bid $140.00, 5.0wk delivery, tier 3 (your cost was $104.80)\n" in block
    assert f"  Round 2: You bid $150.20{RAISED_NOTE}, 4.0wk delivery, tier 4 (your cost was $120.20)\n" in block
    assert block.count(RAISED_NOTE) == 1
    third_event_analyst = [c["messages"][1]["content"] for c in kinds(calls)["analyst"]
                           if c["messages"][1]["content"].startswith("EVENT 3 of 6")][0]
    assert f"    Round 2: Your company bid $150.20{RAISED_NOTE}, 4.0wk, tier 4\n" in third_event_analyst
    assert "    Round 1: Your company bid $140.00, 5.0wk, tier 3\n" in third_event_analyst
    for call in agent_requests(calls):
        text = " ".join(m["content"] for m in call["messages"])
        assert "price set by rule" not in text and "cost + 5%" not in text


def test_the_usage_of_a_floor_run_counts_the_follow_up_requests_under_the_bidder_and_the_analyst(scripted_floor):
    institution, calls, out = scripted_floor
    usage = run_json(out)["usage"]["by_agent"]
    assert set(usage) == {"bidder_floor", "analyst_floor", "mediator" if institution == "mediator" else "buyer_agent"}
    assert usage["bidder_floor"]["requests"] == 2 * EVENTS * ROUNDS + 2          # A was asked again twice
    assert usage["analyst_floor"]["requests"] == 2 * (EVENTS - 1) + 3            # and three analysts were


def test_the_events_file_records_the_floor_of_each_event_for_each_supplier_and_a_run_without_one_has_none(
        scripted_floor, recorded):
    institution, calls, out = scripted_floor
    for event in events_of(out)[1:]:
        for sid, record in event["departures"].items():
            assert set(record) == {"organization", "raw_briefings", "reasked", "follow_up", "floor_line",
                                   "floor_status", "minimum_profit", "bids_returned", "bids_raised"}
            assert record["organization"] == "floor" and record["floor_status"] in ("set", "none", "malformed")
            assert len(record["raw_briefings"]) == (2 if record["reasked"] else 1)
            assert record["follow_up"] == (REASK_ANALYST if record["reasked"] else None)
            assert (record["minimum_profit"] is not None) == (record["floor_status"] == "set")
    calls_b, out_b = recorded["brief", institution]
    assert all("departures" not in event and all("floor" not in b for bids in event["bids"].values() for b in bids)
               for event in events_of(out_b))
    # a departures run has the column
    assert reader_columns_of(out_b) == demo.COLUMNS + ["n_floor_raised_bids"]
    assert {row["n_floor_raised_bids"] for row in rows_of(out_b)} == {"0"}


def test_the_audit_of_a_bid_in_the_events_file_has_these_fields(scripted_floor):
    institution, calls, out = scripted_floor
    audits = [bid["floor"] for event in events_of(out) for bids in event["bids"].values() for bid in bids
              if "floor" in bid]
    assert any(audit["reasked"] for audit in audits) and any(audit["raised"] for audit in audits)
    # a bid has an audit when its supplier's minimum is set for the event: 6 supplier-events of the script, 3 rounds
    with_a_minimum = sum(d["minimum_profit"] is not None for e in events_of(out) for d in e["departures"].values())
    assert len(audits) == with_a_minimum * ROUNDS == 18
    for audit in audits:
        assert set(audit) == ({"minimum_profit", "reasked", "raised", "first", "follow_up", "resent"}
                              if audit["reasked"] else {"minimum_profit", "reasked", "raised"})
        assert isinstance(audit["minimum_profit"], float) and isinstance(audit["reasked"], bool)
        assert isinstance(audit["raised"], bool) and (audit["raised"] is False or audit["reasked"] is True)
        if audit["reasked"]:
            assert set(audit["first"]) == set(audit["resent"]) == {
                "price", "delivery_weeks", "quality_tier", "cost", "reasoning"}
            assert isinstance(audit["first"]["reasoning"], str) and isinstance(audit["resent"]["reasoning"], str)
            assert audit["follow_up"].startswith("Your bid's price, $")


def test_the_platform_is_never_sent_a_bid_that_was_returned_to_the_bidder(scripted_floor):
    institution, calls, out = scripted_floor
    audits = [bid["floor"] for event in events_of(out) for bids in event["bids"].values() for bid in bids
              if "floor" in bid and bid["floor"]["reasked"]]
    returned = [audit["first"] for audit in audits]
    assert [bid["price"] for bid in returned] == [122.0, 124.0]                 # the two bids of A that were returned
    # what the platform's agent is sent: the bids of the round that stood, never the returned one
    sent = " ".join(m["content"] for call in kinds(calls)["platform"] for m in call["messages"])
    for bid in returned:
        assert f"${bid['price']:.2f}" not in sent and bid["reasoning"] not in sent
    fixed, raised = [a["resent"] for a in audits if not a["raised"]], [a["resent"] for a in audits if a["raised"]]
    assert [bid["price"] for bid in fixed] == [140.0] and [bid["price"] for bid in raised] == [126.0]
    assert "$140.00" in sent and "$150.20" in sent                              # the bids that stood: fixed, and raised
    assert "$126.00" not in sent                                                # the bid sent back that was raised


def test_a_run_of_the_earlier_organizations_has_the_columns_it_had_and_no_floor_audit(tmp_path, monkeypatch):
    run_departures(tmp_path, "--organizations", "organized", "single", monkeypatch=monkeypatch)
    assert reader_columns_of(tmp_path) == demo.COLUMNS
    assert all("departures" not in event and all("floor" not in b for bids in event["bids"].values() for b in bids)
               for event in events_of(tmp_path))


def test_a_mixed_market_with_one_floor_organization_enforces_the_floor_of_that_supplier_only(tmp_path, monkeypatch):
    def briefing(event, sid, attempt, user):
        return f"Briefing {sid}{event}." + ("\nMINIMUM PROFIT PER CONTRACT: $40" if sid == "B" else "")

    run_departures(tmp_path, "--mixed", "brief_revenue:floor_revenue", briefing=briefing, monkeypatch=monkeypatch)
    events = events_of(tmp_path)
    assert all(set(event["departures"]) == ({"B"} if event["event"] >= 2 else set()) for event in events)
    assert all("floor" not in bid for event in events for bid in event["bids"]["A"])
    assert all(bid["floor"]["minimum_profit"] == 40.0 for event in events[1:] for bid in event["bids"]["B"])
    assert sum(bid["floor"]["raised"] for event in events[1:] for bid in event["bids"]["B"]) > 0


def test_a_bidder_that_cannot_enforce_a_floor_stops_the_run_before_its_first_bid():
    from orgsim.interfaces import SupplierAgent
    from orgsim.mocks import MockBuyerAgentPlatform, MockPlatformMediator, MockProcurementAnalyst, MockStrategicSupplier
    from orgsim.primitives import get_contract_scenarios
    from orgsim.sequence import pre_draw_event_sequence, run_sequence

    class Plain(SupplierAgent):
        def generate_bid(self, supplier_costs, regime_context, round_num, history):
            raise AssertionError("the run should have stopped before a bid")

    plan = pre_draw_event_sequence(2, get_contract_scenarios(), seed=3)
    agents = {"A": Plain(), "B": MockStrategicSupplier(seed=1)}
    with pytest.raises(NotImplementedError, match="Plain does not enforce a minimum profit"):
        run_sequence("Mediator", 2, 2, agents, plan, mediator=MockPlatformMediator(),
                     analysts={sid: MockProcurementAnalyst(floor_line=True) for sid in "AB"},
                     organization="floor", prompt_set="primitives")


def test_the_runner_tells_the_bidder_the_floor_of_each_event_and_clears_it_after_the_event():
    from orgsim.mocks import MockPlatformMediator, MockProcurementAnalyst, MockStrategicSupplier
    from orgsim.primitives import get_contract_scenarios
    from orgsim.sequence import pre_draw_event_sequence, run_sequence

    class Recording(MockStrategicSupplier):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.in_force = []

        def generate_bid(self, *args, **kwargs):
            self.in_force.append(self.minimum_profit)
            return super().generate_bid(*args, **kwargs)

    plan = pre_draw_event_sequence(3, get_contract_scenarios(), seed=3)
    agents = {"A": Recording(seed=1), "B": Recording(seed=2)}
    run_sequence("Mediator", 3, 2, agents, plan, mediator=MockPlatformMediator(),
                 analysts={sid: MockProcurementAnalyst(floor_line=True) for sid in "AB"},
                 organization="floor", prompt_set="primitives")
    for agent in agents.values():
        # two rounds of each event: no floor in the first event (the analyst writes nothing), then the mock analyst's
        # $10 for each past event on record
        assert agent.in_force == [None, None, 10.0, 10.0, 20.0, 20.0]
        assert agent.minimum_profit is None                    # the floor of the last event does not stay


# ---- the mock run of the floor organizations ----

@pytest.mark.parametrize("organization", ["floor", "floor_revenue"])
def test_a_mock_run_of_floor_sets_a_floor_after_the_first_event_and_the_mock_bidder_obeys_it(organization, tmp_path):
    argv = ["--backend", "mock", "--prompts", "primitives", "--environment", ENVIRONMENT, "--organizations",
            organization, "--events", "4", "--rounds", "2", "--seeds", "1", "--out", str(tmp_path)]
    assert main(argv) == 0
    events, rows = events_of(tmp_path), rows_of(tmp_path)
    for event in events:
        assert set(event["departures"]) == ({"A", "B"} if event["event"] >= 2 else set())
        for record in event["departures"].values():
            assert record["floor_status"] == "set" and record["minimum_profit"] == 10.0 * (event["event"] - 1)
            assert "MINIMUM PROFIT PER CONTRACT: $" in record["raw_briefings"][0] and record["reasked"] is False
    raised = sum(FLOOR_PRICE_TAG in bid["reasoning"]
                 for event in events for bids in event["bids"].values() for bid in bids)
    assert raised > 0 and sum(int(row["n_floor_raised_bids"]) for row in rows) == raised
    for event in events:
        for bids in event["bids"].values():
            for bid in bids:
                if FLOOR_PRICE_TAG in bid["reasoning"]:
                    assert bid["floor"] == {"minimum_profit": bid["floor"]["minimum_profit"], "reasked": False,
                                            "raised": True}
