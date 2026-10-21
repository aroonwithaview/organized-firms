"""The departures organizations of the primitives set: their registry, their texts pinned exactly, the headers of the
message to the bidder, the cost structure in the third person, and the usage labels. Runs through the command, with a
fake chat endpoint, are in test_demo_departures.py."""
import hashlib
import json
import re
from types import SimpleNamespace

import pytest

from orgsim import primitives_prompts as PP
from orgsim.llm import ChatClient, LabeledClient
from orgsim.departures import CAP_REASK, capped_briefing, cut_to_words, follow_up_turns, word_count
from orgsim.organization import (
    ALL_PRIMITIVES_ANALYST_ORGANIZATIONS, ALL_PRIMITIVES_ORGANIZATIONS, ALL_PRIMITIVES_REVENUE_ORGANIZATIONS,
    BRIEF_HEADER, BRIEFING_WORD_LIMIT, CAP_HEADER, FLOOR_HEADER, AnalystBriefing, ProcurementAnalyst,
    DEPARTURE_ANALYST_ORGANIZATIONS, DEPARTURE_BASES, DEPARTURE_ORGANIZATIONS,
    DEPARTURE_REVENUE_ORGANIZATIONS, FIRST_EVENT_NOTE, FORWARD_HEADER, PRIMITIVES_ANALYST_ORGANIZATIONS,
    PRIMITIVES_ORGANIZATIONS, PRIMITIVES_REVENUE_ORGANIZATIONS, build_primitives_bidder_context, departure_base,
    forwards_the_record,
)
from orgsim.interfaces import FLOOR_PRICE_TAG, RULE_PRICE_TAG
from orgsim.llm_agents import _previous_rounds_block
from orgsim.primitives import ContractTerms, SupplierBid
from orgsim.sequence import build_primitives_record, make_supplier
from test_primitives_prompts import (
    ANALYST_PROFIT, BIDDER_ROLE, OPENING_REVENUE, OPENING_TEAM, OUTPUT, REVENUE_ANALYST, REVENUE_BIDDER, ROLE_ANALYST,
    SHARED,
)

ENVIRONMENTS = ("random", "persistent")
# The departures organizations this file covers: all the ones the code has. A base name and its conflict variant.
BASES = DEPARTURE_BASES
CONFLICT = {base: f"{base}_revenue" for base in BASES}
# The organizations whose bidder reads the shared role text (all but floor, which has its own).
SHARED_ROLE_BASES = [base for base in BASES if base != "floor"]
# The role text of the bidder of floor, and the sentences of the floor in the analyst's role text and in what code says.
ROLE_FLOOR = (
    "You are your company's bidder: you decide the price, delivery and quality of each of its bids, within one limit "
    "that your company's procurement analyst may set. The analyst keeps your company's record of past events on this "
    "platform; for each past event, the record shows your company's cost type, the contract, your company's bids, the "
    "platform's replies and the result. From your company's second event on, the analyst sends you a message before "
    "each event, and in it may set a minimum profit per contract for that event. In each round, a bid whose price is "
    "less than its total cost at its delivery and quality plus that minimum is returned to you once, and the platform "
    "does not see it; if the bid you send back is also below it, its price is raised to its total cost at that bid's "
    "delivery and quality plus the minimum. The message is included in what you read in each round of that event, "
    "under a heading that begins MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST.")
ADDITION_ANALYST_FLOOR = (
    "In your briefing you may also set a minimum profit per contract for the coming event. In each round, a bid whose "
    "price is less than its total cost at its delivery and quality plus that minimum is returned to the bidder once; "
    "if the bid it sends back is also below it, its price is raised to its total cost plus the minimum. End your "
    "briefing with one line, either MINIMUM PROFIT PER CONTRACT: $X or MINIMUM PROFIT PER CONTRACT: none. The bidder "
    "reads your briefing without any line that begins with the words MINIMUM PROFIT PER CONTRACT, and a notice of the "
    "minimum.")
ROLE_ANALYST_FLOOR = ROLE_ANALYST + " " + ADDITION_ANALYST_FLOOR
REASK_ANALYST = ("Write your briefing again. End your briefing with one line, either MINIMUM PROFIT PER CONTRACT: $X "
                 "or MINIMUM PROFIT PER CONTRACT: none.")
REASK_BIDDER = (
    "Your bid's price, $P, is less than its total cost at its delivery and quality, $C, plus the minimum profit per "
    "contract of $X that your company's analyst set for this event. Submit your bid again. If the bid you send is also "
    "below the minimum, its price will be raised to its total cost plus $X.")
NOTICE = "MINIMUM PROFIT PER CONTRACT FOR THIS EVENT, SET BY YOUR COMPANY'S ANALYST: "
NOTICE_NOT_SET = "NO MINIMUM PROFIT PER CONTRACT IS SET FOR THIS EVENT"
RAISED_NOTE = " (price raised to its total cost plus the minimum profit your company's analyst set)"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---- the texts of the design, written out again here, wrapped differently from the code ----

ROLE_DEPARTURE = (
    "You are your company's bidder: you decide the price, delivery and quality of each of its bids. Your company's "
    "procurement analyst keeps your company's record of past events on this platform; for each past event, the record "
    "shows your company's cost type, the contract, your company's bids, the platform's replies and the result. From "
    "your company's second event on, the analyst sends you a message before each event. It is included in what you "
    "read in each round of that event, under a heading that begins MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST.")
NOT_SHOWN = "You are not shown the cost figures the bidder is shown."
SHOWN = "Before each briefing you are also shown the cost figures the bidder is shown for that event."
ROLE_ANALYST_COST = ROLE_ANALYST.replace(NOT_SHOWN, SHOWN)
ROLE_ANALYST_CAP = ROLE_ANALYST + " Your briefing may have at most 50 words."
HEADER_BRIEF = "MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST (written by the analyst):"
HEADER_CAP = "MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST (written by the analyst, at most 50 words):"
HEADER_FORWARD = "MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST (your company's full record, forwarded unchanged):"
HEADER_FLOOR = ("MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST (written by the analyst, except the last line, a "
                "notice of the minimum):")
COST_HEADER_BIDDER = "YOUR PRIVATE COST STRUCTURE (you and your company's analyst are shown it):"
COST_HEADER_ORIGINAL = "YOUR PRIVATE COST STRUCTURE (only you know this):"
COST_HEADER_ANALYST = "YOUR COMPANY'S COST STRUCTURE (the bidder is shown the same figures):"

ANALYST_BLOCK_SPEED = (
    "YOUR COMPANY'S COST STRUCTURE (the bidder is shown the same figures):\n"
    "Base production cost: $85.0\n\n"
    "DELIVERY COST (faster = more expensive, quadratic):\n"
    "    1wk: +$24.5\n    2wk: +$18.0\n    3wk: +$12.5\n    4wk: +$8.0\n    5wk: +$4.5\n    6wk: +$2.0\n"
    "    7wk: +$0.5\n    8wk: +$0.0\n\n"
    "QUALITY COST (higher tier = more expensive, quadratic):\n"
    "    T2: +$7.2\n    T3: +$16.2\n    T4: +$28.8\n    T5: +$45.0\n\n"
    "EXAMPLE TOTAL COSTS (base + delivery + quality):\n"
    "    2wk/T2 → total cost $110.2\n    3wk/T3 → total cost $113.7\n    4wk/T4 → total cost $121.8\n"
    "    5wk/T5 → total cost $134.5\n    6wk/T2 → total cost $94.2\n    2wk/T5 → total cost $148.0\n"
    "    7wk/T3 → total cost $101.7\n\n"
    "COST RANGE: The cheapest possible configuration (8wk/T2) costs $92.2. The most expensive (1wk/T5) costs "
    "$154.5.\n\n"
    "Total cost = base + delivery cost + quality cost.")


def expected_analyst_block(cost_type):
    """The bidder's cost structure turned into the analyst's by the changes of the design, one by one."""
    bidder = make_supplier("A", cost_type).describe_for_primitives(cost_type)
    text = bidder.split("\n\n", 1)[1]                         # without the sentence that names the cost type
    for old, new in ((COST_HEADER_ORIGINAL, COST_HEADER_ANALYST),
                     ("Your cheapest possible configuration", "The cheapest possible configuration"),
                     ("Your most expensive", "The most expensive"), ("Your total cost = base", "Total cost = base")):
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    return text


# ---- the registry ----

def test_the_departures_organizations_are_the_bases_and_their_conflict_variants():
    assert DEPARTURE_ORGANIZATIONS == DEPARTURE_BASES + DEPARTURE_REVENUE_ORGANIZATIONS
    assert DEPARTURE_REVENUE_ORGANIZATIONS == tuple(f"{base}_revenue" for base in DEPARTURE_BASES)
    assert {departure_base(name) for name in DEPARTURE_ORGANIZATIONS} == set(DEPARTURE_BASES)
    assert all(departure_base(name) == name.removesuffix("_revenue") for name in DEPARTURE_ORGANIZATIONS)
    assert all(departure_base(name) is None for name in PRIMITIVES_ORGANIZATIONS + ("organized_guided", "hybrid"))


def test_the_six_earlier_organizations_and_their_groups_are_as_they_were():
    assert PRIMITIVES_ORGANIZATIONS == ("single", "single_norecord", "organized", "organized_pass", "single_revenue",
                                        "organized_revenue")
    assert PRIMITIVES_ANALYST_ORGANIZATIONS == ("organized", "organized_revenue")
    assert PRIMITIVES_REVENUE_ORGANIZATIONS == ("single_revenue", "organized_revenue")
    assert ALL_PRIMITIVES_ORGANIZATIONS == PRIMITIVES_ORGANIZATIONS + DEPARTURE_ORGANIZATIONS
    assert ALL_PRIMITIVES_ANALYST_ORGANIZATIONS == PRIMITIVES_ANALYST_ORGANIZATIONS + DEPARTURE_ANALYST_ORGANIZATIONS
    assert ALL_PRIMITIVES_REVENUE_ORGANIZATIONS == PRIMITIVES_REVENUE_ORGANIZATIONS + DEPARTURE_REVENUE_ORGANIZATIONS
    assert not set(DEPARTURE_ORGANIZATIONS) & set(PRIMITIVES_ORGANIZATIONS)


def test_every_departures_organization_has_an_analyst_that_calls_a_model_except_forward():
    assert DEPARTURE_ANALYST_ORGANIZATIONS == tuple(name for name in DEPARTURE_ORGANIZATIONS
                                                    if not name.startswith("forward"))
    assert {name for name in DEPARTURE_ORGANIZATIONS if forwards_the_record(name)} == {"forward", "forward_revenue"}
    assert forwards_the_record("organized_pass") and not forwards_the_record("organized")
    assert not any(forwards_the_record(name) for name in DEPARTURE_ANALYST_ORGANIZATIONS)


# ---- the system prompts ----

def expected_bidder_prompt(organization, environment):
    base = organization.removesuffix("_revenue")
    conflict = organization.endswith("_revenue")
    role = ROLE_FLOOR if base == "floor" else ROLE_DEPARTURE
    opening = OPENING_REVENUE if conflict else OPENING_TEAM
    objective = [REVENUE_BIDDER + " " + ANALYST_PROFIT] if conflict else []
    return "\n\n".join([SHARED[(opening, environment)], role, *objective, OUTPUT])


def expected_analyst_prompt(organization, environment):
    base = organization.removesuffix("_revenue")
    conflict = organization.endswith("_revenue")
    opening = OPENING_REVENUE if conflict else OPENING_TEAM
    role = {"brief_cost": ROLE_ANALYST_COST, "brief_cap": ROLE_ANALYST_CAP, "floor": ROLE_ANALYST_FLOOR}.get(
        base, ROLE_ANALYST)
    return "\n\n".join([SHARED[(opening, environment)], role] + ([REVENUE_ANALYST] if conflict else []))


SHARED_ROLE_ORGANIZATIONS = [name for name in DEPARTURE_ORGANIZATIONS if departure_base(name) in SHARED_ROLE_BASES]
BIDDER_PROMPT_ORGANIZATIONS = list(DEPARTURE_ORGANIZATIONS)
ANALYST_PROMPT_ORGANIZATIONS = list(DEPARTURE_ANALYST_ORGANIZATIONS)


def test_the_role_blocks_of_the_departures_organizations_are_these_exact_texts():
    assert PP.DEPARTURE_ROLE == ROLE_DEPARTURE
    assert PP.ANALYST_COST_ROLE == ROLE_ANALYST_COST and ROLE_ANALYST_COST != ROLE_ANALYST
    assert PP.ANALYST_CAP_ROLE == ROLE_ANALYST_CAP and ROLE_ANALYST_CAP.startswith(ROLE_ANALYST + " ")
    assert PP.FLOOR_ROLE == ROLE_FLOOR and PP.ANALYST_FLOOR_ROLE == ROLE_ANALYST_FLOOR
    assert ROLE_FLOOR.startswith(
        BIDDER_ROLE[:-1] + ", within one limit that your company's procurement analyst may set. ")
    assert PP.ANALYST_ROLE == ROLE_ANALYST
    assert ROLE_DEPARTURE.startswith(BIDDER_ROLE + " ")
    for name in SHARED_ROLE_ORGANIZATIONS:
        assert PP.ROLES[name] == ROLE_DEPARTURE, name
    for name in ("floor", "floor_revenue"):
        assert PP.ROLES[name] == ROLE_FLOOR != ROLE_DEPARTURE
    for name in ANALYST_PROMPT_ORGANIZATIONS:
        assert PP.ANALYST_ROLES[name] == {"brief_cost": ROLE_ANALYST_COST, "brief_cap": ROLE_ANALYST_CAP,
                                          "floor": ROLE_ANALYST_FLOOR}.get(departure_base(name), ROLE_ANALYST)


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("organization", BIDDER_PROMPT_ORGANIZATIONS)
def test_the_bidders_system_prompt_is_the_shared_block_its_role_text_and_the_output_format(
        organization, environment):
    assert PP.bidder_prompt(organization, 6, environment) == expected_bidder_prompt(organization, environment)


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("organization", ANALYST_PROMPT_ORGANIZATIONS)
def test_the_analysts_system_prompt_is_the_shared_block_and_the_analysts_role_text(organization, environment):
    assert PP.analyst_prompt(organization, 6, environment) == expected_analyst_prompt(organization, environment)


@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_the_bidder_prompt_is_the_same_in_brief_forward_and_brief_cost_within_an_objective(environment):
    for suffix in ("", "_revenue"):
        prompts = {PP.bidder_prompt(base + suffix, 6, environment) for base in SHARED_ROLE_BASES}
        assert len(prompts) == 1, suffix
    team = PP.bidder_prompt("brief", 6, environment)
    conflict = PP.bidder_prompt("brief_revenue", 6, environment)
    assert team != conflict and team != PP.bidder_prompt("organized", 6, environment)
    assert PP.bidder_prompt("floor", 6, environment) != team                 # floor has its own role text, from event 1
    assert PP.bidder_prompt("floor_revenue", 6, environment) != conflict


def test_the_conflict_bidder_of_every_departures_organization_is_told_the_analysts_objective_forward_included():
    for base in BASES:
        text = PP.bidder_prompt(f"{base}_revenue", 6, "random")
        assert REVENUE_BIDDER + " " + ANALYST_PROFIT in text and text.count(ANALYST_PROFIT) == 1
        assert OPENING_REVENUE in text and OPENING_TEAM not in text
        team = PP.bidder_prompt(base, 6, "random")
        assert OPENING_TEAM in team and "revenue" not in team
    for base in ("brief", "floor", "brief_cap", "brief_cost"):
        assert REVENUE_ANALYST in PP.analyst_prompt(f"{base}_revenue", 6, "random")
        assert "revenue" not in PP.analyst_prompt(base, 6, "random")


def test_forward_has_no_analyst_prompt_and_unknown_names_are_refused():
    for name in ("forward", "forward_revenue", "single", "organized_guided"):
        with pytest.raises(ValueError, match="unknown organization of the primitives set"):
            PP.analyst_prompt(name, 6, "random")
    with pytest.raises(ValueError, match="unknown organization of the primitives set"):
        PP.bidder_prompt("briefing", 6, "random")


def test_the_system_prompts_of_the_departures_organizations_are_pinned_by_digest():
    found = {f"{kind} {organization} {environment}": digest(prompt(organization, 6, environment))
             for environment in ENVIRONMENTS
             for kind, prompt, organizations in (("bidder", PP.bidder_prompt, BIDDER_PROMPT_ORGANIZATIONS),
                                                 ("analyst", PP.analyst_prompt, ANALYST_PROMPT_ORGANIZATIONS))
             for organization in organizations}
    assert found == EXPECTED_DEPARTURE_PROMPT_DIGESTS


EXPECTED_DEPARTURE_PROMPT_DIGESTS = {
    "bidder brief random": "55675ee635526493",
    "bidder forward random": "55675ee635526493",
    "bidder floor random": "50002be10d7f6605",
    "bidder brief_cap random": "55675ee635526493",
    "bidder brief_cost random": "55675ee635526493",
    "bidder brief_revenue random": "cdb21ab2a9e32493",
    "bidder forward_revenue random": "cdb21ab2a9e32493",
    "bidder floor_revenue random": "3c37449bbfba9523",
    "bidder brief_cap_revenue random": "cdb21ab2a9e32493",
    "bidder brief_cost_revenue random": "cdb21ab2a9e32493",
    "analyst brief random": "c8cacec75b475d07",
    "analyst floor random": "7851f88df717eaee",
    "analyst brief_cap random": "436a558ca0cb4f4f",
    "analyst brief_cost random": "2af9fef121009945",
    "analyst brief_revenue random": "0a666882ae243686",
    "analyst floor_revenue random": "543faa1cdf8edf39",
    "analyst brief_cap_revenue random": "49985115faa836f0",
    "analyst brief_cost_revenue random": "b91dee68caee7bf7",
    "bidder brief persistent": "620e8f7be1e914ec",
    "bidder forward persistent": "620e8f7be1e914ec",
    "bidder floor persistent": "fd0e7a2f20c6be56",
    "bidder brief_cap persistent": "620e8f7be1e914ec",
    "bidder brief_cost persistent": "620e8f7be1e914ec",
    "bidder brief_revenue persistent": "ca0b1f0d243b6e15",
    "bidder forward_revenue persistent": "ca0b1f0d243b6e15",
    "bidder floor_revenue persistent": "ae2d60d838570b57",
    "bidder brief_cap_revenue persistent": "ca0b1f0d243b6e15",
    "bidder brief_cost_revenue persistent": "ca0b1f0d243b6e15",
    "analyst brief persistent": "cbcff0999327e31b",
    "analyst floor persistent": "24cd227c1bd57013",
    "analyst brief_cap persistent": "43e6a3ae648c95a9",
    "analyst brief_cost persistent": "42279b62252259e9",
    "analyst brief_revenue persistent": "5aec536aed1dfb5d",
    "analyst floor_revenue persistent": "a0a6365953eef3b3",
    "analyst brief_cap_revenue persistent": "4f2862cb57b93a44",
    "analyst brief_cost_revenue persistent": "cfcecf80af10c6bc",
}


# ---- what the bidder reads about past events ----

@pytest.mark.parametrize("organization", DEPARTURE_ORGANIZATIONS)
def test_every_departures_bidder_reads_the_same_line_before_its_first_event(organization):
    assert build_primitives_bidder_context(organization, "", None) == FIRST_EVENT_NOTE


def test_the_headings_of_the_message_are_these_exact_texts():
    assert BRIEF_HEADER == HEADER_BRIEF and FORWARD_HEADER == HEADER_FORWARD and CAP_HEADER == HEADER_CAP
    assert FLOOR_HEADER == HEADER_FLOOR and len({HEADER_BRIEF, HEADER_CAP, HEADER_FORWARD, HEADER_FLOOR}) == 4
    assert str(BRIEFING_WORD_LIMIT) in HEADER_CAP and f"at most {BRIEFING_WORD_LIMIT} words" in ROLE_ANALYST_CAP
    for header in (HEADER_BRIEF, HEADER_CAP, HEADER_FORWARD, HEADER_FLOOR):    # all begin with the words the role names
        assert header.startswith("MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST")
        assert ROLE_DEPARTURE.count("MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST") == 1


RECORD = "YOUR COMPANY'S RECORD OF PAST EVENTS ON THIS PLATFORM:\nRECORD LINES\n"


@pytest.mark.parametrize("suffix", ["", "_revenue"])
def test_what_each_departures_bidder_reads_of_past_events_from_the_second_event_on(suffix):
    for base in ("brief", "brief_cost"):
        assert build_primitives_bidder_context(base + suffix, RECORD, "A BRIEFING") == HEADER_BRIEF + "\nA BRIEFING"
    assert build_primitives_bidder_context("brief_cap" + suffix, RECORD, "A BRIEFING") == HEADER_CAP + "\nA BRIEFING"
    assert build_primitives_bidder_context("floor" + suffix, RECORD, "A BRIEFING\n\nA NOTICE") == (
        HEADER_FLOOR + "\nA BRIEFING\n\nA NOTICE")            # the heading of floor says that the last line is a notice
    assert build_primitives_bidder_context("forward" + suffix, RECORD, None) == HEADER_FORWARD + "\n" + RECORD


def test_forward_forwards_the_single_bidders_record_with_a_heading_and_no_briefing():
    single = build_primitives_bidder_context("single", RECORD, None)
    assert single == RECORD
    for name in ("forward", "forward_revenue"):
        assert build_primitives_bidder_context(name, RECORD, None) == HEADER_FORWARD + "\n" + single
        with pytest.raises(ValueError, match="has no analyst briefing"):
            build_primitives_bidder_context(name, RECORD, "a briefing")


def test_the_bidder_contexts_of_the_departures_organizations_refuse_what_does_not_fit():
    for name in DEPARTURE_ANALYST_ORGANIZATIONS:
        with pytest.raises(ValueError, match="needs its analyst's briefing"):
            build_primitives_bidder_context(name, RECORD, None)
        with pytest.raises(ValueError, match="no briefing before its first event"):
            build_primitives_bidder_context(name, "", "a briefing")
    with pytest.raises(ValueError, match="unknown organization of the primitives set"):
        build_primitives_bidder_context("organized_guided", RECORD, None)


# ---- the cost structure ----

def test_the_third_person_cost_structure_is_this_exact_text():
    assert make_supplier("A", "speed").describe_for_analyst() == ANALYST_BLOCK_SPEED


@pytest.mark.parametrize("cost_type", ["speed", "quality"])
def test_the_analysts_cost_structure_is_the_bidders_with_the_changes_of_the_design_and_the_figures_agree(cost_type):
    supplier = make_supplier("A", cost_type)
    analyst, bidder = supplier.describe_for_analyst(), supplier.describe_for_primitives(cost_type)
    assert analyst == expected_analyst_block(cost_type)
    numbers = r"[0-9]+(?:[.][0-9]+)?"
    assert re.findall(numbers, analyst) == re.findall(numbers, bidder.split("\n\n", 1)[1])      # the same figures
    assert not re.findall(r"\b(?:you|your(?! company)|yours|yourself)\b", analyst, re.I)         # no second person left
    assert "Your" not in analyst and "only you know" not in analyst


def test_the_bidders_cost_structure_has_its_original_title_unless_it_is_told_to_have_another():
    supplier = make_supplier("A", "quality")
    original = supplier.describe_for_primitives("quality")
    assert original == supplier.describe_for_primitives("quality", COST_HEADER_ORIGINAL)
    assert original.startswith("In this event your costs are those of a quality specialist.\n\n"
                               + COST_HEADER_ORIGINAL + "\nBase production cost: $85.0\n")
    shared = supplier.describe_for_primitives("quality", COST_HEADER_BIDDER)
    assert shared == original.replace(COST_HEADER_ORIGINAL, COST_HEADER_BIDDER) and shared != original
    with pytest.raises(ValueError, match="'speed' or 'quality'"):
        supplier.describe_for_primitives("balanced", COST_HEADER_BIDDER)


def test_the_analysts_message_with_costs_is_this_exact_text_and_without_them_it_is_as_it_was():
    plain = PP.analyst_message(3, 6, "RULES TEXT", "RECORD TEXT")
    assert plain == ("EVENT 3 of 6 is next. Your company's bidder reads these rules with each event:\n"
                     "RULES TEXT\n\n"
                     "Here is your company's record of past events on this platform:\n"
                     "RECORD TEXT\n\n"
                     "Write your briefing for your company's bidder.")
    assert PP.analyst_message(3, 6, "RULES TEXT", "RECORD TEXT", None) == plain
    shown = PP.analyst_message(3, 6, "RULES TEXT", "RECORD TEXT", ("speed", "A COST BLOCK"))
    assert shown == ("EVENT 3 of 6 is next. Your company's bidder reads these rules with each event:\n"
                     "RULES TEXT\n\n"
                     "Here is your company's record of past events on this platform:\n"
                     "RECORD TEXT\n\n"
                     "In the coming event your company's costs are those of a speed specialist.\n\n"
                     "A COST BLOCK\n\n"
                     "Write your briefing for your company's bidder.")


# ---- usage labels ----

class Counting:
    """A client that answers 'ok' with a usage of 10 prompt tokens and 5 completion tokens."""

    def __init__(self):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @staticmethod
    def _create(**kw):
        message = SimpleNamespace(content="ok", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5))


def test_a_labeled_client_counts_its_requests_under_its_label_whatever_the_system_prompt():
    client = ChatClient(Counting(), agent_labels={"PROMPT": "by_prompt"})
    one, two = client.labeled("bidder_brief"), client.labeled("bidder_forward")
    assert isinstance(one, LabeledClient) and one.root is client and one.label == "bidder_brief"
    messages = [{"role": "system", "content": "PROMPT"}, {"role": "user", "content": "hello"}]
    for view in (one, one, two):
        assert view.chat.completions.create(model="m", messages=messages).choices[0].message.content == "ok"
    client.chat.completions.create(model="m", messages=messages)                 # the client itself: by the prompt
    client.chat.completions.create(model="m", messages=[{"role": "system", "content": "OTHER"}])
    by_agent = client.usage_summary()["by_agent"]
    assert {label: counts["requests"] for label, counts in by_agent.items()} == {
        "bidder_brief": 2, "bidder_forward": 1, "by_prompt": 1, "other": 1}
    assert by_agent["bidder_brief"]["prompt_tokens"] == 20 and by_agent["bidder_forward"]["completion_tokens"] == 5
    assert client.usage_summary()["requests"] == 5                              # every view counts in the one client
    assert not hasattr(one, "usage_summary")                                    # and has no counts of its own


def test_a_labeled_client_sends_what_the_client_sends_and_counts_a_failure_too():
    seen = []

    class Fails:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kw):
            seen.append(kw)
            raise RuntimeError("down")

    client = ChatClient(Fails(), local=False, default_sampling=True, max_tokens=50)
    with pytest.raises(RuntimeError, match="down"):
        client.labeled("analyst_brief").chat.completions.create(
            model="m", temperature=0.3, extra_body={"top_k": 20}, messages=[{"role": "system", "content": "S"}])
    assert seen == [{"model": "m", "messages": [{"role": "system", "content": "S"}], "max_tokens": 50}]
    counts = client.usage_summary()["by_agent"]["analyst_brief"]
    assert (counts["requests"], counts["failed_requests"]) == (1, 1)


# ---- brief_cap: the words of a briefing, the follow-up, the cut ----

def words(n, prefix="w"):
    """A text of n words, `prefix`1 `prefix`2 ... separated by one space."""
    return " ".join(f"{prefix}{i}" for i in range(1, n + 1))


class ScriptedAnalyst(ProcurementAnalyst):
    """An analyst that answers with the replies it is given, one by one, and keeps what it was sent."""

    def __init__(self, *replies):
        self.replies, self.sent = list(replies), []

    def write_briefing(self, supplier_id, message, earlier_turns=()):
        self.sent.append((supplier_id, message, tuple(earlier_turns)))
        return AnalystBriefing(None, self.replies[len(self.sent) - 1])


def cap_reask(count, limit=50):
    """The follow-up to the analyst of brief_cap, written out again."""
    return (f"Your briefing has {count} words; the limit is {limit} words. Write it again within the limit. "
            f"If it still has more than {limit} words, the bidder reads only its first {limit} words.")


def test_the_limit_and_the_follow_up_are_these_exact_texts():
    assert BRIEFING_WORD_LIMIT == 50
    assert CAP_REASK.format(words=73, limit=50) == (
        "Your briefing has 73 words; the limit is 50 words. Write it again within the limit. "
        "If it still has more than 50 words, the bidder reads only its first 50 words.") == cap_reask(73)
    assert follow_up_turns("FIRST", "NEXT") == ({"role": "assistant", "content": "FIRST"},
                                                  {"role": "user", "content": "NEXT"})


def test_words_are_counted_by_splitting_on_white_space_and_a_cut_keeps_what_comes_before_the_last_word():
    text = "one  two\tthree\nfour\u00a0five\r\n\n six   "
    assert word_count(text) == len(text.split()) == 6 and word_count("") == word_count(" \n ") == 0
    assert word_count("a-b c,d e.f") == 3                                          # no tokenizer: pieces between spaces
    for n in range(0, 9):
        assert word_count(cut_to_words(text, n)) == min(n, 6)
    assert cut_to_words(text, 3) == "one  two\tthree" and cut_to_words(text, 4) == "one  two\tthree\nfour"
    # the text itself, if it fits
    assert cut_to_words(text, 6) == text and cut_to_words(text, 7) == text
    assert cut_to_words(text, 5) == "one  two\tthree\nfour\u00a0five" and cut_to_words(text, 0) == ""
    assert cut_to_words(words(60), 50) == words(50) and cut_to_words(words(50), 50) == words(50)


@pytest.mark.parametrize("count", [1, 25, 49, 50])
def test_a_briefing_within_the_limit_is_taken_as_written_and_the_analyst_is_not_asked_again(count):
    analyst = ScriptedAnalyst(words(count))
    briefing, record = capped_briefing(analyst, "A", "THE MESSAGE")
    assert (briefing.posture, briefing.text) == (None, words(count))
    assert analyst.sent == [("A", "THE MESSAGE", ())]
    assert record == {"word_limit": 50, "original": words(count), "original_words": count, "reasked": False,
                      "follow_up": None, "second": None, "second_words": None, "cut": False, "final_words": count}


@pytest.mark.parametrize("count", [51, 73, 400])
def test_a_longer_briefing_is_asked_for_again_in_the_same_conversation_and_a_second_within_the_limit_is_taken(count):
    first, second = words(count), words(30, "s")
    analyst = ScriptedAnalyst(first, second)
    briefing, record = capped_briefing(analyst, "B", "THE MESSAGE")
    follow_up = cap_reask(count)
    assert analyst.sent == [("B", "THE MESSAGE", ()),
                            ("B", "THE MESSAGE", ({"role": "assistant", "content": first},
                                                  {"role": "user", "content": follow_up}))]
    assert briefing.text == second and briefing.posture is None
    assert record == {"word_limit": 50, "original": first, "original_words": count, "reasked": True,
                      "follow_up": follow_up, "second": second, "second_words": 30, "cut": False, "final_words": 30}


def test_a_second_briefing_that_is_still_too_long_is_cut_to_its_first_50_words_and_the_analyst_is_asked_only_twice():
    first = words(60)
    second = "\n".join(words(10, f"line{i}_") for i in range(1, 8))                  # 70 words over 7 lines
    analyst = ScriptedAnalyst(first, second, "NEVER ASKED")
    briefing, record = capped_briefing(analyst, "A", "M")
    assert len(analyst.sent) == 2
    assert briefing.text == cut_to_words(second, 50) and word_count(briefing.text) == 50
    assert briefing.text.endswith("line5_10") and briefing.text.count("\n") == 4      # five whole lines, as written
    assert record["cut"] is True and record["reasked"] is True
    assert (record["original"], record["original_words"], record["second"], record["second_words"],
            record["final_words"]) == (first, 60, second, 70, 50)
    assert record["follow_up"] == cap_reask(60)
    cut_mid_line = ScriptedAnalyst(words(51), words(51, "x"))
    assert capped_briefing(cut_mid_line, "A", "M")[0].text == words(50, "x")


def test_the_first_briefing_is_not_cut_when_the_second_is_within_the_limit_and_the_limit_can_be_given():
    analyst = ScriptedAnalyst(words(8), words(3), words(9))
    briefing, record = capped_briefing(analyst, "A", "M", limit=5)
    assert briefing.text == words(3) and record["word_limit"] == 5 and record["cut"] is False
    assert record["follow_up"] == cap_reask(8, limit=5)


# ---- floor: the line that ends the briefing, the notice, the re-asks ----

def test_the_texts_of_the_floor_are_these_exact_texts():
    from orgsim import departures as D
    assert D.FLOOR_REASK_ANALYST == REASK_ANALYST
    assert D.FLOOR_NOTICE.format(value="$5.00") == NOTICE + "$5.00"
    assert D.FLOOR_NOTICE.format(value="none") == NOTICE + "none"
    assert D.FLOOR_RAISED_NOTE == RAISED_NOTE
    assert D.floor_notice(None) == NOTICE + "none" and D.floor_notice(5) == NOTICE + "$5.00"
    assert D.floor_notice(12.5) == NOTICE + "$12.50" and D.floor_notice(0.0) == NOTICE + "$0.00"
    assert D.floor_notice(7.126) == NOTICE + "$7.13"
    assert D.FLOOR_NOTICE_NOT_SET == NOTICE_NOT_SET and NOTICE_NOT_SET != NOTICE + "none"
    assert D.FLOOR_LINE_PREFIX == "MINIMUM PROFIT PER CONTRACT" and D.FLOOR_LINE_START == D.FLOOR_LINE_PREFIX + ":"
    assert D.FLOOR_RAISED_NOTE.startswith(" (price raised to its total cost plus the minimum profit ")
    expected = (REASK_BIDDER.replace("$P", "$122.00").replace("$C", "$104.80").replace("$X", "$30.00"))
    assert D.floor_reask_text(122.0, 104.8, 30.0) == expected
    # the price and the cost with the digits they have beyond the cent, the minimum to the cent
    assert D.floor_reask_text(99.999, 104.8049, 7.5) == (
        REASK_BIDDER.replace("$P", "$99.999").replace("$C", "$104.8049").replace("$X", "$7.50"))
    assert D.floor_reask_text(140.02, 110.0245, 30.0) == (
        REASK_BIDDER.replace("$P", "$140.02").replace("$C", "$110.0245").replace("$X", "$30.00"))
    assert D.FLOOR_LINE_START == "MINIMUM PROFIT PER CONTRACT:"


SET = "set"
NONE = ("none", None)
MALFORMED = ("malformed", None)


@pytest.mark.parametrize("text, expected", [
    # ---- what has always been read
    ("Hold firm.\nMINIMUM PROFIT PER CONTRACT: $5", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: $5.25", (SET, 5.25)),
    ("x\nMINIMUM PROFIT PER CONTRACT: $0", (SET, 0.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: $0.00", (SET, 0.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: $1234.5", (SET, 1234.5)),
    ("x\nMINIMUM PROFIT PER CONTRACT: none", NONE),
    ("MINIMUM PROFIT PER CONTRACT: $9", (SET, 9.0)),                                  # the only line
    ("x\nMINIMUM PROFIT PER CONTRACT:$7", (SET, 7.0)),                                # no space after the colon
    ("x\nMINIMUM PROFIT PER CONTRACT:    $7", (SET, 7.0)),                             # several spaces
    ("x\nMINIMUM PROFIT PER CONTRACT: $ 7.5", (SET, 7.5)),                            # a space after the dollar sign
    ("x\n  MINIMUM PROFIT PER CONTRACT: $12.5   \n\n\n", (SET, 12.5)),           # white space around, blank lines after
    ("x\r\nMINIMUM PROFIT PER CONTRACT: none\r\n", NONE),                             # Windows line ends
    ("x\nMINIMUM PROFIT PER CONTRACT: none   ", NONE),
    ("x\nMINIMUM PROFIT PER CONTRACT: $5.", (SET, 5.0)),                              # a trailing point of the number
    ("x\nMINIMUM PROFIT PER CONTRACT: $.5", (SET, 0.5)),
    ("x\nMINIMUM PROFIT PER CONTRACT: $12.346", (SET, 12.35)),                    # to the cent, as the notice prints it
    ("x\nMINIMUM PROFIT PER CONTRACT: $0.004", (SET, 0.0)),
    # the last line counts
    ("x\n\nMINIMUM PROFIT PER CONTRACT: $3\nMINIMUM PROFIT PER CONTRACT: none", NONE),
    ("x\nMINIMUM PROFIT PER CONTRACT: none\nMINIMUM PROFIT PER CONTRACT: $3", (SET, 3.0)),
    # ---- 18c: the words and none in any case
    ("x\nminimum profit per contract: $5", (SET, 5.0)),
    ("x\nMinimum Profit Per Contract: $5", (SET, 5.0)),
    ("x\nMinimum profit per contract: $5", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: None", NONE),
    ("x\nMINIMUM PROFIT PER CONTRACT: NONE", NONE),
    ("x\nminimum profit per contract: none", NONE),
    # the dollar sign is optional
    ("x\nMINIMUM PROFIT PER CONTRACT: 5", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: 5.25", (SET, 5.25)),
    ("x\nMINIMUM PROFIT PER CONTRACT: .5", (SET, 0.5)),
    ("x\nMINIMUM PROFIT PER CONTRACT:5", (SET, 5.0)),
    # markdown marks at the ends of the line: emphasis, code, list, heading and quote marks
    ("x\n**MINIMUM PROFIT PER CONTRACT: $5**", (SET, 5.0)),
    ("x\n__MINIMUM PROFIT PER CONTRACT: none__", NONE),
    ("x\n*MINIMUM PROFIT PER CONTRACT: $5*", (SET, 5.0)),
    ("x\n`MINIMUM PROFIT PER CONTRACT: $5`", (SET, 5.0)),
    ("x\n- MINIMUM PROFIT PER CONTRACT: $5", (SET, 5.0)),
    ("x\n+ MINIMUM PROFIT PER CONTRACT: $5", (SET, 5.0)),
    ("x\n• MINIMUM PROFIT PER CONTRACT: $5", (SET, 5.0)),
    ("x\n### MINIMUM PROFIT PER CONTRACT: $5.00", (SET, 5.0)),
    ("x\n> MINIMUM PROFIT PER CONTRACT: $5.", (SET, 5.0)),
    ("x\n> - **MINIMUM PROFIT PER CONTRACT: $5**", (SET, 5.0)),
    ("x\n  *  MINIMUM PROFIT PER CONTRACT: $5  ", (SET, 5.0)),
    ("x\n**Minimum profit per contract: $5**", (SET, 5.0)),
    # marks and white space between the words, the colon and the value
    ("x\n**MINIMUM PROFIT PER CONTRACT:** $5", (SET, 5.0)),
    ("x\n**MINIMUM PROFIT PER CONTRACT**: $5", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: **$5**", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: $**5**", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: *none*", NONE),
    ("x\n**Minimum profit per contract:** none", NONE),
    ("x\n_MINIMUM PROFIT PER CONTRACT_: $ 5", (SET, 5.0)),
    # a period after the value, none included, and marks after the period
    ("x\nMINIMUM PROFIT PER CONTRACT: none.", NONE),
    ("x\n**MINIMUM PROFIT PER CONTRACT: none.**", NONE),
    ("x\nMINIMUM PROFIT PER CONTRACT: None.", NONE),
    ("x\nMINIMUM PROFIT PER CONTRACT: $5.25.", (SET, 5.25)),
    ("x\nMINIMUM PROFIT PER CONTRACT: $5.**", (SET, 5.0)),
    # more than one white-space character between the words of the label, and a space before the colon
    ("x\nMINIMUM  PROFIT   PER\tCONTRACT: $5", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: none", NONE),
    ("x\nMINIMUM PROFIT PER CONTRACT : $5", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT   :   none", NONE),
    ("x\nMINIMUM PROFIT PER CONTRACT\t: $5", (SET, 5.0)),
    # ---- 18d: a period after closing marks
    ("x\n**MINIMUM PROFIT PER CONTRACT: $5**.", (SET, 5.0)),
    ("x\n**MINIMUM PROFIT PER CONTRACT: none**.", NONE),
    ("x\n__MINIMUM PROFIT PER CONTRACT: $5.25__.", (SET, 5.25)),
    ("x\n`MINIMUM PROFIT PER CONTRACT: $5`.", (SET, 5.0)),
    ("x\n*MINIMUM PROFIT PER CONTRACT: none*.", NONE),
    ("x\nMINIMUM PROFIT PER CONTRACT: $5 .", (SET, 5.0)),
    ("x\n**MINIMUM PROFIT PER CONTRACT: $5.**", (SET, 5.0)),
    # ---- 18d: an en dash, a white circle and a list number in front of the line
    ("x\n– MINIMUM PROFIT PER CONTRACT: $5", (SET, 5.0)),
    ("x\n◦ MINIMUM PROFIT PER CONTRACT: none", NONE),
    ("x\n– **MINIMUM PROFIT PER CONTRACT:** $5", (SET, 5.0)),
    ("x\n1. MINIMUM PROFIT PER CONTRACT: $5", (SET, 5.0)),
    ("x\n2) MINIMUM PROFIT PER CONTRACT: $7.50", (SET, 7.5)),
    ("x\n10. **MINIMUM PROFIT PER CONTRACT:** none.", NONE),
    ("x\n3.MINIMUM PROFIT PER CONTRACT: $5", (SET, 5.0)),
    ("x\n  - 1) *MINIMUM PROFIT PER CONTRACT: none*", NONE),
    # ---- 18d: lines of marks after the floor line are skipped, and the floor line is the last other line
    ("x\nMINIMUM PROFIT PER CONTRACT: $5\n---", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: $5\n***", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: $5\n___", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: $5\n```", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: none\n\n   ---   \n\n", NONE),
    ("x\nMINIMUM PROFIT PER CONTRACT: $5\n===\n~~~", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: $5\n- - -\n* * *", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: $5\n———", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: $5\n|---|---|", (SET, 5.0)),
    ("x\nMINIMUM PROFIT PER CONTRACT: $5\n**\n–\n◦\n•", (SET, 5.0)),
    ("```\nMINIMUM PROFIT PER CONTRACT: none\n```", NONE),
    ("---\nMINIMUM PROFIT PER CONTRACT: $5\n---", (SET, 5.0)),
])
def test_the_floor_line_is_read_from_the_last_line_that_is_not_only_marks_with_the_tolerance_of_the_design(
        text, expected):
    from orgsim.departures import parse_floor_line
    assert parse_floor_line(text) == expected


@pytest.mark.parametrize("text", [
    "", "   \n  ", "A briefing and no line.", "***", "---", "```", "x\n**",
    # not a non-negative number
    "x\nMINIMUM PROFIT PER CONTRACT: $-5", "x\nMINIMUM PROFIT PER CONTRACT: -5", "x\nMINIMUM PROFIT PER CONTRACT: $+5",
    "x\nMINIMUM PROFIT PER CONTRACT: $ -5", "x\nMINIMUM PROFIT PER CONTRACT: -$5", "x\nMINIMUM PROFIT PER CONTRACT: +5",
    "x\nMINIMUM PROFIT PER CONTRACT: $", "x\nMINIMUM PROFIT PER CONTRACT:", "x\nMINIMUM PROFIT PER CONTRACT: $.",
    "x\nMINIMUM PROFIT PER CONTRACT: $1,000", "x\nMINIMUM PROFIT PER CONTRACT: $5e3",
    "x\nMINIMUM PROFIT PER CONTRACT: $1.2.3", "x\nMINIMUM PROFIT PER CONTRACT: $abc",
    "x\nMINIMUM PROFIT PER CONTRACT: five", "x\nMINIMUM PROFIT PER CONTRACT: $5%",
    "x\nMINIMUM PROFIT PER CONTRACT: $5..",
    # words, or anything else, after the value
    "x\nMINIMUM PROFIT PER CONTRACT: $5 per contract", "x\nMINIMUM PROFIT PER CONTRACT: 5 per contract",
    "x\nMINIMUM PROFIT PER CONTRACT: $5, because costs are high", "x\nMINIMUM PROFIT PER CONTRACT: none of it",
    "x\nMINIMUM PROFIT PER CONTRACT: none, because ...", "x\nMINIMUM PROFIT PER CONTRACT: $5 USD",
    "x\n**Minimum profit per contract:** $2, because costs are high",
    "x\nMINIMUM PROFIT PER CONTRACT: $5 MINIMUM PROFIT PER CONTRACT: none",
    # ranges, two values, a value in a form of its own
    "x\nMINIMUM PROFIT PER CONTRACT: $5-10", "x\nMINIMUM PROFIT PER CONTRACT: $5 - $10",
    "x\nMINIMUM PROFIT PER CONTRACT: $5 to $10", "x\nMINIMUM PROFIT PER CONTRACT: $5 or $6",
    "x\nMINIMUM PROFIT PER CONTRACT: 5/10", "x\nMINIMUM PROFIT PER CONTRACT: $5 none",
    "x\nMINIMUM PROFIT PER CONTRACT: $none",
    # not the whole line: other words, or a different label before or in the words
    "x MINIMUM PROFIT PER CONTRACT: $5", "x\nMy floor is MINIMUM PROFIT PER CONTRACT: $5",
    "x\nMINIMUM PROFIT: $5", "x\nMINIMUM PROFIT PER CONTRACT",
    "x\nMINIMUM PROFITS PER CONTRACT: $5", "x\nMINIMUM_PROFIT_PER_CONTRACT: $5",
    # no colon
    "x\nMINIMUM PROFIT PER CONTRACT $5", "x\nMINIMUM PROFIT PER CONTRACT - $5", "x\nMINIMUM PROFIT PER CONTRACT; $5",
    # the floor line is not the last line
    "MINIMUM PROFIT PER CONTRACT: $5\nand a last line of text",
    "x\nMINIMUM PROFIT PER CONTRACT: $5\n(this is my floor)",
    # 18d: the line after a rule is the candidate; nothing but marks leaves the line before them
    "x\nMINIMUM PROFIT PER CONTRACT: $5\n---\nand a last line of text", "x\n---\n---", "---", "```", "***\n---",
    "x\n**\n––",
    "x\nMINIMUM PROFIT PER CONTRACT: $5 ---", "x\n**MINIMUM PROFIT PER CONTRACT: $5**..",
    "x\nMINIMUM PROFIT PER CONTRACT: $5**x", "x\n1 MINIMUM PROFIT PER CONTRACT: $5",
    "x\nA. MINIMUM PROFIT PER CONTRACT: $5", "x\n1.5 MINIMUM PROFIT PER CONTRACT: $5",
    "x\nthe MINIMUM PROFIT PER CONTRACT: $5 ---", "x\nMINIMUM PROFIT PER CONTRACT: $5\n```text",
])
def test_a_last_line_that_is_not_a_floor_line_is_malformed(text):
    from orgsim.departures import parse_floor_line
    assert parse_floor_line(text) == MALFORMED


@pytest.mark.parametrize("text, expected", [
    # a floor line at the end, as the analyst writes it
    ("Hold firm.\nKeep to speed.\nMINIMUM PROFIT PER CONTRACT: $5", "Hold firm.\nKeep to speed."),
    ("Hold firm.\n\n\nMINIMUM PROFIT PER CONTRACT: $5  \n\n", "Hold firm."),
    ("MINIMUM PROFIT PER CONTRACT: none", ""),
    ("", ""),
    ("A\r\nB\r\nMINIMUM PROFIT PER CONTRACT: none\r\n", "A\r\nB"),
    # a line is removed if it begins with the words once the white space, the marks and a list number at its start are
    # removed, in any case and with white space between the words, whatever follows, wherever it is
    ("A\nMINIMUM PROFIT PER CONTRACT: $3\nB\n  MINIMUM PROFIT PER CONTRACT: none  \n", "A\nB"),
    ("A\n\tMINIMUM PROFIT PER CONTRACT: $3\n\nB\nMINIMUM PROFIT PER CONTRACT: $4", "A\n\nB"),
    ("MINIMUM PROFIT PER CONTRACT: $3\nA", "A"),
    ("A\nminimum profit per contract: $5\nB", "A\nB"),
    ("A\nMinimum Profit Per Contract: None\nB", "A\nB"),
    ("A\n**MINIMUM PROFIT PER CONTRACT: $5**\nB", "A\nB"),
    ("A\n- MINIMUM PROFIT PER CONTRACT: 5\nB", "A\nB"),
    ("A\n### Minimum profit per contract: $5.00\nB", "A\nB"),
    ("A\n> MINIMUM  PROFIT PER CONTRACT : none.\nB", "A\nB"),
    ("A\n**Minimum profit per contract:** $5\nB", "A\nB"),
    ("A\nMINIMUM PROFIT PER CONTRACT $9 per contract\nB", "A\nB"),
    ("A\nMINIMUM PROFIT PER CONTRACT: $5, because costs are high\nB", "A\nB"),
    ("A\n  **MINIMUM PROFIT PER CONTRACT:** $2, because costs are high\nB", "A\nB"),
    ("A\n**Minimum profit per contract:** $2, because costs are high\nB", "A\nB"),
    ("A\nminimum profit per contract is open\nB", "A\nB"),
    ("A\nminimum profit per contract $9 per contract\nB", "A\nB"),
    ("A\nMINIMUM  PROFIT PER CONTRACT $9 per contract\nB", "A\nB"),
    ("A\nMINIMUM\tPROFIT PER   CONTRACT\nB", "A\nB"),
    ("A\n- MINIMUM PROFIT PER CONTRACTS differ.\nB", "A\nB"),
    ("A\n# MINIMUM PROFIT PER CONTRACT is open\nB", "A\nB"),
    ("A\n> • MINIMUM PROFIT PER CONTRACT: none of it\nB", "A\nB"),
    ("A\n**MINIMUM PROFIT PER CONTRACT** is open\nB", "A\nB"),
    ("A\nMINIMUM PROFIT PER CONTRACT: $x\nB", "A\nB"),
    # the marks and the numbers of 18d: an en dash, a white circle, a list number
    ("A\n– MINIMUM PROFIT PER CONTRACT: $5\nB", "A\nB"),
    ("A\n◦ minimum profit per contract is open\nB", "A\nB"),
    ("A\n1. MINIMUM PROFIT PER CONTRACT: $5\nB", "A\nB"),
    ("A\n2) minimum  profit per contract: none.\nB", "A\nB"),
    ("A\n10. **Minimum profit per contract:** $2, because ...\nB", "A\nB"),
    ("A\n  - 1) *MINIMUM PROFIT PER CONTRACT: none*\nB", "A\nB"),
    # nothing else is removed: other words before the words, a different label, a letter or a decimal number in front, a
    # rule or a fence
    ("A\nI keep the MINIMUM PROFIT PER CONTRACT open.\nB", "A\nI keep the MINIMUM PROFIT PER CONTRACT open.\nB"),
    ("A MINIMUM PROFIT PER CONTRACT: $5", "A MINIMUM PROFIT PER CONTRACT: $5"),
    ("A\nMy MINIMUM PROFIT PER CONTRACT: $5\nB", "A\nMy MINIMUM PROFIT PER CONTRACT: $5\nB"),
    ("A\nMINIMUM PROFIT: $5\nB", "A\nMINIMUM PROFIT: $5\nB"),
    ("A\nMINIMUM PROFITS PER CONTRACT: $5\nB", "A\nMINIMUM PROFITS PER CONTRACT: $5\nB"),
    ("A\nA. MINIMUM PROFIT PER CONTRACT: $5\nB", "A\nA. MINIMUM PROFIT PER CONTRACT: $5\nB"),
    ("A\n1.5 MINIMUM PROFIT PER CONTRACT: $5\nB", "A\n1.5 MINIMUM PROFIT PER CONTRACT: $5\nB"),
    ("A\n1 MINIMUM PROFIT PER CONTRACT: $5\nB", "A\n1 MINIMUM PROFIT PER CONTRACT: $5\nB"),
    ("A\n---\nB", "A\n---\nB"),
    ("A\n```\nMINIMUM PROFIT PER CONTRACT: $5\n```", "A\n```\n```"),
    ("A\nMINIMUM PROFIT PER CONTRACT: $5\n---", "A\n---"),
])
def test_the_briefing_loses_the_lines_that_begin_with_the_words_and_nothing_else(text, expected):
    from orgsim.departures import strip_floor_lines
    assert strip_floor_lines(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("a\nb", "b"), ("a\nb\n\n", "b"), ("a\n b c \n", "b c"), ("", ""), ("   \n ", ""), ("one", "one"),
    ("a\r\nb\r\n", "b"),
    # a line of marks and white space alone is skipped
    ("a\nb\n---", "b"), ("a\nb\n***\n___\n```", "b"), ("a\nb\n  ---  \n\n", "b"), ("a\nb\n- - -\n= = =", "b"),
    ("a\nb\n–\n◦\n•\n+\n>\n#", "b"), ("a\nb\n|---|---|", "b"), ("a\nb\n———", "b"), ("a\nb\n~~~\n```", "b"),
    ("---\n---", ""), ("```", ""), ("a\n**b**\n**", "**b**"),
    # a line with anything else is not skipped
    ("a\nb\n```text", "```text"), ("a\n1.\n", "1."), ("a\n$\n", "$"), ("a\nb\n--- end", "--- end"),
])
def test_the_line_that_is_read_as_the_floor_line_is_the_last_one_that_is_not_only_marks(text, expected):
    from orgsim.departures import floor_candidate
    assert floor_candidate(text) == expected


def test_the_last_line_of_a_text():
    from orgsim.departures import last_line
    assert last_line("a\n\n b c \n\n") == "b c" and last_line("") == "" and last_line("one") == "one"


@pytest.mark.parametrize("value, shown", [
    (122.0, "122.00"), (104.8, "104.80"), (0.0, "0.00"), (7.5, "7.50"), (104.80000000000001, "104.80"),
    (0.1 + 0.2, "0.30"), (140.02, "140.02"),
    (110.0245, "110.0245"), (114.645, "114.645"), (134.7995, "134.7995"), (1234567.125, "1234567.125"),
    (100.0000000004, "100.00"), (99.9999999996, "100.00"), (0.0000000004, "0.00"),      # within 1e-9 of whole cents
    # beyond it: up to nine decimals, without the zeros at the end
    (100.0000000011, "100.000000001"), (0.5000000012, "0.500000001"), (114.6450004, "114.6450004"),
    (100.0000004, "100.0000004"), (100.0000011, "100.0000011"), (0.5000012, "0.5000012"), (134.7999999, "134.7999999"),
    (140.0244999, "140.0244999"),
])
def test_a_price_or_a_cost_in_the_re_ask_has_two_decimals_if_it_is_a_whole_number_of_cents_and_its_digits_otherwise(
        value, shown):
    from orgsim.departures import dollars
    assert dollars(value) == shown


@pytest.mark.parametrize("minimum, price, cost, below", [
    (30.0, 134.8, 104.8, False),                  # exactly at the sum passes
    (30.0, 134.79, 104.8, True),                  # a cent below is below
    (30.0, 134.7999999999, 104.8, False),         # within the float's last places is not below
    (30.0, 134.8000000001, 104.8, False),
    (30.0, 140.0, 104.8, False), (30.0, 122.0, 104.8, True), (0.0, 104.8, 104.8, False), (0.0, 104.79, 104.8, True),
])
def test_a_price_is_below_the_floor_when_it_is_strictly_below_the_cost_plus_the_minimum(minimum, price, cost, below):
    from orgsim.departures import below_floor
    assert below_floor(price, cost, minimum) is below
    cost_in_float = make_supplier("A", "quality").cost(ContractTerms(0, 5.0, 3))            # 85 + 16.2 + 3.6
    assert below_floor(round(cost_in_float + 30.0, 2), cost_in_float, 30.0) is False         # and to the cent it passes


def test_a_valid_floor_line_is_removed_from_the_briefing_and_the_notice_follows_it():
    from orgsim.departures import floor_briefing
    analyst = ScriptedAnalyst("Hold firm on price.\nSpeed matters.\n\nMINIMUM PROFIT PER CONTRACT: $12.5\n")
    briefing, record, minimum = floor_briefing(analyst, "A", "THE MESSAGE")
    assert minimum == 12.5 and briefing.posture is None
    assert briefing.text == "Hold firm on price.\nSpeed matters.\n\n" + NOTICE + "$12.50"
    assert analyst.sent == [("A", "THE MESSAGE", ())]
    assert record == {"raw_briefings": ["Hold firm on price.\nSpeed matters.\n\nMINIMUM PROFIT PER CONTRACT: $12.5\n"],
                      "reasked": False, "follow_up": None, "floor_line": "MINIMUM PROFIT PER CONTRACT: $12.5",
                      "floor_status": "set", "minimum_profit": 12.5}


def test_a_floor_line_of_none_sets_no_floor_and_the_bidder_is_told_so():
    from orgsim.departures import floor_briefing
    briefing, record, minimum = floor_briefing(ScriptedAnalyst("Be bold.\nMINIMUM PROFIT PER CONTRACT: none"), "B", "M")
    assert minimum is None and briefing.text == "Be bold.\n\n" + NOTICE + "none"
    assert (record["floor_status"], record["minimum_profit"], record["reasked"]) == ("none", None, False)


def test_a_missing_floor_line_is_asked_for_once_in_the_same_conversation_and_the_second_briefing_is_the_one_used():
    from orgsim.departures import floor_briefing
    first = "A briefing without the line."
    analyst = ScriptedAnalyst(first, "A second briefing.\nMINIMUM PROFIT PER CONTRACT: $20")
    briefing, record, minimum = floor_briefing(analyst, "A", "THE MESSAGE")
    assert analyst.sent == [("A", "THE MESSAGE", ()),
                            ("A", "THE MESSAGE", ({"role": "assistant", "content": first},
                                                  {"role": "user", "content": REASK_ANALYST}))]
    assert minimum == 20.0 and briefing.text == "A second briefing.\n\n" + NOTICE + "$20.00"
    assert record == {"raw_briefings": [first, "A second briefing.\nMINIMUM PROFIT PER CONTRACT: $20"], "reasked": True,
                      "follow_up": REASK_ANALYST, "floor_line": "MINIMUM PROFIT PER CONTRACT: $20",
                      "floor_status": "set", "minimum_profit": 20.0}


def test_a_malformed_line_twice_means_no_floor_and_the_bidder_reads_the_second_briefing_without_its_floor_lines():
    from orgsim.departures import floor_briefing
    first = "First.\nMINIMUM PROFIT PER CONTRACT: $5 per contract"
    second = "Second.\nMINIMUM PROFIT PER CONTRACT: $5, because costs are high\n"
    analyst = ScriptedAnalyst(first, second, "NEVER")
    briefing, record, minimum = floor_briefing(analyst, "A", "M")
    assert len(analyst.sent) == 2 and minimum is None
    assert briefing.text == "Second.\n\n" + NOTICE_NOT_SET             # the line that begins with the words is gone
    assert record["raw_briefings"] == [first, second]
    assert (record["reasked"], record["floor_status"], record["minimum_profit"], record["floor_line"]) == (
        True, "malformed", None, "MINIMUM PROFIT PER CONTRACT: $5, because costs are high")


def shown_as(body, value):
    """What the bidder of floor reads: the body of the briefing, a blank line (none if it is empty), the notice."""
    return (body + "\n\n" if body else "") + NOTICE + value


@pytest.mark.parametrize("answer, body, value, status, minimum", [
    ("Intro.\n**MINIMUM PROFIT PER CONTRACT: $5**", "Intro.", "$5.00", "set", 5.0),
    ("Intro.\nMinimum profit per contract: $5", "Intro.", "$5.00", "set", 5.0),
    ("Intro.\n- MINIMUM  PROFIT PER CONTRACT : 7.5", "Intro.", "$7.50", "set", 7.5),
    ("Intro.\n### MINIMUM PROFIT PER CONTRACT: $5.00", "Intro.", "$5.00", "set", 5.0),
    ("Intro.\n> MINIMUM PROFIT PER CONTRACT: $5.", "Intro.", "$5.00", "set", 5.0),
    ("Intro.\n**MINIMUM PROFIT PER CONTRACT:** $12", "Intro.", "$12.00", "set", 12.0),
    ("Intro.\nMINIMUM PROFIT PER CONTRACT: None", "Intro.", "none", "none", None),
    ("Intro.\nMINIMUM PROFIT PER CONTRACT: none.", "Intro.", "none", "none", None),
    # 18d: a period after closing marks, an en dash, a white circle and a list number in front
    ("Intro.\n**MINIMUM PROFIT PER CONTRACT: $5**.", "Intro.", "$5.00", "set", 5.0),
    ("Intro.\n– MINIMUM PROFIT PER CONTRACT: $6", "Intro.", "$6.00", "set", 6.0),
    ("Intro.\n◦ MINIMUM PROFIT PER CONTRACT: none", "Intro.", "none", "none", None),
    ("Intro.\n1. MINIMUM PROFIT PER CONTRACT: $7", "Intro.", "$7.00", "set", 7.0),
    ("Intro.\n2) **Minimum profit per contract:** none.", "Intro.", "none", "none", None),
    # 18d: a prose line that begins with the words is removed too
    ("Intro.\n**Minimum profit per contract:** $2, because costs are high.\nMINIMUM PROFIT PER CONTRACT: $3", "Intro.",
     "$3.00", "set", 3.0),
])
def test_a_floor_line_in_markdown_or_another_case_is_read_removed_and_not_asked_for_again(
        answer, body, value, status, minimum):
    from orgsim.departures import floor_briefing
    analyst = ScriptedAnalyst(answer, "NEVER ASKED")
    briefing, record, floor = floor_briefing(analyst, "A", "M")
    assert len(analyst.sent) == 1 and floor == minimum and briefing.text == shown_as(body, value)
    assert (record["reasked"], record["floor_status"], record["follow_up"]) == (False, status, None)
    assert record["floor_line"] == answer.split("\n")[-1]                                # the line that was read
    assert record["raw_briefings"] == [answer]                                           # the answer as written


@pytest.mark.parametrize("answer, body, line, value, status, minimum", [
    ("Intro.\nMINIMUM PROFIT PER CONTRACT: $5\n---", "Intro.\n---", "MINIMUM PROFIT PER CONTRACT: $5", "$5.00", "set",
     5.0),
    ("Intro.\n```\nMINIMUM PROFIT PER CONTRACT: none\n```", "Intro.\n```\n```", "MINIMUM PROFIT PER CONTRACT: none",
     "none", "none", None),
    ("MINIMUM PROFIT PER CONTRACT: $5\n***\n___", "***\n___", "MINIMUM PROFIT PER CONTRACT: $5", "$5.00", "set", 5.0),
    ("Intro.\n1. MINIMUM PROFIT PER CONTRACT: none.\n\n  ---  \n", "Intro.\n\n  ---",
     "1. MINIMUM PROFIT PER CONTRACT: none.", "none", "none", None),
])
def test_lines_of_marks_after_the_floor_line_are_skipped_in_reading_it_and_stay_in_what_the_bidder_reads(
        answer, body, line, value, status, minimum):
    from orgsim.departures import floor_briefing
    analyst = ScriptedAnalyst(answer, "NEVER ASKED")
    briefing, record, floor = floor_briefing(analyst, "A", "M")
    assert len(analyst.sent) == 1 and floor == minimum and briefing.text == shown_as(body, value)
    assert (record["reasked"], record["floor_status"]) == (False, status)
    assert record["floor_line"] == line and record["raw_briefings"] == [answer]


def test_text_after_a_rule_is_the_line_that_is_read_and_a_second_answer_is_asked_for():
    from orgsim.departures import floor_briefing
    first = "Intro.\nMINIMUM PROFIT PER CONTRACT: $5\n---\nand a last line of text"
    analyst = ScriptedAnalyst(first, "Again.\nMINIMUM PROFIT PER CONTRACT: $6\n---")
    briefing, record, minimum = floor_briefing(analyst, "A", "M")
    assert len(analyst.sent) == 2 and minimum == 6.0 and record["reasked"] is True and record["floor_status"] == "set"
    assert briefing.text == "Again.\n---\n\n" + NOTICE + "$6.00"
    assert record["raw_briefings"][0] == first and record["floor_line"] == "MINIMUM PROFIT PER CONTRACT: $6"


def test_a_briefing_that_is_only_a_floor_line_leaves_only_the_notice_and_a_valid_second_line_is_used():
    from orgsim.departures import floor_briefing
    briefing, _, minimum = floor_briefing(ScriptedAnalyst("MINIMUM PROFIT PER CONTRACT: $3"), "A", "M")
    assert minimum == 3.0 and briefing.text == NOTICE + "$3.00"
    briefing, record, minimum = floor_briefing(
        ScriptedAnalyst("MINIMUM PROFIT PER CONTRACT: $x", "none of it"), "A", "M")
    assert minimum is None and briefing.text == "none of it\n\n" + NOTICE_NOT_SET and record["reasked"] is True
    briefing, record, minimum = floor_briefing(
        ScriptedAnalyst("no line", "MINIMUM PROFIT PER CONTRACT: none"), "A", "M")
    assert minimum is None and briefing.text == NOTICE + "none" and record["floor_status"] == "none"


def test_after_two_answers_without_a_valid_line_the_bidder_is_told_that_no_minimum_is_set_not_that_none_was_set():
    from orgsim.departures import floor_briefing
    second = ("Intro.\n  MINIMUM PROFIT PER CONTRACT $9 per contract\nI keep my MINIMUM PROFIT PER CONTRACT open.\n"
              "Done.")
    briefing, record, minimum = floor_briefing(ScriptedAnalyst("no line", second), "A", "M")
    assert minimum is None and briefing.posture is None
    assert briefing.text == "Intro.\nI keep my MINIMUM PROFIT PER CONTRACT open.\nDone.\n\n" + NOTICE_NOT_SET
    assert NOTICE not in briefing.text
    assert (record["floor_status"], record["minimum_profit"], record["floor_line"]) == ("malformed", None, "Done.")
    # a second briefing that holds nothing but lines that begin with the words leaves only the notice
    briefing, record, minimum = floor_briefing(
        ScriptedAnalyst("no line", "MINIMUM PROFIT PER CONTRACT $9\nMINIMUM PROFIT PER CONTRACT: $x"), "A", "M")
    assert briefing.text == NOTICE_NOT_SET and minimum is None and record["floor_status"] == "malformed"


def test_the_notice_of_none_and_the_notice_that_no_minimum_is_set_go_with_different_records():
    from orgsim.departures import floor_briefing
    said_none, none_record, _ = floor_briefing(ScriptedAnalyst("Intro.\nMINIMUM PROFIT PER CONTRACT: none"), "A", "M")
    failed, failed_record, _ = floor_briefing(ScriptedAnalyst("Intro.", "Intro again."), "A", "M")
    assert said_none.text == "Intro.\n\n" + NOTICE + "none"
    assert failed.text == "Intro again.\n\n" + NOTICE_NOT_SET
    assert (none_record["floor_status"], none_record["reasked"]) == ("none", False)
    assert (failed_record["floor_status"], failed_record["reasked"]) == ("malformed", True)
    assert none_record["minimum_profit"] is None and failed_record["minimum_profit"] is None


def test_in_every_case_the_bidder_reads_no_line_that_begins_with_the_words_of_the_floor_line():
    from orgsim.departures import floor_briefing
    for answers, shown in (
            # a valid line, with another line that begins with the words before it
            (("Intro.\nMINIMUM PROFIT PER CONTRACT: $3\nMore.\nMINIMUM PROFIT PER CONTRACT: $12.5",),
             "Intro.\nMore.\n\n" + NOTICE + "$12.50"),
            # none
            (("Intro.\n  MINIMUM PROFIT PER CONTRACT: $3\nMore.\n\tMINIMUM PROFIT PER CONTRACT:none",),
             "Intro.\nMore.\n\n" + NOTICE + "none"),
            # two answers without a valid line
            (("Intro.", "Intro.\nMINIMUM PROFIT PER CONTRACT: $x\nMore.\nMINIMUM PROFIT PER CONTRACT is open"),
             "Intro.\nMore.\n\n" + NOTICE_NOT_SET)):
        briefing, _, _ = floor_briefing(ScriptedAnalyst(*answers), "A", "M")
        assert briefing.text == shown
        written, notice = briefing.text.rsplit("\n\n", 1)
        beginning = re.compile(r"[\s*_`#>\-•+–◦]*(?:[0-9]+[.)])?\s*minimum\s+profit\s+per\s+contract", re.IGNORECASE)
        assert not any(beginning.match(line) for line in written.split("\n"))


# ---- floor: the bidder's side ----

class Replies:
    """A fake endpoint that answers the requests it is sent with the replies it is given, in order, and keeps the
    requests."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        message = SimpleNamespace(content=self.replies[len(self.calls) - 1], tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))


def bid_json(price, weeks=5.0, tier=3, reasoning="r"):
    return json.dumps({"price": price, "delivery_weeks": weeks, "quality_tier": tier, "reasoning": reasoning})


QUALITY_COSTS = make_supplier("A", "quality")                      # 85 + 1.8 * (8 - weeks)^2 ... see below
COST_5_3 = QUALITY_COSTS.cost(ContractTerms(0, 5.0, 3))          # 104.8
COST_4_4 = QUALITY_COSTS.cost(ContractTerms(0, 4.0, 4))          # 120.2


def floor_agent(*replies, minimum=30.0):
    from orgsim.llm_agents import PrimitivesSupplierAgent
    fake = Replies(*replies)
    agent = PrimitivesSupplierAgent(ChatClient(fake), "m", "SYSTEM", n_rounds=3)
    agent.set_minimum_profit(minimum)
    return fake, agent


def test_the_costs_the_floor_tests_use():
    assert round(COST_5_3, 6) == 104.8 and round(COST_4_4, 6) == 120.2


def test_a_bid_exactly_at_the_cost_plus_the_minimum_is_not_returned():
    fake, agent = floor_agent(bid_json(134.8))
    bid = agent.generate_bid(QUALITY_COSTS, "CONTEXT", 1, [])
    assert len(fake.calls) == 1 and bid.terms.price == 134.8
    assert bid.reasoning == "[STATED]: r"
    assert bid.floor_audit == {"minimum_profit": 30.0, "reasked": False, "raised": False}


def test_a_bid_a_twentieth_of_a_cent_below_the_cost_plus_the_minimum_is_below_it():
    # only float noise in the sum is allowed for, not a price that differs from it in the fourth decimal
    fake, agent = floor_agent(bid_json(134.7995), bid_json(135.0))
    bid = agent.generate_bid(QUALITY_COSTS, "CONTEXT", 1, [])
    assert len(fake.calls) == 2 and bid.terms.price == 135.0
    assert bid.floor_audit["reasked"] is True and bid.floor_audit["raised"] is False
    from orgsim.departures import below_floor
    assert below_floor(134.7995, COST_5_3, 30.0) and not below_floor(COST_5_3 + 30.0, COST_5_3, 30.0)
    assert not below_floor(134.8, COST_5_3, 30.0) and below_floor(134.79, COST_5_3, 30.0)


def test_a_bid_below_the_cost_plus_the_minimum_is_returned_once_and_the_bid_sent_back_is_taken_if_it_is_above():
    reply = "Here is my bid: " + bid_json(122.0, reasoning="first")
    fake, agent = floor_agent(reply, bid_json(140.0, reasoning="second"))
    bid = agent.generate_bid(QUALITY_COSTS, "CONTEXT", 1, [])
    follow_up = REASK_BIDDER.replace("$P", "$122.00").replace("$C", "$104.80").replace("$X", "$30.00")
    first, second = (call["messages"] for call in fake.calls)
    assert [m["role"] for m in first] == ["system", "user"] and first[0]["content"] == "SYSTEM"
    assert first[1]["content"] == "CONTEXT\n\n" + (
        'ROUND 1 of 3: Submit your bid as JSON. Keep its "reasoning" field to 1-2 sentences.')
    assert second == first + [{"role": "assistant", "content": reply}, {"role": "user", "content": follow_up}]
    assert bid.terms.price == 140.0 and bid.reasoning == "[STATED]: second"
    assert bid.floor_audit == {
        "minimum_profit": 30.0, "reasked": True, "raised": False, "follow_up": follow_up,
        "first": {"price": 122.0, "delivery_weeks": 5.0, "quality_tier": 3, "cost": COST_5_3, "reasoning": "first"},
        "resent": {"price": 140.0, "delivery_weeks": 5.0, "quality_tier": 3, "cost": COST_5_3, "reasoning": "second"}}


def test_a_bid_that_is_still_below_is_raised_to_the_cost_plus_the_minimum_at_the_terms_of_the_bid_sent_back():
    fake, agent = floor_agent(bid_json(122.0), bid_json(126.0, weeks=4.0, tier=4, reasoning="again"))
    bid = agent.generate_bid(QUALITY_COSTS, "CONTEXT", 2, [])
    assert len(fake.calls) == 2                                   # asked once, not twice
    # the terms of the bid sent back
    assert (bid.terms.delivery_weeks, bid.terms.quality_tier) == (4.0, 4)
    assert bid.terms.price == COST_4_4 + 30.0 and bid.terms.price > 126.0            # exactly the cost plus the minimum
    assert bid.reasoning == "[STATED]: again" + FLOOR_PRICE_TAG and RULE_PRICE_TAG not in bid.reasoning
    audit = bid.floor_audit
    assert (audit["reasked"], audit["raised"], audit["minimum_profit"]) == (True, True, 30.0)
    assert audit["first"] == {"price": 122.0, "delivery_weeks": 5.0, "quality_tier": 3, "cost": COST_5_3,
                              "reasoning": "r"}
    assert audit["resent"] == {"price": 126.0, "delivery_weeks": 4.0, "quality_tier": 4, "cost": COST_4_4,
                               "reasoning": "again"}


def test_the_audit_keeps_the_reasoning_of_each_bid_as_received_and_an_empty_one_for_a_bid_without_any():
    without = json.dumps({"price": 122.0, "delivery_weeks": 5.0, "quality_tier": 3})
    fake, agent = floor_agent(without, bid_json(140.0, reasoning="Raising it: costs are $104.80, \"firm\"."))
    bid = agent.generate_bid(QUALITY_COSTS, "CONTEXT", 1, [])
    assert bid.floor_audit["first"]["reasoning"] == ""
    assert bid.floor_audit["resent"]["reasoning"] == "Raising it: costs are $104.80, \"firm\"."
    assert bid.reasoning == "[STATED]: Raising it: costs are $104.80, \"firm\"."         # the bid's own, as before
    fake, agent = floor_agent(bid_json(122.0, reasoning="we want the contract"), bid_json(126.0, reasoning="still low"))
    bid = agent.generate_bid(QUALITY_COSTS, "CONTEXT", 1, [])
    assert bid.floor_audit["first"]["reasoning"] == "we want the contract"           # the reasoning of the bid returned
    assert bid.floor_audit["resent"]["reasoning"] == "still low"
    assert bid.reasoning == "[STATED]: still low" + FLOOR_PRICE_TAG                    # the tag is not in the audit
    fake, agent = floor_agent(bid_json(140.0, reasoning="fine"))
    assert "first" not in agent.generate_bid(QUALITY_COSTS, "CONTEXT", 1, []).floor_audit        # a bid that passes


@pytest.mark.parametrize("first_price, shown", [(140.02, "140.02"), (140.0244, "140.0244"),
                                                 (140.0244999, "140.0244999")])
def test_the_re_ask_prints_the_price_and_the_cost_with_their_digits_beyond_the_cent_and_the_raise_is_exact(
        first_price, shown):
    cost = QUALITY_COSTS.cost(ContractTerms(0, 4.55, 3))      # 85 + 1.8 * (8 - 4.55)^2 + 3.6 = 110.0245, below the cent
    assert round(cost, 6) == 110.0245
    fake, agent = floor_agent(bid_json(first_price, weeks=4.55), bid_json(140.0, weeks=4.55, reasoning="again"))
    bid = agent.generate_bid(QUALITY_COSTS, "CONTEXT", 1, [])
    follow_up = fake.calls[1]["messages"][3]["content"]
    assert follow_up == (f"Your bid's price, ${shown}, is less than its total cost at its delivery and quality, "
                         "$110.0245, plus the minimum profit per contract of $30.00 that your company's analyst set "
                         "for this event. Submit your bid again. If the bid you send is also below the minimum, its "
                         "price will be raised to its total cost plus $30.00.")
    assert bid.floor_audit["follow_up"] == follow_up
    # still below: the price is the total cost plus the minimum, exactly, and not rounded to the cent
    assert bid.terms.price == cost + 30.0 and bid.terms.price != round(cost + 30.0, 2)
    assert bid.floor_audit["resent"]["cost"] == cost and bid.floor_audit["raised"] is True


def test_the_bid_sent_back_may_have_other_terms_and_is_then_judged_by_its_own_cost():
    # cheap terms: 8wk / T2 costs 86.6
    fake, agent = floor_agent(bid_json(122.0), bid_json(125.0, weeks=8.0, tier=2))
    bid = agent.generate_bid(QUALITY_COSTS, "CONTEXT", 1, [])
    assert bid.terms.price == 125.0 and bid.reasoning == "[STATED]: r" and bid.floor_audit["raised"] is False


def test_no_floor_means_the_bid_is_taken_as_submitted_whatever_its_price():
    fake, agent = floor_agent(bid_json(50.0), minimum=None)
    bid = agent.generate_bid(QUALITY_COSTS, "CONTEXT", 1, [])
    assert len(fake.calls) == 1 and bid.terms.price == 50.0 and bid.floor_audit is None
    assert bid.reasoning == "[STATED]: r"


def test_a_floor_of_zero_returns_a_bid_priced_below_its_cost_and_takes_one_at_its_cost():
    fake, agent = floor_agent(bid_json(104.8), minimum=0.0)
    bid = agent.generate_bid(QUALITY_COSTS, "C", 1, [])
    assert len(fake.calls) == 1 and bid.terms.price == 104.8 and bid.floor_audit["reasked"] is False
    fake, agent = floor_agent(bid_json(100.0), bid_json(104.8), minimum=0.0)
    bid = agent.generate_bid(QUALITY_COSTS, "C", 1, [])
    assert len(fake.calls) == 2 and bid.terms.price == 104.8
    assert bid.floor_audit["reasked"] is True and bid.floor_audit["raised"] is False


def test_the_floor_is_set_for_one_event_and_cleared_and_must_be_a_number_of_dollars_that_is_not_negative():
    from orgsim.llm_agents import PrimitivesSupplierAgent
    agent = PrimitivesSupplierAgent(ChatClient(Replies()), "m", "S", n_rounds=2)
    assert agent.minimum_profit is None
    agent.set_minimum_profit(12.5)
    assert agent.minimum_profit == 12.5
    agent.set_minimum_profit(None)
    assert agent.minimum_profit is None
    for bad in (-0.01, float("nan"), "5"):
        with pytest.raises((ValueError, TypeError)):
            agent.set_minimum_profit(bad)


def test_the_earlier_rounds_block_marks_a_price_the_floor_raised_and_never_a_price_set_by_rule_in_the_primitives_set():
    history = [{"round": 1, "bid": ContractTerms(134.8, 5.0, 3), "feedback": "THE ANSWER", "price_set_by_rule": False,
                "price_raised_by_floor": True},
               {"round": 2, "bid": ContractTerms(130.0, 5.0, 3), "feedback": "", "price_set_by_rule": True,
                "price_raised_by_floor": False}]
    block = _previous_rounds_block(QUALITY_COSTS, history, PP.REPLY_LABEL, primitives=True)
    assert block == ("PREVIOUS ROUNDS:\n"
                     f"  Round 1: You bid $134.80{RAISED_NOTE}, 5.0wk delivery, tier 3 (your cost was $104.80)\n"
                     "  Platform's reply: THE ANSWER\n"
                     "  Round 2: You bid $130.00, 5.0wk delivery, tier 3 (your cost was $104.80)\n\n")
    assert "price set by rule" not in block
    advice = _previous_rounds_block(QUALITY_COSTS, history)
    # the advice set is as it was
    assert "(price set by rule: cost + 5%)" in advice and "price raised" not in advice


def test_the_record_marks_a_price_the_floor_raised_and_never_a_price_set_by_rule():
    raised = _entry("Quality Priority", "quality",
                    [_bid("A", 1, 110.0), _bid("A", 2, 134.8, 5.0, 3, "r" + FLOOR_PRICE_TAG),
                     _bid("A", 3, 100.0, 5.0, 3, "r" + RULE_PRICE_TAG)], ["No.", "Better."], "A", 100.0, -4.8)
    record = build_primitives_record("A", [raised])
    assert f"    Round 2: Your company bid $134.80{RAISED_NOTE}, 5.0wk, tier 3\n" in record
    assert "    Round 1: Your company bid $110.00, 5.0wk, tier 3\n" in record
    assert "    Round 3: Your company bid $100.00, 5.0wk, tier 3\n" in record and "price set by rule" not in record
    assert record.count(RAISED_NOTE) == 1


def _bid(sid, round_num, price, weeks=5.0, tier=3, reasoning="r"):
    return SupplierBid(sid, ContractTerms(price, weeks, tier), reasoning, round_num)


def _entry(scenario, cost_type, own_bids, replies, winner, paid=None, profit=None):
    other = [_bid("B", bid.round_num, 999.0) for bid in own_bids]
    outcome = SimpleNamespace(all_bids={"A": own_bids, "B": other}, winner_id=winner, supplier_profit=profit,
                              winning_bid=None if paid is None else SimpleNamespace(price=paid))
    return {"scenario_name": scenario, "outcome": outcome, "feedback_history": replies, "own_cost_type": cost_type}


def test_the_default_agent_refuses_a_floor_and_the_mock_bidder_and_the_primitives_bidder_take_it():
    from orgsim.interfaces import SupplierAgent
    from orgsim.mocks import MockStrategicSupplier

    class Plain(SupplierAgent):
        def generate_bid(self, supplier_costs, regime_context, round_num, history):
            raise AssertionError

    with pytest.raises(NotImplementedError, match="Plain does not enforce a minimum profit"):
        Plain().set_minimum_profit(5.0)
    mock = MockStrategicSupplier(seed=3)
    plain = mock.generate_bid(QUALITY_COSTS, "ctx", 1, [])
    assert plain.floor_audit is None and FLOOR_PRICE_TAG not in plain.reasoning
    mock = MockStrategicSupplier(seed=3)
    mock.set_minimum_profit(1000.0)
    raised = mock.generate_bid(QUALITY_COSTS, "ctx", 1, [])
    assert raised.floor_audit == {"minimum_profit": 1000.0, "reasked": False, "raised": True}
    assert raised.terms.price == pytest.approx(QUALITY_COSTS.cost(raised.terms) + 1000.0)
    assert raised.reasoning.endswith(FLOOR_PRICE_TAG)
    mock = MockStrategicSupplier(seed=3)
    mock.set_minimum_profit(0.01)
    assert mock.generate_bid(QUALITY_COSTS, "ctx", 1, []).floor_audit["raised"] is False


@pytest.mark.parametrize("institution", ["Mediator", "BuyerAgent"])
def test_the_history_of_a_round_says_which_bids_the_floor_raised(institution):
    from orgsim.institutions import run_buyer_agent, run_mediator
    from orgsim.interfaces import SupplierAgent
    from orgsim.mocks import MockBuyerAgentPlatform, MockPlatformMediator
    from orgsim.primitives import get_contract_scenarios

    class Marking(SupplierAgent):
        def __init__(self, tagged):
            self.tagged, self.histories = tagged, []

        def generate_bid(self, supplier_costs, regime_context, round_num, history):
            self.histories.append([dict(h) for h in history])
            tag = FLOOR_PRICE_TAG if round_num == self.tagged else ""
            return SupplierBid(supplier_costs.supplier_id, ContractTerms(140.0, 4.0, 3), "r" + tag, round_num)

    scenario = get_contract_scenarios()[0]
    suppliers = [make_supplier("A", "speed"), make_supplier("B", "quality")]
    agents = {"A": Marking(1), "B": Marking(0)}
    if institution == "Mediator":
        run_mediator(1, scenario.buyer, suppliers, agents, MockPlatformMediator(), 50.0, "A", num_rounds=3)
    else:
        run_buyer_agent(1, scenario.buyer, suppliers, agents, MockBuyerAgentPlatform(), 50.0, "A", num_rounds=3)
    marks = {sid: [[h["price_raised_by_floor"] for h in history] for history in agent.histories]
             for sid, agent in agents.items()}
    assert marks == {"A": [[], [True], [True, False]], "B": [[], [False], [False, False]]}
