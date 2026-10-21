"""The primitives prompt set: its texts, pinned exactly; what its sentences say, checked against the code; the plan
of the persistent environment; and the agents that send them. Runs through the command are in
test_demo_primitives.py."""
import hashlib
import inspect
import json
import re
from fractions import Fraction
from types import SimpleNamespace

import pytest

from orgsim import primitives_prompts as PP
from orgsim import prompts as P
from orgsim.institutions import (
    AWARD_TEXT, BUYER_AGENT_PLATFORM_TEXT, MEDIATOR_PLATFORM_TEXT, PLATFORM_TEXTS, PRIMITIVES_OUTSIDE_OPTION,
    primitives_rules, run_buyer_agent, run_mediator,
)
from orgsim.interfaces import RULE_PRICE_TAG
from orgsim.llm import ChatClient
from orgsim.llm_agents import (
    DELIVERY_WEEKS_RANGE, QUALITY_TIERS, LocalProcurementAnalyst, LocalSupplierAgent, PrimitivesSupplierAgent, _is_bid,
    _previous_rounds_block,
)
from orgsim.mocks import MockBuyerAgentPlatform, MockPlatformMediator, MockProcurementAnalyst, MockStrategicSupplier
from orgsim.organization import (
    ANALYST_ORGANIZATIONS, BRIEFING_HEADER, FIRST_EVENT_NOTE, FORWARDED_HEADER, PRIMITIVES_ANALYST_ORGANIZATIONS,
    PRIMITIVES_ORGANIZATIONS, PRIMITIVES_REVENUE_ORGANIZATIONS, AnalystBriefing, ProcurementAnalyst,
    build_primitives_bidder_context,
)
from orgsim.primitives import ContractTerms, SupplierBid, get_contract_scenarios
from orgsim.sequence import (
    P_DIFFERENT, SUPPLIER_IDS, DrawnPlan, build_primitives_record, build_supplier_history, make_supplier,
    pre_draw_event_sequence, run_sequence,
)

ENVIRONMENTS = ("random", "persistent")
SIX = [name for name in PRIMITIVES_ORGANIZATIONS]


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ---- the blocks, exactly (written out again here from the design, with 6 events) ----

OPENING_TEAM = (
    "Your company is a supplier. It takes part in a sequence of 6 procurement events on a platform. Everyone in your "
    "company has the same objective: the company's total profit across the 6 events. Everyone in the company knows "
    "this, and knows who holds which information and who decides.")
OPENING_REVENUE = (
    "Your company is a supplier. It takes part in a sequence of 6 procurement events on a platform. Your company's "
    "owner wants the company's total profit across the 6 events. Everyone in the company knows each member's "
    "objective, and who holds which information and who decides.")
PAYOFFS_AND_COSTS = (
    "In each event, if your company wins, it is paid the price of its final bid, and its profit is that price minus "
    "its total cost at that bid's delivery and quality; if it does not win, its profit from that event is zero. Your "
    "company's costs are private. In each event they are those of either a speed specialist or a quality "
    "specialist: faster delivery costs a speed specialist less than it costs a quality specialist, and a higher "
    "quality tier costs a speed specialist more. One other supplier takes part in each event; your company is not "
    "told its costs or its bids. The same other supplier takes part in every event.")
CONTRACTS_RANDOM = (
    "Each event is for one of two contracts. A bid states a price, a delivery time of 1 to 8 weeks, and a quality "
    "tier of 2, 3, 4 or 5. For both contracts the buyer values faster delivery and a higher quality tier; how much "
    "it values each differs between the two contracts. The contract that is up is not named when the event starts.")
CONTRACTS_PERSISTENT = (
    "Each event is for one of two contracts. A bid states a price, a delivery time of 1 to 8 weeks, and a quality "
    "tier of 2, 3, 4 or 5. For both contracts the buyer values faster delivery and a higher quality tier; how much "
    "it values each differs between the two contracts. The contract is not named when the first event starts.")
DRAWN_PERSISTENT = (
    "How the events are drawn: before the first event, the contract is drawn at random, each of the two with "
    "probability 1 in 2; your company's cost type is drawn at random, speed or quality with probability 1 in 2; and "
    "the other supplier's cost type is the opposite of yours with probability 3 in 4. The contract and both cost "
    "types then stay the same in all 6 events.")
DRAWN_RANDOM = (
    "How the events are drawn: for each event, independently, the contract is drawn at random, each of the two with "
    "probability 1 in 2; your company's cost type is drawn at random, speed or quality with probability 1 in 2; and "
    "the other supplier's cost type is the opposite of yours with probability 3 in 4.")
BIDDER_ROLE = "You are your company's bidder: you decide the price, delivery and quality of each of its bids."
ROLE_U = BIDDER_ROLE + (" From your company's second event on, your company's full record of past events on "
                        "this platform appears in the message below.")
ROLE_U0 = BIDDER_ROLE + " Your company keeps no record of past events on this platform."
ROLE_O = BIDDER_ROLE + (
    " Your company's record of past events on this platform is kept by your company's procurement analyst. From "
    "your company's second event on, the analyst writes you a briefing before each event, and it appears in the "
    "message below. You do not see the record; you see the briefing. The analyst holds the record, which shows, for "
    "each past event, your company's cost type, the contract, your company's bids, the platform's replies and the "
    "result; it also reads the rules you read, and it is not shown the cost figures you are shown.")
ROLE_PASS = BIDDER_ROLE + (
    " Your company's record of past events on this platform is kept by your company's procurement analyst. From "
    "your company's second event on, the analyst forwards you the full record before each event, unchanged, and it "
    "appears in the message below.")
ROLE_ANALYST = (
    "You are your company's procurement analyst. You hold your company's record of past events on this platform. "
    "From your company's second event on, you write a briefing before each event for your company's bidder, who "
    "decides the price, delivery and quality of each bid. The bidder does not see the record; it sees your "
    "briefing, its costs for the event, the rules below and, during the event, the platform's replies. You are not "
    "shown the cost figures the bidder is shown. Write the briefing as plain text.")
REVENUE_BIDDER = ("Your own objective, as the company's bidder, is the company's revenue: the total price of the "
                  "contracts it wins across the 6 events.")
ANALYST_PROFIT = "The analyst's objective is the company's total profit across the 6 events."
REVENUE_ANALYST = (
    "Your own objective, as the company's analyst, is the company's total profit across the 6 events. The bidder's "
    "objective is the company's revenue: the total price of the contracts it wins across the 6 events.")
OUTPUT = ('Respond with ONLY a JSON object with these exact fields: {"price": <number>, "delivery_weeks": <number '
          'between 1.0 and 8.0>, "quality_tier": <integer 2-5>, "reasoning": "<brief reasoning>"}')

SHARED = {(opening, environment): "\n\n".join([opening, PAYOFFS_AND_COSTS, contracts, drawn])
          for opening in (OPENING_TEAM, OPENING_REVENUE)
          for environment, contracts, drawn in (("persistent", CONTRACTS_PERSISTENT, DRAWN_PERSISTENT),
                                                ("random", CONTRACTS_RANDOM, DRAWN_RANDOM))}
BIDDER_PARTS = {
    "single": (OPENING_TEAM, ROLE_U), "single_norecord": (OPENING_TEAM, ROLE_U0),
    "organized": (OPENING_TEAM, ROLE_O), "organized_pass": (OPENING_TEAM, ROLE_PASS),
    "single_revenue": (OPENING_REVENUE, ROLE_U, REVENUE_BIDDER),
    "organized_revenue": (OPENING_REVENUE, ROLE_O, REVENUE_BIDDER + " " + ANALYST_PROFIT),
}


def expected_bidder_prompt(organization, environment):
    opening, role, *objective = BIDDER_PARTS[organization]
    return "\n\n".join([SHARED[(opening, environment)], role, *objective, OUTPUT])


def expected_analyst_prompt(organization, environment):
    opening = OPENING_REVENUE if organization == "organized_revenue" else OPENING_TEAM
    return "\n\n".join([SHARED[(opening, environment)], ROLE_ANALYST]
                       + ([REVENUE_ANALYST] if organization == "organized_revenue" else []))


def test_the_blocks_of_the_system_prompts_are_these_exact_texts():
    assert PP.OPENING.format(n=6) + " " + PP.TEAM_OBJECTIVE.format(n=6) + " " + PP.TEAM_KNOWLEDGE == OPENING_TEAM
    assert PP.OPENING.format(n=6) + " " + PP.REVENUE_OWNER.format(n=6) + " " + PP.REVENUE_KNOWLEDGE == OPENING_REVENUE
    assert PP.PAYOFFS_AND_COSTS == PAYOFFS_AND_COSTS
    assert PP.CONTRACTS_RANDOM == CONTRACTS_RANDOM and PP.CONTRACTS_PERSISTENT == CONTRACTS_PERSISTENT
    assert PP.CONTRACTS == {"random": CONTRACTS_RANDOM, "persistent": CONTRACTS_PERSISTENT}
    assert PP.DRAWN_PERSISTENT.format(n=6) == DRAWN_PERSISTENT and PP.DRAWN_RANDOM == DRAWN_RANDOM
    assert PP.SINGLE_ROLE == ROLE_U and PP.NORECORD_ROLE == ROLE_U0 and PP.ORGANIZED_ROLE == ROLE_O
    assert PP.PASS_ROLE == ROLE_PASS and PP.ANALYST_ROLE == ROLE_ANALYST
    assert PP.REVENUE_BIDDER_OBJECTIVE.format(n=6) == REVENUE_BIDDER
    assert PP.ANALYST_PROFIT_OBJECTIVE.format(n=6) == ANALYST_PROFIT
    assert PP.REVENUE_ANALYST_OBJECTIVES.format(n=6) == REVENUE_ANALYST
    assert PP.BID_OUTPUT == OUTPUT
    assert PP.ROUND_INSTRUCTION == 'Submit your bid as JSON. Keep its "reasoning" field to 1-2 sentences.'
    assert PP.REPLY_LABEL == "Platform's reply"


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("organization", SIX)
def test_the_bidders_system_prompt_is_the_shared_block_the_role_block_and_the_output_format(organization, environment):
    assert PP.bidder_prompt(organization, 6, environment) == expected_bidder_prompt(organization, environment)


@pytest.mark.parametrize("environment", ENVIRONMENTS)
@pytest.mark.parametrize("organization", PRIMITIVES_ANALYST_ORGANIZATIONS)
def test_the_analysts_system_prompt_is_the_shared_block_and_the_analysts_role_block(organization, environment):
    assert PP.analyst_prompt(organization, 6, environment) == expected_analyst_prompt(organization, environment)


def test_the_system_prompts_are_pinned_by_digest():
    found = {f"{kind} {organization} {environment}": digest(prompt(organization, 6, environment))
             for environment in ENVIRONMENTS
             for kind, prompt, organizations in (("bidder", PP.bidder_prompt, SIX),
                                                 ("analyst", PP.analyst_prompt, PRIMITIVES_ANALYST_ORGANIZATIONS))
             for organization in organizations}
    assert found == EXPECTED_PROMPT_DIGESTS


EXPECTED_PROMPT_DIGESTS = {
    "bidder single random": "c189458555d284b6",
    "bidder single_norecord random": "1d1e09cdffac9d23",
    "bidder organized random": "cac5b221a6581bb1",
    "bidder organized_pass random": "253fb98cd7f1aaf4",
    "bidder single_revenue random": "2d0276aff8b7dff8",
    "bidder organized_revenue random": "1e2953d79a8c6fed",
    "analyst organized random": "c8cacec75b475d07",
    "analyst organized_revenue random": "0a666882ae243686",
    "bidder single persistent": "f01672fa4d8b2bad",
    "bidder single_norecord persistent": "9e978ee0671c9b22",
    "bidder organized persistent": "db9930ee7a5bb15b",
    "bidder organized_pass persistent": "b88a8fc06db5c115",
    "bidder single_revenue persistent": "1c19562779531231",
    "bidder organized_revenue persistent": "d9fc9e17aa60757c",
    "analyst organized persistent": "cbcff0999327e31b",
    "analyst organized_revenue persistent": "5aec536aed1dfb5d",
}


def test_each_system_prompt_is_its_own_so_that_usage_can_be_counted_by_prompt():
    prompts = [PP.bidder_prompt(organization, 6, environment) for organization in SIX for environment in ENVIRONMENTS]
    analysts = [PP.analyst_prompt(organization, 6, environment) for organization in PRIMITIVES_ANALYST_ORGANIZATIONS
                for environment in ENVIRONMENTS]
    assert len(set(prompts + analysts)) == len(prompts + analysts) == 12 + 4
    advice = {P.executive_prompt(organization) for organization in P.ORGANIZATIONS} | {P.sealed_bid_prompt(),
                                                                                       P.ANALYST_PROMPT}
    assert not advice & set(prompts + analysts)


@pytest.mark.parametrize("events", [2, 3, 12])
def test_the_number_of_events_is_the_runs(events):
    for organization in SIX:
        text = PP.bidder_prompt(organization, events, "persistent")
        assert f"a sequence of {events} procurement events" in text and f"across the {events} events" in text
        assert f"stay the same in all {events} events" in text
        assert "6" not in text.replace(str(events), "") or events == 6
    assert "stay the same" not in PP.bidder_prompt("single", events, "random")


def test_the_prompts_refuse_what_their_texts_would_make_untrue():
    with pytest.raises(ValueError, match="at least 2 events"):
        PP.bidder_prompt("single", 1, "random")
    with pytest.raises(ValueError, match="environment"):
        PP.bidder_prompt("single", 6, "sometimes")
    with pytest.raises(ValueError, match="unknown organization of the primitives set"):
        PP.bidder_prompt("organized_guided", 6, "random")
    with pytest.raises(ValueError, match="unknown organization of the primitives set"):
        PP.analyst_prompt("single", 6, "random")                       # a single firm has no analyst


# ---- the event: costs, rules, what the bidder knows, the round ----

def test_the_cost_description_of_a_speed_specialist_exactly():
    text = make_supplier("A", "speed").describe_for_primitives("speed")
    assert text == (
        "In this event your costs are those of a speed specialist.\n\n"
        "YOUR PRIVATE COST STRUCTURE (only you know this):\n"
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
        "COST RANGE: Your cheapest possible configuration (8wk/T2) costs $92.2. Your most expensive (1wk/T5) costs "
        "$154.5.\n\n"
        "Your total cost = base + delivery cost + quality cost.")


def test_the_cost_description_of_a_quality_specialist_exactly():
    text = make_supplier("B", "quality").describe_for_primitives("quality")
    assert text == (
        "In this event your costs are those of a quality specialist.\n\n"
        "YOUR PRIVATE COST STRUCTURE (only you know this):\n"
        "Base production cost: $85.0\n\n"
        "DELIVERY COST (faster = more expensive, quadratic):\n"
        "    1wk: +$88.2\n    2wk: +$64.8\n    3wk: +$45.0\n    4wk: +$28.8\n    5wk: +$16.2\n    6wk: +$7.2\n"
        "    7wk: +$1.8\n    8wk: +$0.0\n\n"
        "QUALITY COST (higher tier = more expensive, quadratic):\n"
        "    T2: +$1.6\n    T3: +$3.6\n    T4: +$6.4\n    T5: +$10.0\n\n"
        "EXAMPLE TOTAL COSTS (base + delivery + quality):\n"
        "    2wk/T2 → total cost $151.4\n    3wk/T3 → total cost $133.6\n    4wk/T4 → total cost $120.2\n"
        "    5wk/T5 → total cost $111.2\n    6wk/T2 → total cost $93.8\n    2wk/T5 → total cost $159.8\n"
        "    7wk/T3 → total cost $90.4\n\n"
        "COST RANGE: Your cheapest possible configuration (8wk/T2) costs $86.6. Your most expensive (1wk/T5) costs "
        "$183.2.\n\n"
        "Your total cost = base + delivery cost + quality cost.")


def test_the_cost_description_leaves_out_what_the_other_descriptions_say_about_bidding_and_earning():
    for cost_type in ("speed", "quality"):
        text = make_supplier("A", cost_type).describe_for_primitives(cost_type)
        for removed in ("guaranteed", "below cost", "MUST", "Your profit", "you earn nothing", "STRATEGIC", "Play to",
                        "bid price", "margin"):
            assert removed not in text, removed
        assert text.startswith(f"In this event your costs are those of a {cost_type} specialist.")
    with pytest.raises(ValueError, match="'speed' or 'quality'"):
        make_supplier("A", "speed").describe_for_primitives("balanced")


def _table(text, label):
    """The numbers of a table of the cost description (the +$ amounts under the heading `label`)."""
    block = text.split(label, 1)[1].split("\n\n", 1)[0]
    return [float(x) for x in re.findall(r"\+\$([0-9.]+)", block)]


def test_the_cost_statements_of_the_shared_block_hold_for_the_two_types_and_their_tables_are_the_cost_function():
    speed, quality = make_supplier("A", "speed"), make_supplier("B", "quality")
    sd, sq = (_table(speed.describe_for_primitives("speed"), h) for h in ("DELIVERY COST", "QUALITY COST"))
    qd, qq = (_table(quality.describe_for_primitives("quality"), h) for h in ("DELIVERY COST", "QUALITY COST"))
    # "faster delivery costs a speed specialist less than it costs a quality specialist": at every week but the
    # slowest, where both cost nothing, and so is the cost of each further week of acceleration
    assert all(s < q for s, q in zip(sd[:-1], qd[:-1])) and sd[-1] == qd[-1] == 0.0
    assert all(b - a > 0 for a, b in zip(sd[::-1], sd[::-1][1:]))               # faster is more expensive
    assert all(s - t < q - r for s, t, q, r in zip(sd, sd[1:], qd, qd[1:]))     # each week costs the quality type more
    # "a higher quality tier costs a speed specialist more", than it costs a quality specialist
    assert all(s > q for s, q in zip(sq, qq)) and all(b > a for a, b in zip(sq, sq[1:]))
    for supplier, (delivery, tier) in ((speed, (sd, sq)), (quality, (qd, qq))):
        for week in range(1, 9):
            assert delivery[week - 1] == pytest.approx(supplier.gamma_delivery * (8 - week) ** 2)
        for index, t in enumerate((2, 3, 4, 5)):
            assert tier[index] == pytest.approx(supplier.gamma_quality * t ** 2)
        assert supplier.c_base == 85.0


MEDIATOR_SENTENCES = (
    "Only final bids are used to decide the award; earlier bids and the replies are not. "
    "The platform is instructed to work for both the buyer and the suppliers. It sees every supplier's bids in the "
    "event and the buyer's scoring, and it remembers past events on this platform. ")
BUYER_AGENT_SENTENCES = (
    "Only final bids are used to decide the award; earlier bids and the replies are not. "
    "The platform is instructed to work on behalf of the buyer, to get the best deal for the buyer. It knows the "
    "buyer's priorities and sees your bids in this event. It does not see past events. When it replies to you, it "
    "does not see the other supplier's bids. ")


@pytest.mark.parametrize("institution, sentences", [("Mediator", MEDIATOR_SENTENCES),
                                                    ("BuyerAgent", BUYER_AGENT_SENTENCES)])
def test_the_rules_exactly(institution, sentences):
    assert primitives_rules(institution, 3) == (
        "RULES: Each event has 3 rounds. In each round you submit a bid; after rounds 1 and 2 you receive a reply "
        "from the platform. Your bid in round 3 is your final bid. " + sentences
        + "A final bid's score is the buyer's value for its delivery and quality minus its price; the higher-scoring "
          "final bid wins, and the winner is paid its price.\n\n"
          "BUYER'S OUTSIDE OPTION: The buyer is not obligated to award the contract: if both final bids are priced "
          "above the buyer's value for their terms, no contract is awarded.")
    assert AWARD_TEXT + " " + PLATFORM_TEXTS[institution] + " " == sentences


def test_the_two_institutions_rules_differ_in_the_sentences_about_the_platform_only():
    assert MEDIATOR_PLATFORM_TEXT != BUYER_AGENT_PLATFORM_TEXT
    mediator, agent = primitives_rules("Mediator", 3), primitives_rules("BuyerAgent", 3)
    assert mediator.replace(MEDIATOR_PLATFORM_TEXT, "X") == agent.replace(BUYER_AGENT_PLATFORM_TEXT, "X")
    assert AWARD_TEXT == "Only final bids are used to decide the award; earlier bids and the replies are not."
    assert mediator.count(AWARD_TEXT) == agent.count(AWARD_TEXT) == 1         # the one award sentence of both
    assert "It does not see past events. When it replies to you, it does not see the other supplier's bids." in agent
    assert "past events" not in agent.replace("It does not see past events.", "")
    assert "does not see past events" not in mediator and "other supplier's bids" not in mediator
    assert "shown to the buyer" not in mediator + agent                          # neither says what the buyer sees


def test_the_rules_call_the_platforms_answer_a_reply_and_never_feedback():
    for institution in ("Mediator", "BuyerAgent"):
        for rounds in (2, 3, 6):
            text = primitives_rules(institution, rounds)
            assert "feedback" not in text.lower() and "you receive a reply from the platform" in text
            assert "earlier bids and the replies are not" in text


def test_the_rules_name_the_rounds_of_the_run_and_the_rounds_that_get_a_reply():
    for rounds, replies in ((2, "after round 1 you"), (3, "after rounds 1 and 2 you"), (4, "after rounds 1 to 3 you"),
                            (6, "after rounds 1 to 5 you")):
        for institution in ("Mediator", "BuyerAgent"):
            text = primitives_rules(institution, rounds)
            assert text.startswith(f"RULES: Each event has {rounds} rounds. In each round you submit a bid; {replies} "
                                   "receive a reply from the platform.")
            assert f"Your bid in round {rounds} is your final bid." in text
    with pytest.raises(ValueError, match="at least 2 rounds"):
        primitives_rules("Mediator", 1)
    with pytest.raises(ValueError, match="Mediator or the BuyerAgent"):
        primitives_rules("SecondPrice", 3)
    assert PRIMITIVES_OUTSIDE_OPTION.startswith("BUYER'S OUTSIDE OPTION:")


def test_the_headers_the_event_line_the_round_line_and_the_analysts_message_exactly():
    assert FIRST_EVENT_NOTE == "PAST EVENTS: none yet. This is your company's first event on this platform."
    assert BRIEFING_HEADER == "BRIEFING FROM YOUR COMPANY'S PROCUREMENT ANALYST:"
    assert FORWARDED_HEADER == "RECORD FORWARDED BY YOUR COMPANY'S PROCUREMENT ANALYST:"
    assert PP.event_line(4, 6) == "EVENT 4 of 6"
    assert PP.round_line(2, 3) == 'ROUND 2 of 3: Submit your bid as JSON. Keep its "reasoning" field to 1-2 sentences.'
    assert PP.event_preamble(2, 6, "COSTS", "PAST") == "EVENT 2 of 6\n\nCOSTS\n\nPAST"
    assert PP.event_preamble(2, 6, "COSTS", "") == "EVENT 2 of 6\n\nCOSTS"
    assert PP.analyst_message(3, 6, "RULES TEXT", "RECORD TEXT") == (
        "EVENT 3 of 6 is next. Your company's bidder reads these rules with each event:\n"
        "RULES TEXT\n\n"
        "Here is your company's record of past events on this platform:\n"
        "RECORD TEXT\n\n"
        "Write your briefing for your company's bidder.")


# ---- what the bidder of each organization knows of past events ----

RECORD = "YOUR COMPANY'S RECORD OF PAST EVENTS ON THIS PLATFORM:\nRECORD LINES\n"


@pytest.mark.parametrize("organization", SIX)
def test_every_bidder_reads_the_same_line_before_its_first_event(organization):
    assert build_primitives_bidder_context(organization, "", None) == FIRST_EVENT_NOTE


@pytest.mark.parametrize("organization, expected", [
    ("single", RECORD), ("single_revenue", RECORD), ("single_norecord", ""),
    ("organized", BRIEFING_HEADER + "\nBRIEFING"), ("organized_revenue", BRIEFING_HEADER + "\nBRIEFING"),
    ("organized_pass", FORWARDED_HEADER + "\n" + RECORD)])
def test_what_each_bidder_reads_of_past_events_from_the_second_event_on(organization, expected):
    briefing = "BRIEFING" if organization in PRIMITIVES_ANALYST_ORGANIZATIONS else None
    assert build_primitives_bidder_context(organization, RECORD, briefing) == expected


def test_the_forwarded_record_is_the_single_bidders_record_with_a_title():
    assert build_primitives_bidder_context("organized_pass", RECORD, None) == \
        FORWARDED_HEADER + "\n" + build_primitives_bidder_context("single", RECORD, None)


def test_the_bidder_contexts_of_the_primitives_set_refuse_what_does_not_fit():
    for organization in ("single", "single_norecord", "single_revenue", "organized_pass"):
        with pytest.raises(ValueError, match="has no analyst briefing"):
            build_primitives_bidder_context(organization, RECORD, "a briefing")
    for organization in PRIMITIVES_ANALYST_ORGANIZATIONS:
        with pytest.raises(ValueError, match="needs its analyst's briefing"):
            build_primitives_bidder_context(organization, RECORD, None)
        with pytest.raises(ValueError, match="no briefing before its first event"):
            build_primitives_bidder_context(organization, "", "a briefing")
    for organization in ("organized_guided", "hybrid"):
        with pytest.raises(ValueError, match="unknown organization of the primitives set"):
            build_primitives_bidder_context(organization, RECORD, None)


def test_the_organizations_of_the_two_sets():
    assert PRIMITIVES_ORGANIZATIONS == ("single", "single_norecord", "organized", "organized_pass", "single_revenue",
                                        "organized_revenue")
    assert PRIMITIVES_ANALYST_ORGANIZATIONS == ("organized", "organized_revenue")
    assert PRIMITIVES_REVENUE_ORGANIZATIONS == ("single_revenue", "organized_revenue")
    assert ANALYST_ORGANIZATIONS == ("organized", "organized_guided")                         # the advice set's
    assert P.ORGANIZATIONS == ("single", "organized", "organized_guided")


# ---- the plan: the random environment as before, the persistent environment ----

SCENARIOS = get_contract_scenarios()


def _names(plan):
    return [(sc.name, a, b) for sc, a, b in plan]


def test_the_random_plan_is_what_it_was():
    # Written out from the function before the environment existed: seed + 1000 * seq_id seeds the generator, and each
    # event draws the contract, A's type and then whether B's type differs.
    assert _names(pre_draw_event_sequence(6, SCENARIOS, seed=1)) == [
        ("Speed Priority", "speed", "speed"), ("Speed Priority", "quality", "quality"),
        ("Quality Priority", "quality", "speed"), ("Quality Priority", "speed", "quality"),
        ("Speed Priority", "speed", "quality"), ("Quality Priority", "speed", "speed")]
    assert _names(pre_draw_event_sequence(3, SCENARIOS, seed=7)) == [
        ("Speed Priority", "speed", "speed"), ("Speed Priority", "speed", "quality"),
        ("Speed Priority", "speed", "quality")]
    assert _names(pre_draw_event_sequence(3, SCENARIOS, seed=7, seq_id=2)) == [
        ("Speed Priority", "speed", "speed"), ("Quality Priority", "speed", "quality"),
        ("Speed Priority", "speed", "quality")]
    for seed in (1, 7, 42, 99):
        assert pre_draw_event_sequence(6, SCENARIOS, seed=seed, environment="random") == \
            pre_draw_event_sequence(6, SCENARIOS, seed=seed)


@pytest.mark.parametrize("n_events", [1, 2, 6, 11])
def test_the_persistent_plan_repeats_event_1_of_the_random_plan_at_every_event(n_events):
    for seed in range(40):
        random_plan = pre_draw_event_sequence(n_events, SCENARIOS, seed=seed)
        persistent = pre_draw_event_sequence(n_events, SCENARIOS, seed=seed, environment="persistent")
        assert persistent == [random_plan[0]] * n_events
        assert pre_draw_event_sequence(n_events, SCENARIOS, seed=seed) == random_plan       # no state is kept
        assert pre_draw_event_sequence(99, SCENARIOS, seed=seed, environment="persistent")[0] == random_plan[0]


def test_the_persistent_plan_of_the_other_arguments_is_the_random_plans_event_1():
    for kwargs in ({"seq_id": 3}, {"p_different": 0.5}, {"p_different": 1.0}):
        random_plan = pre_draw_event_sequence(5, SCENARIOS, seed=4, **kwargs)
        persistent = pre_draw_event_sequence(5, SCENARIOS, seed=4, environment="persistent", **kwargs)
        assert persistent == [random_plan[0]] * 5


def test_an_unknown_environment_or_no_event_is_handled():
    with pytest.raises(ValueError, match="random, persistent"):
        pre_draw_event_sequence(3, SCENARIOS, seed=1, environment="static")
    assert pre_draw_event_sequence(0, SCENARIOS, seed=1, environment="persistent") == []


def _frequencies(n_seeds=4000):
    """Over the first event of n_seeds seeds (independent draws): how often each contract, each type of A and of B,
    and a pair of different types occur, and, for A's and B's perspective, how often the rival's type is the other."""
    plans = [pre_draw_event_sequence(1, SCENARIOS, seed=seed)[0] for seed in range(n_seeds)]
    n = len(plans)
    return {
        "contract": sum(sc.name == "Speed Priority" for sc, _, _ in plans) / n,
        "A speed": sum(a == "speed" for _, a, _ in plans) / n,
        "B speed": sum(b == "speed" for _, _, b in plans) / n,
        "different": sum(a != b for _, a, b in plans) / n,
        "different, A speed": sum(a != b for _, a, b in plans if a == "speed") / sum(a == "speed" for _, a, _ in plans),
        "different, B speed": sum(a != b for _, a, b in plans if b == "speed") / sum(b == "speed" for _, _, b in plans),
        "different, A quality": sum(a != b for _, a, b in plans if a == "quality")
        / sum(a == "quality" for _, a, _ in plans),
        "different, B quality": sum(a != b for _, a, b in plans if b == "quality")
        / sum(b == "quality" for _, _, b in plans),
    }


def test_the_probabilities_in_the_text_are_those_of_the_plan_for_supplier_a_and_for_supplier_b():
    stated = re.findall(r"probability (\d) in (\d)", PP.DRAWN_RANDOM)
    assert stated == re.findall(r"probability (\d) in (\d)", PP.DRAWN_PERSISTENT)
    assert stated == [("1", "2"), ("1", "2"), ("3", "4")]
    contract, own_type, opposite = (int(a) / int(b) for a, b in stated)
    assert len(SCENARIOS) == 2                                     # "each of the two contracts"
    found = _frequencies()
    tolerance = 0.03                                               # five standard errors or more for 4000 draws
    for name in ("contract", "A speed", "B speed"):
        assert found[name] == pytest.approx(contract, abs=tolerance), name
    assert contract == own_type == 0.5
    for name in ("different", "different, A speed", "different, B speed", "different, A quality",
                 "different, B quality"):
        assert found[name] == pytest.approx(opposite, abs=tolerance), name        # the same from either supplier's side


def test_events_of_the_random_plan_are_drawn_independently():
    plans = [pre_draw_event_sequence(2, SCENARIOS, seed=seed) for seed in range(4000)]
    for index, name in ((0, "contract"), (1, "A's type"), (2, "B's type")):
        def key(event):
            return (event[0].name if index == 0 else event[index]) == ("Speed Priority" if index == 0 else "speed")
        both = sum(key(first) and key(second) for first, second in plans) / len(plans)
        assert both == pytest.approx(0.25, abs=0.03), name                        # P(both) = P(first) * P(second)


# ---- the probabilities that the texts state: the plan is drawn with them ----

def test_the_stated_probability_is_the_default_of_the_draw_and_the_plan_records_the_arguments_of_its_draw():
    stated = [Fraction(int(a), int(b)) for a, b in re.findall(r"probability (\d) in (\d)", PP.DRAWN_RANDOM)]
    assert stated == [Fraction(1, 2), Fraction(1, 2), Fraction(3, 4)]
    assert P_DIFFERENT == 0.75 and Fraction(P_DIFFERENT) == stated[2]            # "3 in 4" is the plan's p_different
    assert inspect.signature(pre_draw_event_sequence).parameters["p_different"].default == P_DIFFERENT
    plan = pre_draw_event_sequence(4, SCENARIOS, seed=5, seq_id=2, environment="persistent")
    assert isinstance(plan, list) and isinstance(plan, DrawnPlan)
    assert (plan.seed, plan.seq_id, plan.p_different, plan.environment) == (5, 2, 0.75, "persistent")
    assert plan == [plan[0]] * 4 == list(plan) and plan[:2] == [plan[0]] * 2        # a list in every other respect
    assert pre_draw_event_sequence(4, SCENARIOS, seed=5, p_different=0.5).p_different == 0.5


def test_the_primitives_set_refuses_a_draw_with_another_probability_and_the_advice_set_does_not():
    for p in (0.5, 0.0, 1.0, 0.7499, 0.76):
        with pytest.raises(ValueError, match="probability 3 in 4, so the draw's p_different must be 0.75"):
            pre_draw_event_sequence(3, SCENARIOS, seed=1, p_different=p, prompt_set="primitives")
        assert len(pre_draw_event_sequence(3, SCENARIOS, seed=1, p_different=p)) == 3      # the advice set states none
        assert len(pre_draw_event_sequence(3, SCENARIOS, seed=1, p_different=p, prompt_set="advice")) == 3
    assert (pre_draw_event_sequence(3, SCENARIOS, seed=1, p_different=0.75, prompt_set="primitives")
            == pre_draw_event_sequence(3, SCENARIOS, seed=1))                              # and the plan is the same
    with pytest.raises(ValueError, match="prompt set is one of advice, primitives"):
        pre_draw_event_sequence(3, SCENARIOS, seed=1, prompt_set="neutral")


def test_run_sequence_refuses_a_plan_drawn_with_another_probability_under_the_primitives_set():
    agents = {sid: MockStrategicSupplier(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    other = pre_draw_event_sequence(2, SCENARIOS, seed=7, p_different=0.5)
    kwargs = dict(mediator=MockPlatformMediator(), organization="single")
    with pytest.raises(ValueError, match="the plan's p_different must be 0.75, not 0.5"):
        run_sequence("Mediator", 2, 2, agents, other, prompt_set="primitives", **kwargs)
    assert len(run_sequence("Mediator", 2, 2, agents, other, prompt_set="advice", **kwargs)) == 2   # the advice set
    stated = pre_draw_event_sequence(2, SCENARIOS, seed=7, prompt_set="primitives")
    assert len(run_sequence("Mediator", 2, 2, agents, stated, prompt_set="primitives", **kwargs)) == 2
    plain = list(other)                         # a plain list records no probability: it is taken as drawn with 3 in 4
    assert not hasattr(plain, "p_different")
    assert len(run_sequence("Mediator", 2, 2, agents, plain, prompt_set="primitives", **kwargs)) == 2


# ---- the platforms' own prompts are unchanged, and the rules are put in place of theirs only when given ----

class Silent:
    """A client that records the requests and answers a good bid or a short text, by what is asked."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        content = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        message = SimpleNamespace(content=content, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                               usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    def user_messages(self):
        return [call["messages"][1]["content"] for call in self.calls]


BID = json.dumps({"price": 130, "delivery_weeks": 5, "quality_tier": 3, "reasoning": "r"})


class RecordsContexts(MockStrategicSupplier):
    def __init__(self, seed, contexts):
        super().__init__(seed=seed)
        self.contexts = contexts

    def generate_bid(self, supplier_costs, regime_context, round_num, history):
        self.contexts.append(regime_context)
        return super().generate_bid(supplier_costs, regime_context, round_num, history)


@pytest.mark.parametrize("institution", ["mediator", "buyer_agent"])
def test_the_institutions_show_the_rules_they_are_given_after_each_suppliers_own_context(institution):
    s, q = make_supplier("A", "speed"), make_supplier("B", "quality")
    sc = SCENARIOS[0]
    contexts = []
    agents = {"A": RecordsContexts(1, contexts), "B": RecordsContexts(2, contexts)}
    kwargs = dict(run_id=1, buyer=sc.buyer, suppliers=[s, q], agents=agents, efficient_surplus=50.0,
                  efficient_winner_id="A", num_rounds=2, cross_event_history={"A": "CONTEXT OF A"},
                  rules="THE RULES")
    if institution == "mediator":
        run_mediator(mediator=MockPlatformMediator(), **kwargs)
    else:
        run_buyer_agent(buyer_agent=MockBuyerAgentPlatform(), **kwargs)
    assert contexts[0::2] == ["CONTEXT OF A\n\nTHE RULES"] * 2          # A: its context, then the rules
    assert contexts[1::2] == ["THE RULES"] * 2                         # B has no entry: only the rules
    assert not any("RULES:" in c or "BUYER'S OUTSIDE OPTION" in c for c in contexts)    # the advice rules are replaced


def test_the_second_price_auction_and_the_old_rules_are_untouched_by_the_rules_argument():
    s, q = make_supplier("A", "speed"), make_supplier("B", "quality")
    sc = SCENARIOS[0]
    contexts = []
    agents = {"A": RecordsContexts(1, contexts), "B": RecordsContexts(2, contexts)}
    run_mediator(1, sc.buyer, [s, q], agents, MockPlatformMediator(), 50.0, "A", num_rounds=2)
    assert contexts[0].startswith("RULES: You submit a bid, then receive feedback from a platform facilitator.")
    assert "BUYER'S OUTSIDE OPTION: The buyer is NOT obligated" in contexts[0]


# ---- run_sequence in the primitives set: what it refuses ----

def _sequence(organization="single", prompt_set="primitives", regime="Mediator", n_events=2, n_rounds=2, **kwargs):
    agents = {sid: MockStrategicSupplier(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    plan = pre_draw_event_sequence(max(n_events, 2), SCENARIOS, seed=7)
    return run_sequence(regime, n_events, n_rounds, agents, plan, mediator=MockPlatformMediator(),
                        buyer_agent_platform=MockBuyerAgentPlatform(), organization=organization,
                        prompt_set=prompt_set, **kwargs)


def test_run_sequence_runs_the_primitives_set_and_leaves_the_advice_set_the_default():
    assert len(_sequence()) == 2 and len(_sequence(prompt_set="advice")) == 2
    for organization in SIX:
        analysts = ({sid: MockProcurementAnalyst() for sid in SUPPLIER_IDS}
                    if organization in PRIMITIVES_ANALYST_ORGANIZATIONS else None)
        assert len(_sequence(organization, analysts=analysts)) == 2


def test_run_sequence_refuses_what_the_primitives_texts_would_make_untrue():
    with pytest.raises(ValueError, match="prompt set is one of advice, primitives"):
        _sequence(prompt_set="neutral")
    with pytest.raises(ValueError, match="unknown organization of the primitives set: 'organized_guided'"):
        _sequence("organized_guided", analysts={sid: MockProcurementAnalyst() for sid in SUPPLIER_IDS})
    with pytest.raises(ValueError, match="unknown organization of the primitives set: 'hybrid'"):
        _sequence({"A": "single", "B": "hybrid"})
    with pytest.raises(ValueError, match="at least 2 events"):
        _sequence(n_events=1)
    with pytest.raises(ValueError, match="no max_history_events"):
        _sequence(max_history_events=2)
    agents = {sid: MockStrategicSupplier(seed=i) for i, sid in enumerate(SUPPLIER_IDS)}
    with pytest.raises(ValueError, match="second-price auction's bidder has a prompt of its own"):
        run_sequence("SecondPrice", 1, 2, agents, pre_draw_event_sequence(1, SCENARIOS, seed=1), organization="single",
                     prompt_set="primitives")


def test_a_bidder_that_names_a_number_of_rounds_must_name_the_runs():
    agents = {sid: PrimitivesSupplierAgent(ChatClient(Silent(BID)), "m", "SYSTEM", n_rounds=3)
              for sid in SUPPLIER_IDS}
    plan = pre_draw_event_sequence(2, SCENARIOS, seed=7)
    with pytest.raises(ValueError, match="the bidder of supplier A names 3 rounds in an event, not 2"):
        run_sequence("Mediator", 2, 2, agents, plan, mediator=MockPlatformMediator(), organization="single",
                     prompt_set="primitives")
    # the mock bidders name none, and the advice set's bidders are not asked
    assert len(_sequence()) == 2


def test_the_primitives_set_refuses_analysts_where_the_organization_has_none_and_needs_them_where_it_has_one():
    for organization in ("single", "single_norecord", "organized_pass", "single_revenue"):
        with pytest.raises(ValueError, match="a supplier without an analyst model has no analysts"):
            _sequence(organization, analysts={"A": MockProcurementAnalyst()})
    for organization in PRIMITIVES_ANALYST_ORGANIZATIONS:
        with pytest.raises(ValueError, match="an organized firm needs one analyst per supplier"):
            _sequence(organization, analysts={"A": MockProcurementAnalyst()})
    # the advice set's message is the one the advice tests pin
    with pytest.raises(ValueError, match="a single-agent firm has no analysts"):
        _sequence("single", prompt_set="advice", analysts={"A": MockProcurementAnalyst()})


class CountsAnalystCalls(ProcurementAnalyst):
    def __init__(self):
        self.messages = []

    def write_briefing(self, supplier_id, message):
        self.messages.append((supplier_id, message))
        return AnalystBriefing(None, f"BRIEFING {len(self.messages)}")


def test_the_analyst_of_the_primitives_set_is_sent_the_rules_and_the_record_and_its_reply_is_the_briefing():
    analysts = {sid: CountsAnalystCalls() for sid in SUPPLIER_IDS}
    contexts = []
    agents = {sid: RecordsContexts(i, contexts) for i, sid in enumerate(SUPPLIER_IDS)}
    plan = pre_draw_event_sequence(3, SCENARIOS, seed=7)
    results = run_sequence("Mediator", 3, 2, agents, plan, mediator=MockPlatformMediator(), analysts=analysts,
                           organization="organized", prompt_set="primitives")
    for sid, analyst in analysts.items():
        assert [m[0] for m in analyst.messages] == [sid, sid]                 # before events 2 and 3 only
        second, third = (m[1] for m in analyst.messages)
        rules = primitives_rules("Mediator", 2)
        assert second.startswith("EVENT 2 of 3 is next. Your company's bidder reads these rules with each event:\n"
                                 + rules + "\n\nHere is your company's record of past events on this platform:\n"
                                 "YOUR COMPANY'S RECORD OF PAST EVENTS ON THIS PLATFORM:\nYour company has "
                                 "participated in 1 previous procurement event(s)")
        assert second.endswith("\n\nWrite your briefing for your company's bidder.")
        assert "Your company has participated in 2 previous procurement event(s)" in third
    assert [r["analyst_briefings"] for r in results] == [{}, {"A": "BRIEFING 1", "B": "BRIEFING 1"},
                                                         {"A": "BRIEFING 2", "B": "BRIEFING 2"}]
    assert [r["analyst_postures"] for r in results] == [{}, {"A": None, "B": None}, {"A": None, "B": None}]


# ---- the bidder agent ----

def _agent(*replies, n_rounds=3):
    client = Silent(*replies)
    return client, PrimitivesSupplierAgent(ChatClient(client), "m", "SYSTEM PROMPT", n_rounds)


def test_the_bidder_sends_its_context_the_earlier_rounds_and_the_round_line_and_nothing_else():
    client, agent = _agent(BID)
    costs = make_supplier("A", "speed")
    first = agent.generate_bid(costs, "CONTEXT AND RULES", 1, [])
    history = [{"round": 1, "bid": ContractTerms(130.0, 5.0, 3), "feedback": "FEEDBACK ONE",
                "price_set_by_rule": False}]
    agent.generate_bid(costs, "CONTEXT AND RULES", 2, history)
    one, two = client.calls
    assert [m["role"] for m in one["messages"]] == ["system", "user"] and one["messages"][0]["content"] == \
        "SYSTEM PROMPT"
    assert one["messages"][1]["content"] == (
        'CONTEXT AND RULES\n\nROUND 1 of 3: Submit your bid as JSON. Keep its "reasoning" field to 1-2 sentences.')
    cost = costs.cost(ContractTerms(0, 5.0, 3))
    assert two["messages"][1]["content"] == (
        "CONTEXT AND RULES\n\nPREVIOUS ROUNDS:\n"
        f"  Round 1: You bid $130.00, 5.0wk delivery, tier 3 (your cost was ${cost:.2f})\n"
        "  Platform's reply: FEEDBACK ONE\n\n"
        'ROUND 2 of 3: Submit your bid as JSON. Keep its "reasoning" field to 1-2 sentences.')
    assert "tools" not in one and "tools" not in two                          # no cost tool
    assert (first.terms, first.round_num, first.supplier_id) == (ContractTerms(130.0, 5.0, 3), 1, "A")
    assert first.reasoning == "[STATED]: r"


def test_the_bidder_takes_a_bid_below_cost_as_submitted_and_asks_once():
    below = json.dumps({"price": 50, "delivery_weeks": 5, "quality_tier": 3, "reasoning": "low"})
    client, agent = _agent(below, BID)
    costs = make_supplier("A", "speed")
    bid = agent.generate_bid(costs, "CONTEXT", 1, [])
    assert bid.terms.price == 50.0 and bid.terms.price < costs.cost(bid.terms)
    assert len(client.calls) == 1                                              # no correction request
    assert "CORRECTION" not in client.user_messages()[0] and bid.reasoning == "[STATED]: low"
    assert "PRICE SET BY RULE" not in bid.reasoning


def test_the_bidder_knows_the_number_of_rounds_and_is_not_the_tool_bidder():
    client, agent = _agent(BID, n_rounds=2)
    assert agent.use_tools is False and agent.sealed_bid is False and agent.n_rounds == 2
    assert isinstance(agent, LocalSupplierAgent)
    for round_num in (0, 3):
        with pytest.raises(ValueError, match="is not a round of an event of 2 rounds"):
            agent.generate_bid(make_supplier("A", "speed"), "CONTEXT", round_num, [])
    assert client.calls == []


def test_the_bidder_whose_reply_has_no_usable_bid_stops_the_run_after_three_attempts():
    client, agent = _agent("no bid here")
    with pytest.raises(RuntimeError, match="no usable bid from the model in 3 attempts"):
        agent.generate_bid(make_supplier("A", "speed"), "CONTEXT", 1, [])
    assert len(client.calls) == 3


# ---- the analyst agent ----

def test_the_analyst_of_the_primitives_set_sends_the_message_as_given_and_returns_the_reply_as_the_briefing():
    client = Silent("<think>hidden</think>  The record is short.  ")
    analyst = LocalProcurementAnalyst(ChatClient(client), "m", "ANALYST SYSTEM")
    briefing = analyst.write_briefing("A", "THE MESSAGE")
    assert (briefing.posture, briefing.text) == (None, "The record is short.")
    assert client.calls[0]["messages"] == [{"role": "system", "content": "ANALYST SYSTEM"},
                                           {"role": "user", "content": "THE MESSAGE"}]
    assert client.calls[0]["temperature"] == 0.3                              # the analyst's settings as before
    assert "Posture" not in briefing.text and "ANALYST BRIEFING" not in briefing.text


def test_the_analyst_of_the_primitives_set_keeps_a_posture_word_in_its_text_and_reads_none_from_it():
    analyst = LocalProcurementAnalyst(ChatClient(Silent("Posture: DEFENSIVE. I recommend GUARDED.")), "m", "S")
    briefing = analyst.write_briefing("A", "MESSAGE")
    assert briefing.posture is None and briefing.text == "Posture: DEFENSIVE. I recommend GUARDED."
    assert not briefing.is_defensive


def test_the_analyst_of_the_primitives_set_stops_the_run_on_an_empty_reply_or_message():
    analyst = LocalProcurementAnalyst(ChatClient(Silent("<think>only thoughts</think>")), "m", "S")
    with pytest.raises(ValueError, match="returned an empty briefing for B"):
        analyst.write_briefing("B", "MESSAGE")
    with pytest.raises(ValueError, match="is sent a message"):
        analyst.write_briefing("B", "")


def test_the_advice_analyst_still_sends_its_message_and_adds_its_title_and_posture():
    client = Silent("The record is thin. Posture: TRUSTING.")
    analyst = LocalProcurementAnalyst(ChatClient(client), "m", P.ANALYST_PROMPT)
    briefing = analyst.generate_briefing("A", "HISTORY")
    assert client.calls[0]["messages"][1]["content"] == (
        "You are the procurement analyst for Supplier A.\n\nHere is the interaction history with this platform:\n\n"
        "HISTORY\n\nProduce your strategic briefing.")
    assert briefing.text == "ANALYST BRIEFING — A: The record is thin. Posture: TRUSTING."
    assert briefing.posture == AnalystBriefing.TRUSTING


def test_the_interface_asks_for_the_new_method_and_the_mock_analyst_counts_past_events():
    with pytest.raises(NotImplementedError):
        ProcurementAnalyst().write_briefing("A", "MESSAGE")
    record = ("YOUR COMPANY'S RECORD OF PAST EVENTS ON THIS PLATFORM:\n  Event 1 (Speed Priority, your company's "
              "costs: speed specialist):\n    Round 1: Your company bid $1\n\n  Event 2 (Quality Priority, your "
              "company's costs: speed specialist):\n")
    briefing = MockProcurementAnalyst().write_briefing("A", PP.analyst_message(3, 6, "RULES", record))
    assert (briefing.posture, briefing.text) == (None, "Mock briefing: 2 past event(s) on record.")


# ---- the contracts paragraph: the bid space, and what each environment does not name ----

BID_SPACE_SENTENCE = "A bid states a price, a delivery time of 1 to 8 weeks, and a quality tier of 2, 3, 4 or 5."


def test_the_contracts_paragraph_differs_between_the_environments_in_its_last_sentence_only():
    random_first, random_last = CONTRACTS_RANDOM.rsplit(" The contract", 1)
    persistent_first, persistent_last = CONTRACTS_PERSISTENT.rsplit(" The contract", 1)
    assert random_first == persistent_first
    assert random_last == " that is up is not named when the event starts."
    assert persistent_last == " is not named when the first event starts."
    for environment, other in (("random", "persistent"), ("persistent", "random")):
        for organization in SIX:
            text = PP.bidder_prompt(organization, 6, environment)
            assert PP.CONTRACTS[environment] in text and PP.CONTRACTS[other] not in text
        for organization in PRIMITIVES_ANALYST_ORGANIZATIONS:
            text = PP.analyst_prompt(organization, 6, environment)
            assert PP.CONTRACTS[environment] in text and PP.CONTRACTS[other] not in text


def test_the_bid_space_the_contracts_paragraph_states_is_the_one_a_bid_must_be_in_to_be_used():
    for text in (CONTRACTS_RANDOM, CONTRACTS_PERSISTENT):                    # after the paragraph's first sentence
        assert text.startswith("Each event is for one of two contracts. " + BID_SPACE_SENTENCE + " For both contracts")
    stated = re.fullmatch(r"A bid states a price, a delivery time of ([0-9]+) to ([0-9]+) weeks, and a quality tier "
                          r"of ([0-9]), ([0-9]), ([0-9]) or ([0-9])\.", BID_SPACE_SENTENCE)
    low, high, *tiers = (int(number) for number in stated.groups())
    assert (float(low), float(high)) == DELIVERY_WEEKS_RANGE and tuple(tiers) == QUALITY_TIERS

    def bid(weeks, tier, price=100.0):
        return {"price": price, "delivery_weeks": weeks, "quality_tier": tier, "reasoning": "r"}

    for weeks in (1, 1.0, 4.5, 7.99, 8, 8.0):                                  # _is_bid is what keeps a bid
        for tier in (2, 3, 4, 5):
            assert _is_bid(bid(weeks, tier)), (weeks, tier)
    for weeks, tier in ((0.99, 3), (0, 3), (8.01, 3), (9, 3), (5, 1), (5, 6), (5, 2.5), (5, 0)):
        assert not _is_bid(bid(weeks, tier)), (weeks, tier)
    assert _is_bid(bid(5, 3, price=0.01)) and not _is_bid(bid(5, 3, price=0)) and not _is_bid(bid(5, 3, price=-1))
    assert "between 1.0 and 8.0" in PP.BID_OUTPUT and "<integer 2-5>" in PP.BID_OUTPUT   # and so does the format


def test_the_system_prompts_say_replies_and_no_longer_a_member_of_your_team_or_what_the_analyst_is_not_told():
    for environment in ENVIRONMENTS:
        texts = ([PP.bidder_prompt(organization, 6, environment) for organization in SIX]
                 + [PP.analyst_prompt(organization, 6, environment)
                    for organization in PRIMITIVES_ANALYST_ORGANIZATIONS])
        for text in texts:
            assert "feedback" not in text.lower() and "member of your team" not in text
            assert "coming event" not in text and "which contract is up" not in text
        assert ("the platform's replies" in PP.ORGANIZED_ROLE and "the platform's replies" in PP.ANALYST_ROLE
                and "the platform's feedback" not in PP.ORGANIZED_ROLE + PP.ANALYST_ROLE)


def test_the_role_blocks_say_the_full_record_what_the_analyst_reads_and_no_team_member():
    assert "your company's full record of past events" in PP.SINGLE_ROLE            # R1, like R4 but for forwarding
    assert "forwards you the full record" in PP.PASS_ROLE
    for role in (PP.ORGANIZED_ROLE, PP.PASS_ROLE, PP.ANALYST_ROLE):
        assert "a member of your team" not in role
    assert PP.ORGANIZED_ROLE.endswith("it also reads the rules you read, and it is not shown the cost figures you are "
                                      "shown.")
    assert "who decides the price, delivery and quality of each bid" in PP.ANALYST_ROLE
    assert "You are not shown the cost figures the bidder is shown." in PP.ANALYST_ROLE


# ---- the record in the third person ----

def _bid(sid, round_num, price, weeks=5.0, tier=3, reasoning="r"):
    return SupplierBid(sid, ContractTerms(price, weeks, tier), reasoning, round_num)


def _entry(scenario, cost_type, own_bids, replies, winner, paid=None, profit=None):
    """One past event as run_sequence keeps it: A's bids by round, the platform's replies to them, and the result.
    The other supplier's bids are kept in the outcome (at $999), though not in the record."""
    other = [_bid("B", bid.round_num, 999.0) for bid in own_bids]
    outcome = SimpleNamespace(all_bids={"A": own_bids, "B": other}, winner_id=winner, supplier_profit=profit,
                              winning_bid=None if paid is None else SimpleNamespace(price=paid))
    return {"scenario_name": scenario, "outcome": outcome, "feedback_history": replies, "own_cost_type": cost_type}


WON = _entry("Speed Priority", "speed",
             [_bid("A", 1, 112.0), _bid("A", 2, 114.0, 4.5, 3, "r" + RULE_PRICE_TAG), _bid("A", 3, 116.0, 4.0, 4)],
             ["Hold on price.", "Better."], "A", 116.0, 10.3)
LOST = _entry("Quality Priority", "quality", [_bid("A", 1, 130.0, 6.0, 5), _bid("A", 2, 128.0, 6.0, 5)],
              ["Move on delivery."], "B", 999.0, 5.0)
WALKED_AWAY = _entry("Speed Priority", "speed", [_bid("A", 1, 150.0), _bid("A", 2, 149.5)], ["Too high."], None)
LOSS = _entry("Quality Priority", "quality", [_bid("A", 1, 90.0, 8.0, 2)], [], "A", 90.0, -52.45)
POSITIVE_ENTRIES = [WON, LOST, WALKED_AWAY]


def test_the_primitives_record_exactly_for_a_win_a_loss_a_walk_away_and_a_loss_of_money():
    assert build_primitives_record("A", POSITIVE_ENTRIES + [LOSS]) == (
        "YOUR COMPANY'S RECORD OF PAST EVENTS ON THIS PLATFORM:\n"
        "Your company has participated in 4 previous procurement event(s) through this same platform. Here is what "
        "happened:\n"
        "\n"
        "  Event 1 (Speed Priority, your company's costs: speed specialist):\n"
        "    Round 1: Your company bid $112.00, 5.0wk, tier 3\n"
        "    Platform's reply: \"Hold on price.\"\n"
        "    Round 2: Your company bid $114.00, 4.5wk, tier 3\n"                    # no note of a price set by rule
        "    Platform's reply: \"Better.\"\n"
        "    Round 3: Your company bid $116.00, 4.0wk, tier 4\n"
        "    Result: your company won. It was paid $116.00; its profit was $10.30.\n"
        "\n"
        "  Event 2 (Quality Priority, your company's costs: quality specialist):\n"
        "    Round 1: Your company bid $130.00, 6.0wk, tier 5\n"
        "    Platform's reply: \"Move on delivery.\"\n"
        "    Round 2: Your company bid $128.00, 6.0wk, tier 5\n"
        "    Result: your company lost. Another supplier won.\n"
        "\n"
        "  Event 3 (Speed Priority, your company's costs: speed specialist):\n"
        "    Round 1: Your company bid $150.00, 5.0wk, tier 3\n"
        "    Platform's reply: \"Too high.\"\n"
        "    Round 2: Your company bid $149.50, 5.0wk, tier 3\n"
        "    Result: No contract was awarded; the buyer walked away.\n"
        "\n"
        "  Event 4 (Quality Priority, your company's costs: quality specialist):\n"
        "    Round 1: Your company bid $90.00, 8.0wk, tier 2\n"
        "    Result: your company won. It was paid $90.00; its profit was a loss of $52.45.\n")
    assert build_primitives_record("A", []) == ""


@pytest.mark.parametrize("profit, written", [(10.3, "$10.30"), (0.0, "$0.00"), (-52.45, "a loss of $52.45"),
                                             (-0.004, "$0.00"), (-0.006, "a loss of $0.01"), (1234.5, "$1234.50")])
def test_a_profit_is_written_as_dollars_and_a_negative_one_as_a_loss_of_dollars(profit, written):
    record = build_primitives_record("A", [_entry("Speed Priority", "speed", [_bid("A", 1, 100.0)], [], "A", 100.0,
                                                  profit)])
    assert record.endswith(f"    Result: your company won. It was paid $100.00; its profit was {written}.\n")
    assert "$-" not in record                                    # never a minus sign inside a dollar amount


def _as_primitives(advice_record):
    """The advice set's record turned into the primitives set's by the changes of the design, one by one."""
    text = advice_record.replace("YOUR PAST PROCUREMENT HISTORY WITH THIS PLATFORM:",
                                 "YOUR COMPANY'S RECORD OF PAST EVENTS ON THIS PLATFORM:")
    text = text.replace("You have participated in", "Your company has participated in")
    text = text.replace(", your costs: ", ", your company's costs: ")
    text = text.replace(": You bid ", ": Your company bid ").replace("    Feedback: ", "    Platform's reply: ")
    text = text.replace(" (price set by rule: cost + 5%)", "")          # no rule sets a price in the primitives set
    text = re.sub(r"Result: You WON\. You were paid (\$[0-9.]+)\. Your profit was (\$[0-9.]+)\.",
                  r"Result: your company won. It was paid \1; its profit was \2.", text)
    return text.replace("Result: You LOST. Another supplier won.", "Result: your company lost. Another supplier won.")


def test_the_primitives_record_is_the_advice_record_with_those_changes_only_and_every_number_is_kept():
    advice = build_supplier_history("A", POSITIVE_ENTRIES)
    assert build_primitives_record("A", POSITIVE_ENTRIES) == _as_primitives(advice)
    assert advice != build_primitives_record("A", POSITIVE_ENTRIES)
    for entries in (POSITIVE_ENTRIES, POSITIVE_ENTRIES + [LOSS], [LOSS], [WON], [LOST]):
        texts = (build_supplier_history("A", entries).replace(" (price set by rule: cost + 5%)", ""),
                 # (the rule's note, and its 5, are the advice set's)
                 build_primitives_record("A", entries))
        numbers = [re.findall(r"[0-9]+(?:[.][0-9]+)?", text) for text in texts]
        assert numbers[0] == numbers[1] and numbers[0], entries             # the same numbers in the same order
    assert "$-52.45" in build_supplier_history("A", [LOSS])                     # the advice record is as it was
    assert "(price set by rule: cost + 5%)" in advice and "price set by rule" not in build_primitives_record(
        # but the primitives record has no such rule
        "A", POSITIVE_ENTRIES)
    assert "$-52.45" not in build_primitives_record("A", [LOSS])


def test_the_advice_set_keeps_its_record_in_the_second_person_and_with_feedback():
    record = build_supplier_history("A", POSITIVE_ENTRIES + [LOSS])
    assert record.startswith(
        "YOUR PAST PROCUREMENT HISTORY WITH THIS PLATFORM:\n"
        "You have participated in 4 previous procurement event(s) through this same platform. Here is what "
        "happened:\n\n  Event 1 (Speed Priority, your costs: speed specialist):\n"
        "    Round 1: You bid $112.00, 5.0wk, tier 3\n    Feedback: \"Hold on price.\"\n"
        "    Round 2: You bid $114.00 (price set by rule: cost + 5%), 4.5wk, tier 3\n    Feedback: \"Better.\"\n"
        "    Round 3: You bid $116.00, 4.0wk, tier 4\n"
        "    Result: You WON. You were paid $116.00. Your profit was $10.30.\n")
    assert "    Result: You LOST. Another supplier won.\n" in record
    assert "    Result: No contract was awarded; the buyer walked away.\n" in record
    assert "Your profit was $-52.45." in record


def test_the_primitives_record_is_true_of_the_analyst_who_reads_it_too_and_shows_nothing_of_the_other_supplier():
    record = build_primitives_record("A", POSITIVE_ENTRIES + [LOSS])
    assert re.findall(r"\b(?:you|your(?! company)|yours|yourself)\b", record, re.I) == []     # no second person
    assert record.count("Your company bid ") == 8 and record.count("your company's costs: ") == 4
    assert "999" not in record and "Supplier" not in record and "margin" not in record.lower()
    assert "feedback" not in record.lower()
    assert "$999.00" in build_primitives_record("B", POSITIVE_ENTRIES)      # B's record holds B's own bids


def test_the_previous_rounds_block_says_reply_in_the_primitives_set_and_feedback_in_the_advice_set():
    costs = make_supplier("A", "speed")
    history = [{"round": 1, "bid": ContractTerms(130.0, 5.0, 3), "feedback": "THE ANSWER", "price_set_by_rule": False},
               {"round": 2, "bid": ContractTerms(128.0, 5.0, 3), "feedback": "", "price_set_by_rule": False}]
    primitives = _previous_rounds_block(costs, history, PP.REPLY_LABEL)
    advice = _previous_rounds_block(costs, history)
    assert "  Platform's reply: THE ANSWER\n" in primitives and "Feedback" not in primitives
    assert "  Feedback received: THE ANSWER\n" in advice and "Platform's reply" not in advice
    assert primitives.replace("Platform's reply", "Feedback received") == advice     # the label is the only difference
    assert _previous_rounds_block(costs, [], PP.REPLY_LABEL) == "" == _previous_rounds_block(costs, [])
