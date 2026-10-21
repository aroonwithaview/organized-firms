"""The primitives-only prompt set (--prompts primitives): every text that an agent of a supplier reads, apart from
the platforms' own prompts and the rules, states the agent's objective, what it knows and does not know, who decides
and the payoffs, and gives no advice. The rules are in orgsim/institutions.py (primitives_rules), the cost
description in orgsim/primitives.py (SupplierCosts.describe_for_primitives), the record of past events in
orgsim/sequence.py (build_primitives_record, which speaks of "your company" so that it is true of the analyst who
reads it as well as of the bidder) and what the bidder is told about past events in orgsim/organization.py
(build_primitives_bidder_context). The advice set (orgsim/prompts.py) is unchanged and stays the default; this set
shares only the platforms' prompts (MEDIATOR_PROMPT, BUYER_AGENT_PROMPT) with it.

The set has six organizations (orgsim.organization.PRIMITIVES_ORGANIZATIONS). A bidder's system prompt is
the shared block, its role block, for a revenue organization the sentences that state its objective, and the output
format; an analyst's is the shared block, the analyst's role block and, in organized_revenue, the sentences that state
its objective. The shared block holds the number of events, the paragraph on the two contracts (whose last
sentence says when the contract is not named) and the paragraph on how the events are drawn, which differ between
the two environments (orgsim.sequence.ENVIRONMENTS), so a system prompt is built for the run's number of events and
its environment. What the bidder reads with each event (the event line, the costs, what it
knows of past events, the rules, the earlier rounds of the event and the round line) and the analyst's message are
built by the functions below and by the runner (orgsim.sequence.run_sequence).

Every sentence is true of the code in both environments, which tests/test_primitives_prompts.py checks where a
test can: the probabilities against pre_draw_event_sequence, the cost statements against the parameters of the two
types, the rules against the institutions, and the statements about who is shown what against the requests of a run.
The revenue sentences state the objective the agent is given, not a payoff that the code computes.
"""
from orgsim.organization import (
    ALL_PRIMITIVES_ANALYST_ORGANIZATIONS, ALL_PRIMITIVES_ORGANIZATIONS, ALL_PRIMITIVES_REVENUE_ORGANIZATIONS,
    BRIEFING_WORD_LIMIT,
)

PROMPT_SETS = ("advice", "primitives")
ENVIRONMENTS = ("random", "persistent")

# The statements of objective. The team arms (every organization but the two revenue ones) give everyone the
# company's objective; in a revenue arm the owner wants the company's profit and the bidder is judged on revenue.
TEAM_OBJECTIVE = "Everyone in your company has the same objective: the company's total profit across the {n} events."
TEAM_KNOWLEDGE = "Everyone in the company knows this, and knows who holds which information and who decides."
REVENUE_OWNER = "Your company's owner wants the company's total profit across the {n} events."
REVENUE_KNOWLEDGE = ("Everyone in the company knows each member's objective, and who holds which information and who "
                     "decides.")

# The shared block, in its paragraphs. {n} is the run's number of events.
OPENING = "Your company is a supplier. It takes part in a sequence of {n} procurement events on a platform."
PAYOFFS_AND_COSTS = (
    "In each event, if your company wins, it is paid the price of its final bid, and its profit is that price minus "
    "its total cost at that bid's delivery and quality; if it does not win, its profit from that event is zero. Your "
    "company's costs are private. In each event they are those of either a speed specialist or a quality "
    "specialist: faster delivery costs a speed specialist less than it costs a quality specialist, and a higher "
    "quality tier costs a speed specialist more. One other supplier takes part in each event; your company is not "
    "told its costs or its bids. The same other supplier takes part in every event.")
# Items 2 and 3: the bid space is stated here, for every role, the analyst included; and what the persistent environment
# does not name is the contract of its first event, since the record names the contract of the later ones.
CONTRACTS_RANDOM = (
    "Each event is for one of two contracts. A bid states a price, a delivery time of 1 to 8 weeks, and a quality tier "
    "of 2, 3, 4 or 5. For both contracts the buyer values faster delivery and a higher quality tier; how much it "
    "values each differs between the two contracts. The contract that is up is not named when the event starts.")
CONTRACTS_PERSISTENT = (
    "Each event is for one of two contracts. A bid states a price, a delivery time of 1 to 8 weeks, and a quality tier "
    "of 2, 3, 4 or 5. For both contracts the buyer values faster delivery and a higher quality tier; how much it "
    "values each differs between the two contracts. The contract is not named when the first event starts.")
CONTRACTS = {"random": CONTRACTS_RANDOM, "persistent": CONTRACTS_PERSISTENT}
DRAWN_PERSISTENT = (
    "How the events are drawn: before the first event, the contract is drawn at random, each of the two with "
    "probability 1 in 2; your company's cost type is drawn at random, speed or quality with probability 1 in 2; and "
    "the other supplier's cost type is the opposite of yours with probability 3 in 4. The contract and both cost types "
    "then stay the same in all {n} events.")
DRAWN_RANDOM = (
    "How the events are drawn: for each event, independently, the contract is drawn at random, each of the two with "
    "probability 1 in 2; your company's cost type is drawn at random, speed or quality with probability 1 in 2; and "
    "the other supplier's cost type is the opposite of yours with probability 3 in 4.")

# The role blocks.
BIDDER_ROLE = "You are your company's bidder: you decide the price, delivery and quality of each of its bids."
SINGLE_ROLE = (
    BIDDER_ROLE + " From your company's second event on, your company's full record of past events on this platform "
    "appears in the message below.")
NORECORD_ROLE = BIDDER_ROLE + " Your company keeps no record of past events on this platform."
ORGANIZED_ROLE = (
    BIDDER_ROLE + " Your company's record of past events on this platform is kept by your company's procurement "
    "analyst. From your company's second event on, the analyst writes you a briefing before each event, and it "
    "appears in the message below. You do not see the record; you see the briefing. The analyst holds the record, "
    "which shows, for each past event, your company's cost type, the contract, your company's bids, the platform's "
    "replies and the result; it also reads the rules you read, and it is not shown the cost figures you are shown.")
PASS_ROLE = (
    BIDDER_ROLE + " Your company's record of past events on this platform is kept by your company's procurement "
    "analyst. From your company's second event on, the analyst forwards you the full record before each event, "
    "unchanged, and it appears in the message below.")
ANALYST_ROLE = (
    "You are your company's procurement analyst. You hold your company's record of past events on this platform. From "
    "your company's second event on, you write a briefing before each event for your company's bidder, who decides "
    "the price, delivery and quality of each bid. The bidder does not see the record; it sees your briefing, its "
    "costs for the event, the rules below and, during the event, the platform's replies. You are not shown the cost "
    "figures the bidder is shown. Write the briefing as plain text.")
# The role text of the bidder of brief, forward and brief_cost (and their revenue variants), from event 1. It is the
# same for all of them, so that their first events are identical; what differs from the second event on is the message,
# and its heading (orgsim.organization.BRIEF_HEADER, FORWARD_HEADER) says how it was made.
DEPARTURE_ROLE = (
    BIDDER_ROLE + " Your company's procurement analyst keeps your company's record of past events on this platform; "
    "for each past event, the record shows your company's cost type, the contract, your company's bids, the "
    "platform's replies and the result. From your company's second event on, the analyst sends you a message before "
    "each event. It is included in what you read in each round of that event, under a heading that begins MESSAGE "
    "FROM YOUR COMPANY'S PROCUREMENT ANALYST.")
# The analyst's role text in brief_cost: the sentence that says it is not shown the bidder's cost figures is replaced.
ANALYST_COST_ROLE = (
    "You are your company's procurement analyst. You hold your company's record of past events on this platform. From "
    "your company's second event on, you write a briefing before each event for your company's bidder, who decides "
    "the price, delivery and quality of each bid. The bidder does not see the record; it sees your briefing, its "
    "costs for the event, the rules below and, during the event, the platform's replies. Before each briefing you are "
    "also shown the cost figures the bidder is shown for that event. Write the briefing as plain text.")
# The role text of the bidder of floor, from event 1: it names the one limit that the analyst may set (a minimum profit
# per contract) and how the code enforces it. Its first sentence is the usual one with that limit added.
FLOOR_ROLE = (
    "You are your company's bidder: you decide the price, delivery and quality of each of its bids, within one "
    "limit that your company's procurement analyst may set. The analyst keeps your company's record of past events "
    "on this platform; for each past event, the record shows your company's cost type, the contract, your company's "
    "bids, the platform's replies and the result. From your company's second event on, the analyst sends you a "
    "message before each event, and in it may set a minimum profit per contract for that event. In each round, a "
    "bid whose price is less than its total cost at its delivery and quality plus that minimum is returned to you "
    "once, and the platform does not see it; if the bid you send back is also below it, its price is raised to its "
    "total cost at that bid's delivery and quality plus the minimum. The message is included in what you read in each "
    "round of that event, under a heading that begins MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST.")
# The analyst's role text in floor: the usual one, the minimum profit that it may set, the line that ends its briefing,
# and what the bidder reads of the briefing.
ANALYST_FLOOR_ROLE = (
    ANALYST_ROLE + " In your briefing you may also set a minimum profit per contract for the coming event. In each "
    "round, a bid whose price is less than its total cost at its delivery and quality plus that minimum is returned "
    "to the bidder once; if the bid it sends back is also below it, its price is raised to its total cost plus the "
    "minimum. End your briefing with one line, either MINIMUM PROFIT PER CONTRACT: $X or MINIMUM PROFIT PER "
    "CONTRACT: none. The bidder reads your briefing without any line that begins with the words MINIMUM PROFIT PER "
    "CONTRACT, and a notice of the minimum.")
# The analyst's role text in brief_cap: the limit of the briefing is added.
ANALYST_CAP_ROLE = ANALYST_ROLE + f" Your briefing may have at most {BRIEFING_WORD_LIMIT} words."
ROLES = {"single": SINGLE_ROLE, "single_norecord": NORECORD_ROLE, "organized": ORGANIZED_ROLE,
         "organized_pass": PASS_ROLE, "single_revenue": SINGLE_ROLE, "organized_revenue": ORGANIZED_ROLE}
ROLES.update({name: DEPARTURE_ROLE for name in ("brief", "forward", "brief_cap", "brief_cost", "brief_revenue",
                                                "forward_revenue", "brief_cap_revenue", "brief_cost_revenue")})
ROLES.update({"floor": FLOOR_ROLE, "floor_revenue": FLOOR_ROLE})
# The analyst's role text by organization (those with an analyst that calls a model).
ANALYST_ROLES = {"organized": ANALYST_ROLE, "organized_revenue": ANALYST_ROLE, "brief": ANALYST_ROLE,
                 "brief_revenue": ANALYST_ROLE, "brief_cap": ANALYST_CAP_ROLE, "brief_cap_revenue": ANALYST_CAP_ROLE,
                 "brief_cost": ANALYST_COST_ROLE, "brief_cost_revenue": ANALYST_COST_ROLE,
                 "floor": ANALYST_FLOOR_ROLE, "floor_revenue": ANALYST_FLOOR_ROLE}

# The sentences that state a revenue organization's objectives, after its role block.
REVENUE_BIDDER_OBJECTIVE = ("Your own objective, as the company's bidder, is the company's revenue: the total price of "
                            "the contracts it wins across the {n} events.")
ANALYST_PROFIT_OBJECTIVE = "The analyst's objective is the company's total profit across the {n} events."
REVENUE_ANALYST_OBJECTIVES = (
    "Your own objective, as the company's analyst, is the company's total profit across the {n} events. The bidder's "
    "objective is the company's revenue: the total price of the contracts it wins across the {n} events.")

# What both bidders must answer with. BID_OUTPUT of the advice set is a different block and is unchanged.
BID_OUTPUT = ('Respond with ONLY a JSON object with these exact fields: {"price": <number>, "delivery_weeks": <number '
              'between 1.0 and 8.0>, "quality_tier": <integer 2-5>, "reasoning": "<brief reasoning>"}')

# The round instruction, which names the round and the number of rounds.
ROUND_INSTRUCTION = 'Submit your bid as JSON. Keep its "reasoning" field to 1-2 sentences.'

# What the earlier rounds of an event and the record call the platform's answer to a bid. The rules call it a reply too.
REPLY_LABEL = "Platform's reply"


def _check(organization: str, allowed: tuple[str, ...], n_events: int, environment: str) -> None:
    if organization not in allowed:
        raise ValueError(f"unknown organization of the primitives set: {organization!r}")
    if n_events < 2:
        raise ValueError("the primitives set is for a sequence of at least 2 events")
    if environment not in ENVIRONMENTS:
        raise ValueError(f"the environment is one of {', '.join(ENVIRONMENTS)}, not {environment!r}")


def shared_block(organization: str, n_events: int, environment: str) -> str:
    """The shared block of an organization's agents, for a run of `n_events` events in the given environment."""
    _check(organization, ALL_PRIMITIVES_ORGANIZATIONS, n_events, environment)
    if organization in ALL_PRIMITIVES_REVENUE_ORGANIZATIONS:
        objective, knowledge = REVENUE_OWNER, REVENUE_KNOWLEDGE
    else:
        objective, knowledge = TEAM_OBJECTIVE, TEAM_KNOWLEDGE
    opening = f"{OPENING} {objective} {knowledge}".format(n=n_events)
    drawn = DRAWN_PERSISTENT.format(n=n_events) if environment == "persistent" else DRAWN_RANDOM
    return "\n\n".join([opening, PAYOFFS_AND_COSTS, CONTRACTS[environment], drawn])


def bidder_prompt(organization: str, n_events: int, environment: str) -> str:
    """System prompt of the bidder of this organization: the shared block, the role block, in a revenue organization
    the sentences that state the objectives, and the output format."""
    parts = [shared_block(organization, n_events, environment), ROLES[organization]]
    if organization == "single_revenue":
        parts.append(REVENUE_BIDDER_OBJECTIVE.format(n=n_events))
    # the bidder is told the analyst's objective, forward too
    elif organization in ALL_PRIMITIVES_REVENUE_ORGANIZATIONS:
        parts.append(f"{REVENUE_BIDDER_OBJECTIVE} {ANALYST_PROFIT_OBJECTIVE}".format(n=n_events))
    parts.append(BID_OUTPUT)
    return "\n\n".join(parts)


def analyst_prompt(organization: str, n_events: int, environment: str) -> str:
    """System prompt of the analyst of this organization (one that has an analyst that calls a model; the analyst of a
    revenue variant also reads the sentences that state the objectives): the shared block and the analyst's role
    block."""
    _check(organization, ALL_PRIMITIVES_ANALYST_ORGANIZATIONS, n_events, environment)
    parts = [shared_block(organization, n_events, environment), ANALYST_ROLES[organization]]
    if organization in ALL_PRIMITIVES_REVENUE_ORGANIZATIONS:
        parts.append(REVENUE_ANALYST_OBJECTIVES.format(n=n_events))
    return "\n\n".join(parts)


def event_line(event: int, n_events: int) -> str:
    """The first line of what a bidder reads with each event."""
    return f"EVENT {event} of {n_events}"


def event_preamble(event: int, n_events: int, costs: str, past_events: str) -> str:
    """What a bidder reads before the rules: the event line, the costs and, unless there is none, what it knows of past
    events (orgsim.organization.build_primitives_bidder_context)."""
    return "\n\n".join([event_line(event, n_events), costs] + ([past_events] if past_events else []))


def round_line(round_num: int, n_rounds: int) -> str:
    """The last line of what a bidder reads in a round."""
    return f"ROUND {round_num} of {n_rounds}: {ROUND_INSTRUCTION}"


def analyst_message(event: int, n_events: int, rules: str, record: str, costs: tuple[str, str] | None = None) -> str:
    """What an analyst reads before event `event` (the second or later): the rules its bidder reads, verbatim, and
    the company's record. The analyst of brief_cost is also given `costs`, the cost type of the coming event
    ("speed" or "quality") and the bidder's cost structure in the third person (SupplierCosts.describe_for_analyst),
    which come after the record."""
    shown = "" if costs is None else (f"In the coming event your company's costs are those of a {costs[0]} specialist."
                                      f"\n\n{costs[1]}\n\n")
    return (f"{event_line(event, n_events)} is next. Your company's bidder reads these rules with each event:\n"
            f"{rules}\n\n"
            f"Here is your company's record of past events on this platform:\n"
            f"{record}\n\n"
            f"{shown}"
            f"Write your briefing for your company's bidder.")
