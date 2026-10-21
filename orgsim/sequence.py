"""Repeated procurement events with persistent supplier memory."""
import random
from collections.abc import Callable

from orgsim.primitives import (
    PRIVATE_COST_HEADER, SHARED_COST_HEADER, BuyerPreferences, ContractScenario, RunOutcome, SupplierCosts,
    compute_max_efficient_surplus,
)
from orgsim.departures import FLOOR_RAISED_NOTE, capped_briefing, floor_briefing
from orgsim.interfaces import FLOOR_PRICE_TAG, RULE_PRICE_TAG, BuyerAgentPlatform, PlatformMediator, SupplierAgent
from orgsim.organization import (ALL_PRIMITIVES_ANALYST_ORGANIZATIONS, ALL_PRIMITIVES_ORGANIZATIONS,
                                 ANALYST_ORGANIZATIONS, AnalystBriefing, ProcurementAnalyst, build_bidder_context,
                                 build_primitives_bidder_context, departure_base, forwards_the_record,
                                 organization_of_each)
from orgsim.institutions import primitives_rules, run_buyer_agent, run_mediator, run_second_price
from orgsim.primitives_prompts import ENVIRONMENTS, PROMPT_SETS, REPLY_LABEL, analyst_message, event_preamble

SPEED_SPECIALIST = {"c_base": 85.0, "gamma_delivery": 0.5, "gamma_quality": 1.8}
QUALITY_SPECIALIST = {"c_base": 85.0, "gamma_delivery": 1.8, "gamma_quality": 0.4}
SUPPLIER_IDS = ["A", "B"]


def make_supplier(supplier_id: str, cost_type: str) -> SupplierCosts:
    """Create a SupplierCosts with the given type for this event."""
    params = SPEED_SPECIALIST if cost_type == "speed" else QUALITY_SPECIALIST
    label = "SpeedSpec" if cost_type == "speed" else "QualSpec"
    return SupplierCosts(
        supplier_id=supplier_id,
        name=label,
        c_base=params["c_base"],
        gamma_delivery=params["gamma_delivery"],
        gamma_quality=params["gamma_quality"],
    )


# =============================================================================
# Supplier cross-event history builder (minimal private-sector disclosure)
# =============================================================================

def build_supplier_history(
    supplier_id: str,
    past_outcomes: list[dict],
    max_events: int = 0,
) -> str:
    """Build a natural-language history of past procurement events for a supplier.

    Realistic private-sector disclosure: the supplier learns only their own
    result (it won and was paid, another supplier won, or no contract was
    awarded) and the platform feedback received. No score margins, competitor
    bids, or participation counts are disclosed.

    Args:
        max_events: If > 0, truncate to the most recent N events (for local
            LLMs with limited context). Earlier events are summarised as
            "[N earlier event(s) omitted for brevity]". If 0, include all events.
    """
    if not past_outcomes:
        return ""

    total_events = len(past_outcomes)
    if max_events > 0 and total_events > max_events:
        omitted = total_events - max_events
        past_outcomes = past_outcomes[-max_events:]
    else:
        omitted = 0

    lines = [
        f"YOUR PAST PROCUREMENT HISTORY WITH THIS PLATFORM:",
        f"You have participated in {total_events} previous procurement "
        f"event(s) through this same platform. Here is what happened:\n",
    ]
    if omitted > 0:
        lines.append(f"  [{omitted} earlier event(s) omitted for brevity]\n")

    # Event numbering: use original indices so the supplier sees consistent numbering
    start_idx = omitted + 1
    for i, entry in enumerate(past_outcomes, start_idx):
        outcome = entry["outcome"]
        scenario = entry["scenario_name"]
        feedback_list = entry.get("feedback_history", [])
        cost_type = entry.get("own_cost_type", "unknown")

        won = outcome.winner_id == supplier_id
        bids = outcome.all_bids.get(supplier_id, [])

        lines.append(f"  Event {i} ({scenario}, your costs: {cost_type} specialist):")

        for bid in bids:
            rule_note = " (price set by rule: cost + 5%)" if RULE_PRICE_TAG in bid.reasoning else ""
            lines.append(
                f"    Round {bid.round_num}: You bid ${bid.terms.price:.2f}{rule_note}, "
                f"{bid.terms.delivery_weeks}wk, tier {bid.terms.quality_tier}"
            )
            fb_idx = bid.round_num - 1
            if fb_idx < len(feedback_list):
                lines.append(f"    Feedback: \"{feedback_list[fb_idx]}\"")

        if won:
            profit = outcome.supplier_profit
            lines.append(
                f"    Result: You WON. You were paid ${outcome.winning_bid.price:.2f}. "
                f"Your profit was ${profit:.2f}."
            )
        elif outcome.winner_id is None:
            lines.append("    Result: No contract was awarded; the buyer walked away.")
        else:
            lines.append(f"    Result: You LOST. Another supplier won.")

        # POST-EVENT DISCLOSURE: minimal, realistic for private-sector procurement.
        # The platform does NOT disclose: score margins, competitor bids,
        # or number of participants. The supplier only knows their own result:
        # it won and was paid, another supplier won, or no contract was awarded.
        # This is standard in private-sector procurement — opacity is the norm.
        #
        # The supplier CAN infer competitive conditions from:
        # - Whether they won or lost
        # - The price at which they won (their own bid)
        # - The pattern of wins/losses at various price levels over time

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# The record of the primitives prompt set
# =============================================================================

PRIMITIVES_RECORD_HEADER = "YOUR COMPANY'S RECORD OF PAST EVENTS ON THIS PLATFORM:"


def _profit_phrase(profit: float) -> str:
    """A profit as the primitives record writes it: "$10.30", or "a loss of $52.45" when it is negative. A negative
    profit that rounds to $0.00 is written "$0.00", not as a loss of nothing."""
    cents = f"{abs(profit):.2f}"
    return f"a loss of ${cents}" if profit < 0 and float(cents) > 0 else f"${cents}"


def build_primitives_record(supplier_id: str, past_outcomes: list[dict]) -> str:
    """The record of a supplier's past events in the primitives prompt set, or "" before its first event.

    It holds the facts of build_supplier_history (the advice set's record, which is unchanged) with the same numbers,
    event and contract labels and cost-type labels, and the same disclosure: the supplier's own bids, the platform's
    replies, and its own result, never a score margin, a competitor's bid or the number of participants. It differs
    in how it says them:
      - it speaks of "your company" in the third person ("Your company bid ...", "Result: your company won. It was
        paid $X; its profit was $Y."), so that every sentence is true of the analyst who reads it for the bidder
        as well as of the bidder who reads it itself;
      - a negative profit is written "a loss of $52.45", not "$-52.45";
      - the platform's answer to a bid is called "Platform's reply" (primitives_prompts.REPLY_LABEL), as the rules
        call it;
      - no rule sets a price in the primitives set, so the note of a price set by the below-cost rule is never printed;
        a price that the floor raised (FLOOR_PRICE_TAG in the bid's reasoning) is followed by FLOOR_RAISED_NOTE;
      - it is always the full record: the primitives set takes no max_history_events (its texts say the record is full).
    """
    if not past_outcomes:
        return ""
    lines = [
        PRIMITIVES_RECORD_HEADER,
        f"Your company has participated in {len(past_outcomes)} previous procurement "
        f"event(s) through this same platform. Here is what happened:\n",
    ]
    for i, entry in enumerate(past_outcomes, 1):
        outcome = entry["outcome"]
        feedback_list = entry.get("feedback_history", [])
        cost_type = entry.get("own_cost_type", "unknown")

        lines.append(f"  Event {i} ({entry['scenario_name']}, your company's costs: {cost_type} specialist):")
        for bid in outcome.all_bids.get(supplier_id, []):
            floor_note = FLOOR_RAISED_NOTE if FLOOR_PRICE_TAG in bid.reasoning else ""
            lines.append(
                f"    Round {bid.round_num}: Your company bid ${bid.terms.price:.2f}{floor_note}, "
                f"{bid.terms.delivery_weeks}wk, tier {bid.terms.quality_tier}"
            )
            fb_idx = bid.round_num - 1
            if fb_idx < len(feedback_list):
                lines.append(f"    {REPLY_LABEL}: \"{feedback_list[fb_idx]}\"")

        if outcome.winner_id == supplier_id:
            lines.append(
                f"    Result: your company won. It was paid ${outcome.winning_bid.price:.2f}; "
                f"its profit was {_profit_phrase(outcome.supplier_profit)}."
            )
        elif outcome.winner_id is None:
            lines.append("    Result: No contract was awarded; the buyer walked away.")
        else:
            lines.append("    Result: your company lost. Another supplier won.")
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Mediator cross-event history builder
# =============================================================================

def build_mediator_cross_event_history(
    supplier_id: str,
    past_outcomes: list[dict],
    max_events: int = 0,
) -> str:
    """Build the mediator's memory of a supplier across past procurement events.

    Args:
        max_events: If > 0, keep only the most recent N events (for local LLMs with limited
            context), as build_supplier_history does for the supplier's own record. Earlier
            events are summarised as "[N earlier event(s) omitted for brevity]". If 0, include
            all events.
    """
    if not past_outcomes:
        return ""

    total_events = len(past_outcomes)
    if max_events > 0 and total_events > max_events:
        omitted = total_events - max_events
        past_outcomes = past_outcomes[-max_events:]
    else:
        omitted = 0

    lines = [
        f"PLATFORM MEMORY — SUPPLIER {supplier_id}:",
        f"You have worked with this supplier in {total_events} previous "
        f"procurement event(s). Here is what you observed:\n",
    ]
    if omitted > 0:
        lines.append(f"  [{omitted} earlier event(s) omitted for brevity]\n")

    # Event numbering: use original indices so the mediator sees the numbers of the full memory
    for i, entry in enumerate(past_outcomes, omitted + 1):
        outcome: RunOutcome = entry["outcome"]
        scenario = entry["scenario_name"]
        feedback_list = entry.get("feedback_history", [])

        bids = outcome.all_bids.get(supplier_id, [])
        won = outcome.winner_id == supplier_id

        lines.append(f"  Event {i} ({scenario}):")

        for bid in bids:
            lines.append(
                f"    Round {bid.round_num}: Bid ${bid.terms.price:.2f}, "
                f"{bid.terms.delivery_weeks}wk, tier {bid.terms.quality_tier}"
            )
            fb_idx = bid.round_num - 1
            if fb_idx < len(feedback_list):
                lines.append(f"    Your feedback to them: \"{feedback_list[fb_idx]}\"")

        if won:
            lines.append(f"    Outcome: They WON at ${outcome.winning_bid.price:.2f}.")
        elif outcome.winner_id is None:
            lines.append("    Outcome: No contract was awarded; the buyer walked away.")
        else:
            lines.append(f"    Outcome: They LOST.")

        if len(bids) >= 2:
            first, last = bids[0].terms, bids[-1].terms
            delivery_moved = abs(first.delivery_weeks - last.delivery_weeks)
            quality_moved = abs(first.quality_tier - last.quality_tier)
            price_moved = first.price - last.price

            signals = []
            if delivery_moved > 0.5:
                signals.append(f"delivery flexibility ({first.delivery_weeks}→{last.delivery_weeks}wk)")
            if quality_moved >= 1:
                signals.append(f"quality flexibility (tier {first.quality_tier}→{last.quality_tier})")
            if price_moved > 3:
                signals.append(f"price concession (${first.price:.0f}→${last.price:.0f})")
            if price_moved < -3:
                signals.append(f"price increase (${first.price:.0f}→${last.price:.0f})")

            if signals:
                lines.append(f"    Revealed: {', '.join(signals)}")
            else:
                lines.append(f"    Revealed: Held firm across rounds.")

        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Event plan: pre-draw contracts and type assignments
# =============================================================================

# Each event: (contract_scenario, type_for_A, type_for_B)
EventPlan = list[tuple[ContractScenario, str, str]]

# The probability that supplier B's cost type is the opposite of supplier A's, which pre_draw_event_sequence uses
# unless it is given another. The primitives prompt set states it ("3 in 4"), so that set refuses a plan drawn with
# any other (the command has no option for it).
P_DIFFERENT = 0.75


class DrawnPlan(list):
    """An event plan as pre_draw_event_sequence returns it: the list of (contract, type for A, type for B), one for
    each event, which also records the arguments of its draw. Under the primitives prompt set run_sequence reads
    `p_different` from it, because that set's texts state the probability. A plan made another way, a plain list,
    has no such record and is taken as drawn with P_DIFFERENT."""

    def __init__(self, events, *, seed: int, seq_id: int, p_different: float, environment: str):
        super().__init__(events)
        self.seed, self.seq_id, self.p_different, self.environment = seed, seq_id, p_different, environment


def _require_stated_p_different(p_different: float, whose: str) -> None:
    """The primitives texts say that the other supplier's cost type is the opposite of one's own with probability 3
    in 4, which is P_DIFFERENT; `whose` p_different is the draw's or the plan's."""
    if p_different != P_DIFFERENT:
        raise ValueError("the primitives set states that the other supplier's cost type is the opposite of yours "
                         f"with probability 3 in 4, so {whose} p_different must be {P_DIFFERENT}, not {p_different}")


def pre_draw_event_sequence(
    n_events: int,
    scenarios: list[ContractScenario],
    seed: int = 42,
    seq_id: int = 1,
    p_different: float = P_DIFFERENT,
    environment: str = "random",
    prompt_set: str = "advice",
) -> DrawnPlan:
    """Draw each event's contract and the two suppliers' cost types.

    For each event, the contract is drawn uniformly from `scenarios` and supplier A's type is
    "speed" or "quality" with equal probability. Supplier B gets the other type with
    probability `p_different` and A's type otherwise. Each supplier is therefore a speed
    specialist with probability 1/2, the two types differ in a share `p_different` of events
    (the two type indicators have correlation 1 - 2 * p_different, -0.5 by default), and
    events are drawn independently. The plan depends only on seed + 1000 * seq_id. Pass the
    same plan to each institution so that all of them face the same contracts and costs.

    `environment` is "random", the plan above, or "persistent": the plan is drawn as in the random
    environment and its first event (the contract and both types) is repeated at every event. For a
    given seed and the same other arguments, event 1 is the same in the two environments, and the
    random plan does not depend on this argument.

    `prompt_set` is the prompt set the plan is drawn for. The primitives set states the probabilities of the draw
    (1 in 2, and 3 in 4 for `p_different`), so it refuses any `p_different` but P_DIFFERENT; the advice set states
    none. The plan returned is a DrawnPlan, a list that also records the arguments of the draw, so that
    run_sequence can refuse a plan drawn with another `p_different` under the primitives set.
    """
    if environment not in ENVIRONMENTS:
        raise ValueError(f"the environment is one of {', '.join(ENVIRONMENTS)}, not {environment!r}")
    if prompt_set not in PROMPT_SETS:
        raise ValueError(f"the prompt set is one of {', '.join(PROMPT_SETS)}, not {prompt_set!r}")
    if prompt_set == "primitives":
        _require_stated_p_different(p_different, "the draw's")
    rng = random.Random(seed + seq_id * 1000)
    type_choices = ["speed", "quality"]
    opposite = {"speed": "quality", "quality": "speed"}
    plan = []
    for _ in range(n_events):
        sc = rng.choice(scenarios)
        type_a = rng.choice(type_choices)
        # Negatively correlated: B gets opposite type with P=p_different
        if rng.random() < p_different:
            type_b = opposite[type_a]
        else:
            type_b = type_a
        plan.append((sc, type_a, type_b))
    if environment == "persistent":
        plan = plan[:1] * n_events
    return DrawnPlan(plan, seed=seed, seq_id=seq_id, p_different=p_different, environment=environment)


def compute_score_margin(buyer: BuyerPreferences, final_bids: dict) -> float:
    """Compute the percentage score margin between winner and loser."""
    scores = {}
    for sid, bid in final_bids.items():
        scores[sid] = buyer.score(bid.terms)
    if len(scores) < 2:
        return 0.0
    sorted_scores = sorted(scores.values(), reverse=True)
    winner_score = sorted_scores[0]
    loser_score = sorted_scores[1]
    if winner_score <= 0:
        return 0.0
    return (winner_score - loser_score) / winner_score * 100


# =============================================================================
# Repeated events
# =============================================================================

def run_sequence(
    regime_name: str,
    n_events: int,
    n_rounds: int,
    supplier_agents: dict[str, SupplierAgent],
    event_plan: EventPlan,
    mediator: PlatformMediator | None = None,
    buyer_agent_platform: BuyerAgentPlatform | None = None,
    analysts: dict[str, ProcurementAnalyst] | None = None,
    seq_id: int = 1,
    max_history_events: int = 0,
    organization: str | dict[str, str] = "organized",
    on_event: Callable[[dict], None] | None = None,
    prompt_set: str = "advice",
) -> list[dict]:
    """Run `n_events` procurement events in order under one institution.

    regime_name is "Mediator" (pass `mediator`) or "BuyerAgent" (pass `buyer_agent_platform`), the
    AI institutions, or "SecondPrice", the second-score sealed-bid auction (below). Event k uses
    event_plan[k - 1]: the contract and each supplier's cost type. Each event of an AI institution
    has `n_rounds` >= 2 rounds, because both institutions' rules promise feedback between rounds.

    "SecondPrice" is a one-time event: it needs n_events == 1, organization "single" and no analysts.
    Its rules tell the bidder that it is a one-time event, which is true only when no record of
    earlier events exists, and with one event there is no record for an analyst to read. `n_rounds`
    is not used (the auction has one sealed bid). The bidder is given no record, no first-event
    note and no feedback: run_second_price takes none.

    Before each event the runner writes each supplier's record of its past events
    (build_supplier_history; build_primitives_record in the primitives set); `max_history_events` > 0
    keeps only the last N events in it, and in the mediator's memory of each supplier
    (build_mediator_cross_event_history) too.
    Who reads the record depends on the supplier's organization. `organization` is one name for both
    suppliers (a homogeneous market) or a mapping from each supplier id to its name (a mixed market,
    in which the two suppliers differ); each supplier is treated by its own, and nothing that a
    supplier, an analyst or a platform is sent depends on the other supplier's organization:
      "single":    the bidder reads it. The supplier has no analyst.
      "organized", "organized_guided" (ANALYST_ORGANIZATIONS): the supplier's analyst reads it and
                   writes a briefing; the bidder reads only the briefing. `analysts` maps each
                   such supplier, and only such a supplier, to its analyst, who is first called
                   before the second event. The two differ only in the bidder's system prompt
                   (orgsim.prompts), which this function does not see.
    Before the first event every bidder of an AI institution reads FIRST_EVENT_NOTE. No agent of the supplier
    is told the upcoming contract's name; the record names past contracts.

    Every supplier bids in every event.

    The mediator receives the buyer's scoring function (description_for_platform_detailed,
    or the ordinal description if the scenario has none) and its own record of each
    supplier's past events; the buyer agent receives the ordinal description
    (description_for_platform) and no record. The buyer walks away, awarding nothing, when every
    final bid is priced above its value for the terms, as both institutions' rules say; the
    event's surpluses are then 0 and the record tells each supplier no contract was awarded.

    seq_id labels the printout. Returns one dict per event with the plan's draws, the outcome,
    the welfare measures, and the analysts' briefings and postures (analyst_briefings and
    analyst_postures). A model analyst's posture is read from its briefing text by a keyword rule
    (LocalProcurementAnalyst._classify_posture) that does not understand a negation, so check
    analyst_postures against the stored briefings before relying on it.

    `on_event`, if given, is called with each event's result dict (the one in the returned list)
    right after the event ends, before the next event starts. When a later event raises there is
    no list to return, so a caller that wants the finished events keeps them from this call. An
    exception raised by `on_event` stops the sequence.

    `prompt_set` is "advice", which all of the above describes, or "primitives", the prompt set of
    orgsim/primitives_prompts.py, for the Mediator and the BuyerAgent only. In the primitives set:
      - the organizations are orgsim.organization.ALL_PRIMITIVES_ORGANIZATIONS. Only the suppliers of
        ALL_PRIMITIVES_ANALYST_ORGANIZATIONS ("organized", "brief", "brief_cost" and their revenue variants) have an
        analyst (a model); the analyst is sent primitives_prompts.analyst_message and its reply is the briefing, with
        no posture (the posture is None). The analyst of "brief_cost" is also sent the cost type of the coming event
        and the bidder's cost structure in the third person, and the bidder of "brief_cost" reads a title of its cost
        structure that says its analyst is shown it, from the second event on. The suppliers of "organized_pass" and
        "forward" (and "forward_revenue") have no analyst: from the second event on the code puts the record into the
        bidder's message, and that record is what analyst_briefings holds for it. The supplier of "single_norecord" is
        never shown a past event, though its record is written (the platforms' memory does not depend on it).
      - the record is build_primitives_record's: the facts of build_supplier_history said of "your company" in
        the third person, so that it is true of the analyst who reads it as well as of the bidder, with the
        platform's answers called "Platform's reply" as the rules call them.
      - floor: the analyst may end its briefing with a minimum profit per contract for the event
        (orgsim.departures.floor_briefing: asked again once if the last line is not a floor line, and then no floor is
        set), the bidder reads the briefing without its floor lines (orgsim.departures.strip_floor_lines) and then a
        notice (the minimum, none, or that none is set because the analyst gave no valid line), and the runner tells
        the bidder's agent before the event (SupplierAgent.set_minimum_profit), which returns a bid priced below its
        total cost plus the minimum once and then raises the price (PrimitivesSupplierAgent).
        The result of the event holds "departures" for each supplier of floor: the briefings as the analyst wrote them,
        whether it was asked again, the status of the line, the minimum, and how many of the supplier's bids were
        returned and raised; each of its bids holds floor_audit.
      - brief_cap: the analyst's briefing has at most BRIEFING_WORD_LIMIT words (orgsim.departures.capped_briefing): an
        analyst whose briefing is longer is asked once more, in the same conversation, and a second briefing that is
        still too long is cut. The result of the event then holds "departures": for each supplier of brief_cap, the
        original briefing, whether the analyst was asked again, the second briefing, whether it was cut and the final
        word count. A run with no supplier of brief_cap has no such key.
      - each bidder's context, which run_mediator and run_buyer_agent put before the rules, is the event
        line, the costs of the supplier's type in this event and what the bidder knows of past events
        (build_primitives_bidder_context), and the rules are primitives_rules(regime_name, n_rounds).
      - no price rule is applied: a bid stands as submitted (the bidders of the set, PrimitivesSupplierAgent,
        send no correction), and the bidders' round line names the number of rounds, which must be n_rounds.
      - at least 2 events, and no max_history_events: the texts say that the record is full and that the
        platform remembers past events. The system prompts (primitives_prompts) name the number of events,
        and the caller builds them for n_events.
      - a plan drawn with P_DIFFERENT, because the texts state the probabilities of the draw: a DrawnPlan that
        records another `p_different` is refused, and a plain list, which records none, is taken as drawn with it.
    """
    platforms = {"Mediator": mediator, "BuyerAgent": buyer_agent_platform}
    sealed_bid = regime_name == "SecondPrice"
    if regime_name not in platforms and not sealed_bid:
        raise ValueError(f"unknown institution: {regime_name!r}")
    if not sealed_bid and platforms[regime_name] is None:
        raise ValueError(f"the {regime_name} institution needs its platform agent")
    if len(event_plan) < n_events:
        raise ValueError(f"the event plan has {len(event_plan)} events, fewer than {n_events}")
    organizations = organization_of_each(organization, SUPPLIER_IDS)
    if prompt_set not in PROMPT_SETS:
        raise ValueError(f"the prompt set is one of {', '.join(PROMPT_SETS)}, not {prompt_set!r}")
    primitives = prompt_set == "primitives"
    if primitives:
        if sealed_bid:
            raise ValueError("the primitives set is for the Mediator and the BuyerAgent: the second-price auction's "
                             "bidder has a prompt of its own")
        unknown = sorted({name for name in organizations.values() if name not in ALL_PRIMITIVES_ORGANIZATIONS})
        if unknown:
            raise ValueError(f"unknown organization of the primitives set: {unknown[0]!r}")
        _require_stated_p_different(getattr(event_plan, "p_different", P_DIFFERENT), "the plan's")
        if n_events < 2:
            raise ValueError("the primitives set is for a sequence of at least 2 events")
        if max_history_events:
            raise ValueError("the primitives set takes no max_history_events: its texts say that the record is full "
                             "and that the platform remembers past events")
        for sid, agent in supplier_agents.items():  # a bidder that names the rounds must name these
            if getattr(agent, "n_rounds", n_rounds) != n_rounds:
                raise ValueError(f"the bidder of supplier {sid} names {agent.n_rounds} rounds in an event, "
                                 f"not {n_rounds}")
    if sealed_bid:
        if n_events != 1:
            raise ValueError("the second-price auction is a one-time event, so it needs n_events == 1")
        if any(name != "single" for name in organizations.values()):
            raise ValueError("the second-price auction has no record for an analyst, so it needs organization 'single'")
    elif n_rounds < 2:
        raise ValueError("the AI institutions need at least 2 rounds")
    # The analysts are exactly those of the suppliers whose organization has one.
    analyst_organizations = ALL_PRIMITIVES_ANALYST_ORGANIZATIONS if primitives else ANALYST_ORGANIZATIONS
    needs_analyst = {sid for sid in SUPPLIER_IDS if organizations[sid] in analyst_organizations}
    given = set(analysts or ())
    if given & (set(SUPPLIER_IDS) - needs_analyst):
        raise ValueError("a supplier without an analyst model has no analysts" if primitives
                         else "a single-agent firm has no analysts")
    if given - set(SUPPLIER_IDS):
        raise ValueError("an analyst is for a supplier of the market (A or B)")
    if needs_analyst - given:
        raise ValueError("an organized firm needs one analyst per supplier")
    rules = primitives_rules(regime_name, n_rounds) if primitives else None
    # The runs in which code does something to an analyst's briefing or to a bid (brief_cap, floor) record it in
    # "departures".
    tracked = primitives and any(departure_base(organizations[sid]) in ("brief_cap", "floor")
                                 for sid in SUPPLIER_IDS)
    supplier_histories: dict[str, list[dict]] = {sid: [] for sid in SUPPLIER_IDS}
    mediator_histories: dict[str, list[dict]] = {sid: [] for sid in SUPPLIER_IDS}
    event_results = []

    for event_idx in range(1, n_events + 1):
        # Use the pre-drawn plan (same across all institutions)
        sc, type_a, type_b = event_plan[event_idx - 1]
        type_map = {"A": type_a, "B": type_b}

        # Create per-event SupplierCosts based on drawn types
        event_suppliers = [
            make_supplier("A", type_a),
            make_supplier("B", type_b),
        ]

        same_type = "SAME" if type_a == type_b else "DIFF"
        print(f"\n  {'─' * 70}")
        print(f"  [{regime_name}] Event {event_idx}/{n_events} (seq {seq_id})")
        print(f"    Contract: {sc.name} | A={type_a}, B={type_b} ({same_type})")
        print(f"  {'─' * 70}")

        eff_surplus, eff_winner, _ = compute_max_efficient_surplus(
            sc.buyer, event_suppliers
        )
        costs_of = {supplier.supplier_id: supplier for supplier in event_suppliers}

        # Build cross-event history for each supplier
        cross_event_history = {}
        for sid in SUPPLIER_IDS:
            if primitives:
                hist_str = build_primitives_record(sid, supplier_histories[sid])
            else:
                hist_str = build_supplier_history(
                    sid, supplier_histories[sid],
                    max_events=max_history_events,
                )
            if hist_str:
                cross_event_history[sid] = hist_str

        # Generate analyst briefings (if analysts provided and history exists)
        # These are AnalystBriefing objects with structured .posture + .text
        analyst_briefings: dict[str, AnalystBriefing] = {}
        departures: dict[str, dict] = {}  # primitives set: what code did for each supplier of brief_cap and floor
        # primitives set: the minimum profit per contract that an analyst set, by supplier
        floors: dict[str, float] = {}
        forwarded: set[str] = set()  # primitives set: the suppliers whose analyst passes the record on unchanged
        if primitives:
            for sid in SUPPLIER_IDS:
                record = cross_event_history.get(sid)
                if record is None:
                    continue  # the first event: nothing to brief
                if sid in (analysts or {}):
                    # The analyst of brief_cost is also shown the bidder's costs for the coming event.
                    costs = ((type_map[sid], costs_of[sid].describe_for_analyst())
                             if departure_base(organizations[sid]) == "brief_cost" else None)
                    message = analyst_message(event_idx, n_events, rules, record, costs)
                    if departure_base(organizations[sid]) == "brief_cap":
                        briefing, mechanics = capped_briefing(analysts[sid], sid, message)
                        departures[sid] = {"organization": organizations[sid], **mechanics}
                    elif departure_base(organizations[sid]) == "floor":
                        briefing, mechanics, minimum = floor_briefing(analysts[sid], sid, message)
                        departures[sid] = {"organization": organizations[sid], **mechanics}
                        if minimum is not None:
                            floors[sid] = minimum
                    else:
                        briefing = analysts[sid].write_briefing(sid, message)
                    analyst_briefings[sid] = briefing
                    print(f"    Analyst ({sid}): {briefing.text[:60]}...")
                elif forwards_the_record(organizations[sid]):
                    analyst_briefings[sid] = AnalystBriefing(None, record)
                    forwarded.add(sid)
                    print(f"    Analyst ({sid}): forwards the record unchanged")
        elif analysts:
            for sid in SUPPLIER_IDS:
                if sid in cross_event_history and sid in analysts:
                    briefing = analysts[sid].generate_briefing(
                        supplier_id=sid,
                        history_text=cross_event_history[sid],
                    )
                    analyst_briefings[sid] = briefing
                    print(f"    Analyst ({sid}): [{briefing.posture}] {briefing.text[:60]}...")

        # What each bidder sees about past events depends on its own supplier's organization: the single firm's
        # bidder reads the raw history; the bidder of an organized firm reads only the analyst's memo. Neither
        # bidder nor analyst is told which contract is up; suppliers learn a past contract's name from their record.
        if primitives:
            # The context of each bidder is what it reads before the rules: the event line, its costs in this event
            # and what it knows of past events (nothing, for the firm that keeps no record, from the second event on).
            for sid in SUPPLIER_IDS:
                briefing = analyst_briefings.get(sid)
                past_events = build_primitives_bidder_context(
                    organizations[sid],
                    cross_event_history.get(sid, ""),
                    briefing.text if briefing and sid not in forwarded else None,
                )
                # From the second event on, the bidder of brief_cost has an analyst who is shown its cost structure,
                # and its title says so; in the first event the analyst writes nothing and the title is the usual one.
                header = (SHARED_COST_HEADER if event_idx >= 2 and departure_base(organizations[sid]) == "brief_cost"
                          else PRIVATE_COST_HEADER)
                cross_event_history[sid] = event_preamble(
                    event_idx, n_events, costs_of[sid].describe_for_primitives(type_map[sid], header), past_events)
        else:
            for sid in SUPPLIER_IDS:
                briefing = analyst_briefings.get(sid)
                context = build_bidder_context(
                    organizations[sid],
                    cross_event_history.get(sid, ""),
                    briefing.text if briefing else None,
                )
                cross_event_history[sid] = context

        # Build mediator cross-event history (Mediator only)
        mediator_cross_event_history = {}
        if regime_name == "Mediator":
            for sid in SUPPLIER_IDS:
                med_hist_str = build_mediator_cross_event_history(
                    sid, mediator_histories[sid],
                    max_events=max_history_events,
                )
                if med_hist_str:
                    mediator_cross_event_history[sid] = med_hist_str

        # The bidders of the floor organizations are told the minimum profit of the event (none, before the second
        # event).
        floor_suppliers = [sid for sid in SUPPLIER_IDS if departure_base(organizations[sid]) == "floor"]
        for sid in floor_suppliers:
            supplier_agents[sid].set_minimum_profit(floors.get(sid))

        # ── Run the appropriate institution ──
        if regime_name == "Mediator":
            mediator_buyer_desc = (
                sc.description_for_platform_detailed
                if sc.description_for_platform_detailed
                else sc.description_for_platform
            )
            outcome = run_mediator(
                run_id=event_idx,
                buyer=sc.buyer,
                suppliers=event_suppliers,
                agents=supplier_agents,
                mediator=mediator,
                efficient_surplus=eff_surplus,
                efficient_winner_id=eff_winner,
                num_rounds=n_rounds,
                buyer_description=mediator_buyer_desc,
                cross_event_history=cross_event_history,
                mediator_cross_event_history=mediator_cross_event_history or None,
                rules=rules,
            )
        elif regime_name == "BuyerAgent":
            outcome = run_buyer_agent(
                run_id=event_idx,
                buyer=sc.buyer,
                suppliers=event_suppliers,
                agents=supplier_agents,
                buyer_agent=buyer_agent_platform,
                efficient_surplus=eff_surplus,
                efficient_winner_id=eff_winner,
                num_rounds=n_rounds,
                buyer_description=sc.description_for_platform,
                cross_event_history=cross_event_history,
                rules=rules,
            )
        elif sealed_bid:
            # No record, first-event note or feedback reaches the bidder: run_second_price takes none
            # (the bidder context built above is not used).
            outcome = run_second_price(
                run_id=event_idx,
                buyer=sc.buyer,
                suppliers=event_suppliers,
                agents=supplier_agents,
                efficient_surplus=eff_surplus,
                efficient_winner_id=eff_winner,
            )
        else:
            raise ValueError(f"Unknown regime: {regime_name}")
        for sid in floor_suppliers:
            supplier_agents[sid].set_minimum_profit(None)
            if sid in departures:  # how many of the supplier's bids the floor returned, and how many it raised
                audits = [bid.floor_audit for bid in outcome.all_bids[sid] if bid.floor_audit]
                departures[sid]["bids_returned"] = sum(audit["reasked"] for audit in audits)
                departures[sid]["bids_raised"] = sum(audit["raised"] for audit in audits)

        # Compute score margin
        final_bids = {
            sid: bids[-1]
            for sid, bids in outcome.all_bids.items()
        }
        score_margin = compute_score_margin(sc.buyer, final_bids)

        # Update histories for both suppliers
        for sid in SUPPLIER_IDS:
            feedback_for_supplier = outcome.mediator_feedback.get(sid, [])

            entry = {
                "scenario_name": sc.name,
                "outcome": outcome,
                "feedback_history": feedback_for_supplier,
                "own_cost_type": type_map[sid],
            }
            supplier_histories[sid].append(entry)
            mediator_histories[sid].append(entry)

        winner_text = (f"{outcome.winner_id} ({'✓' if outcome.correct_winner else '✗'})"
                       if outcome.awarded else "none (buyer walked away)")
        print(f"    Winner: {winner_text} "
              f"eff={outcome.efficiency_ratio:.1%} "
              f"buyer=${outcome.buyer_surplus:.1f} "
              f"supplier=${outcome.supplier_profit:.1f} "
              f"margin={score_margin:.1f}%")

        event_result = {
            "event": event_idx,
            "contract": sc.name,
            "type_a": type_a,
            "type_b": type_b,
            "same_type": type_a == type_b,
            "winner": outcome.winner_id,
            "awarded": outcome.awarded,
            "correct_winner": outcome.correct_winner,
            "efficiency": outcome.efficiency_ratio,
            "buyer_surplus": outcome.buyer_surplus,
            "supplier_profit": outcome.supplier_profit,
            "joint_surplus": outcome.joint_surplus,
            "score_margin_pct": score_margin,
            "outcome": outcome,
            "analyst_briefings": {sid: b.text for sid, b in analyst_briefings.items()},
            "analyst_postures": {sid: b.posture for sid, b in analyst_briefings.items()},
        }
        if tracked:
            event_result["departures"] = departures
        event_results.append(event_result)
        if on_event is not None:
            on_event(event_result)

    return event_results
