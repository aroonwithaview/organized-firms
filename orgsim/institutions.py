"""Institutions: first-price RFQ, second-price, AI mediator, AI buyer agent."""
from orgsim.primitives import BuyerPreferences, ContractTerms, RunOutcome, SupplierBid, SupplierCosts
from orgsim.interfaces import FLOOR_PRICE_TAG, RULE_PRICE_TAG, BuyerAgentPlatform, PlatformMediator, SupplierAgent


def _true_costs_and_inflation(
    suppliers: list[SupplierCosts],
    final_bids: dict[str, SupplierBid],
) -> tuple[dict[str, float], dict[str, float]]:
    """Each bidder's true cost at its final terms, and its price's markup over that cost."""
    all_true_costs = {}
    all_inflation = {}
    for s in suppliers:
        if s.supplier_id in final_bids:
            bid_terms = final_bids[s.supplier_id].terms
            tc = s.cost(bid_terms)
            all_true_costs[s.supplier_id] = tc
            if tc > 0:
                all_inflation[s.supplier_id] = (bid_terms.price - tc) / tc
            else:
                all_inflation[s.supplier_id] = 0.0
    return all_true_costs, all_inflation


def _compute_outcome(
    regime: str,
    run_id: int,
    buyer: BuyerPreferences,
    suppliers: list[SupplierCosts],
    final_bids: dict[str, SupplierBid],
    all_round_bids: dict[str, list[SupplierBid]],
    efficient_surplus: float,
    efficient_winner_id: str,
    use_vickrey_pricing: bool = False,
    buyer_walks_away: bool = False,
) -> RunOutcome:
    """Select the winner and compute the outcome's metrics.

    The highest score wins. With buyer_walks_away (the mediator and the buyer agent, whose
    rules announce it), the buyer awards no contract when every final bid is priced above
    its value for the terms (every score negative); the outcome then has no winner and zero
    surpluses. A score of exactly 0 is accepted. The sealed-bid institutions announce no
    walk-away and always award. Ties go to the first supplier in `final_bids` order, which
    every institution builds in `suppliers` order. Under second-price pricing, winning_bid,
    buyer_surplus and supplier_profit use the price paid; all_bids and all_inflation_rates
    use the submitted price; all_true_costs is the cost of the submitted delivery and
    quality, which does not depend on price.
    """

    # Score all final bids
    scores = {}
    for sid, bid in final_bids.items():
        scores[sid] = buyer.score(bid.terms)

    # The AI institutions' rules tell suppliers the buyer walks away from a winning bid priced above
    # its value for the terms; that is every final bid when the best score is negative.
    if buyer_walks_away and max(scores.values()) < 0:
        all_true_costs, all_inflation = _true_costs_and_inflation(suppliers, final_bids)
        return RunOutcome(
            regime=regime, run_id=run_id, winner_id=None, winning_bid=None, winner_true_cost=0.0,
            buyer_surplus=0.0, supplier_profit=0.0, joint_surplus=0.0,
            efficient_surplus=efficient_surplus, efficient_winner_id=efficient_winner_id,
            all_bids={sid: bids for sid, bids in all_round_bids.items()},
            all_true_costs=all_true_costs, all_inflation_rates=all_inflation,
        )

    # Select winner (highest score)
    winner_id = max(scores, key=scores.get)
    winning_bid = final_bids[winner_id]

    # Vickrey pricing: adjust winner's price up so their score = second-best
    if use_vickrey_pricing and len(scores) > 1:
        sorted_scores = sorted(scores.values(), reverse=True)
        second_best = sorted_scores[1]
        # winner_score = v(q) - p_orig. We want v(q) - p_new = second_best
        # p_new = v(q) - second_best
        vickrey_price = buyer.value(winning_bid.terms) - second_best
        winning_bid = SupplierBid(
            supplier_id=winning_bid.supplier_id,
            terms=ContractTerms(
                price=vickrey_price,
                delivery_weeks=winning_bid.terms.delivery_weeks,
                quality_tier=winning_bid.terms.quality_tier,
            ),
            reasoning=winning_bid.reasoning + f" [Vickrey price: ${vickrey_price:.2f}]",
            round_num=winning_bid.round_num,
        )

    # Find winner's true cost
    winner_costs = next(s for s in suppliers if s.supplier_id == winner_id)
    winner_true_cost = winner_costs.cost(winning_bid.terms)

    # Compute surpluses
    buyer_surplus = buyer.score(winning_bid.terms)
    supplier_profit = winning_bid.terms.price - winner_true_cost
    joint_surplus = buyer.value(winning_bid.terms) - winner_true_cost

    # Compute inflation rates for all suppliers
    all_true_costs, all_inflation = _true_costs_and_inflation(suppliers, final_bids)

    return RunOutcome(
        regime=regime,
        run_id=run_id,
        winner_id=winner_id,
        winning_bid=winning_bid.terms,
        winner_true_cost=winner_true_cost,
        buyer_surplus=buyer_surplus,
        supplier_profit=supplier_profit,
        joint_surplus=joint_surplus,
        efficient_surplus=efficient_surplus,
        efficient_winner_id=efficient_winner_id,
        all_bids={sid: bids for sid, bids in all_round_bids.items()},
        all_true_costs=all_true_costs,
        all_inflation_rates=all_inflation,
    )


# =============================================================================
# First-price RFQ
# =============================================================================

def run_first_price(
    run_id: int,
    buyer: BuyerPreferences,
    suppliers: list[SupplierCosts],
    agents: dict[str, SupplierAgent],
    efficient_surplus: float,
    efficient_winner_id: str,
) -> RunOutcome:
    """First-price sealed bid.

    Each supplier submits one bid on price, delivery and quality; the highest buyer score
    (value minus price) wins and is paid its bid. Suppliers are told there are two bidders and
    that all terms are scored, but not the scoring weights. Expects exactly two suppliers.

    The bidder prompts in orgsim/prompts.py are written for the AI institutions; parts of
    them do not apply to a sealed bid.
    """
    if len(suppliers) != 2:
        raise ValueError("the sealed-bid rules tell suppliers there are two bidders")

    regime_context = (
        "RULES (first-price auction): You submit one bid. Both suppliers "
        "bid at the same time. The buyer scores each bid based on all terms "
        "(price, delivery, quality). The highest-scoring bid wins. "
        "The winner is paid the price they bid. There are no further rounds.\n\n"
        "This is a one-time procurement event. Your bid does not affect "
        "any future interactions."
    )

    final_bids = {}
    all_round_bids = {}

    for s in suppliers:
        agent = agents[s.supplier_id]
        bid = agent.generate_bid(s, regime_context, round_num=1, history=[])
        final_bids[s.supplier_id] = bid
        all_round_bids[s.supplier_id] = [bid]

    return _compute_outcome(
        regime="FirstPrice",
        run_id=run_id,
        buyer=buyer,
        suppliers=suppliers,
        final_bids=final_bids,
        all_round_bids=all_round_bids,
        efficient_surplus=efficient_surplus,
        efficient_winner_id=efficient_winner_id,
    )


# =============================================================================
# Second-price (Vickrey) multi-attribute auction
# =============================================================================

def run_second_price(
    run_id: int,
    buyer: BuyerPreferences,
    suppliers: list[SupplierCosts],
    agents: dict[str, SupplierAgent],
    efficient_surplus: float,
    efficient_winner_id: str,
) -> RunOutcome:
    """Second-score sealed bid (multi-attribute Vickrey).

    Each supplier submits one bid; the highest score wins. The winner keeps its delivery and
    quality and is paid the price at which its score equals the second-highest score. Suppliers
    are told this pricing rule but not the buyer's scoring weights. Expects exactly two suppliers.

    The bidder prompts of the AI institutions in orgsim/prompts.py do not apply to a sealed bid.
    The demo gives this institution's bidders sealed_bid_prompt() and a LocalSupplierAgent with
    sealed_bid=True, whose other texts state the primitives and the rules only. The rules below
    are part of what they read, so each statement is true of this code: the winner is paid the
    highest price at which its score is still at least the other bid's score (the price at which
    the two are equal), which is at least the price it bid and equals it when the two scores are
    already equal. The text calls the event a one-time event, which is true only when no record
    of earlier events exists, so the demo runs this institution for one event.
    """
    if len(suppliers) != 2:
        raise ValueError("the sealed-bid rules tell suppliers there are two bidders")

    regime_context = (
        "RULES (second-price auction): You submit one bid. Both suppliers "
        "bid at the same time. The buyer scores each bid based on all terms "
        "(price, delivery, quality). The highest-scoring bid wins.\n\n"
        "PRICING: If you win, you are paid the highest price at which your "
        "terms would still score at least as high as the other supplier's "
        "bid. This price is at least the price you bid.\n\n"
        "There are no further rounds.\n\n"
        "This is a one-time procurement event. Your bid does not affect "
        "any future interactions."
    )

    final_bids = {}
    all_round_bids = {}

    for s in suppliers:
        agent = agents[s.supplier_id]
        bid = agent.generate_bid(s, regime_context, round_num=1, history=[])
        final_bids[s.supplier_id] = bid
        all_round_bids[s.supplier_id] = [bid]

    return _compute_outcome(
        regime="SecondPrice",
        run_id=run_id,
        buyer=buyer,
        suppliers=suppliers,
        final_bids=final_bids,
        all_round_bids=all_round_bids,
        efficient_surplus=efficient_surplus,
        efficient_winner_id=efficient_winner_id,
        use_vickrey_pricing=True,
    )


# =============================================================================
# Rules of the primitives prompt set
# =============================================================================

def _rounds_with_reply(num_rounds: int) -> str:
    """The rounds after which a bidder gets a reply from the platform, which are all but the last: "round 1",
    "rounds 1 and 2", "rounds 1 to 3", ..."""
    last = num_rounds - 1
    if last == 1:
        return "round 1"
    return "rounds 1 and 2" if last == 2 else f"rounds 1 to {last}"


# The sentence about what decides the award, which both institutions share, and the sentences about the platform, which
# differ between them. The award sentence does not say what the buyer sees: the buyer's agent acts for the buyer and
# sees the supplier's bids of the event, so "the buyer does not see earlier bids" could be read as untrue of it.
AWARD_TEXT = "Only final bids are used to decide the award; earlier bids and the replies are not."
MEDIATOR_PLATFORM_TEXT = (
    "The platform is instructed to work for both the buyer and the suppliers. It sees every supplier's bids in the "
    "event and the buyer's scoring, and it remembers past events on this platform.")
BUYER_AGENT_PLATFORM_TEXT = (
    "The platform is instructed to work on behalf of the buyer, to get the best deal for the buyer. It knows the "
    "buyer's priorities and sees your bids in this event. It does not see past events. When it replies to you, it "
    "does not see the other supplier's bids.")
PRIMITIVES_OUTSIDE_OPTION = (
    "BUYER'S OUTSIDE OPTION: The buyer is not obligated to award the contract: if both final bids are priced above "
    "the buyer's value for their terms, no contract is awarded.")
PLATFORM_TEXTS = {"Mediator": MEDIATOR_PLATFORM_TEXT, "BuyerAgent": BUYER_AGENT_PLATFORM_TEXT}


def primitives_rules(institution: str, num_rounds: int) -> str:
    """The rules that the bidder of the primitives prompt set reads in every round, for the institution "Mediator" or
    "BuyerAgent" with `num_rounds` rounds in an event. They state what the code does and give no advice. Each
    statement and the code that makes it true:

    - "Each event has N rounds." and "Your bid in round N is your final bid.": run_mediator and run_buyer_agent run
      `num_rounds` rounds, and the last round's bid is the final bid (final_bids = the last of all_round_bids).
    - "In each round you submit a bid": every supplier bids in every round.
    - "after rounds 1 to N-1 you receive a reply from the platform": the mediator's feedback and the buyer agent's
      counteroffer message are generated when round_num < num_rounds, for each supplier.
    - "Only final bids are used to decide the award; earlier bids and the replies are not.": _compute_outcome scores
      only the final bids. The buyer's agent sees the supplier's earlier bids of the event and both platforms write
      the replies, but _compute_outcome uses neither.
    - Mediator: "instructed to work for both the buyer and the suppliers" is the first line of MEDIATOR_PROMPT. "It
      sees every supplier's bids in the event and the buyer's scoring": generate_feedback gets all current bids, every
      supplier's earlier bids in the event and the scoring function (description_for_platform_detailed). "it remembers
      past events on this platform": mediator_cross_event_history holds every supplier's earlier events, which
      run_sequence builds before each event (in full: the primitives set takes no max_history_events).
    - Buyer agent: "instructed to work on behalf of the buyer, to get the best deal for the buyer" is the first
      sentence and the goal in BUYER_AGENT_PROMPT. "It knows the buyer's priorities": it gets the ordinal description
      of the buyer's priorities. "sees your bids in this event": it gets this supplier's current bid and its earlier
      bids in the event (not the other supplier's). "It does not see past events": run_buyer_agent starts each event
      with an empty history, passes the agent only the buyer's preferences, the ordinal description, this
      supplier's bid and its bids of the event so far, and run_sequence builds a memory for the Mediator only; the
      agent keeps no state between calls. "When it replies to you, it does not see the other supplier's bids": the
      call that writes a supplier's reply gets that supplier's bid and history only (the platform as a whole does
      see the other supplier's bids, when it replies to that supplier).
    - "A final bid's score is the buyer's value for its delivery and quality minus its price" is
      BuyerPreferences.score; "the higher-scoring final bid wins" is the winner selection in _compute_outcome, which
      gives a tie to the first supplier (the rules do not mention ties); "the winner is paid its price": these
      institutions pay the winner's final bid (no second-price pricing).
    - BUYER'S OUTSIDE OPTION: with buyer_walks_away, no contract is awarded when the highest score is below 0, that
      is when every final bid is priced above the buyer's value for its terms; the market has two suppliers, so
      "both final bids".
    """
    if num_rounds < 2:
        raise ValueError("the AI institutions' rules promise feedback between rounds, so they need at least 2 rounds")
    try:
        platform = PLATFORM_TEXTS[institution]
    except KeyError:
        raise ValueError(f"the primitives rules are for the Mediator or the BuyerAgent, not {institution!r}") from None
    return (
        f"RULES: Each event has {num_rounds} rounds. In each round you submit a bid; after "
        f"{_rounds_with_reply(num_rounds)} you receive a reply from the platform. Your bid in round {num_rounds} "
        f"is your final bid. {AWARD_TEXT} {platform} A final bid's score is the buyer's value for its delivery and "
        f"quality minus its price; the higher-scoring final bid wins, and the winner is paid its price.\n\n"
        + PRIMITIVES_OUTSIDE_OPTION
    )


# =============================================================================
# AI mediator (coaches suppliers; only final bids reach the buyer)
# =============================================================================

def run_mediator(
    run_id: int,
    buyer: BuyerPreferences,
    suppliers: list[SupplierCosts],
    agents: dict[str, SupplierAgent],
    mediator: PlatformMediator,
    efficient_surplus: float,
    efficient_winner_id: str,
    num_rounds: int = 5,
    buyer_description: str = "",
    cross_event_history: dict[str, str] | None = None,
    mediator_cross_event_history: dict[str, str] | None = None,
    rules: str | None = None,
) -> RunOutcome:
    """AI mediator: suppliers revise bids over several rounds with the mediator's feedback;
    only final bids reach the buyer, and the winner is paid its bid.

    The rules shown to suppliers describe the process and say nothing about whether the
    platform can be trusted. They also say the buyer walks away from a winning bid priced
    above its value, and the buyer does (see _compute_outcome). The mediator receives all
    current bids, every supplier's bids and its own feedback in earlier rounds of this
    event, and its memory of every supplier (mediator_cross_event_history), which comes before
    the buyer's description (buyer.describe_for_platform() when none is given). Within a round
    it answers the suppliers in turn, and every call in the round gets the same earlier
    feedback (a copy taken before the first call), so the reply to one supplier does not
    appear in the call for the next.

    `rules`, if given, is the text the suppliers read as the rules, in place of the one below (the primitives
    prompt set passes primitives_rules("Mediator", num_rounds)). Each supplier reads its entry of
    cross_event_history, if it has one, and then the rules.
    """
    if num_rounds < 2:
        raise ValueError("the AI institutions' rules promise feedback between rounds, "
                         "so they need at least 2 rounds")

    _buyer_outside_option = (
        "\n\nBUYER'S OUTSIDE OPTION: The buyer is NOT obligated to award this "
        "contract. If the winning bid's price exceeds the buyer's value for "
        "those terms, the buyer will walk away rather than accept a "
        "value-destroying deal. As a supplier, this means you want to "
        "capture as much of the deal's value as you can — pricing as high "
        "as the buyer will accept — while staying below the point where "
        "they would prefer no deal. The more competition you face, the "
        "less room you have to price up; with less competition, you have "
        "more room — but the buyer's walkaway point is always the ceiling."
    )

    # Regime context is the SAME regardless of participation count.
    # The platform does NOT disclose how many suppliers are bidding.
    # The supplier must infer competitive conditions from the platform's
    # feedback during the event and from their own historical patterns.
    regime_context = (
        "RULES: You submit a bid, then receive feedback from a platform "
        "facilitator. You can revise your bid over multiple rounds. "
        "Only your FINAL bid is shown to the buyer. The buyer does NOT "
        "see your earlier bids or any process information.\n\n"
        "The buyer scores final bids on all terms (price, delivery, quality). "
        "The highest-scoring final bid wins. The winner is paid "
        "the price they bid. The platform does not disclose how many "
        "suppliers are participating in this event."
        + _buyer_outside_option
    )
    if rules is not None:
        regime_context = rules

    # Inject per-supplier cross-event history if provided
    _supplier_contexts: dict[str, str] = {}
    for s in suppliers:
        ctx = regime_context
        if cross_event_history and s.supplier_id in cross_event_history:
            ctx = cross_event_history[s.supplier_id] + "\n\n" + ctx
        _supplier_contexts[s.supplier_id] = ctx

    all_round_bids: dict[str, list[SupplierBid]] = {s.supplier_id: [] for s in suppliers}
    histories: dict[str, list[dict]] = {s.supplier_id: [] for s in suppliers}
    all_feedback: dict[str, list[str]] = {s.supplier_id: [] for s in suppliers}

    # Track all bids and feedback for mediator memory (within-event)
    previous_bids: dict[str, list[ContractTerms]] = {s.supplier_id: [] for s in suppliers}
    previous_feedback_history: dict[str, list[str]] = {s.supplier_id: [] for s in suppliers}

    current_bids: dict[str, ContractTerms] = {}

    for round_num in range(1, num_rounds + 1):
        for s in suppliers:
            agent = agents[s.supplier_id]
            bid = agent.generate_bid(
                s, _supplier_contexts[s.supplier_id], round_num=round_num,
                history=histories[s.supplier_id],
            )
            all_round_bids[s.supplier_id].append(bid)
            current_bids[s.supplier_id] = bid.terms

        # Generate feedback (except after last round)
        if round_num < num_rounds:
            for s in suppliers:
                previous_bids[s.supplier_id].append(current_bids[s.supplier_id])

            # Every call in this round gets the feedback of earlier rounds only. A copy keeps the
            # reply written to one supplier out of the call for the next one.
            earlier_feedback = {sid: list(texts) for sid, texts in previous_feedback_history.items()}

            for s in suppliers:
                # Build mediator context: the memory of every supplier + buyer description
                mediator_desc = buyer_description
                if mediator_cross_event_history:
                    memories = [mediator_cross_event_history[sid]
                                for sid in sorted(mediator_cross_event_history)]
                    mediator_desc = ("\n\n".join(memories) + "\n\n"
                                     + (buyer_description or buyer.describe_for_platform()))

                feedback = mediator.generate_feedback(
                    buyer, current_bids, s.supplier_id, round_num,
                    buyer_description=mediator_desc,
                    previous_bids=previous_bids,
                    previous_feedback=earlier_feedback,
                )
                all_feedback[s.supplier_id].append(feedback)
                previous_feedback_history[s.supplier_id].append(feedback)
                histories[s.supplier_id].append({
                    "round": round_num,
                    "bid": current_bids[s.supplier_id],
                    "feedback": feedback,
                    "price_set_by_rule": RULE_PRICE_TAG in all_round_bids[s.supplier_id][-1].reasoning,
                    "price_raised_by_floor": FLOOR_PRICE_TAG in all_round_bids[s.supplier_id][-1].reasoning,
                })

    # Final bids = last round
    final_bids = {
        sid: bids[-1] for sid, bids in all_round_bids.items()
    }

    outcome = _compute_outcome(
        regime="Mediator",
        run_id=run_id,
        buyer=buyer,
        suppliers=suppliers,
        final_bids=final_bids,
        all_round_bids=all_round_bids,
        efficient_surplus=efficient_surplus,
        efficient_winner_id=efficient_winner_id,
        buyer_walks_away=True,
    )
    outcome.mediator_feedback = all_feedback
    return outcome


# =============================================================================
# AI buyer agent (negotiates for the buyer)
# =============================================================================

def run_buyer_agent(
    run_id: int,
    buyer: BuyerPreferences,
    suppliers: list[SupplierCosts],
    agents: dict[str, SupplierAgent],
    buyer_agent: BuyerAgentPlatform,
    efficient_surplus: float,
    efficient_winner_id: str,
    num_rounds: int = 5,
    buyer_description: str = "",
    cross_event_history: dict[str, str] | None = None,
    rules: str | None = None,
) -> RunOutcome:
    """AI buyer agent: suppliers revise bids over several rounds with feedback from an agent
    that negotiates for the buyer; the winner is paid its bid.

    The rules shown to suppliers say that the agent works for the buyer and sees all their
    earlier bids, but not how it negotiates. They also say the buyer walks away from a winning
    bid priced above its value, and the buyer does (see _compute_outcome). Besides
    the buyer's preferences and description, the agent receives only this supplier's bids and
    its own replies in the current event, not the other supplier's bids and not past events.

    `rules`, if given, is the text the suppliers read as the rules, in place of the one below (the primitives
    prompt set passes primitives_rules("BuyerAgent", num_rounds)). Each supplier reads its entry of
    cross_event_history, if it has one, and then the rules.
    """
    if num_rounds < 2:
        raise ValueError("the AI institutions' rules promise feedback between rounds, "
                         "so they need at least 2 rounds")

    _buyer_outside_option = (
        "\n\nBUYER'S OUTSIDE OPTION: The buyer is NOT obligated to award this "
        "contract. If the winning bid's price exceeds the buyer's value for "
        "those terms, the buyer will walk away rather than accept a "
        "value-destroying deal. As a supplier, this means you want to "
        "capture as much of the deal's value as you can — pricing as high "
        "as the buyer will accept — while staying below the point where "
        "they would prefer no deal. The more competition you face, the "
        "less room you have to price up; with less competition, you have "
        "more room — but the buyer's walkaway point is always the ceiling."
    )

    # Regime context is the SAME regardless of participation count.
    # The platform does NOT disclose how many suppliers are bidding.
    regime_context = (
        "RULES: You submit a bid, then receive feedback from the platform. "
        "You can revise your bid over multiple rounds. The platform is "
        "working on behalf of the buyer — its goal is to get the best "
        "deal for the buyer. The platform can see all your previous bids in this event "
        "and will use that information in its feedback.\n\n"
        "The buyer scores final bids on all terms (price, delivery, quality). "
        "The highest-scoring final bid wins. The winner is paid "
        "the price they bid. The platform does not disclose how many "
        "suppliers are participating in this event."
        + _buyer_outside_option
    )
    if rules is not None:
        regime_context = rules

    # Inject per-supplier cross-event history if provided
    _supplier_contexts: dict[str, str] = {}
    for s in suppliers:
        ctx = regime_context
        if cross_event_history and s.supplier_id in cross_event_history:
            ctx = cross_event_history[s.supplier_id] + "\n\n" + ctx
        _supplier_contexts[s.supplier_id] = ctx

    all_round_bids: dict[str, list[SupplierBid]] = {s.supplier_id: [] for s in suppliers}
    histories: dict[str, list[dict]] = {s.supplier_id: [] for s in suppliers}
    all_feedback: dict[str, list[str]] = {s.supplier_id: [] for s in suppliers}

    for round_num in range(1, num_rounds + 1):
        for s in suppliers:
            agent = agents[s.supplier_id]
            bid = agent.generate_bid(
                s, _supplier_contexts[s.supplier_id], round_num=round_num,
                history=histories[s.supplier_id],
            )
            all_round_bids[s.supplier_id].append(bid)

            # The buyer agent answers each bid, except in the last round.
            if round_num < num_rounds:
                feedback = buyer_agent.generate_counteroffer_feedback(
                    buyer, bid.terms, s.supplier_id, round_num,
                    histories[s.supplier_id],
                    buyer_description=buyer_description,
                )
                all_feedback[s.supplier_id].append(feedback)
                histories[s.supplier_id].append({
                    "round": round_num,
                    "bid": bid.terms,
                    "feedback": feedback,
                    "price_set_by_rule": RULE_PRICE_TAG in bid.reasoning,
                    "price_raised_by_floor": FLOOR_PRICE_TAG in bid.reasoning,
                })

    final_bids = {sid: bids[-1] for sid, bids in all_round_bids.items()}

    outcome = _compute_outcome(
        regime="BuyerAgent",
        run_id=run_id,
        buyer=buyer,
        suppliers=suppliers,
        final_bids=final_bids,
        all_round_bids=all_round_bids,
        efficient_surplus=efficient_surplus,
        efficient_winner_id=efficient_winner_id,
        buyer_walks_away=True,
    )
    outcome.mediator_feedback = all_feedback
    return outcome
