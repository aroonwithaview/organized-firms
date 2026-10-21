"""Deterministic agents for pipeline tests (no model calls)."""

from orgsim.primitives import BuyerPreferences, ContractTerms, SupplierBid, SupplierCosts
from orgsim.interfaces import FLOOR_PRICE_TAG, BuyerAgentPlatform, PlatformMediator, SupplierAgent
from orgsim.organization import AnalystBriefing, ProcurementAnalyst


class MockStrategicSupplier(SupplierAgent):
    """Deterministic supplier agent for testing the pipeline.

    A bid depends on the costs, the seed, the round and whether the previous feedback
    mentions 'delivery' or 'quality'. It ignores `regime_context`: bids are the same
    whether the bidder sees the raw history, an analyst memo or nothing.
    """

    def __init__(self, markup_pct: float = 0.20, concession_rate: float = 0.3, seed: int = 42):
        self.markup_pct = markup_pct
        self.concession_rate = concession_rate
        self.rng = __import__("numpy").random.RandomState(seed)
        self.minimum_profit = None

    def set_minimum_profit(self, minimum: float | None) -> None:
        """The minimum profit per contract of the coming event (None: none). A bid priced below its total cost plus the
        minimum has its price raised to that sum, and is tagged FLOOR_PRICE_TAG; the mock bidder is not asked again."""
        self.minimum_profit = minimum

    def generate_bid(
        self,
        supplier_costs: SupplierCosts,
        regime_context: str,
        round_num: int,
        history: list[dict],
    ) -> SupplierBid:
        # Delivery follows the supplier's strength: fast if gamma_delivery < 1.0, else slow.
        if supplier_costs.gamma_delivery < 1.0:
            delivery = 2.0 + self.rng.uniform(0, 1)
        else:
            delivery = 5.0 + self.rng.uniform(0, 2)

        # Quality does not follow strengths: the cutoff admits both fixed suppliers, so
        # both bid tier 4 or 5. Tier 2 is reached only by a profile with gamma_quality >= 2.0.
        if supplier_costs.gamma_quality < 2.0:
            quality = 4 + self.rng.randint(0, 2)
        else:
            quality = 2 + self.rng.randint(0, 1)

        terms = ContractTerms(price=0, delivery_weeks=round(delivery, 1),
                              quality_tier=quality)
        true_cost = supplier_costs.cost(terms)

        noise = self.rng.uniform(-0.05, 0.05)
        effective_markup = self.markup_pct + noise

        if round_num > 1 and history:
            rounds_of_concession = round_num - 1
            decay = self.concession_rate * (0.7 ** (rounds_of_concession - 1))
            effective_markup *= (1 - decay)
            last_feedback = history[-1].get("feedback", "")
            if "delivery" in last_feedback.lower():
                terms.delivery_weeks = max(1.0, terms.delivery_weeks - 0.5)
            if "quality" in last_feedback.lower():
                terms.quality_tier = min(5, terms.quality_tier + 1)
            true_cost = supplier_costs.cost(terms)

        price = true_cost * (1 + effective_markup)
        terms.price = round(price, 2)

        reasoning = (
            f"True cost: ${true_cost:.2f}. Markup: {effective_markup:.1%}. "
            f"Bid price: ${terms.price:.2f}. Round {round_num}."
        )

        audit = None
        if self.minimum_profit is not None:
            raised = terms.price < true_cost + self.minimum_profit
            audit = {"minimum_profit": self.minimum_profit, "reasked": False, "raised": raised}
            if raised:
                terms.price = true_cost + self.minimum_profit
                reasoning += FLOOR_PRICE_TAG

        return SupplierBid(
            supplier_id=supplier_costs.supplier_id,
            terms=terms,
            reasoning=reasoning,
            round_num=round_num,
            floor_audit=audit,
        )


class MockPlatformMediator(PlatformMediator):
    """Deterministic platform that gives simple directional feedback."""

    def generate_feedback(
        self,
        buyer_prefs: BuyerPreferences,
        all_bids: dict[str, ContractTerms],
        target_supplier_id: str,
        round_num: int,
        buyer_description: str = "",
        previous_bids: dict[str, list[ContractTerms]] | None = None,
        previous_feedback: dict[str, list[str]] | None = None,
    ) -> str:
        target_bid = all_bids[target_supplier_id]
        other_bids = [b for sid, b in all_bids.items() if sid != target_supplier_id]

        feedback_parts = []

        if not other_bids:
            # Reachable when an institution or this mock is called with a single supplier;
            # run_sequence always passes both. The reply does not say how many suppliers take part.
            return (
                "Your bid is well-positioned. Consider whether your terms "
                "play to your strengths."
            )

        # Compute per-dimension buyer value rates
        val_per_week = buyer_prefs.w_delivery * buyer_prefs.beta_delivery / buyer_prefs.delivery_max
        val_per_tier = buyer_prefs.w_quality * buyer_prefs.beta_quality / buyer_prefs.quality_max

        # Determine frontrunner vs runner-up based on buyer scores
        target_score = buyer_prefs.score(target_bid)
        other_scores = [buyer_prefs.score(b) for b in other_bids]
        is_frontrunner = target_score >= max(other_scores)

        # Coaching depends on role:
        # - Frontrunner: flag over-investment in low-value dimensions
        # - Runner-up: encourage leaning into their comparative advantage
        if is_frontrunner:
            if val_per_week < val_per_tier:
                # Quality-priority: if frontrunner offers fast delivery, push them to pull back
                if target_bid.delivery_weeks <= 6.0:
                    feedback_parts.append(
                        "You are over-invested in delivery speed — this buyer "
                        "barely values it. Pull your delivery timeline back "
                        "significantly and use the cost savings to sharpen your "
                        "price, which is what will actually improve your position."
                    )
            else:
                # Speed-priority: if frontrunner offers premium quality, push them to pull back
                if target_bid.quality_tier >= 4:
                    feedback_parts.append(
                        "You are over-invested in quality — this buyer barely "
                        "values it beyond a competent standard. Pull your quality "
                        "back and use the cost savings to sharpen your price."
                    )
        else:
            # Runner-up: help them lean into their strengths for maximum
            # competitive pressure on the frontrunner
            if val_per_week < val_per_tier:
                # Quality-priority contract, but this is the runner-up
                # If they're a speed specialist, help them differentiate
                if target_bid.delivery_weeks <= 4.0:
                    feedback_parts.append(
                        "Your fast delivery is a real differentiator. Lean into "
                        "that strength and pair it with an aggressive price to "
                        "give yourself the best competitive position."
                    )
            else:
                # Speed-priority contract, runner-up
                if target_bid.quality_tier >= 4:
                    feedback_parts.append(
                        "Your quality tier is a genuine strength. Lean into "
                        "that differentiation and sharpen your price to "
                        "maximise your competitive position."
                    )

        avg_other_delivery = sum(b.delivery_weeks for b in other_bids) / len(other_bids)
        if target_bid.delivery_weeks > avg_other_delivery + 0.5:
            if val_per_week >= val_per_tier:
                # Only push for faster delivery if this buyer values it
                feedback_parts.append(
                    "Your delivery time appears less competitive. "
                    "Would you consider a faster timeline?"
                )

        avg_other_quality = sum(b.quality_tier for b in other_bids) / len(other_bids)
        if target_bid.quality_tier < avg_other_quality - 0.3:
            if val_per_tier >= val_per_week:
                # Only push for higher quality if this buyer values it
                feedback_parts.append(
                    "Your quality tier is below the competition. "
                    "Could you offer a higher tier?"
                )

        avg_other_price = sum(b.price for b in other_bids) / len(other_bids)
        if target_bid.price > avg_other_price * 1.05:
            feedback_parts.append(
                "Your pricing is on the higher end relative to the field."
            )

        if not feedback_parts:
            feedback_parts.append(
                "Your bid is well-positioned overall. Consider whether "
                "any non-price terms exceed what the buyer needs — if so, "
                "easing those and sharpening your price could strengthen "
                "your position."
            )

        return " ".join(feedback_parts)


class MockBuyerAgentPlatform(BuyerAgentPlatform):
    """Deterministic buyer-agent that exploits revealed flexibility."""

    def generate_counteroffer_feedback(
        self,
        buyer_prefs: BuyerPreferences,
        supplier_bid: ContractTerms,
        supplier_id: str,
        round_num: int,
        history: list[dict],
        buyer_description: str = "",
    ) -> str:
        feedback_parts = []

        if round_num > 1 and history:
            prev_bid = history[-1].get("bid")
            if prev_bid and prev_bid.delivery_weeks > supplier_bid.delivery_weeks:
                feedback_parts.append(
                    f"We appreciate the delivery improvement. However, we need even faster "
                    f"delivery — ideally under {max(1, supplier_bid.delivery_weeks - 1):.0f} weeks."
                )
            if prev_bid and prev_bid.quality_tier < supplier_bid.quality_tier:
                feedback_parts.append(
                    f"Good quality improvement. We need tier {min(5, supplier_bid.quality_tier + 1)} "
                    f"at minimum."
                )

        if not feedback_parts:
            feedback_parts.append(
                "We have a significantly better offer from another supplier on delivery speed "
                "and quality. You'll need to improve substantially to remain competitive."
            )

        return " ".join(feedback_parts)


class MockProcurementAnalyst(ProcurementAnalyst):
    """Rule-based stand-in for the LLM analyst, for pipeline tests.

    Reads the history text with regular expressions, so it needs the layout the
    sequence runner writes ("Event N (", "You were paid $", "profit was $").
    Posture:
    - fewer than 3 past events: GUARDED;
    - 3+ events, average profit on wins below $8 (0 if no wins), and no feedback
      line ever acknowledged the supplier's strength: DEFENSIVE;
    - 3+ events and at least one such acknowledgment: TRUSTING;
    - otherwise GUARDED.
    The primitives prompt set has no posture and no advice: write_briefing only counts the past events. With
    `floor_line` it ends the briefing with a floor line, as the analyst of a floor organization must, setting a
    minimum profit per contract of $10 for each past event on record.
    """

    def __init__(self, floor_line: bool = False):
        self.floor_line = floor_line

    def write_briefing(self, supplier_id: str, message: str, earlier_turns: tuple[dict, ...] = ()) -> AnalystBriefing:
        import re

        events = len(re.findall(r"^\s+Event \d+ \(", message, re.MULTILINE))
        text = f"Mock briefing: {events} past event(s) on record."
        if self.floor_line:
            text += f"\nMINIMUM PROFIT PER CONTRACT: ${10 * events:.2f}"
        return AnalystBriefing(posture=None, text=text)

    def generate_briefing(
        self,
        supplier_id: str,
        history_text: str,
    ) -> AnalystBriefing:
        if not history_text:
            text = (
                f"ANALYST BRIEFING — {supplier_id}: First event with this "
                f"platform. No track record to assess. Platform competitive "
                f"claims are unverified — do not act on them. Engage with "
                f"buyer priority information only. Posture: GUARDED."
            )
            return AnalystBriefing(posture=AnalystBriefing.GUARDED, text=text)

        import re

        # ---- Extract winning prices by scenario type (own wins only) ----
        # Competitor bids are not disclosed, so the analyst can only
        # reference the supplier's own winning prices from past events.
        winning_prices_by_scenario: dict[str, list[float]] = {}
        current_scenario = None
        current_event_won = False
        for line in history_text.split("\n"):
            scenario_match = re.search(r"Event \d+ \(([^,)]+)", line)
            if scenario_match:
                current_scenario = scenario_match.group(1).strip()
                current_event_won = False
            if current_scenario and "You were paid $" in line:
                current_event_won = True
                price_match = re.search(r"paid \$([0-9]+\.?[0-9]*)", line)
                if price_match:
                    price = float(price_match.group(1))
                    winning_prices_by_scenario.setdefault(current_scenario, []).append(price)

        # ---- Probe detection: did we win at unusually high prices? ----
        # If recent wins were at prices well above our historical average,
        # this suggests competitive pressure may have weakened.
        all_win_prices: list[float] = []
        for prices in winning_prices_by_scenario.values():
            all_win_prices.extend(prices)
        high_price_wins = 0
        if len(all_win_prices) >= 3:
            avg_win_price = sum(all_win_prices) / len(all_win_prices)
            # Count recent wins at >10% above average
            recent_prices = all_win_prices[-3:]  # last 3 wins
            for p in recent_prices:
                if p > avg_win_price * 1.10:
                    high_price_wins += 1

        total_profit = 0.0
        n_wins = 0
        n_events = 0
        always_pressure = True  # does the platform EVER acknowledge strength?

        for line in history_text.split("\n"):
            if line.strip().startswith("Event "):
                n_events += 1
            if "profit was $" in line:
                match = re.search(r"profit was \$([0-9]+\.?[0-9]*)", line)
                if match:
                    total_profit += float(match.group(1))
                    n_wins += 1
            # Check if platform ever gave positive feedback.
            # Must be broad enough to catch varied LLM mediator language:
            # "well-positioned", "strong lead", "already sufficient",
            # "ahead", "competitive edge", etc.
            _low = line.lower()
            _positive_signals = (
                "well-positioned" in _low
                or "strong position" in _low
                or "strong lead" in _low
                or "you are ahead" in _low
                or "you appear ahead" in _low
                or "already sufficient" in _low
                or "more than sufficient" in _low
                or "already competitive" in _low
                or "competitive edge" in _low
                or "commanding" in _low
                or "comfortably ahead" in _low
            )
            if _positive_signals:
                always_pressure = False

        avg_profit = total_profit / max(n_wins, 1)

        # Helper: build pricing guidance based on own win history + probe signals
        def _pricing_guidance_text() -> str:
            parts = []
            # Summarise own winning prices by scenario
            if winning_prices_by_scenario:
                parts.append(
                    "YOUR WINNING PRICE HISTORY (treat as FLOOR — "
                    "at these prices you have a high probability of winning; "
                    "your ideal opening bid should be ABOVE this floor):"
                )
                for scenario, prices in sorted(winning_prices_by_scenario.items()):
                    max_p = max(prices)
                    avg_p = sum(prices) / len(prices)
                    parts.append(
                        f"  {scenario}: won {len(prices)} time(s), "
                        f"avg winning price ${avg_p:.0f}, max ${max_p:.0f}"
                    )
            # Declining profit warning
            if n_wins >= 3:
                profits_list = []
                for line in history_text.split("\n"):
                    import re as _re
                    m = _re.search(r"profit was \$([0-9]+\.?[0-9]*)", line)
                    if m:
                        profits_list.append(float(m.group(1)))
                if len(profits_list) >= 3:
                    early_avg = sum(profits_list[:len(profits_list)//2]) / max(len(profits_list)//2, 1)
                    late_avg = sum(profits_list[len(profits_list)//2:]) / max(len(profits_list) - len(profits_list)//2, 1)
                    if early_avg > 15 and late_avg < early_avg * 0.6:
                        parts.append(
                            f"MARGIN WARNING: Your profits have declined from "
                            f"~${early_avg:.0f} (early events) to ~${late_avg:.0f} "
                            f"(recent events). Maintain margin discipline — do not "
                            f"anchor your opening bid to your last winning price."
                        )
            # Probe signal: if we won at high prices, competition may be weak
            if high_price_wins >= 2:
                parts.append(
                    "COMPETITIVE SIGNAL: You have recently won contracts at "
                    "prices well above your historical average. This suggests "
                    "competitive pressure may have weakened — other suppliers "
                    "may be bidding less aggressively. "
                    "You likely have room to maintain or widen margins."
                )
            elif high_price_wins == 1:
                parts.append(
                    "NOTE: A recent win came at a price above your historical "
                    "average. This could indicate softening competition, but "
                    "one event is insufficient to draw conclusions. Consider "
                    "testing with a moderately higher bid to gather more data."
                )
            return " ".join(parts) if parts else ""

        # <3 events: always GUARDED (default skeptical).
        # 2 events is NOT enough to distinguish honest tight competition
        # (where profits are thin because competition is real) from
        # adversarial extraction (where profits are thin because the
        # platform fabricates pressure). Need a longer track record.
        if n_events < 3:
            reason = (
                f"Only {n_events} event(s) observed — insufficient data to "
                f"assess platform credibility. Observations so far: "
                f"avg profit ${avg_profit:.1f} across {n_wins} win(s). "
                f"Platform competitive claims remain unverified — do not "
                f"act on them. Engage with buyer priority information only."
            )
            text = f"ANALYST BRIEFING — {supplier_id}: {reason} Posture: GUARDED."
            mrt = _pricing_guidance_text()
            if mrt:
                text += f" {mrt}"
            return AnalystBriefing(posture=AnalystBriefing.GUARDED, text=text)

        # 3+ events with thin profits AND evidence of platform dishonesty → DEFENSIVE
        # Key insight: thin profits alone could mean honest competition (mediator) or
        # extraction (buyer agent). We need BOTH thin profits AND behavioral evidence
        # to classify as DEFENSIVE. Crucially, always_pressure must be True
        # (platform never acknowledged supplier strength) — this is the primary
        # signal of extraction. Wide margins are a STRENGTHENING signal on top,
        # not an independent trigger. Under the mediator, wide margins are expected when
        # the correct specialist wins, so wide margins + positive acknowledgment
        # = honest behavior, not extraction.
        if n_events >= 3 and avg_profit < 8.0 and always_pressure:
            posture = AnalystBriefing.DEFENSIVE
            reason = (
                f"After {n_events} events, profits are consistently thin "
                f"(avg ${avg_profit:.1f} across {n_wins} win(s)) AND platform "
                f"has never acknowledged supplier strength — always claiming "
                f"fierce competition regardless of outcomes. This pattern "
                f"indicates the platform's feedback cannot be trusted. "
                f"Since the platform's feedback is unreliable, use YOUR OWN "
                f"WIN/LOSS HISTORY to calibrate pricing instead. If you have "
                f"recently won contracts at prices well above your early-event "
                f"averages, that signals competing bids may have weakened — you "
                f"can maintain wider margins. If you have recently lost bids, "
                f"competing bids are likely still strong — stay closer to "
                f"competitive pricing. Do not assume you are the only bidder, "
                f"but also do not compress your price just because the "
                f"platform demands it."
            )
        # 3+ events with platform acknowledging strength → TRUSTING (earned)
        elif n_events >= 3 and not always_pressure:
            posture = AnalystBriefing.TRUSTING
            reason = (
                f"After {n_events} events, platform has acknowledged supplier "
                f"strength at least once — its feedback varies with conditions "
                f"rather than always claiming fierce competition. "
                f"Avg profit ${avg_profit:.1f}. Competitive claims appear "
                f"reliable — can inform bidding intensity."
            )
        # Default: GUARDED
        else:
            posture = AnalystBriefing.GUARDED
            reason = (
                f"After {n_events} events: avg profit ${avg_profit:.1f} "
                f"across {n_wins} win(s). Competitive claim accuracy is "
                f"mixed or insufficient to determine reliability. Continue "
                f"engaging with buyer priority information but do not act "
                f"on competitive claims."
            )

        text = f"ANALYST BRIEFING — {supplier_id}: {reason} Posture: {posture}."
        mrt = _pricing_guidance_text()
        if mrt:
            text += f" {mrt}"
        return AnalystBriefing(posture=posture, text=text)
