"""Agent interfaces and bid parsing."""
import json
import re
from abc import ABC, abstractmethod

from orgsim.primitives import BuyerPreferences, ContractTerms, SupplierBid, SupplierCosts

# Appended to a bid's reasoning when code, not the model, shaped the bid: the below-cost rule set the
# price (RULE_PRICE_TAG), or the bid was made without the cost tool after the tool path failed
# (NO_TOOL_TAG), or, in the floor organizations of the primitives set, the price was raised to the total cost
# plus the minimum profit that the firm's analyst set (FLOOR_PRICE_TAG). The advice set has the first two, the
# primitives set the third only.
RULE_PRICE_TAG = " [PRICE SET BY RULE: COST + 5%]"
NO_TOOL_TAG = " [NO TOOL]"
FLOOR_PRICE_TAG = " [PRICE RAISED TO THE MINIMUM PROFIT]"


class SupplierAgent(ABC):
    """Interface for a supplier agent that produces bids."""

    @abstractmethod
    def generate_bid(
        self,
        supplier_costs: SupplierCosts,
        regime_context: str,
        round_num: int,
        history: list[dict],
    ) -> SupplierBid:
        """Return this supplier's bid for round `round_num`.

        `history` has one dict per earlier round of the event, with keys `round`,
        `bid` (a ContractTerms), `feedback` (the platform's text for that round),
        `price_set_by_rule` (True when the below-cost rule set that bid's price, which
        RULE_PRICE_TAG marks in the bid's reasoning) and `price_raised_by_floor` (True when the
        price was raised to the minimum profit that the firm's analyst set, which FLOOR_PRICE_TAG
        marks).
        """
        pass

    def set_minimum_profit(self, minimum: float | None) -> None:
        """Tell the agent the minimum profit per contract that its firm's analyst sets for the coming event, in
        dollars, or None if none is set. The runner calls this before every event of a supplier of a floor
        organization (primitives set), and an agent that does not enforce a minimum raises here, so that a floor is
        never set silently and left unenforced. A supplier without a floor organization is never told."""
        raise NotImplementedError(f"{type(self).__name__} does not enforce a minimum profit")


class PlatformMediator(ABC):
    """Interface for the platform that generates feedback to suppliers."""

    @abstractmethod
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
        """Return the feedback text for `target_supplier_id` after round `round_num`.

        In repeated events `buyer_description` also carries the mediator's memory of every
        supplier, sorted by supplier id, before the buyer's description. `previous_bids[sid]`
        already includes the current round's bid when this is called. `previous_feedback[sid]`
        holds the replies of earlier rounds only, never one from this round, and every call
        in a round gets the same copy.
        """
        pass


class BuyerAgentPlatform(ABC):
    """Interface for a buyer-agent platform that negotiates for the buyer."""

    @abstractmethod
    def generate_counteroffer_feedback(
        self,
        buyer_prefs: BuyerPreferences,
        supplier_bid: ContractTerms,
        supplier_id: str,
        round_num: int,
        history: list[dict],
        buyer_description: str = "",
    ) -> str:
        """Return the buyer agent's counteroffer text for `supplier_id` after round `round_num`.

        `history` has the entries that SupplierAgent.generate_bid receives: one dict per earlier
        round of the event, with keys `round`, `bid` (a ContractTerms), `feedback`,
        `price_set_by_rule` and `price_raised_by_floor`. The current bid is `supplier_bid`.
        """
        pass


def parse_bid_json(text: str, supplier_id: str, round_num: int) -> SupplierBid:
    """Parse a bid from model text. The whole text is read as JSON; failing that, the first flat JSON
    object in it with a "price" key is used (an object that itself contains an object is not found, and
    ValueError is raised). A missing field raises KeyError, JSON that is not an object raises TypeError,
    and a fractional tier is truncated. Used by tests only.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{[^{}]*"price"[^{}]*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse bid from: {text[:200]}")

    terms = ContractTerms(
        price=float(data["price"]),
        delivery_weeks=float(data["delivery_weeks"]),
        quality_tier=int(data["quality_tier"]),
    )
    return SupplierBid(
        supplier_id=supplier_id,
        terms=terms,
        reasoning=data.get("reasoning", ""),
        round_num=round_num,
    )
