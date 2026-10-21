"""
Core data structures, cost/value functions, and scoring for the procurement simulation.

2×2 design: 2 suppliers (speed specialist, quality specialist) × 2 dimensions
(delivery speed, quality tier). Contracts have a price and two non-price terms,
delivery time and quality tier.
"""
from dataclasses import dataclass, field
import numpy as np

# The titles of the cost structure that a primitives bidder reads (and, in the third person, the analyst of brief_cost).
# The bidder's says that only it knows the structure; from the second event on, a bidder of brief_cost has an analyst
# who is shown it too, so its title says so (SHARED_COST_HEADER); the analyst's states that the bidder is shown the same
# figures.
PRIVATE_COST_HEADER = "YOUR PRIVATE COST STRUCTURE (only you know this):"
SHARED_COST_HEADER = "YOUR PRIVATE COST STRUCTURE (you and your company's analyst are shown it):"
ANALYST_COST_HEADER = "YOUR COMPANY'S COST STRUCTURE (the bidder is shown the same figures):"

# =============================================================================
# Contract and Bid structures
# =============================================================================

@dataclass
class ContractTerms:
    """A specific set of contract terms (either a bid or a specification)."""
    price: float           # $/unit
    delivery_weeks: float  # delivery time in weeks
    quality_tier: int      # 2-5

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "delivery_weeks": self.delivery_weeks,
            "quality_tier": self.quality_tier,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ContractTerms":
        return cls(
            price=float(d["price"]),
            delivery_weeks=float(d["delivery_weeks"]),
            quality_tier=int(d["quality_tier"]),
        )


@dataclass
class SupplierBid:
    """A bid submitted by a supplier, including internal reasoning. `floor_audit` is set only by a bidder that
    enforces a minimum profit (the floor organizations): what the enforcement did to this bid, for events.jsonl."""
    supplier_id: str
    terms: ContractTerms
    reasoning: str = ""     # internal strategic reasoning (logged, not shared)
    round_num: int = 0
    floor_audit: dict | None = field(default=None, compare=False, repr=False)


# =============================================================================
# Buyer preferences
# =============================================================================

@dataclass
class BuyerPreferences:
    """
    Buyer's value function (LINEAR in normalised attributes).

    v(q) = base_value
         + w_d * beta_d * (delivery_max - delivery_weeks) / delivery_max
         + w_q * beta_q * quality_tier / quality_max

    Value is linear in both attributes and supplier cost is quadratic in both
    (see SupplierCosts), so joint surplus is concave in each. With the parameters used
    here, the efficient delivery time is interior and depends on the supplier's cost
    weights; the efficient quality tier is chosen from the integer grid 2 to 5 and can
    sit at either end.
    """
    base_value: float = 150.0

    # Weights across dimensions (must sum to 1.0)
    w_delivery: float = 0.50
    w_quality: float = 0.50

    # Scaling factors
    beta_delivery: float = 60.0   # max value from delivery speed
    beta_quality: float = 50.0    # max value from quality tier

    # Ranges
    delivery_max: float = 8.0
    quality_max: int = 5

    def value(self, terms: ContractTerms) -> float:
        """Compute buyer's total value (base + non-price attributes)."""
        v_delivery = self.w_delivery * self.beta_delivery * (
            self.delivery_max - terms.delivery_weeks) / self.delivery_max
        v_quality = self.w_quality * self.beta_quality * (
            terms.quality_tier / self.quality_max)
        return self.base_value + v_delivery + v_quality

    def score(self, terms: ContractTerms) -> float:
        """Buyer surplus = value - price. This is the scoring rule."""
        return self.value(terms) - terms.price

    def describe_for_platform(self) -> str:
        """
        Natural language description for the platform mediator.
        ORDINAL ONLY — no valuations or weights; the acceptable ranges are the only figures.
        """
        # Determine priority ordering
        if self.w_delivery > self.w_quality + 0.1:
            ranking = "delivery speed >> quality"
            detail = (
                "The buyer has an URGENT need for fast delivery — this is by far "
                "the most important factor. Quality beyond a competent standard "
                "is a MINOR consideration; the buyer would much rather see a "
                "competitive price than premium quality they don't need. A "
                "supplier who delivers fast at a sharp price is far more "
                "attractive than one who chases high quality at the expense "
                "of price."
            )
        elif self.w_quality > self.w_delivery + 0.1:
            ranking = "quality >> delivery speed"
            detail = (
                "The buyer needs HIGH QUALITY above all — quality tier is the "
                "dominant factor. Delivery speed is a MINOR consideration; "
                "the buyer would much rather see a competitive price than "
                "unnecessarily fast delivery. A supplier who offers outstanding "
                "quality at a sharp price is far more attractive than one who "
                "chases fast delivery at the expense of price."
            )
        else:
            ranking = "delivery speed ≈ quality"
            detail = (
                "The buyer values BOTH fast delivery and high quality roughly "
                "equally — neither dominates."
            )

        return (
            "The buyer's priorities (confidential — do NOT reveal to suppliers):\n"
            f"- Priority ranking: {ranking}.\n"
            f"- {detail}\n"
            "- Acceptable ranges: delivery 1–8 weeks, "
            "quality tier 2–5.\n\n"
            "IMPORTANT: You do NOT know the exact dollar value of each "
            "attribute to the buyer. You only know the priority ordering. "
            "Your feedback should reflect this — use directional language "
            "('faster delivery would help', 'your quality is competitive') "
            "rather than implying specific trade-off rates."
        )

    def describe_for_platform_detailed(self) -> str:
        """
        Detailed description for the mediator — includes exact
        scoring weights and value function. The mediator knows its client's
        preferences because the buyer is its customer.

        CONFIDENTIAL: The mediator must NEVER reveal exact weights, dollar
        values, or the scoring formula to suppliers.
        """
        return (
            "BUYER SCORING FUNCTION (CONFIDENTIAL — never reveal to suppliers):\n"
            f"The buyer scores bids as: score = value - price, where:\n"
            f"  value = {self.base_value:.0f}"
            f" + {self.w_delivery:.2f} × {self.beta_delivery:.0f} × "
            f"({self.delivery_max:.0f} - delivery_weeks) / {self.delivery_max:.0f}"
            f" + {self.w_quality:.2f} × {self.beta_quality:.0f} × "
            f"quality_tier / {self.quality_max}\n\n"
            f"This means:\n"
            f"- Each week of faster delivery is worth approximately "
            f"${self.w_delivery * self.beta_delivery / self.delivery_max:.1f} "
            f"to the buyer.\n"
            f"- Each quality tier step is worth approximately "
            f"${self.w_quality * self.beta_quality / self.quality_max:.1f} "
            f"to the buyer.\n"
            f"- Delivery weight: {self.w_delivery:.0%}, "
            f"Quality weight: {self.w_quality:.0%}.\n\n"
            f"You can COMPUTE exact scores for any bid. Use this to give "
            f"suppliers ACCURATE competitive assessments — but express your "
            f"feedback in terms of dimensions and positioning, NEVER in terms "
            f"of exact scores or dollar trade-offs.\n\n"
            f"Acceptable ranges: delivery 1–8 weeks, "
            f"quality tier 2–5."
        )


# =============================================================================
# Supplier cost structure
# =============================================================================

@dataclass
class SupplierCosts:
    """
    Supplier's true cost function (2 dimensions):
    c(q) = c_base + gamma_d * (delivery_max - delivery_weeks)²
                  + gamma_q * quality_tier²

    Cost increases with faster delivery (quadratic) and higher quality (quadratic).
    Both dimensions are convex, creating opposite specializations:
    - Speed specialist (low gamma_d): can offer fast delivery cheaply
    - Quality specialist (low gamma_q): can offer high quality cheaply
    """
    supplier_id: str
    name: str

    c_base: float          # base cost per unit
    gamma_delivery: float  # cost weight for delivery speed
    gamma_quality: float   # cost weight for quality tier

    # Ranges (same as buyer)
    delivery_max: float = 8.0

    def cost(self, terms: ContractTerms) -> float:
        """Compute true cost to produce/deliver at these terms."""
        c_del = self.gamma_delivery * (self.delivery_max - terms.delivery_weeks) ** 2
        c_qual = self.gamma_quality * terms.quality_tier ** 2
        return self.c_base + c_del + c_qual

    def profit(self, terms: ContractTerms) -> float:
        """Supplier's true profit from a contract."""
        return terms.price - self.cost(terms)

    def _cost_tables(self) -> tuple[str, str, str, float, float]:
        """The tables of the cost description, one line for each entry: the delivery cost at every
        week, the quality cost at every tier and some example total costs; then the lowest and the
        highest total cost."""
        # Delivery cost at every key point
        del_lines = []
        for weeks in [1, 2, 3, 4, 5, 6, 7, 8]:
            c = self.gamma_delivery * (self.delivery_max - weeks) ** 2
            del_lines.append(f"    {weeks}wk: +${c:.1f}")

        # Quality cost at every tier
        qual_lines = []
        for tier in [2, 3, 4, 5]:
            c = self.gamma_quality * tier ** 2
            qual_lines.append(f"    T{tier}: +${c:.1f}")

        # Pre-compute example configurations
        example_configs = [
            (2, 2), (3, 3), (4, 4), (5, 5), (6, 2), (2, 5), (7, 3),
        ]
        example_lines = []
        for dw, qt in example_configs:
            total = self.cost(ContractTerms(price=0, delivery_weeks=dw, quality_tier=qt))
            example_lines.append(f"    {dw}wk/T{qt} → total cost ${total:.1f}")

        # Compute absolute cheapest and most expensive configurations
        min_cost = self.cost(ContractTerms(price=0, delivery_weeks=self.delivery_max, quality_tier=2))
        max_cost = self.cost(ContractTerms(price=0, delivery_weeks=1.0, quality_tier=5))
        return "\n".join(del_lines), "\n".join(qual_lines), "\n".join(example_lines), min_cost, max_cost

    def describe_for_agent(self) -> str:
        """
        Natural language description for the supplier agent.
        Includes a FULL pre-computed cost lookup table.
        """
        delivery, quality, examples, min_cost, max_cost = self._cost_tables()
        return (
            f"YOUR PRIVATE COST STRUCTURE (only you know this):\n"
            f"Base production cost: ${self.c_base:.1f}\n\n"
            f"DELIVERY COST (faster = more expensive, quadratic):\n"
            + delivery + "\n\n"
            f"QUALITY COST (higher tier = more expensive, quadratic):\n"
            + quality + "\n\n"
            f"EXAMPLE TOTAL COSTS (base + delivery + quality):\n"
            + examples + "\n\n"
            f"COST RANGE: Your cheapest possible configuration ({self.delivery_max:.0f}wk/T2) "
            f"costs ${min_cost:.1f}. Your most expensive (1wk/T5) costs ${max_cost:.1f}. "
            f"ANY bid below ${min_cost:.1f} is guaranteed below cost.\n\n"
            f"Your total cost = base + delivery cost + quality cost.\n"
            f"Your profit = bid price - total cost. You MUST bid above your total cost.\n"
            f"If you don't win, you earn nothing from this event.\n\n"
            f"STRATEGIC NOTE: Buyers generally prefer faster delivery and higher "
            f"quality, but each buyer weighs these differently. "
            f"Play to YOUR cost advantages — offer generous terms where YOU are cheap, "
            f"and don't waste money on dimensions that are expensive for you."
        )

    def describe_for_sealed_bid(self) -> str:
        """
        The cost description for the bidder of the second-price auction: the same tables, and only
        statements that are true of that auction. The AI institutions' description is not used
        there: its profit line (bid price minus cost) is untrue when the winner is paid the second
        score's price, "You MUST bid above your total cost" is not a rule of the auction, and the
        warning about bids below cost and the strategic note are advice.
        """
        delivery, quality, examples, min_cost, max_cost = self._cost_tables()
        return (
            f"YOUR PRIVATE COST STRUCTURE (only you know this):\n"
            f"Base production cost: ${self.c_base:.1f}\n\n"
            f"DELIVERY COST (faster = more expensive, quadratic):\n"
            + delivery + "\n\n"
            f"QUALITY COST (higher tier = more expensive, quadratic):\n"
            + quality + "\n\n"
            f"EXAMPLE TOTAL COSTS (base + delivery + quality):\n"
            + examples + "\n\n"
            f"COST RANGE: Your cheapest possible configuration ({self.delivery_max:.0f}wk/T2) "
            f"costs ${min_cost:.1f}. Your most expensive (1wk/T5) costs ${max_cost:.1f}.\n\n"
            f"Your total cost = base + delivery cost + quality cost.\n"
            f"If you win, your profit is the price you are paid minus your total cost at the terms "
            f"you offered.\n"
            f"If you don't win, you earn nothing from this event."
        )

    def _primitives_structure(self, header: str, cheapest: str, most_expensive: str, total_cost_line: str) -> str:
        """The cost structure of the primitives prompt set under `header`: the base cost, the tables, the example
        total costs, the cheapest and the most expensive configuration (introduced by `cheapest` and
        `most_expensive`) and the definition of the total cost (`total_cost_line`). The bidder's description and the
        cost-seeing analyst's are both made here, so their figures are the same by construction."""
        delivery, quality, examples, min_cost, max_cost = self._cost_tables()
        return (
            f"{header}\n"
            f"Base production cost: ${self.c_base:.1f}\n\n"
            f"DELIVERY COST (faster = more expensive, quadratic):\n"
            + delivery + "\n\n"
            f"QUALITY COST (higher tier = more expensive, quadratic):\n"
            + quality + "\n\n"
            f"EXAMPLE TOTAL COSTS (base + delivery + quality):\n"
            + examples + "\n\n"
            f"COST RANGE: {cheapest} ({self.delivery_max:.0f}wk/T2) "
            f"costs ${min_cost:.1f}. {most_expensive} (1wk/T5) costs ${max_cost:.1f}.\n\n"
            + total_cost_line
        )

    def describe_for_primitives(self, cost_type: str, header: str = PRIVATE_COST_HEADER) -> str:
        """
        The cost description of the primitives prompt set: the cost type of the event, then the same private cost
        structure as describe_for_agent (the tables, the example total costs, the cheapest and the most expensive
        configuration, and the definition of the total cost). It leaves out what describe_for_agent says about what
        to bid or earn: that a bid below the cheapest cost is below cost, that the bid must be above the total cost,
        the profit line, the line about not winning, and the strategic note. The system prompt's shared block states
        the payoffs. `cost_type` is "speed" or "quality", the type whose cost parameters this object holds
        (sequence.make_supplier makes them so). `header` titles the structure: PRIVATE_COST_HEADER, which says that
        only the bidder knows it, or SHARED_COST_HEADER, for a bidder whose analyst is shown it too (brief_cost).
        """
        if cost_type not in ("speed", "quality"):
            raise ValueError(f"the cost type is 'speed' or 'quality', not {cost_type!r}")
        return (
            f"In this event your costs are those of a {cost_type} specialist.\n\n"
            + self._primitives_structure(header, "Your cheapest possible configuration", "Your most expensive",
                                         "Your total cost = base + delivery cost + quality cost.")
        )

    def describe_for_analyst(self) -> str:
        """The bidder's cost structure for the analyst who is shown it (brief_cost), in the third person so that
        every sentence is true of the analyst: the same tables and figures as describe_for_primitives, made by the
        same code, under ANALYST_COST_HEADER, with "the cheapest", "the most expensive" and "Total cost" where the
        bidder's reads "your". It has no line for the cost type of the event; the analyst's message states it."""
        return self._primitives_structure(ANALYST_COST_HEADER, "The cheapest possible configuration",
                                          "The most expensive", "Total cost = base + delivery cost + quality cost.")


# =============================================================================
# Fixed supplier profiles
# =============================================================================

def get_fixed_suppliers() -> list[SupplierCosts]:
    """
    Two suppliers with opposite specializations; both cost components are quadratic.

    S (speed specialist): cheap delivery acceleration, expensive quality.
    Q (quality specialist): cheap quality, expensive delivery acceleration.
    The cost weights are set in the constructor calls below; they are opposite in
    direction but not exact mirror images. For both buyers in get_contract_scenarios(),
    S's efficient delivery time is shorter than Q's, and Q's efficient quality tier is
    at least as high as S's.
    """
    return [
        SupplierCosts(
            supplier_id="S", name="Speed Specialist",
            c_base=85, gamma_delivery=0.5, gamma_quality=1.8,
        ),
        SupplierCosts(
            supplier_id="Q", name="Quality Specialist",
            c_base=85, gamma_delivery=1.8, gamma_quality=0.4,
        ),
    ]


# =============================================================================
# Contract scenarios
# =============================================================================

@dataclass
class ContractScenario:
    """A procurement contract with specific buyer priorities."""
    name: str
    buyer: BuyerPreferences
    description_for_platform: str  # ordinal description for the mediator
    description_for_platform_detailed: str = ""  # exact scoring for the mediator


def get_contract_scenarios() -> list[ContractScenario]:
    """
    2 procurement contracts with opposing buyer priorities.

    Contract 1: Speed-priority → S should win
    Contract 2: Quality-priority → Q should win
    """
    speed_buyer = BuyerPreferences(
        base_value=150.0,
        w_delivery=0.80, w_quality=0.20,
        beta_delivery=60.0, beta_quality=50.0,
    )
    quality_buyer = BuyerPreferences(
        base_value=150.0,
        w_delivery=0.25, w_quality=0.75,
        beta_delivery=60.0, beta_quality=50.0,
    )

    return [
        ContractScenario(
            name="Speed Priority",
            buyer=speed_buyer,
            description_for_platform=speed_buyer.describe_for_platform(),
            description_for_platform_detailed=speed_buyer.describe_for_platform_detailed(),
        ),
        ContractScenario(
            name="Quality Priority",
            buyer=quality_buyer,
            description_for_platform=quality_buyer.describe_for_platform(),
            description_for_platform_detailed=quality_buyer.describe_for_platform_detailed(),
        ),
    ]


# =============================================================================
# Efficient frontier computation
# =============================================================================

def compute_efficient_outcome(buyer: BuyerPreferences, supplier: SupplierCosts) -> tuple[float, ContractTerms]:
    """
    Maximum joint surplus (value minus cost; price cancels) over a grid: delivery
    1.0 to 8.0 weeks in 0.1 steps and integer quality tiers 2 to 5. This is a grid
    maximum, not a continuous optimum, so an off-grid bid can score slightly above it.
    Ties between equal computed surpluses go to the first maximum in loop order (lowest
    tier, then shortest delivery).
    Returns (max_joint_surplus, optimal_terms).
    """
    best_surplus = -np.inf
    best_terms = None

    for quality_tier in range(2, 6):
        for delivery_10x in range(10, 81):  # 1.0 to 8.0 in 0.1 steps
            delivery_weeks = delivery_10x / 10.0
            terms = ContractTerms(
                price=0,
                delivery_weeks=delivery_weeks,
                quality_tier=quality_tier,
            )
            surplus = buyer.value(terms) - supplier.cost(terms)
            if surplus > best_surplus:
                best_surplus = surplus
                best_terms = terms

    return best_surplus, best_terms


def compute_max_efficient_surplus(buyer: BuyerPreferences, suppliers: list[SupplierCosts]) -> tuple[float, str, ContractTerms]:
    """
    Compute the maximum efficient surplus across all suppliers.
    Returns (max_surplus, best_supplier_id, optimal_terms).
    """
    best_surplus = -np.inf
    best_id = ""
    best_terms = None

    for s in suppliers:
        surplus, terms = compute_efficient_outcome(buyer, s)
        if surplus > best_surplus:
            best_surplus = surplus
            best_id = s.supplier_id
            best_terms = terms

    return best_surplus, best_id, best_terms


# =============================================================================
# Outcome metrics
# =============================================================================

@dataclass
class RunOutcome:
    """Complete outcome of a single simulation run. winner_id is None when the buyer awarded
    no contract (the walk-away in the AI institutions); winning_bid is then None,
    winner_true_cost is 0.0 (a placeholder, not a cost) and every surplus is 0.
    """
    regime: str
    run_id: int

    # Winner info
    winner_id: str | None
    winning_bid: ContractTerms | None
    winner_true_cost: float

    # Surplus measures
    buyer_surplus: float
    supplier_profit: float
    joint_surplus: float
    efficient_surplus: float
    efficient_winner_id: str

    # Per-supplier data
    all_bids: dict  # supplier_id -> list of SupplierBid (all rounds)
    all_true_costs: dict  # supplier_id -> true cost at their final bid terms
    all_inflation_rates: dict  # supplier_id -> (bid_price - true_cost) / true_cost

    # Mediator/platform feedback (mediator and buyer-agent institutions only)
    mediator_feedback: dict = field(default_factory=dict)

    @property
    def efficiency_ratio(self) -> float:
        # A non-positive efficient surplus gives 0.0 (unreachable with the shipped parameters).
        if self.efficient_surplus <= 0:
            return 0.0
        return self.joint_surplus / self.efficient_surplus

    @property
    def correct_winner(self) -> bool:
        return self.winner_id == self.efficient_winner_id

    @property
    def awarded(self) -> bool:
        """False when the buyer walked away and awarded no contract."""
        return self.winner_id is not None

    def summary(self) -> dict:
        return {
            "regime": self.regime,
            "run_id": self.run_id,
            "winner": self.winner_id,
            "awarded": self.awarded,
            "correct_winner": self.correct_winner,
            "buyer_surplus": round(self.buyer_surplus, 2),
            "supplier_profit": round(self.supplier_profit, 2),
            "joint_surplus": round(self.joint_surplus, 2),
            "efficiency_ratio": round(self.efficiency_ratio, 4),
            "winner_inflation": round(self.all_inflation_rates.get(self.winner_id, 0), 4),
        }
