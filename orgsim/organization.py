"""Organization of the supplier firm: analyst briefings and what the bidder sees. Each supplier has its own
organization; a market is homogeneous when both have the same one and mixed when they differ. The advice set has
three organizations (ANALYST_ORGANIZATIONS are those with an analyst) and the primitives set has six
(PRIMITIVES_ORGANIZATIONS) and, run only when named, the departures organizations (DEPARTURE_ORGANIZATIONS)."""

# The organizations whose supplier has a procurement analyst: the analyst reads the record and writes the
# briefing, and the bidder reads only the briefing. The single firm has none. The organizations in this
# tuple differ in what their bidder is told (orgsim.prompts.BIDDER_PARTS), not in this routing.
ANALYST_ORGANIZATIONS = ("organized", "organized_guided")


# The organizations of the primitives prompt set (orgsim.primitives_prompts), which has its own texts for them.
# single and organized have the names of the advice set's but not its texts. The advice set's organizations
# (orgsim.prompts.ORGANIZATIONS) are not these, and organized_guided is not among them.
#   single           the bidder holds the firm's record; it has no analyst (U)
#   single_norecord  the firm keeps no record: the bidder is never shown a past event (U0)
#   organized        the analyst holds the record and writes the bidder a briefing (O)
#   organized_pass   the analyst passes the record to the bidder unchanged: the code forwards it, no model is
#                    called (O_pass)
#   single_revenue   as single, and the bidder is judged on the firm's revenue (U_i)
#   organized_revenue  as organized, and the bidder is judged on the firm's revenue; the analyst on its profit (O_i)
PRIMITIVES_ORGANIZATIONS = ("single", "single_norecord", "organized", "organized_pass", "single_revenue",
                            "organized_revenue")
# The primitives organizations whose supplier has an analyst that calls a model.
PRIMITIVES_ANALYST_ORGANIZATIONS = ("organized", "organized_revenue")
# The primitives organizations whose bidder is judged on revenue.
PRIMITIVES_REVENUE_ORGANIZATIONS = ("single_revenue", "organized_revenue")


# The departures organizations of the primitives set. In each, the firm has an analyst who holds the record and sends
# the bidder one message before each event from the second, under a heading that begins MESSAGE FROM YOUR COMPANY'S
# PROCUREMENT ANALYST; they differ in how the message is made and in what the analyst or the code may do:
#   brief       the analyst writes a briefing (B)
#   forward     no analyst call: the message is the full record, forwarded unchanged (F)
#   floor       as brief, and the analyst may set a minimum profit per contract, which the code enforces
#   brief_cap   as brief, with the briefing at most BRIEFING_WORD_LIMIT words
#   brief_cost  as brief, and the analyst is also shown the bidder's cost figures
# Each has a conflict variant with the suffix _revenue: the bidder is judged on revenue and the analyst on profit. The
# six organizations above are not among them, and they stay as they were (PRIMITIVES_ORGANIZATIONS is still the default
# of a run of the primitives set that names none).
DEPARTURE_BASES = ("brief", "forward", "floor", "brief_cap", "brief_cost")
# The most words that the briefing of brief_cap may have. The analyst's role text and the heading of the message say it.
BRIEFING_WORD_LIMIT = 50
DEPARTURE_REVENUE_ORGANIZATIONS = tuple(f"{base}_revenue" for base in DEPARTURE_BASES)
DEPARTURE_ORGANIZATIONS = DEPARTURE_BASES + DEPARTURE_REVENUE_ORGANIZATIONS


def departure_base(organization: str) -> str | None:
    """The base of a departures organization ("brief_cost_revenue" -> "brief_cost"), None for any other name."""
    if organization not in DEPARTURE_ORGANIZATIONS:
        return None
    return organization.removesuffix("_revenue")


# The departures organizations whose firm has an analyst that calls a model (all but forward).
DEPARTURE_ANALYST_ORGANIZATIONS = tuple(name for name in DEPARTURE_ORGANIZATIONS if departure_base(name) != "forward")
# Every organization of the primitives set, and those with an analyst or with a bidder judged on revenue.
ALL_PRIMITIVES_ORGANIZATIONS = PRIMITIVES_ORGANIZATIONS + DEPARTURE_ORGANIZATIONS
ALL_PRIMITIVES_ANALYST_ORGANIZATIONS = PRIMITIVES_ANALYST_ORGANIZATIONS + DEPARTURE_ANALYST_ORGANIZATIONS
ALL_PRIMITIVES_REVENUE_ORGANIZATIONS = PRIMITIVES_REVENUE_ORGANIZATIONS + DEPARTURE_REVENUE_ORGANIZATIONS


def forwards_the_record(organization: str) -> bool:
    """Whether the code, not a model, passes the record to the bidder unchanged: organized_pass and forward."""
    return organization == "organized_pass" or departure_base(organization) == "forward"


class AnalystBriefing:
    """Structured analyst output: posture classification + free-form briefing text. The primitives prompt set
    classifies no posture, so the posture of its briefings is None."""

    __slots__ = ("posture", "text")

    DEFENSIVE = "DEFENSIVE"
    TRUSTING = "TRUSTING"
    GUARDED = "GUARDED"

    def __init__(self, posture: str | None, text: str):
        self.posture = posture  # one of DEFENSIVE / TRUSTING / GUARDED, or None in the primitives set
        self.text = text        # free-form briefing for the bidder

    @property
    def is_defensive(self) -> bool:
        return self.posture == self.DEFENSIVE


class ProcurementAnalyst:
    """Base class for the analyst who holds the firm's history."""

    def generate_briefing(self, supplier_id: str, history_text: str) -> AnalystBriefing:
        raise NotImplementedError

    def write_briefing(self, supplier_id: str, message: str, earlier_turns: tuple[dict, ...] = ()) -> AnalystBriefing:
        """The primitives prompt set: send `message` (orgsim.primitives_prompts.analyst_message, which holds the
        rules and the record) and return the reply as the briefing, with no posture and no title. `earlier_turns`, if
        given, are the turns that follow the message in the conversation, chat messages ({"role", "content"}) in order:
        the analyst's own first reply and the follow-up that asks again (see orgsim.departures)."""
        raise NotImplementedError


# Title of the analyst's memo: the earlier simulation's title without its last sentence ("It is
# confidential and not shared with the platform or buyer."), which has no counterpart for the
# single firm. "Working in your best interest" states the organized firm's aligned objectives.
MEMO_HEADER = (
    "INTERNAL MEMO FROM YOUR PROCUREMENT ANALYST\n"
    "(This memo is prepared by a member of YOUR team, working in "
    "your best interest.)\n\n"
)


# New text. Before the first event there is nothing to route, so every organization sees this line.
FIRST_EVENT_NOTE = "PAST EVENTS: none yet. This is your company's first event on this platform."


def organization_of_each(organization, supplier_ids) -> dict[str, str]:
    """The organization of each supplier, as a new dict. `organization` is one name, which every supplier has
    (a homogeneous market), or a mapping from each supplier id to its name (a mixed market). The names are not
    checked here: build_bidder_context refuses one it does not know."""
    if isinstance(organization, str):
        return {sid: organization for sid in supplier_ids}
    if not hasattr(organization, "keys"):
        raise ValueError(f"unknown organization: {organization!r}")
    if set(organization) != set(supplier_ids):
        raise ValueError(f"organization must name one organization for each of {list(supplier_ids)}: "
                         f"{dict(organization)!r}")
    return {sid: organization[sid] for sid in supplier_ids}


def build_bidder_context(organization: str, history_text: str, briefing_text: str | None) -> str:
    """What the bidding agent of a supplier with this organization sees about past events: its own record or its
    own analyst's memo, and never the other supplier's.

    single:    the raw history; the firm has no analyst.
    organized, organized_guided (ANALYST_ORGANIZATIONS): only the analyst's memo; the raw history
               stays with the analyst.
    Before the first event every bidder of an AI institution sees FIRST_EVENT_NOTE, and an organized firm may
    not send a briefing then, so the firms start with identical information. (The bidder of the second-price
    auction has no record to route and is given neither this note nor a record.)
    """
    if organization == "single":
        if briefing_text is not None:
            raise ValueError("a single-agent firm has no analyst briefing")
        return history_text or FIRST_EVENT_NOTE
    if organization in ANALYST_ORGANIZATIONS:
        if not history_text:
            if briefing_text is not None:
                raise ValueError("an organized firm gets no briefing before its first event")
            return FIRST_EVENT_NOTE
        if briefing_text is None:
            raise ValueError("an organized firm with past events needs its analyst's briefing")
        return MEMO_HEADER + briefing_text
    raise ValueError(f"unknown organization: {organization!r}")


# The titles of what the bidder of an organized firm reads about past events in the primitives set (new text).
BRIEFING_HEADER = "BRIEFING FROM YOUR COMPANY'S PROCUREMENT ANALYST:"
FORWARDED_HEADER = "RECORD FORWARDED BY YOUR COMPANY'S PROCUREMENT ANALYST:"
# The headings of the message that the bidder of a departures organization reads from the second event on. Each begins
# with the words that the bidder's role text names, and says how the message was made.
BRIEF_HEADER = "MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST (written by the analyst):"
CAP_HEADER = ("MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST "
              f"(written by the analyst, at most {BRIEFING_WORD_LIMIT} words):")
FORWARD_HEADER = "MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST (your company's full record, forwarded unchanged):"
# The heading of floor: the last line of the message is the notice of the minimum, which the code writes.
FLOOR_HEADER = ("MESSAGE FROM YOUR COMPANY'S PROCUREMENT ANALYST "
                "(written by the analyst, except the last line, a notice of the minimum):")


def _message_header(organization: str) -> str:
    """The heading above the analyst's message to the bidder of this organization (one that has an analyst, or
    forward)."""
    if organization in PRIMITIVES_ANALYST_ORGANIZATIONS:
        return BRIEFING_HEADER
    base = departure_base(organization)
    return {"forward": FORWARD_HEADER, "brief_cap": CAP_HEADER, "floor": FLOOR_HEADER}.get(base, BRIEF_HEADER)


def build_primitives_bidder_context(organization: str, history_text: str, briefing_text: str | None) -> str:
    """What the bidder of a supplier with this organization (ALL_PRIMITIVES_ORGANIZATIONS) reads about past events in
    the primitives set, or "" when it reads nothing; never the other supplier's record or briefing. `history_text` is
    the supplier's record (build_primitives_record), "" before its first event.

    First event (no record): every organization, the one without a record included, reads FIRST_EVENT_NOTE, so the
    bidders' messages of the first event differ only in their system prompts.
    From the second event on:
      single, single_revenue:           the record, as written.
      single_norecord:                  nothing. The firm keeps no record; the runner still writes the record (the
                                        platform's memory does not depend on it), and it is not given to this bidder.
      organized, organized_revenue:     BRIEFING_HEADER and the analyst's briefing (`briefing_text`). The record stays
                                        with the analyst.
      organized_pass:                   FORWARDED_HEADER and the record, the very text that single reads, passed on by
                                        the code: there is no analyst call and no `briefing_text`.
      brief, brief_cost (and _revenue): BRIEF_HEADER and the analyst's briefing (`briefing_text`).
      brief_cap (and _revenue):         CAP_HEADER, which names the limit, and the analyst's briefing, at most
                                        that long.
      floor (and _revenue):             FLOOR_HEADER, which says that the last line is a notice of the minimum, and
                                        the analyst's briefing without its floor lines followed by that notice
                                        (`briefing_text` holds both: orgsim.departures.floor_briefing).
      forward (and _revenue):           FORWARD_HEADER and the record, the very text that single reads, passed on by the
                                        code: there is no analyst call and no `briefing_text`.
    """
    if organization not in ALL_PRIMITIVES_ORGANIZATIONS:
        raise ValueError(f"unknown organization of the primitives set: {organization!r}")
    if organization in ALL_PRIMITIVES_ANALYST_ORGANIZATIONS:
        if not history_text:
            if briefing_text is not None:
                raise ValueError("an organized firm gets no briefing before its first event")
            return FIRST_EVENT_NOTE
        if briefing_text is None:
            raise ValueError("an organized firm with past events needs its analyst's briefing")
        return _message_header(organization) + "\n" + briefing_text
    if briefing_text is not None:
        raise ValueError(f"{organization} has no analyst briefing")
    if not history_text:
        return FIRST_EVENT_NOTE
    if organization == "single_norecord":
        return ""
    if organization == "organized_pass":
        return FORWARDED_HEADER + "\n" + history_text
    if departure_base(organization) == "forward":
        return _message_header(organization) + "\n" + history_text
    return history_text
