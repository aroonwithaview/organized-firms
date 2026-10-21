"""The mechanisms of the departures organizations of the primitives set that code, not a model, carries out.

floor: the analyst may set a minimum profit per contract for the coming event, on the last line of its briefing;
parse_floor_line reads it (white space, case and markdown marks around the words do not matter, and lines of marks
after it are skipped), floor_briefing asks again once if the line is missing or malformed, and the bidder reads the
briefing without any line that begins with the words (strip_floor_lines), then a notice: the floor (floor_notice),
none, or, when the analyst gave no valid line in two answers, that no floor is set (FLOOR_NOTICE_NOT_SET). A bidder
whose price is below its total cost plus the minimum is asked again once, and if the bid it sends back is also below,
the price is raised to the total cost plus the minimum (orgsim.llm_agents.PrimitivesSupplierAgent, which uses the texts
below).

brief_cap: the briefing may have at most BRIEFING_WORD_LIMIT words (counted by splitting on white space). An analyst
whose briefing is longer is asked once more, in the same conversation, with a follow-up turn that says how long the
briefing was; a second answer that is still too long is cut to its first BRIEFING_WORD_LIMIT words. capped_briefing
does this and returns what the bidder reads and what to record.
"""
import re

from orgsim.organization import BRIEFING_WORD_LIMIT, AnalystBriefing, ProcurementAnalyst

# ---- floor ----
# The words that begin the line that ends the briefing of the analyst of floor: the minimum profit per contract in
# dollars, or none. The bidder does not read a line that begins with them, in any case and with white space between
# them.
FLOOR_LINE_PREFIX = "MINIMUM PROFIT PER CONTRACT"
FLOOR_LINE_START = FLOOR_LINE_PREFIX + ":"
_WORDS = re.compile(r"\s+".join(FLOOR_LINE_PREFIX.split()), re.IGNORECASE)
# The marks of markdown that an analyst may put around the line. Before a line is read, white space and the marks of
# _END_MARKS are removed at both its ends, and at its start the marks of _START_MARKS (list, heading and quote marks
# as well) and the numbers of a numbered list ("1." or "1)"). White space and the marks of _END_MARKS may also stand
# between the words and the colon, between the colon and the value, between the dollar sign and the number, and between
# the value and the period that may end it.
_START_MARKS = "*_`#>-•+–◦"
_END_MARKS = "*_`"
_NUMBERING = re.compile(r"[0-9]+[.)]")
_SEPARATOR = r"[\s*_`]*"
# A line of these marks (those of _START_MARKS and = ~ | —) and white space alone (a rule such as "---" or "***", a
# code fence) is not a candidate for the floor line: the candidate is the last other non-empty line (floor_candidate).
_RULE_MARKS = _START_MARKS + "=~|—"
# A floor line, once its ends are cleaned: the words (in any case, with white space between them), a colon, then a
# non-negative number with an optional dollar sign, or none (in any case); one point may end the value. Anything else
# after the value (words, a comma, a second number, a range) makes the line malformed, and so do a missing colon and a
# value of another form (a sign, a thousands separator, an exponent).
_FLOOR_LINE = re.compile(
    _WORDS.pattern + _SEPARATOR + ":" + _SEPARATOR
    + r"(?:(?:\$" + _SEPARATOR + r")?(?P<number>[0-9]+(?:\.[0-9]+)?|\.[0-9]+)|(?P<none>none))" + _SEPARATOR + r"\.?",
    re.IGNORECASE)
# The follow-up turn that asks the analyst again when its last line is not a floor line.
FLOOR_REASK_ANALYST = ("Write your briefing again. End your briefing with one line, either MINIMUM PROFIT PER "
                       "CONTRACT: $X or MINIMUM PROFIT PER CONTRACT: none.")
# What the bidder is told of the floor of the event, after the briefing: $X to two decimals, or none.
FLOOR_NOTICE = "MINIMUM PROFIT PER CONTRACT FOR THIS EVENT, SET BY YOUR COMPANY'S ANALYST: {value}"
# What it is told when the analyst gave no valid floor line in two answers: no floor is set, and the analyst is not
# said to have set none.
FLOOR_NOTICE_NOT_SET = "NO MINIMUM PROFIT PER CONTRACT IS SET FOR THIS EVENT"
# The follow-up turn that returns a bid whose price is below its total cost plus the minimum profit. {price}, {cost} and
# {floor} are the bid's price, its total cost at its delivery and quality, and the minimum profit. The minimum is to two
# decimals (it was rounded to the cent when read); the price and the cost are written by dollars().
FLOOR_REASK_BIDDER = (
    "Your bid's price, ${price}, is less than its total cost at its delivery and quality, ${cost}, plus the minimum "
    "profit per contract of ${floor} that your company's analyst set for this event. Submit your bid again. If the bid "
    "you send is also below the minimum, its price will be raised to its total cost plus ${floor}.")
# What the record and the earlier-rounds block print after the price of a bid that the floor raised.
FLOOR_RAISED_NOTE = " (price raised to its total cost plus the minimum profit your company's analyst set)"
# A price within this of the total cost plus the minimum is not below it: a price that is that sum to the cent can
# differ from it in the last place of a float.
FLOOR_TOLERANCE = 1e-9

# The follow-up turn that asks the analyst of brief_cap for a briefing within the limit again. {words} is the number of
# words of the briefing it wrote, and {limit} the limit.
CAP_REASK = ("Your briefing has {words} words; the limit is {limit} words. Write it again within the limit. If it "
             "still has more than {limit} words, the bidder reads only its first {limit} words.")


def word_count(text: str) -> int:
    """The number of words of a text: the pieces between white space."""
    return len(text.split())


def cut_to_words(text: str, limit: int = BRIEFING_WORD_LIMIT) -> str:
    """The text up to the end of its `limit`-th word, as written (its line breaks and spaces are kept), or the whole
    text if it has no more than `limit` words."""
    if limit <= 0:
        return ""
    ends = [match.end() for match in re.finditer(r"\S+", text)]
    return text if len(ends) <= limit else text[:ends[limit - 1]]


def follow_up_turns(first_reply: str, follow_up: str) -> tuple[dict, ...]:
    """The turns that follow the message when an agent is asked again: its own first reply, then the follow-up."""
    return ({"role": "assistant", "content": first_reply}, {"role": "user", "content": follow_up})


def capped_briefing(analyst: ProcurementAnalyst, supplier_id: str, message: str,
                    limit: int = BRIEFING_WORD_LIMIT) -> tuple[AnalystBriefing, dict]:
    """The briefing of the analyst of brief_cap, within `limit` words, and the record of how it came to be.

    The analyst writes its briefing. If it has more than `limit` words, the analyst is asked once more with CAP_REASK
    as a follow-up turn; if the second briefing has more than `limit` words, it is cut to its first `limit` words. The
    record holds the original briefing and its word count, whether the analyst was asked again and the follow-up, the
    second briefing and its word count, whether it was cut, and the word count of the briefing that the bidder reads."""
    first = analyst.write_briefing(supplier_id, message)
    record = {"word_limit": limit, "original": first.text, "original_words": word_count(first.text), "reasked": False,
              "follow_up": None, "second": None, "second_words": None, "cut": False}
    final = first.text
    if record["original_words"] > limit:
        follow_up = CAP_REASK.format(words=record["original_words"], limit=limit)
        second = analyst.write_briefing(supplier_id, message, follow_up_turns(first.text, follow_up))
        final = second.text
        record.update(reasked=True, follow_up=follow_up, second=second.text, second_words=word_count(second.text))
        if record["second_words"] > limit:
            final, record["cut"] = cut_to_words(second.text, limit), True
    record["final_words"] = word_count(final)
    return AnalystBriefing(None, final), record


def last_line(text: str) -> str:
    """The last non-empty line of a text, without the white space around it ("" if there is none)."""
    lines = text.rstrip().split("\n")
    return lines[-1].strip()


def _without_leading_marks(line: str) -> str:
    """The line without the white space, the marks of _START_MARKS and the numbers of a numbered list at its start."""
    while True:
        cleaned = line.lstrip().lstrip(_START_MARKS)
        numbering = _NUMBERING.match(cleaned)
        if numbering:
            cleaned = cleaned[numbering.end():]
        if cleaned == line:
            return line
        line = cleaned


def _cleaned(line: str) -> str:
    """The line without the white space and the marks of markdown at its ends (_START_MARKS at the start,
    _END_MARKS at the end)."""
    line = _without_leading_marks(line)
    while True:
        cleaned = line.rstrip().rstrip(_END_MARKS)
        if cleaned == line:
            return line
        line = cleaned


def _parse_line(line: str) -> tuple[str, float | None]:
    """Read one line as a floor line: ("set", X), ("none", None) or ("malformed", None)."""
    match = _FLOOR_LINE.fullmatch(_cleaned(line))
    if match is None:
        return "malformed", None
    if match.group("none"):
        return "none", None
    return "set", round(float(match.group("number")), 2)


def floor_candidate(text: str) -> str:
    """The line of a text that is read as its floor line: the last non-empty line that is not only marks (a rule such as
    "---" or "***", a code fence), without the white space around it; "" if there is none."""
    for line in reversed(text.split("\n")):
        stripped = line.strip()
        if stripped and not all(ch in _RULE_MARKS or ch.isspace() for ch in stripped):
            return stripped
    return ""


def parse_floor_line(text: str) -> tuple[str, float | None]:
    """Read the floor from the last line of a briefing that is not only marks (floor_candidate): ("set", X) for a line
    like MINIMUM PROFIT PER CONTRACT: $X, X a non-negative number (rounded to the cent, as the notice and the re-ask
    print it), ("none", None) for a line like MINIMUM PROFIT PER CONTRACT: none, and ("malformed", None) for any other
    line, or a text with none. The words and "none" may be in any case, and white space and markdown marks around the
    line or between its parts do not matter (see _FLOOR_LINE for what is a floor line and what is not)."""
    return _parse_line(floor_candidate(text))


def strip_floor_lines(text: str) -> str:
    """The briefing without the lines that the bidder is not to read, and without the white space after what is left.
    A line is removed, wherever it is, if it begins, once the white space, the marks and the list numbers at its start
    are removed (_without_leading_marks), with the words FLOOR_LINE_PREFIX in any case and with white space between
    them, whatever follows; nothing else is removed. Every floor line begins that way."""
    kept = [line for line in text.split("\n") if not _WORDS.match(_without_leading_marks(line))]
    return "\n".join(kept).rstrip()


def floor_notice(minimum: float | None) -> str:
    """The notice that tells the bidder the floor of the event: FLOOR_NOTICE with $X to two decimals, or none."""
    return FLOOR_NOTICE.format(value="none" if minimum is None else f"${minimum:.2f}")


def floor_briefing(analyst: ProcurementAnalyst, supplier_id: str,
                   message: str) -> tuple[AnalystBriefing, dict, float | None]:
    """The briefing of the analyst of floor, what the bidder reads and the floor of the event.

    The analyst writes its briefing, whose last line that is not only marks must be a floor line (parse_floor_line). If
    it is not, the analyst is asked once more with FLOOR_REASK_ANALYST as a follow-up turn, and if the second briefing's
    last line is not a floor line either, no floor is set. The briefing that the bidder reads is the last one the
    analyst wrote, in every case without the lines that strip_floor_lines removes, then a blank line and the notice:
    the floor (floor_notice), none, or FLOOR_NOTICE_NOT_SET when the analyst gave no valid line twice.
    Returns that briefing (the posture is None), the record and the floor (X, or None for none, a malformed line
    included). The record holds the briefings as the analyst wrote them, whether it was asked again, the status of the
    last line ("set", "none" or "malformed", which tells the last two cases apart) and the floor."""
    first = analyst.write_briefing(supplier_id, message)
    raw, final, reasked = [first.text], first, False
    status, minimum = parse_floor_line(first.text)
    if status == "malformed":
        final = analyst.write_briefing(supplier_id, message, follow_up_turns(first.text, FLOOR_REASK_ANALYST))
        raw.append(final.text)
        reasked = True
        status, minimum = parse_floor_line(final.text)
    body = strip_floor_lines(final.text)
    notice = FLOOR_NOTICE_NOT_SET if status == "malformed" else floor_notice(minimum)
    shown = (body + "\n\n" if body else "") + notice
    record = {"raw_briefings": raw, "reasked": reasked, "follow_up": FLOOR_REASK_ANALYST if reasked else None,
              "floor_line": floor_candidate(final.text), "floor_status": status, "minimum_profit": minimum}
    return AnalystBriefing(None, shown), record, minimum


def dollars(value: float) -> str:
    """A price or a cost as the re-ask of the bidder prints it, without the dollar sign: two decimals if the value is a
    whole number of cents (within 1e-9, the tolerance of below_floor), otherwise up to nine decimals, without the zeros
    at the end ("114.645"), so that the figures show why the price is below the total cost plus the minimum."""
    if abs(value - round(value, 2)) <= 1e-9:
        return f"{round(value, 2):.2f}"
    return f"{value:.9f}".rstrip("0")


def floor_reask_text(price: float, cost: float, minimum: float) -> str:
    """The follow-up turn that returns a bid priced below its total cost plus the minimum profit (FLOOR_REASK_BIDDER):
    the price and the cost as dollars() writes them, the minimum to two decimals."""
    return FLOOR_REASK_BIDDER.format(price=dollars(price), cost=dollars(cost), floor=f"{minimum:.2f}")


def below_floor(price: float, cost: float, minimum: float) -> bool:
    """Whether a price is below the total cost plus the minimum profit (strictly, up to FLOOR_TOLERANCE)."""
    return price < cost + minimum - FLOOR_TOLERANCE
