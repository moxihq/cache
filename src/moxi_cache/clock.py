"""What a cache read gives back: the `Clock`, its kinds, its reason codes, and the rules
every harness's clock shares.

A read gives one of five kinds:

* `warm`: a clock under its lifetime;
* `cold`: a clock at or past its lifetime, or a next request that starts a cache of its
  own (a changed model, a rollout Codex compressed);
* `fresh`: positive evidence that the session has made no request, so there is no cache
  to have expired;
* `no-cache`: nothing to count (a free model, a model with no lifetime row, a harness
  cache has no clock for);
* `unknown`: no verdict could be read with evidence.

Every `no-cache` and `unknown` carries a `code` from `CODES`, and so does a `cold` that
has a reason. A caller branches on the code, never on the reason's words.

**One skew rule for every clock.** A last request stamped more than `SKEW_LIMIT` seconds
after now is `unknown`: the system clock may have stepped back, and a negative age read as
zero would read warm. Within the limit the age is zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

WARM, COLD, FRESH, NO_CACHE, UNKNOWN = "warm", "cold", "fresh", "no-cache", "unknown"

#: The kinds a read can give.
KINDS = (WARM, COLD, FRESH, NO_CACHE, UNKNOWN)


class _Missing:
    """An argument the caller did not pass, kept apart from None."""

    def __repr__(self):
        return "MISSING"


MISSING = _Missing()

#: How far in the future a last request may be stamped before the read is `unknown`.
SKEW_LIMIT = 5.0

#: The backward read's block, and how much of a record it reads before it gives up. A
#: read that reaches the limit without the line it needs is `unknown`, never `fresh`: a
#: long tail with no request is not evidence of no request. Read at each call, so a test
#: may lower them.
CLOCK_BLOCK = 64 * 1024
CLOCK_SCAN_LIMIT = 64 * 1024 * 1024

#: Where a Claude read's lifetime came from (`Clock.ttl_source`): the cache tier the
#: requests wrote, or config `claude.ttl_seconds` when no line read named one.
TTL_TIER, TTL_CONFIG = "tier", "config"

# ---------------------------------------------------------------------------- codes
# Every reason code a read can carry. A `no-cache` or `unknown` always carries one.

#: `no-cache`: the harness is not one cache has a clock for.
UNKNOWN_HARNESS = "unknown-harness"
#: `no-cache`: the next request runs on a free model.
FREE = "free"
#: `no-cache`: the model has no lifetime row, so there is no clock to count by.
NO_ROW = "no-row"

#: `unknown`: an exception inside the read.
READ_FAILED = "read-failed"
#: `unknown`: no session id, and no word from the caller that the next delivery starts one.
NO_SESSION_YET = "no-session-yet"
#: `unknown`: an id with a path separator, a glob character or a leading dot.
BAD_ID = "bad-id"
#: `unknown`: the caller named no place to read this harness's record from.
NO_SOURCE = "no-source"
#: `unknown`: the place to read from is missing or cannot be read.
SOURCE_UNREADABLE = "source-unreadable"
#: `unknown`: no record of the session, and nothing shows it has made no request.
NO_RECORD = "no-record"
#: `unknown`: the record was found but could not be read.
UNREADABLE = "unreadable"
#: `unknown`: the read reached its byte limit before the evidence it needs.
SCAN_LIMIT = "scan-limit"
#: `unknown`: the last request has no readable time.
NO_STAMP = "no-stamp"
#: `unknown`: lines that will not parse, and no request that can be read.
MALFORMED = "malformed"
#: `unknown`: the last request is stamped too far in the future.
FUTURE_STAMP = "future-stamp"
#: `unknown`: cache's own config file could not be read.
CONFIG_ERROR = "config-error"
#: `unknown`: the lifetime table (shipped rows and the user's) did not load.
TABLE_NOT_LOADED = "table-not-loaded"
#: `unknown`: a record line or row in a shape cache does not recognise.
UNRECOGNISED = "unrecognised"
#: `unknown`: a Codex rollout that is not the thread's whole history.
NOT_WHOLE = "not-whole"
#: `unknown`: a thread's history split across more than one rollout file.
SPLIT = "split"
#: `unknown`: requests in the record, none under the session's own key.
NOT_OWN = "not-own"
#: `unknown`: a record written by a harness version whose lines are not read as a clock.
OLDER_HARNESS = "older-harness"
#: `unknown`: no request shows, but nothing confirms none was made.
NO_REQUEST = "no-request"
#: `unknown`: which model the next request will use cannot be told.
NO_PAIR = "no-pair"
#: `unknown`: a Codex rollout holding a settings line of another thread.
OTHER_THREAD = "other-thread"
#: `unknown`: an OpenCode session with a revert pending.
REVERT = "revert"
#: `unknown`: a measured lifetime, and no drift record to read it from.
NO_DRIFT_RECORD = "no-drift-record"
#: `unknown`: the drift record cannot be accepted, or a lowering could not be written.
DRIFT_HELD = "drift-held"

#: `cold`: the model or provider changed since the last request.
CHANGED = "changed"
#: `cold`: a Codex rollout compressed after a week idle, so there is no age.
COMPRESSED = "compressed"

#: Every reason code a read can carry.
CODES = frozenset({
    UNKNOWN_HARNESS, FREE, NO_ROW,
    READ_FAILED, NO_SESSION_YET, BAD_ID, NO_SOURCE, SOURCE_UNREADABLE, NO_RECORD,
    UNREADABLE, SCAN_LIMIT, NO_STAMP, MALFORMED, FUTURE_STAMP, CONFIG_ERROR,
    TABLE_NOT_LOADED, UNRECOGNISED, NOT_WHOLE, SPLIT, NOT_OWN, OLDER_HARNESS, NO_REQUEST,
    NO_PAIR, OTHER_THREAD, REVERT, NO_DRIFT_RECORD, DRIFT_HELD,
    CHANGED, COMPRESSED,
})


@dataclass
class Clock:
    """What a read found: a kind, the cache's age when there is a clock, the lifetime the
    read used (`ttl_seconds`, None when it used none) and where it came from
    (`ttl_source`, Claude only), what the lifetime was looked up by (`key`, in words),
    and why (`reason` in words, `code` from `CODES`) when the read is `unknown`,
    `no-cache`, or a `cold` with no lifetime.

    A Codex read on a measured row may also carry `lowering`, a new lowering of that
    model's lifetime: through a learning drift store it was written before the read
    returned; through a read-only store it was observed and counted in this read only,
    and nothing was written;
    `notes`, journal lines about cold landings under the documented minimum; and
    `judged`, (thread id, the newest turn the drift scan judged)."""

    kind: str
    age_seconds: float | None = None
    reason: str | None = None
    ttl_seconds: float | None = None
    key: str | None = None
    basis: str | None = None             # "documented" or "measured", with a lifetime
    provider: str | None = None
    model: str | None = None
    lowered_from: float | None = None    # a measured row's shipped number, when lowered
    lowering: dict | None = None
    notes: list | None = None
    judged: tuple | None = None
    code: str | None = None
    ttl_source: str | None = None

    def as_dict(self) -> dict:
        out = asdict(self)
        if out["judged"] is not None:
            out["judged"] = list(out["judged"])
        return out


def unknown(code: str, reason: str) -> Clock:
    return Clock(UNKNOWN, reason=reason, code=code)


def pair_text(provider, model) -> str:
    return f"{provider}/{model}"


def future_stamp(ts: float, now: float) -> str | None:
    """The reason a stamp is too far in the future to read, or None."""
    if ts - now > SKEW_LIMIT:
        return ("the last request is stamped in the future; the system clock may have "
                "stepped back")
    return None


def span(seconds: float) -> str:
    """Compact: `58m`, `2h10m`, `2d3h`."""
    minutes = int(max(0.0, seconds) // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h{minutes:02d}m" if minutes else f"{hours}h"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours}h" if hours else f"{days}d"


#: How the Codex and OpenCode readers' reason for a model with no lifetime row begins.
NO_ROW_WORDS = "no documented cache lifetime is known for "


def no_row(provider, model) -> str:
    return f"{NO_ROW_WORDS}{provider}/{model}"


def free_reason(provider, model) -> str:
    return (f"{provider}/{model} is a free model, so a cold cache costs nothing and is not "
            "counted")


def bad_id(what: str, value) -> Clock:
    return unknown(BAD_ID, f"the {what} {value!r} is not one that can be looked up")


__all__ = ["WARM", "COLD", "FRESH", "NO_CACHE", "UNKNOWN", "KINDS", "MISSING", "SKEW_LIMIT",
           "CLOCK_BLOCK", "CLOCK_SCAN_LIMIT", "TTL_TIER", "TTL_CONFIG", "CODES", "Clock",
           "unknown", "pair_text", "future_stamp", "span", "no_row", "free_reason",
           "NO_ROW_WORDS", "bad_id"]
