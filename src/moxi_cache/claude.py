"""The Claude clock: the end of a session's last request, read from its own transcript.

**The clock** is the session's main conversation: the timestamp of the last line of its
transcript with `type == "assistant"`, not `isSidechain: true`, and a `message` whose
`model` is not `<synthetic>` and that has a `usage` dict. A subagent writes its own file
and its lines are sidechain, so a subagent's work never moves it. cache finds the file and
reads it backwards through sessions (`probe_root`, `read_back`), and judges each line
itself, an unterminated last line included. The clock reads young by the length of the
last response, because the stamp is written when a response ends and the lifetime runs
from when the request started; a read inside that margin past the lifetime reads warm.
Declared.

**What each ending of the read means.** The newest line that meets the rule is the clock;
one with no readable stamp is `unknown`. A read that ended at the start of the file with
no such line is `fresh`, unless a line on the way would not parse, which is `unknown`. A
read that ended at its byte limit with no such line is `unknown`, never `fresh`: a long
tail with no request is not evidence of no request.

**`fresh` needs positive evidence.** The absence of evidence is `unknown`. Three things
are evidence: (1) a transcript found and read to its start, with no line that meets the
rule; (2) the caller's word that its next delivery starts a new session, with no session
id (`new_session`); (3) no transcript, on the session's own word: the caller passed the
root the session itself stated (`projects_root`), that root was read and holds no file for
the id, and no prompt or turn end has been seen in the session (`turned` false). A Claude
session writes no transcript until its first prompt. A missing transcript with anything
less is `unknown`, and its reason names the root searched. For a session that stated no
root, cache reads under sessions' root for it (config `claude.projects`).

**The lifetime** comes from the cache tier the requests wrote. Every usage block names
it: `usage.cache_creation.ephemeral_1h_input_tokens` or `ephemeral_5m_input_tokens`, above
zero. The lifetime is the tier of the newest line that meets the rule and names a tier, at
or before the clock line: one hour or five minutes, the shorter when a line shows both. A
request that only read the cache wrote no tier, so when the clock line names none the same
backward read goes on past it, within the same byte limit, to the first line that does. A
read that reaches the start or the limit with no tiered line uses config
`claude.ttl_seconds`, and the `Clock` says which it used (`ttl_source`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from moxi_sessions.api import claude as transcripts
from moxi_sessions.api import projects_root as sessions_projects_root
from moxi_sessions.api import read_back

from . import clock as _clock
from .clock import (COLD, FRESH, TTL_CONFIG, TTL_TIER, WARM, Clock, bad_id, future_stamp,
                    unknown)

HARNESS = "claude"

#: The two tiers a usage block can name, and their lifetimes.
TIER_1H = ("ephemeral_1h_input_tokens", 3600)
TIER_5M = ("ephemeral_5m_input_tokens", 300)

#: The reason a read with no session id is `unknown`.
NO_SESSION_YET = "no session id reported yet"

#: What one line says: it meets the rule (with its stamp, maybe None, and its tier, maybe
#: None), it does not, or it will not parse.
_MEETS, _OTHER, _MALFORMED = "meets", "other", "malformed"


def _count(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def tier_of(message: dict) -> int | None:
    """The lifetime, in seconds, of the cache tier a request wrote; None when it wrote
    none. The shorter when it names both."""
    usage = message.get("usage")
    creation = usage.get("cache_creation") if isinstance(usage, dict) else None
    if not isinstance(creation, dict):
        return None
    found = [seconds for key, seconds in (TIER_5M, TIER_1H) if _count(creation.get(key))]
    return min(found) if found else None


def judge(raw: bytes) -> tuple:
    """One transcript line against the clock's rule: (verdict, stamp, tier)."""
    obj = transcripts.parse_line(raw)
    if obj is None:
        return _MALFORMED, None, None
    if obj.get("type") != "assistant" or obj.get("isSidechain") is True:
        return _OTHER, None, None
    message = obj.get("message")
    if not isinstance(message, dict):
        return _MALFORMED, None, None
    if message.get("model") == "<synthetic>" or not isinstance(message.get("usage"), dict):
        return _OTHER, None, None
    return _MEETS, transcripts.parse_timestamp(obj.get("timestamp")), tier_of(message)


@dataclass(frozen=True)
class Read:
    """What one backward read of a transcript found.

    `outcome` is `clock` (with `ts`, and `tier` when a line at or before the clock line
    named one), `none` (read to the start, no line meets the rule), `limit` (the limit
    stopped the read with no such line), `nostamp` (the newest such line has no readable
    stamp) or `malformed` (read to the start, no such line, and a line that would not
    parse). `tier_ended` says how the tier read ended when the clock line named none:
    `start` or `limit`."""

    outcome: str
    ts: float | None = None
    tier: int | None = None
    tier_ended: str | None = None


def last_request(path, *, limit: int | None = None) -> Read:
    """Read a transcript backwards to its clock line, and past it to the tier. Raises
    OSError on a read that fails."""
    limit = _clock.CLOCK_SCAN_LIMIT if limit is None else limit
    malformed = False
    with read_back(path, block=_clock.CLOCK_BLOCK, limit=limit) as reader:
        lines = iter(reader)
        for raw, _unterminated in lines:
            verdict, ts, tier = judge(raw)
            if verdict == _MALFORMED:
                malformed = True
                continue
            if verdict != _MEETS:
                continue
            if ts is None:
                return Read("nostamp")
            if tier is not None:
                return Read("clock", ts, tier)
            for older, _ in lines:
                verdict, _ts, tier = judge(older)
                if verdict == _MEETS and tier is not None:
                    return Read("clock", ts, tier)
            return Read("clock", ts, None, tier_ended=reader.ended)
        if reader.ended == "limit":
            return Read("limit")
    return Read("malformed") if malformed else Read("none")


def read(session_id, *, now: float, projects_root=None, turned: bool = True,
         new_session: bool = False) -> Clock:
    """One Claude session's clock. Raises nothing the caller does not catch."""
    if not session_id:
        if new_session:
            return Clock(FRESH)          # the caller's next delivery starts a new session
        return unknown(_clock.NO_SESSION_YET, NO_SESSION_YET)
    if not isinstance(session_id, str) or not transcripts.SAFE_ID_RE.match(session_id):
        return bad_id("session id", session_id)
    own = bool(projects_root)
    if own:
        root = Path(projects_root)
    else:
        from moxi.config import ConfigError

        try:
            root = Path(sessions_projects_root(None))
        except ConfigError as e:
            return unknown(_clock.CONFIG_ERROR, f"sessions' config could not be read ({e})")
    probe = transcripts.probe_root(root, session_id)
    if not probe.hit:
        readable = probe.outcome == transcripts.MISS
        if own and readable and not turned:
            return Clock(FRESH)          # no transcript, on the session's own word
        where = root if readable else f"{root} (missing or unreadable)"
        return unknown(_clock.NO_RECORD, f"no transcript for session {session_id} under "
                                         f"{where}, and nothing shows it has made no "
                                         "request")
    try:
        found = last_request(probe.path)
    except OSError as e:
        return unknown(_clock.UNREADABLE,
                       f"the transcript could not be read ({e.strerror or e})")
    if found.outcome == "none":
        return Clock(FRESH)              # read whole, and no request in it
    if found.outcome == "limit":
        return unknown(_clock.SCAN_LIMIT,
                       f"no request found in the last "
                       f"{_clock.CLOCK_SCAN_LIMIT // (1024 * 1024)} MB of the transcript")
    if found.outcome == "nostamp":
        return unknown(_clock.NO_STAMP, "the transcript's last request has no readable time")
    if found.outcome == "malformed":
        return unknown(_clock.MALFORMED, "the transcript has lines that will not parse and "
                                         "no request that can be read")
    skew = future_stamp(found.ts, now)
    if skew:
        return unknown(_clock.FUTURE_STAMP, skew)
    if found.tier is not None:
        ttl, source = found.tier, TTL_TIER
    else:
        from moxi.config import ConfigError

        from .settings import current

        try:
            ttl, source = current()["claude.ttl_seconds"], TTL_CONFIG
        except ConfigError as e:
            return unknown(_clock.CONFIG_ERROR, f"cache's config could not be read ({e})")
    age = max(0.0, now - found.ts)
    return Clock(COLD if age >= ttl else WARM, age_seconds=age, ttl_seconds=ttl,
                 ttl_source=source)


__all__ = ["HARNESS", "TIER_1H", "TIER_5M", "Read", "judge", "tier_of", "last_request",
           "read"]
