"""The Codex clock: the end of a thread's last request, read from its rollout.

**THE EVIDENCE RULE.** A Codex cache reads `warm` or `fresh` only from a whole-history
rollout recognised line by line, in the format of the Codex version the reader was checked
against. Anything it does not recognise is `unknown`. Recognition is a whitelist, never a
deny-list.

* **Finding it.** sessions lists every file under `<cache_source>/*/*/*/` whose name
  parses by Codex's own rule as `rollout-<timestamp>-<thread>[_<rollout id>].jsonl[.zst]`
  for this thread (`find_rollout`), at every read. None: no rollout. One `.jsonl`: read.
  One `.jsonl.zst`: Codex compressed it after 7 idle days, so `cold` with no age. More than
  one: the thread's history is split (a revert, or one rollout in two states), `unknown`.
* **The whole-history check, on line 1.** Line 1 is the file's one `session_meta`, its
  `id` and `session_id` are the thread id, its `history_mode` is `legacy` or `paginated`,
  and every payload key is on `META_KEYS`. A key that refers to history kept elsewhere
  (`history_base`, `forked_from_id`, ...) is not on it, and neither is any key not seen.
* **Reading it.** Backwards through sessions' `read_back` in 64 KB blocks, to the earliest
  line the verdict needs, and line 1 (`head=True`), with a 64 MB limit. Every line covered
  must parse as a JSON object with a `timestamp` and a `type` on `TOP_KINDS`, and an
  `event_msg` or `response_item` must carry a `payload.type` on its list. The one
  exception is a last line with no newline that does not parse: Codex is writing it, and
  it is skipped.
* **The last request** is the last `token_usage_record` whose `thread_id` and
  `session_id` are both the thread id: the thread's own cache key. Its `timestamp`, the
  end of the response, is the clock's stamp. A guardian review or an internal child uses
  another key and never moves the clock.
* **The two pairs.** The last model is the `model` of the record's own turn's
  `turn_context`; the last provider is the latest `thread_settings_applied` of this thread
  at or before the record, else line 1's `model_provider`. The next model and provider are
  the latest of each in the whole file. Pairs that differ read `cold`; equal pairs are
  looked up in the lifetime table.
* **`fresh`** only for a thread the launcher's own start attempt created after a read
  that found no record (`turned` false), with no rollout yet or a whole-history rollout of
  fresh kinds alone, made by a Codex version the reader was checked against; or, with no
  thread id at all, on the caller's word that its attempt starts one (`new_session`).

**Drift.** On a read whose last pair is on a measured row, the scan also judges the
thread's newest turns: a turn whose first request found the cache cold (cached tokens
under half of the previous request's input) at a gap the row still counted warm lowers
that model's lifetime (`judge_drift`). The read reports the observation on the `Clock`; a
learning drift store writes it before the read returns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from moxi_sessions.api import claude as transcripts
from moxi_sessions.api import codex as rollouts
from moxi_sessions.api import read_back

from . import clock as _clock
from . import table as _table
from .clock import (COLD, FRESH, MISSING, NO_CACHE, WARM, Clock, bad_id, free_reason,
                    future_stamp, no_row, pair_text, span, unknown)
from .drift import DriftView

HARNESS = "codex"

#: The Codex versions whose new threads read as `fresh`. A thread made by any other
#: version reads `unknown` until its first request completes.
CHECKED_VERSIONS = ("0.153.0",)

#: Every top-level line kind recognised in a rollout. Codex 0.153.0 can also write
#: `inter_agent_communication`, `security_risk_score` and `realtime_item`; none has been
#: seen, so each stays unrecognised until the reader is checked against one.
TOP_KINDS = ("session_meta", "turn_context", "world_state", "token_usage_record",
             "compacted", "event_msg", "response_item")

#: `event_msg` payload types Codex 0.153.0 persists in a rollout (its rollout policy's
#: `should_persist_event_msg`), by their serialized names.
EVENT_KINDS = ("item_completed", "token_count", "thread_goal_updated", "thread_rolled_back",
               "turn_aborted", "task_started", "task_complete", "thread_settings_applied",
               "user_message", "agent_message", "agent_reasoning",
               "agent_reasoning_raw_content", "entered_review_mode", "exited_review_mode",
               "patch_apply_end", "context_compacted", "mcp_tool_call_end", "web_search_end",
               "image_generation_end", "sub_agent_activity")

#: `response_item` payload types Codex 0.153.0 persists (`should_persist_response_item`).
RESPONSE_KINDS = ("message", "agent_message", "reasoning", "local_shell_call",
                  "function_call", "tool_search_call", "function_call_output",
                  "tool_search_output", "custom_tool_call", "custom_tool_call_output",
                  "web_search_call", "image_generation_call", "compaction",
                  "context_compaction")

#: The `session_meta` payload keys of a whole-history rollout: every key on the rollouts
#: of 0.153.0 the reader was checked against. A key that refers to history kept elsewhere
#: is not here.
META_KEYS = ("session_id", "id", "timestamp", "cwd", "originator", "cli_version", "source",
             "thread_source", "model_provider", "base_instructions", "history_mode",
             "multi_agent_version", "context_window", "git")

HISTORY_MODES = ("legacy", "paginated")

#: Drift: the smallest previous prompt an observation is made from, and the share of it a
#: request must read from the cache to count as warm.
DRIFT_MIN_INPUT = 16_384
DRIFT_SHARE = 0.5
#: A cold landing sets the lifetime to this share of the gap, rounded down to the minute.
DRIFT_LOWER_TO = 0.8

# The reasons.
NO_SESSION_YET = "no thread id reported yet"
NO_ROOT = "no Codex sessions directory was named for this thread"
SPLIT = "this thread's history is split across more than one rollout file"
COMPRESSED = "its rollout has been idle at least 7 days"
NOT_WHOLE = "this rollout is not the thread's whole history"
UNRECOGNISED = "the rollout has a line the reader does not recognise"
NOT_OWN = "the rollout has requests, none under this thread's own key"
OLDER_CODEX = ("this rollout was written by a Codex older than 0.153.0, whose lines are not "
               "read as a clock")
NO_REQUEST = "this thread shows no request, but nothing confirms it has made none"
NO_PAIR = "which model the next request will use cannot be told"
CHANGED = ("the model or provider changed since the last request, so the next request "
           "starts a cache of its own")
OTHER_THREAD = "the rollout has a settings line of another thread"
NO_STAMP = "the rollout's last request has no readable time"
NO_DRIFT = ("no cache drift record was given, so the measured lifetime cannot be read")


# --------------------------------------------------------------------------- lines

@dataclass
class Line:
    """What the reader keeps of one rollout line."""
    kind: str
    sub: str | None
    ts: float | None
    raw_ts: object
    payload: dict


class Unrecognised(Exception):
    pass


def judge_line(raw: bytes) -> Line:
    """One complete line -> Line, or Unrecognised."""
    try:
        obj = json.loads(raw)
    except ValueError as e:
        raise Unrecognised("will not parse") from e
    if not isinstance(obj, dict):
        raise Unrecognised("not an object")
    kind = obj.get("type")
    stamp = obj.get("timestamp")
    if kind not in TOP_KINDS or not isinstance(stamp, str):
        raise Unrecognised(f"kind {kind!r}")
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        raise Unrecognised("no payload")
    sub = None
    if kind == "event_msg":
        sub = payload.get("type")
        if sub not in EVENT_KINDS:
            raise Unrecognised(f"event {sub!r}")
    elif kind == "response_item":
        sub = payload.get("type")
        if sub not in RESPONSE_KINDS:
            raise Unrecognised(f"response item {sub!r}")
    return Line(kind=kind, sub=sub, ts=transcripts.parse_timestamp(stamp), raw_ts=stamp,
                payload=_slim(kind, sub, payload))


def _slim(kind, sub, payload: dict) -> dict:
    """The fields the clock reads; the rest of a line (prompts, outputs) is dropped."""
    if kind == "token_usage_record":
        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        return {"thread_id": payload.get("thread_id"),
                "session_id": payload.get("session_id"),
                "turn_id": payload.get("turn_id"),
                "response_id": payload.get("response_id"),
                "input": usage.get("input_tokens"),
                "cached": usage.get("cached_input_tokens")}
    if kind == "turn_context":
        return {"turn_id": payload.get("turn_id"), "model": payload.get("model")}
    if kind == "session_meta":
        return dict(payload)
    if sub == "thread_settings_applied":
        settings = payload.get("thread_settings")
        settings = settings if isinstance(settings, dict) else {}
        return {"thread_id": payload.get("thread_id"),
                "model": settings.get("model"),
                "provider": settings.get("model_provider_id")}
    if sub in ("task_started", "task_complete", "turn_aborted"):
        return {"turn_id": payload.get("turn_id")}
    if sub == "item_completed":
        item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
        return {"item_type": item.get("type")}
    if kind == "response_item" and sub == "message":
        return {"role": payload.get("role")}
    return {}


def _is_tsa(line: Line) -> bool:
    return line.kind == "event_msg" and line.sub == "thread_settings_applied"


def _is_record(line: Line) -> bool:
    return line.kind == "token_usage_record"


def _own(line: Line, thread_id: str) -> bool:
    return (_is_record(line) and line.payload.get("thread_id") == thread_id
            and line.payload.get("session_id") == thread_id)


def _is_start(line: Line) -> bool:
    return line.kind == "event_msg" and line.sub == "task_started"


def _breaks_continuation(line: Line) -> bool:
    """A line after which the next request may read a low cache for a reason other than
    expiry: a compaction, a rollback, an aborted turn."""
    return (line.kind == "compacted"
            or (line.kind == "event_msg"
                and line.sub in ("context_compacted", "thread_rolled_back", "turn_aborted")))


@dataclass
class Scan:
    """What one read covered: line 1, and the covered lines oldest first."""
    meta: Line | None = None
    lines: list = field(default_factory=list)
    reached_start: bool = False
    hit_limit: bool = False


# --------------------------------------------------------------------------- the read

def read(thread_id, *, cache_source, turned, new_session, now, drift=MISSING,
         bound_at=None, judged=None, limit=None) -> Clock:
    """One Codex thread's clock. `drift` is the lowerings in force (a `DriftView`, a
    mapping, or MISSING, which reads a measured row `unknown`); `judged` the newest turn id
    already judged for drift on this thread, or None. Raises nothing the caller does not
    catch."""
    try:
        table = _table.load_table()
    except _table.TableError as e:
        return unknown(_clock.TABLE_NOT_LOADED, f"{_table.NOT_LOADED} ({e})")
    if not thread_id:
        if new_session:
            return Clock(FRESH)          # no thread yet: the launcher's own attempt starts it
        return unknown(_clock.NO_SESSION_YET, NO_SESSION_YET)
    if not isinstance(thread_id, str) or not transcripts.SAFE_ID_RE.match(thread_id):
        return bad_id("thread id", thread_id)
    if not cache_source:
        return unknown(_clock.NO_SOURCE, NO_ROOT)
    root = Path(cache_source)
    try:
        found = rollouts.find_rollout(root, thread_id)
    except OSError:
        return unknown(_clock.SOURCE_UNREADABLE,
                       f"the Codex sessions directory {root} is missing or unreadable")
    if not found:
        if not turned:
            return Clock(FRESH)          # the launcher's own new thread: no rollout yet
        return unknown(_clock.NO_RECORD, f"no rollout for thread {thread_id} under {root}, "
                                         "and nothing shows it has made no request")
    if len(found) > 1:
        return unknown(_clock.SPLIT, SPLIT)
    path = found[0]
    if path.name.endswith(".zst"):
        return Clock(COLD, reason=COMPRESSED, code=_clock.COMPRESSED)
    limit = _clock.CLOCK_SCAN_LIMIT if limit is None else limit
    try:
        with read_back(path, block=_clock.CLOCK_BLOCK, limit=limit, head=True) as reader:
            return _read_open(reader, thread_id, table=table, turned=turned, now=now,
                              drift=drift, bound_at=bound_at, judged=judged)
    except OSError as e:
        return unknown(_clock.UNREADABLE, f"the rollout could not be read ({e.strerror or e})")


def _whole_history(meta: Line | None, thread_id: str) -> bool:
    if meta is None or meta.kind != "session_meta":
        return False
    p = meta.payload
    if set(p) - set(META_KEYS):
        return False
    return (p.get("id") == thread_id and p.get("session_id") == thread_id
            and p.get("history_mode") in HISTORY_MODES)


class _Need:
    """When the backwards scan has covered every line the verdict needs: the last own
    record, its turn's `turn_context`, and a settings line at or before it (else the scan
    runs to line 1). With drift on, also the previous own record of the oldest turn to
    judge, that record's turn start, and a settings line at or before it."""

    def __init__(self, thread_id, judged):
        self.thread_id = thread_id
        self.judged = judged
        self.record = None
        self.context = False
        self.settings_before = False
        self.drift = False
        self.newest_turn = None
        self.boundary_seen = False
        self.prev_record = None
        self.prev_start = False
        self.prev_settings = False

    def feed(self, line: Line) -> None:
        if self.record is None:
            if _own(line, self.thread_id):
                self.record = line
            return
        if (line.kind == "turn_context"
                and line.payload.get("turn_id") == self.record.payload.get("turn_id")):
            self.context = True
        if _is_tsa(line):
            self.settings_before = True
        if self.drift:
            self.feed_drift(line)

    def feed_drift(self, line: Line) -> None:
        turn = line.payload.get("turn_id")
        if self.prev_record is None:
            if not self.boundary_seen:
                if self.judged is None:
                    if _is_start(line) and turn == self.newest_turn:
                        self.boundary_seen = True
                    return
                if turn is not None and turn == self.judged:
                    self.boundary_seen = True
            if self.boundary_seen and _own(line, self.thread_id):
                self.prev_record = line
            return
        if _is_start(line) and turn == self.prev_record.payload.get("turn_id"):
            self.prev_start = True
        if _is_tsa(line):
            self.prev_settings = True

    def done(self) -> bool:
        if self.record is None or not self.context or not self.settings_before:
            return False
        if not self.drift:
            return True
        return self.prev_record is not None and self.prev_start and self.prev_settings


def _limit_mb() -> int:
    return _clock.CLOCK_SCAN_LIMIT // (1024 * 1024)


def _read_open(reader, thread_id, *, table, turned, now, drift, bound_at, judged) -> Clock:
    head = reader.first
    try:
        meta = judge_line(head) if head is not None else None
    except Unrecognised:
        meta = None
    if meta is None or not _whole_history(meta, thread_id):
        return unknown(_clock.NOT_WHOLE, NOT_WHOLE)
    need = _Need(thread_id, judged)
    newest_first: list = []
    measured_checked = False
    for raw, last in reader:
        try:
            line = judge_line(raw)
        except Unrecognised:
            if last:
                try:
                    json.loads(raw)
                except ValueError:
                    continue             # the line Codex is writing as the read runs
            return unknown(_clock.UNRECOGNISED, UNRECOGNISED)
        if line.kind == "session_meta":
            return unknown(_clock.NOT_WHOLE, NOT_WHOLE)
        if _is_tsa(line) and line.payload.get("thread_id") != thread_id:
            return unknown(_clock.OTHER_THREAD, OTHER_THREAD)
        newest_first.append(line)
        need.feed(line)
        if (need.record is not None and need.context and need.settings_before
                and not measured_checked):
            measured_checked = True
            need.newest_turn = need.record.payload.get("turn_id")
            if _last_pair_measured(newest_first, meta, thread_id, table):
                need.drift = True
                _refeed_for_drift(need, newest_first)
        if need.done():
            break
    reached_start = reader.ended == "start"
    lines = list(reversed(newest_first))
    scan = Scan(meta=meta, lines=lines, reached_start=reached_start,
                hit_limit=reader.ended == "limit")
    if need.record is None:
        if not reached_start:
            return unknown(_clock.SCAN_LIMIT, f"no request found in the last {_limit_mb()} MB "
                                              "of the rollout")
        return _no_record(scan, thread_id, turned)
    if not (need.context and need.settings_before) and not reached_start:
        return unknown(_clock.SCAN_LIMIT, f"the rollout's last {_limit_mb()} MB end before "
                                          "the evidence the reader needs")
    return _clock_from(scan, need.record, thread_id, table=table, now=now, drift=drift,
                       bound_at=bound_at, judged=judged)


def _refeed_for_drift(need: _Need, newest_first: list) -> None:
    """Drift was switched on mid-scan: replay the lines already read past the record."""
    past = False
    for line in newest_first:
        if past:
            need.feed_drift(line)
        elif line is need.record:
            past = True


def _no_record(scan: Scan, thread_id, turned) -> Clock:
    """The whole file read and no record of the thread's own."""
    records = [ln for ln in scan.lines if _is_record(ln)]
    if records:
        return unknown(_clock.NOT_OWN, NOT_OWN)
    if any(ln.kind == "event_msg" and ln.sub == "token_count" for ln in scan.lines):
        return unknown(_clock.OLDER_HARNESS, OLDER_CODEX)
    if turned:
        return unknown(_clock.NO_REQUEST, NO_REQUEST)
    if scan.meta.payload.get("cli_version") not in CHECKED_VERSIONS:
        return unknown(_clock.NO_REQUEST, NO_REQUEST)
    if all(_fresh_kind(ln, thread_id) for ln in scan.lines):
        return Clock(FRESH)
    return unknown(_clock.NO_REQUEST, NO_REQUEST)


def _fresh_kind(line: Line, thread_id) -> bool:
    """A line a new thread writes before its first response."""
    if line.kind in ("turn_context", "world_state"):
        return True
    if line.kind == "event_msg":
        if line.sub in ("task_started", "user_message"):
            return True
        if line.sub == "thread_settings_applied":
            return line.payload.get("thread_id") == thread_id
        if line.sub == "item_completed":
            return line.payload.get("item_type") == "UserMessage"
        return False
    if line.kind == "response_item" and line.sub == "message":
        return line.payload.get("role") in ("user", "developer")
    return False


def _pair_at(lines: list, index: int, meta: Line, whole: bool) -> tuple:
    """The (provider, model) the record at `lines[index]` used, by the last-pair rule: the
    model of its own turn's `turn_context`, the provider of the latest settings line at
    or before it, else line 1's. A half that cannot be read is None; so is line 1's
    provider when the covered lines do not reach back to it."""
    record = lines[index]
    turn = record.payload.get("turn_id")
    model = None
    for ln in reversed(lines[:index]):
        if ln.kind == "turn_context" and ln.payload.get("turn_id") == turn:
            model = ln.payload.get("model")
            break
    provider = None
    for ln in reversed(lines[:index]):
        if _is_tsa(ln):
            provider = ln.payload.get("provider")
            break
    else:
        provider = meta.payload.get("model_provider") if whole else None
    return (provider if isinstance(provider, str) and provider else None,
            model if isinstance(model, str) and model else None)


def _next_pair(lines: list, meta: Line, whole: bool) -> tuple:
    """The pair the thread's next request will use: the model of the latest
    `turn_context` or settings line, the provider of the latest settings line, else line
    1's."""
    model = provider = None
    for ln in reversed(lines):
        if model is None and (ln.kind == "turn_context" or _is_tsa(ln)):
            model = ln.payload.get("model")
            if not isinstance(model, str) or not model:
                return None, None
        if provider is None and _is_tsa(ln):
            provider = ln.payload.get("provider")
            if not isinstance(provider, str) or not provider:
                return None, None
        if model is not None and provider is not None:
            break
    if provider is None and whole:
        provider = meta.payload.get("model_provider")
    return (provider if isinstance(provider, str) and provider else None,
            model if isinstance(model, str) and model else None)


def _last_pair_measured(newest_first: list, meta: Line, thread_id, table) -> bool:
    lines = list(reversed(newest_first))
    index = next(i for i in range(len(lines) - 1, -1, -1) if _own(lines[i], thread_id))
    provider, model = _pair_at(lines, index, meta, False)
    if provider is None or model is None:
        return False
    row = table.lookup(HARNESS, provider, model)
    return row is not None and row.measured


def _clock_from(scan: Scan, record: Line, thread_id, *, table, now, drift, bound_at,
                judged) -> Clock:
    lines = scan.lines
    index = next(i for i in range(len(lines) - 1, -1, -1) if lines[i] is record)
    whole = scan.reached_start
    next_provider, next_model = _next_pair(lines, scan.meta, whole)
    if next_provider is None or next_model is None:
        return unknown(_clock.NO_PAIR, NO_PAIR)
    if table.is_free(HARNESS, next_provider, next_model):
        return Clock(NO_CACHE, reason=free_reason(next_provider, next_model),
                     key=f"{HARNESS} {pair_text(next_provider, next_model)} (free)",
                     code=_clock.FREE)
    last_provider, last_model = _pair_at(lines, index, scan.meta, whole)
    if last_provider is None or last_model is None:
        return unknown(_clock.NO_PAIR, NO_PAIR)
    ts = record.ts
    if ts is None:
        return unknown(_clock.NO_STAMP, NO_STAMP)
    skew = future_stamp(ts, now)
    if skew:
        return unknown(_clock.FUTURE_STAMP, skew)
    age = max(0.0, now - ts)
    if (last_provider, last_model) != (next_provider, next_model):
        return Clock(COLD, age_seconds=age, reason=CHANGED, code=_clock.CHANGED)
    row = table.lookup(HARNESS, last_provider, last_model)
    key = f"{HARNESS} {pair_text(last_provider, last_model)}"
    if row is None:
        # A model with no lifetime row has no clock to count by.
        return Clock(NO_CACHE, age_seconds=age, reason=no_row(last_provider, last_model),
                     code=_clock.NO_ROW)
    if row.free:
        return Clock(NO_CACHE, reason=free_reason(last_provider, last_model),
                     key=f"{key} (free)", code=_clock.FREE)
    if not row.measured:
        return _verdict(age, row.seconds, key=key, basis="documented",
                        provider=last_provider, model=last_model)
    # A measured row: the lifetime in force after drift.
    if drift is MISSING:
        return unknown(_clock.NO_DRIFT_RECORD, NO_DRIFT)
    state = DriftView.of(drift)
    entry_key = (HARNESS, last_provider, last_model)
    blocked = state.blocked_reason(entry_key)
    if blocked:
        return unknown(_clock.DRIFT_HELD, blocked)
    lifetime, lowered_at = state.lifetime(entry_key, row)
    observed = judge_drift(lines, scan.meta, thread_id, row=row,
                           pair=(last_provider, last_model), lifetime=lifetime,
                           bound_at=bound_at, judged=judged, whole=whole)
    if observed.lowered is not None:
        lifetime = observed.lowered["lifetime"]
        lowered_at = observed.lowered["at_iso"]
    clock = _verdict(age, lifetime, key=_measured_key(key, row, lifetime, lowered_at),
                     basis="measured", provider=last_provider, model=last_model,
                     lowered_from=row.seconds if lifetime < row.seconds else None)
    clock.lowering = observed.lowered
    clock.notes = observed.notes
    clock.judged = (thread_id, observed.newest) if observed.newest else None
    return clock


def _measured_key(key, row, lifetime, lowered_at) -> str:
    if lifetime < row.seconds:
        when = f" on {lowered_at[:10]}" if isinstance(lowered_at, str) and lowered_at else ""
        return f"{key} (measured on this box, lowered from {span(row.seconds)}{when})"
    return f"{key} (measured on this box)"


def _verdict(age, lifetime, *, key, basis, provider, model, lowered_from=None) -> Clock:
    kind = COLD if age >= lifetime else WARM
    return Clock(kind, age_seconds=age, ttl_seconds=lifetime, key=key, basis=basis,
                 provider=provider, model=model, lowered_from=lowered_from)


# --------------------------------------------------------------------------- drift

@dataclass
class Judged:
    lowered: dict | None = None
    notes: list = field(default_factory=list)
    newest: str | None = None


def lower_to(gap: float, floor: int) -> int:
    """80% of the gap, rounded down to the minute, never below the floor."""
    return max(int(floor), int(gap * DRIFT_LOWER_TO) // 60 * 60)


def judge_drift(lines: list, meta: Line, thread_id, *, row, pair, lifetime, bound_at,
                judged, whole: bool) -> Judged:
    """The drift scan over the covered lines: each turn newer than `judged` (the newest
    turn alone when None), oldest first, against the lifetime in force."""
    out = Judged()
    starts = [i for i, ln in enumerate(lines) if _is_start(ln)]
    own = [i for i, ln in enumerate(lines) if _own(ln, thread_id)]
    if not own:
        return out
    # turn id -> index of its first own record
    first_of: dict = {}
    for i in own:
        first_of.setdefault(lines[i].payload.get("turn_id"), i)
    turns = []                           # (start index, first record index), oldest first
    for s in starts:
        turn = lines[s].payload.get("turn_id")
        r = first_of.get(turn)
        if r is not None and r > s:
            turns.append((s, r))
    if not turns:
        return out
    out.newest = lines[turns[-1][1]].payload.get("turn_id")
    if judged is None:
        todo = turns[-1:]
    else:
        newer = []
        for s, r in reversed(turns):
            if lines[s].payload.get("turn_id") == judged:
                break
            newer.append((s, r))
        todo = list(reversed(newer))
    in_force = lifetime
    for s, r in todo:
        obs = _observe(lines, meta, thread_id, s, r, row=row, pair=pair,
                       bound_at=bound_at, whole=whole)
        if obs is None:
            continue
        gap, prev, rec = obs
        if gap < row.floor:
            out.notes.append(
                f"cache clock: {pair[1]} on codex found its cache cold {span(gap)} after "
                f"the request before, under the documented {span(row.floor)}; not counted "
                f"(thread {thread_id}, cached {rec.payload.get('cached')} of "
                f"{prev.payload.get('input')} tokens)")
            continue
        if gap >= in_force:
            continue
        new = lower_to(gap, row.floor)
        if new >= in_force:
            continue
        in_force = new
        out.lowered = {
            "harness": HARNESS, "provider": pair[0], "model": pair[1], "lifetime": new,
            "from": lifetime if out.lowered is None else out.lowered["from"],
            "gap": gap, "thread": thread_id, "response_id": rec.payload.get("response_id"),
            "at": rec.ts, "at_iso": rec.raw_ts, "prev_at": prev.ts,
            "prev_at_iso": prev.raw_ts,
            "cached": rec.payload.get("cached"), "input": prev.payload.get("input"),
            "row_seconds": row.seconds, "row_checked": row.checked}
    return out


def _observe(lines, meta, thread_id, s, r, *, row, pair, bound_at, whole):
    """(gap, R_prev, R) when turn (start `s`, first record `r`) is a cold landing at a gap
    that may be counted; None otherwise."""
    if bound_at is None:
        return None                       # no registration time: no observation at all
    prev = None
    for i in range(s - 1, -1, -1):
        if _own(lines[i], thread_id):
            prev = i
            break
    if prev is None:
        return None
    rec_line, prev_line, start_line = lines[r], lines[prev], lines[s]
    if None in (rec_line.ts, prev_line.ts, start_line.ts):
        return None
    if _pair_at(lines, r, meta, whole) != tuple(pair):
        return None
    if _pair_at(lines, prev, meta, whole) != tuple(pair):
        return None
    prev_turn = prev_line.payload.get("turn_id")
    prev_start = None
    for i in range(prev, -1, -1):
        if _is_start(lines[i]) and lines[i].payload.get("turn_id") == prev_turn:
            prev_start = i
            break
    if prev_start is None:
        return None
    for ln in lines[prev_start:r]:
        if _breaks_continuation(ln):
            return None
    for ln in lines[prev + 1:s]:
        if _is_tsa(ln):
            return None                   # a resume, or a settings change
    if prev_line.ts < bound_at:
        return None                       # the launcher registered after R_prev: a resume
    inp, cached = prev_line.payload.get("input"), rec_line.payload.get("cached")
    if not isinstance(inp, int) or not isinstance(cached, int) or isinstance(inp, bool):
        return None
    if inp < DRIFT_MIN_INPUT:
        return None
    if cached >= inp * DRIFT_SHARE:
        return None                       # warm
    gap = start_line.ts - prev_line.ts
    return gap, prev_line, rec_line


__all__ = ["HARNESS", "CHECKED_VERSIONS", "TOP_KINDS", "EVENT_KINDS", "RESPONSE_KINDS",
           "META_KEYS", "read", "judge_line", "judge_drift", "lower_to"]
