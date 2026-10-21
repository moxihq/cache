"""The OpenCode clock: the start of a session's last request, read from `opencode.db`.

The launcher asks OpenCode where its database is (`opencode db path`, under the server's
environment) and passes it as `cache_source`. cache opens it read only, with a 200 ms busy
timeout and an authorizer that allows reads of `session`, `message` and `part` and their
`table_info`, and nothing else. The database also holds accounts and credentials, which
cache has no reason to read. cache is the database's one reader in Moxi.

**The order of the read.** The next pair first: the pair OpenCode will use for a prompt
that names no model and no agent. It is the default agent's configured model as the
launcher reports it (`agent_model`); else the session row's `model`; else the latest user
message's `model`; else not known. Each step reads only what it needs. A next pair on a
free row of the lifetime table is `no-cache` at any age, whatever the last request was: a
cold landing on a free model costs nothing. A next pair that cannot be read is `unknown`.

Otherwise the whole read runs. On every read: the shape check (`PRAGMA table_info` on the
three tables), every message row classified (`role` `user` or `assistant`), and no revert
pending. The last request is the latest `step-start` part of a qualifying assistant
message (this session's own, not a compaction, not a summary). Its stamp is the latest
time known to be at or before the request's start: the message's creation for its first
step, the end of the step before it for a later one. So the clock errs old. The last pair
is that message's `providerID` and `modelID`. Pairs that differ read `cold`; equal pairs
are looked up in the lifetime table, `"*"` matching any model of the provider.

**`fresh`** only for a session the launcher's own start attempt created after a read that
found no session, until its first turn ends (`turned` false), whose rows are all on the
fresh whitelist (user messages with text parts only), made by an OpenCode version the
reader was checked against, with zero token and cost totals on its session row; or, with
no session id at all, on the caller's word that its attempt creates one (`new_session`).
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

from moxi_sessions.api import claude as transcripts

from . import clock as _clock
from . import table as _table
from .clock import (COLD, FRESH, MISSING, NO_CACHE, WARM, Clock, bad_id, free_reason,
                    future_stamp, no_row, pair_text, unknown)

HARNESS = "opencode"

#: The OpenCode versions whose new sessions read as `fresh`.
CHECKED_VERSIONS = ("1.18.34",)

#: The columns the reader reads, checked on every read.
SHAPE = {
    "session": ("id", "parent_id", "version", "model", "revert", "tokens_input",
                "tokens_output", "tokens_reasoning", "tokens_cache_read",
                "tokens_cache_write", "cost"),
    "message": ("id", "session_id", "time_created", "data"),
    "part": ("id", "message_id", "session_id", "time_created", "data"),
}
TOTALS = ("tokens_input", "tokens_output", "tokens_reasoning", "tokens_cache_read",
          "tokens_cache_write", "cost")
BUSY_MS = 200

NO_SESSION_YET = "no session id reported yet"
NO_SOURCE = "no OpenCode database was named for this session"
NO_PAIR = "which model the next request will use cannot be told"
NO_AGENT = ("which model the next request will use cannot be told: the launcher could not "
            "name the session's default agent")
UNSHAPED_MESSAGE = "a message in this session is not in a shape the reader knows"
REVERT = "this session has a revert pending"
UNSHAPED_ROWS = "this session's rows are not in a shape the reader knows"
NO_REQUEST = "this session shows no request, but nothing confirms it has made none"
CHANGED = ("the model or provider changed since the last request, so the next request "
           "starts a cache of its own")


class Unread(Exception):
    """A read that cannot go on: `unknown`, with a code and the reason."""

    def __init__(self, reason: str, code: str = _clock.UNRECOGNISED):
        super().__init__(reason)
        self.code = code


def _authorize(action, arg1, arg2, _db, _source):
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ and arg1 in SHAPE:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA and arg1 == "table_info" and arg2 in SHAPE:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def connect(path) -> sqlite3.Connection:
    """The database, read only, with the busy timeout and the authorizer. Raises Unread."""
    path = Path(path)
    if not path.is_file():
        raise Unread(f"the OpenCode database {path} is missing", _clock.SOURCE_UNREADABLE)
    if not os.access(path, os.R_OK):
        raise Unread(f"the OpenCode database {path} cannot be read",
                     _clock.SOURCE_UNREADABLE)
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=BUSY_MS / 1000)
    except sqlite3.Error as e:
        raise Unread(f"the OpenCode database {path} could not be opened ({e})",
                     _clock.SOURCE_UNREADABLE) from e
    conn.set_authorizer(_authorize)
    return conn


def _parse_model(value) -> tuple | None:
    """`"provider/model"` -> (provider, model), else None."""
    if not isinstance(value, str) or "/" not in value:
        return None
    provider, _, model = value.partition("/")
    return (provider, model) if provider and model else None


class _Db:
    """One read's connection and what it has checked, opened on first use."""

    def __init__(self, path, session_id):
        self.path = path
        self.session_id = session_id
        self.conn = None
        self.row = None
        self.messages = None
        self.failed: Unread | None = None

    def close(self):
        if self.conn is not None:
            self.conn.close()

    def query(self, sql, args=()):
        try:
            return self.conn.execute(sql, args).fetchall()
        except sqlite3.Error as e:
            raise Unread(f"the OpenCode database could not be read ({e})",
                         _clock.UNREADABLE) from e

    def session(self):
        """Open, check the shape, and read the session row. Raises Unread."""
        if self.row is not None:
            return self.row
        if self.failed is not None:
            raise self.failed
        try:
            self.row = self._open()
        except Unread as e:
            self.failed = e
            raise
        return self.row

    def _open(self) -> dict:
        if not self.path:
            raise Unread(NO_SOURCE, _clock.NO_SOURCE)
        if not self.session_id:
            raise Unread(NO_SESSION_YET, _clock.NO_SESSION_YET)
        self.conn = connect(self.path)
        self.check_shape("session")
        cols = SHAPE["session"]
        rows = self.query(f"SELECT {', '.join(cols)} FROM session WHERE id = ?",
                          (self.session_id,))
        if not rows:
            raise Unread(f"no session {self.session_id} in {self.path}", _clock.NO_RECORD)
        return dict(zip(cols, rows[0]))

    def check_shape(self, *tables) -> None:
        """Each column the reader reads from `tables` exists. Raises Unread naming the
        first that does not. The session row's own step needs only `session`'s."""
        for table in tables:
            have = {r[1] for r in self.query(f"PRAGMA table_info({table})")}
            if not have:
                raise Unread(f"the OpenCode database has no `{table}` table")
            missing = [c for c in SHAPE[table] if c not in have]
            if missing:
                raise Unread(f"the OpenCode database's `{table}` table has no "
                             f"`{missing[0]}` column")

    def classified(self):
        """Every message row of the session, each a JSON object with role user or
        assistant; oldest first as (id, time_created, data). Raises Unread."""
        if self.messages is not None:
            return self.messages
        self.session()
        self.check_shape("message", "part")
        out = []
        for mid, created, raw in self.query(
                "SELECT id, time_created, data FROM message WHERE session_id = ? "
                "ORDER BY time_created, id", (self.session_id,)):
            try:
                data = json.loads(raw)
            except (TypeError, ValueError):
                raise Unread(UNSHAPED_MESSAGE) from None
            if not isinstance(data, dict) or data.get("role") not in ("user", "assistant"):
                raise Unread(UNSHAPED_MESSAGE)
            out.append((mid, created, data))
        self.messages = out
        return out

    def no_revert(self):
        if self.session().get("revert") is not None:
            raise Unread(REVERT, _clock.REVERT)


def next_pair(db: _Db, agent_model):
    """(provider, model) of the next request, by the first step that applies; raises
    Unread when it cannot be known."""
    if agent_model is MISSING or agent_model is None:
        raise Unread(NO_AGENT, _clock.NO_PAIR)
    if not isinstance(agent_model, dict) or not isinstance(agent_model.get("agent"), str) \
            or not agent_model.get("agent"):
        raise Unread(NO_AGENT, _clock.NO_PAIR)
    configured = agent_model.get("model")
    if configured is not None:
        pair = _parse_model(configured)
        if pair is None:
            raise Unread(f"{NO_PAIR} (the default agent's model {configured!r} does not "
                         "parse)", _clock.NO_PAIR)
        return pair
    row = db.session()
    raw = row.get("model")
    if raw is not None:
        try:
            value = json.loads(raw) if isinstance(raw, str) else None
        except ValueError:
            value = None
        if (not isinstance(value, dict) or not isinstance(value.get("providerID"), str)
                or not isinstance(value.get("id"), str) or not value["providerID"]
                or not value["id"]):
            raise Unread(f"{NO_PAIR} (the session's model does not parse)", _clock.NO_PAIR)
        return value["providerID"], value["id"]
    messages = db.classified()
    db.no_revert()
    for _mid, _created, data in reversed(messages):
        if data.get("role") != "user" or "model" not in data:
            continue
        model = data.get("model")
        if (not isinstance(model, dict) or not isinstance(model.get("providerID"), str)
                or not isinstance(model.get("modelID"), str) or not model["providerID"]
                or not model["modelID"]):
            raise Unread(f"{NO_PAIR} (the latest user message's model does not parse)",
                         _clock.NO_PAIR)
        return model["providerID"], model["modelID"]
    raise Unread(NO_PAIR, _clock.NO_PAIR)


def read(session_id, *, cache_source, turned, new_session, agent_model=MISSING,
         now) -> Clock:
    """One OpenCode session's clock."""
    try:
        table = _table.load_table()
    except _table.TableError as e:
        return unknown(_clock.TABLE_NOT_LOADED, f"{_table.NOT_LOADED} ({e})")
    if session_id and (not isinstance(session_id, str)
                       or not transcripts.SAFE_ID_RE.match(session_id)):
        return bad_id("session id", session_id)
    db = _Db(cache_source, session_id)
    try:
        return _read(db, session_id, table=table, cache_source=cache_source, turned=turned,
                     new_session=new_session, agent_model=agent_model, now=now)
    except Unread as e:
        return unknown(e.code, str(e))
    finally:
        db.close()


def _read(db: _Db, session_id, *, table, cache_source, turned, new_session, agent_model,
          now) -> Clock:
    # The next pair first: on a free row nothing else is read or checked.
    try:
        upcoming = next_pair(db, agent_model)
        unknown_next = None
    except Unread as e:
        upcoming, unknown_next = None, e
    if upcoming is not None and table.is_free(HARNESS, *upcoming):
        return Clock(NO_CACHE, reason=free_reason(*upcoming),
                     key=f"{HARNESS} {pair_text(*upcoming)} (free)", code=_clock.FREE)
    if not session_id:
        if new_session:
            return Clock(FRESH)          # the launcher's own attempt creates the session
        return unknown(_clock.NO_SESSION_YET, NO_SESSION_YET)
    if not cache_source:
        return unknown(_clock.NO_SOURCE, NO_SOURCE)
    row = db.session()
    messages = db.classified()
    db.no_revert()
    last = _last_request(db, messages)
    if last is None:
        return _no_request(db, row, messages, turned)
    stamp_ms, data = last
    provider, model = data.get("providerID"), data.get("modelID")
    if unknown_next is not None:
        return unknown(unknown_next.code, str(unknown_next))
    if not (isinstance(provider, str) and provider and isinstance(model, str) and model):
        return unknown(_clock.NO_PAIR, NO_PAIR)
    ts = stamp_ms / 1000.0
    skew = future_stamp(ts, now)
    if skew:
        return unknown(_clock.FUTURE_STAMP, skew)
    age = max(0.0, now - ts)
    if (provider, model) != tuple(upcoming):
        return Clock(COLD, age_seconds=age, reason=CHANGED, code=_clock.CHANGED)
    row_ = table.lookup(HARNESS, provider, model)
    key = f"{HARNESS} {pair_text(provider, model)}"
    if row_ is None:
        # A model with no lifetime row has no clock to count by.
        return Clock(NO_CACHE, age_seconds=age, reason=no_row(provider, model),
                     code=_clock.NO_ROW)
    if row_.free:
        return Clock(NO_CACHE, reason=free_reason(provider, model), key=f"{key} (free)",
                     code=_clock.FREE)
    kind = COLD if age >= row_.seconds else WARM
    return Clock(kind, age_seconds=age, ttl_seconds=row_.seconds, key=key,
                 basis="documented", provider=provider, model=model)


def _qualifies(data: dict) -> bool:
    return (data.get("role") == "assistant" and data.get("agent") != "compaction"
            and data.get("summary") is not True)


def _last_request(db: _Db, messages):
    """(stamp in epoch ms, the assistant message's data) for the session's last request,
    or None when the session has no assistant message. Raises Unread when it has
    assistant messages and none with a step the reader can use."""
    assistants = [m for m in messages if m[2].get("role") == "assistant"]
    if not assistants:
        return None
    for mid, created, data in reversed(assistants):
        if not _qualifies(data):
            continue
        parts = []
        for _pid, pcreated, raw in db.query(
                "SELECT id, time_created, data FROM part WHERE message_id = ? "
                "AND session_id = ? ORDER BY time_created, id", (mid, db.session_id)):
            try:
                pdata = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(pdata, dict):
                parts.append((pcreated, pdata.get("type")))
        starts = [i for i, (_c, t) in enumerate(parts) if t == "step-start"]
        if not starts:
            continue
        last = starts[-1]
        stamp = created
        if last != starts[0]:
            for c, t in reversed(parts[:last]):
                if t == "step-finish":
                    stamp = c
                    break
        if not isinstance(stamp, (int, float)) or isinstance(stamp, bool):
            raise Unread("the session's last request has no readable time", _clock.NO_STAMP)
        return stamp, data
    raise Unread(UNSHAPED_ROWS)


def _no_request(db: _Db, row: dict, messages, turned) -> Clock:
    """No assistant message: `fresh` only on every witness of a new session."""
    if turned:
        return unknown(_clock.NO_REQUEST, NO_REQUEST)
    if row.get("version") not in CHECKED_VERSIONS:
        return unknown(_clock.NO_REQUEST, NO_REQUEST)
    if any(row.get(k) not in (0, 0.0) for k in TOTALS):
        return unknown(_clock.NO_REQUEST, NO_REQUEST)
    users = {mid for mid, _c, data in messages if data.get("role") == "user"}
    if len(users) != len(messages):
        return unknown(_clock.NO_REQUEST, NO_REQUEST)
    for message_id, raw in db.query("SELECT message_id, data FROM part WHERE session_id = ?",
                                    (db.session_id,)):
        try:
            pdata = json.loads(raw)
        except (TypeError, ValueError):
            return unknown(_clock.NO_REQUEST, NO_REQUEST)
        if (not isinstance(pdata, dict) or pdata.get("type") != "text"
                or message_id not in users):
            return unknown(_clock.NO_REQUEST, NO_REQUEST)
    return Clock(FRESH)


__all__ = ["HARNESS", "CHECKED_VERSIONS", "SHAPE", "connect", "read", "next_pair", "Unread"]
