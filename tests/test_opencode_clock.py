"""The OpenCode clock: a session's rows in `opencode.db`, the next pair first, free models,
the authorizer, and `fresh` on every witness of a new session.

The fixture database is built from OpenCode 1.18.34's schema, not copied from a live one;
every read is against tmp paths.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path

import pytest

from moxi_cache import api, opencode, table

NOW = 1_791_300_000.0                    # a fixed "now" for the pure reads
SID = "ses_0000000000000TestSessionA"
SID_B = "ses_0000000000000TestSessionB"


# ================================================================ opencode fixtures

#: OpenCode 1.18.34's schema for the three tables the reader reads, as read on
#: 2026-10-05, and the two it must never read.
SCHEMA = """
CREATE TABLE `session` (
  `id` text PRIMARY KEY, `project_id` text NOT NULL, `workspace_id` text, `parent_id` text,
  `slug` text NOT NULL, `directory` text NOT NULL, `path` text, `title` text NOT NULL,
  `version` text NOT NULL, `share_url` text, `summary_additions` integer,
  `summary_deletions` integer, `summary_files` integer, `summary_diffs` text,
  `metadata` text, `cost` real DEFAULT 0 NOT NULL,
  `tokens_input` integer DEFAULT 0 NOT NULL, `tokens_output` integer DEFAULT 0 NOT NULL,
  `tokens_reasoning` integer DEFAULT 0 NOT NULL,
  `tokens_cache_read` integer DEFAULT 0 NOT NULL,
  `tokens_cache_write` integer DEFAULT 0 NOT NULL, `revert` text, `permission` text,
  `agent` text, `model` text, `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL, `time_compacting` integer, `time_archived` integer);
CREATE TABLE `message` (
  `id` text PRIMARY KEY, `session_id` text NOT NULL, `time_created` integer NOT NULL,
  `time_updated` integer NOT NULL, `data` text NOT NULL);
CREATE TABLE `part` (
  `id` text PRIMARY KEY, `message_id` text NOT NULL, `session_id` text NOT NULL,
  `time_created` integer NOT NULL, `time_updated` integer NOT NULL, `data` text NOT NULL);
CREATE INDEX `message_session_time_created_id_idx` ON `message` (`session_id`,`time_created`,`id`);
CREATE INDEX `part_message_id_id_idx` ON `part` (`message_id`,`id`);
CREATE INDEX `part_session_idx` ON `part` (`session_id`);
CREATE TABLE `account` (`id` text PRIMARY KEY, `email` text);
CREATE TABLE `credential` (`id` text PRIMARY KEY, `secret` text);
INSERT INTO `account` VALUES ('acc', 'someone@example.invalid');
INSERT INTO `credential` VALUES ('cred', 'SECRET-DO-NOT-READ');
"""

BUILD_NULL = {"agent": "build", "model": None}
FREE_AGENT = {"agent": "build", "model": "opencode/big-pickle"}


class OcDb:
    """A fixture `opencode.db` in the 1.18.34 schema."""

    def __init__(self, path: Path, schema: str = SCHEMA):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0
        with self.conn() as c:
            c.executescript(schema)

    def conn(self):
        """A connection in autocommit, closed when the `with` block ends."""
        c = sqlite3.connect(self.path)
        c.isolation_level = None
        return contextlib.closing(c)

    def mint(self, prefix):
        self.n += 1
        return f"{prefix}_{self.n:026d}"

    def session(self, sid=SID, *, model=("opencode", "big-pickle"), version="1.18.34",
                totals=0, revert=None, parent=None, raw_model=...):
        if raw_model is ...:
            raw_model = None if model is None else json.dumps(
                {"id": model[1], "providerID": model[0], "variant": "default"})
        with self.conn() as c:
            c.execute("INSERT INTO session (id, project_id, parent_id, slug, directory, "
                      "title, version, cost, tokens_input, tokens_output, "
                      "tokens_reasoning, tokens_cache_read, tokens_cache_write, revert, "
                      "model, time_created, time_updated) VALUES "
                      "(?, 'global', ?, 's', '/tmp', 't', ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0)",
                      (sid, parent, version, totals, totals, totals, totals, totals, totals,
                       revert, raw_model))

    def message(self, sid, created_s, data, *, raw=None):
        mid = self.mint("msg")
        with self.conn() as c:
            c.execute("INSERT INTO message VALUES (?, ?, ?, ?, ?)",
                      (mid, sid, int(created_s * 1000), int(created_s * 1000),
                       raw if raw is not None else json.dumps(data)))
        return mid

    def part(self, sid, mid, created_s, data):
        with self.conn() as c:
            c.execute("INSERT INTO part VALUES (?, ?, ?, ?, ?, ?)",
                      (self.mint("prt"), mid, sid, int(created_s * 1000),
                       int(created_s * 1000), json.dumps(data)))

    def user(self, sid, t, *, model=("opencode", "big-pickle"), raw_model=...):
        data = {"role": "user", "time": {"created": int(t * 1000)}, "agent": "build",
                "summary": {"diffs": []}}
        if raw_model is not ...:
            data["model"] = raw_model
        elif model is not None:
            data["model"] = {"providerID": model[0], "modelID": model[1]}
        mid = self.message(sid, t, data)
        self.part(sid, mid, t + 0.01, {"type": "text", "text": "hello"})
        return mid

    def request(self, sid, t, provider, model, *, steps=((0.0, 3.0),), agent="build",
                summary=None, created=None, user=True):
        """A user message and its assistant reply whose first step starts at `t`, each
        step `(start offset, end offset)` from `t`. The message is created at
        `created` (default `t`)."""
        if user:
            self.user(sid, (created or t) - 0.5, model=(provider, model))
        data = {"role": "assistant", "agent": agent, "mode": agent, "providerID": provider,
                "modelID": model, "time": {"created": int((created or t) * 1000)},
                "finish": "stop"}
        if summary is not None:
            data["summary"] = summary
        mid = self.message(sid, created or t, data)
        for start, end in steps:
            self.part(sid, mid, t + start, {"type": "step-start"})
            if end is not None:
                self.part(sid, mid, t + end, {"type": "step-finish", "reason": "stop"})
        return mid


def oc_db(tmp_path, **session) -> OcDb:
    db = OcDb(tmp_path / "opencode" / "opencode.db")
    db.session(**session)
    return db


def oc_read(db, sid=SID, *, turned=True, new_session=False, agent_model=BUILD_NULL,
            now=NOW, source=...):
    path = str(db.path) if source is ... else source
    return api.read("opencode", sid, cache_source=path, turned=turned,
                    new_session=new_session, agent_model=agent_model, now=now)



# ================================================================ the opencode reader

def test_anthropic_counts_5_minutes(tmp_path):
    db = oc_db(tmp_path, model=("anthropic", "claude-sonnet-4"))
    db.request(SID, NOW - 299, "anthropic", "claude-sonnet-4")
    clock = oc_read(db)
    assert (clock.kind, clock.ttl_seconds, clock.key) == (
        "warm", 300, "opencode anthropic/claude-sonnet-4")
    db2 = oc_db(tmp_path / "b", model=("anthropic", "claude-sonnet-4"))
    db2.request(SID, NOW - 300, "anthropic", "claude-sonnet-4")
    assert oc_read(db2).kind == "cold"


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-6-astra", "gpt-5.6-terra",
                                   "gpt-5.6-luna"])
def test_openai_counts_30_minutes_and_no_drift_lowers_it(tmp_path, model):
    for age, kind in ((1799, "warm"), (1800, "cold")):
        db = oc_db(tmp_path / str(age), model=("openai", model))
        db.request(SID, NOW - age, "openai", model)
        clock = oc_read(db)
        assert (clock.kind, clock.ttl_seconds) == (kind, 1800)


@pytest.mark.parametrize("model", ["big-pickle", "space-bunny-free"])
@pytest.mark.parametrize("age", [0, 30 * 86400])
def test_a_free_next_pair_is_no_cache_at_any_age(tmp_path, model, age):
    db = oc_db(tmp_path, model=("opencode", model))
    db.request(SID, NOW - age, "opencode", model)
    clock = oc_read(db)
    assert clock.kind == "no-cache" and clock.ttl_seconds is None
    assert clock.reason == (f"opencode/{model} is a free model, so a cold cache costs "
                            "nothing and is not counted")
    assert clock.key == f"opencode opencode/{model} (free)"


def test_an_unknown_provider_reads_no_cache(tmp_path):
    """No lifetime row for the pair: no clock, `no-cache`."""
    db = oc_db(tmp_path, model=("deepseek", "deepseek-chat"))
    db.request(SID, NOW - 5, "deepseek", "deepseek-chat")
    clock = oc_read(db)
    assert clock.kind == "no-cache" and clock.reason == opencode.no_row(
        "deepseek", "deepseek-chat")


def test_the_stamp_errs_old(tmp_path):
    # First step: the message's creation, not the step-start 3 s later.
    db = oc_db(tmp_path / "a", model=("anthropic", "claude-x"))
    db.request(SID, NOW - 297, "anthropic", "claude-x", created=NOW - 300)
    assert oc_read(db).kind == "cold"
    # A later step reads from the step-finish before it.
    db = oc_db(tmp_path / "b", model=("anthropic", "claude-x"))
    db.request(SID, NOW - 1000, "anthropic", "claude-x",
               steps=((0, 10), (700, 710)))            # second step starts at NOW-300
    clock = oc_read(db)
    assert clock.age_seconds == pytest.approx(990, abs=0.01)
    # A step that started 6 minutes ago and finished 2 minutes ago is cold on 300 s.
    db = oc_db(tmp_path / "c", model=("anthropic", "claude-x"))
    db.request(SID, NOW - 360, "anthropic", "claude-x", steps=((0, 240),))
    assert oc_read(db).kind == "cold"
    # A request in flight reads by its start.
    db = oc_db(tmp_path / "d", model=("anthropic", "claude-x"))
    db.request(SID, NOW - 100, "anthropic", "claude-x", steps=((0, None),))
    clock = oc_read(db)
    assert clock.kind == "warm" and clock.age_seconds == pytest.approx(100, abs=0.01)


def test_a_compaction_and_a_child_session_do_not_move_the_clock(tmp_path):
    db = oc_db(tmp_path, model=("anthropic", "claude-x"))
    db.request(SID, NOW - 400, "anthropic", "claude-x")
    db.request(SID, NOW - 10, "anthropic", "claude-x", agent="compaction", user=False)
    db.request(SID, NOW - 9, "anthropic", "claude-x", summary=True, user=False)
    db.session(SID_B, model=("anthropic", "claude-x"), parent=SID)
    db.request(SID_B, NOW - 5, "anthropic", "claude-x")
    clock = oc_read(db)
    assert clock.kind == "cold" and clock.age_seconds == pytest.approx(400, abs=0.01)


def fresh_session(tmp_path, **kw) -> OcDb:
    db = oc_db(tmp_path, model=("openai", "gpt-5.6-sol"), **kw)
    db.user(SID, NOW - 5, model=("openai", "gpt-5.6-sol"))
    return db


def test_the_launchers_own_new_session_is_fresh(tmp_path):
    db = fresh_session(tmp_path)
    assert oc_read(db, turned=False).kind == "fresh"
    assert oc_read(db, turned=True).kind == "unknown"


@pytest.mark.parametrize("case", ["assistant-no-step", "bad-role", "bad-data",
                                  "file-part", "step-on-user", "totals", "version",
                                  "missing-column", "no-session", "missing-db",
                                  "unreadable-db", "locked-db", "no-part-table",
                                  "no-source"])
def test_each_witness_that_fails_is_unknown(tmp_path, case):
    if case == "missing-column":
        db = OcDb(tmp_path / "db" / "o.db",
                  SCHEMA.replace("`tokens_cache_write` integer DEFAULT 0 NOT NULL,", ""))
        db.session = None
        with db.conn() as c:
            c.execute("INSERT INTO session (id, project_id, slug, directory, title, "
                      "version, time_created, time_updated) VALUES (?, 'g', 's', '/', "
                      "'t', '1.18.34', 0, 0)", (SID,))
    elif case == "no-part-table":
        db = OcDb(tmp_path / "db" / "o.db",
                  SCHEMA.split("CREATE TABLE `part`")[0]
                  + "CREATE TABLE `account` (`id` text); CREATE TABLE `credential` (`id` text);")
        OcDb.session(db, model=("openai", "gpt-5.6-sol"))
    else:
        db = fresh_session(tmp_path,
                           totals=5 if case == "totals" else 0,
                           version="1.18.35" if case == "version" else "1.18.34")
    source = ...
    if case == "assistant-no-step":
        db.message(SID, NOW - 4, {"role": "assistant", "agent": "build",
                                  "providerID": "openai", "modelID": "gpt-5.6-sol"})
    elif case == "bad-role":
        db.message(SID, NOW - 4, {"role": "system"})
    elif case == "bad-data":
        db.message(SID, NOW - 4, None, raw="{not json")
    elif case == "file-part":
        mid = db.user(SID, NOW - 3)
        db.part(SID, mid, NOW - 3, {"type": "file", "url": "x"})
    elif case == "step-on-user":
        mid = db.user(SID, NOW - 3)
        db.part(SID, mid, NOW - 3, {"type": "step-start"})
    elif case == "no-session":
        source = str(db.path)
        clock = oc_read(db, sid=SID_B, turned=False, source=source)
        assert clock.kind == "unknown" and "no session" in clock.reason
        return
    elif case == "missing-db":
        source = str(tmp_path / "nowhere.db")
    elif case == "unreadable-db":
        db.path.chmod(0)
    elif case == "no-source":
        source = None
    try:
        if case == "locked-db":
            holder = sqlite3.connect(db.path)
            holder.execute("BEGIN EXCLUSIVE")
            try:
                start = time.monotonic()
                clock = oc_read(db, turned=False)
                assert time.monotonic() - start < 2
            finally:
                holder.rollback()
                holder.close()
        else:
            clock = oc_read(db, turned=False, source=source)
    finally:
        if case == "unreadable-db":
            db.path.chmod(0o600)
    assert clock.kind == "unknown", (case, clock)
    if case == "assistant-no-step":
        assert clock.reason == opencode.UNSHAPED_ROWS


def test_the_session_rows_step_needs_only_the_session_tables_shape(tmp_path):
    """A free next pair from the session row's model needs the shape of `session`, not
    of `message` or `part`; the full read needs all three."""
    schema = SCHEMA.split("CREATE TABLE `part`")[0] + \
        "CREATE TABLE `account` (`id` text); CREATE TABLE `credential` (`id` text);"
    db = OcDb(tmp_path / "free" / "o.db", schema)
    OcDb.session(db, model=("opencode", "big-pickle"))
    assert oc_read(db).kind == "no-cache"
    db = OcDb(tmp_path / "paid" / "o.db", schema)
    OcDb.session(db, model=("anthropic", "claude-A"))
    clock = oc_read(db)
    assert clock.kind == "unknown" and "no `part` table" in clock.reason


def test_the_pairs_under_the_wildcard_row(tmp_path):
    def read(**kw):
        return oc_read(db, **kw)
    # A last request on claude-A with the session's model now claude-B: cold.
    db = oc_db(tmp_path / "a", model=("anthropic", "claude-B"))
    db.request(SID, NOW - 10, "anthropic", "claude-A")
    assert read().reason == opencode.CHANGED
    # A provider change: cold.
    db = oc_db(tmp_path / "b", model=("bedrock", "claude-A"))
    db.request(SID, NOW - 10, "anthropic", "claude-A")
    assert read().reason == opencode.CHANGED
    # The default agent's model differs from the session's: cold.
    db = oc_db(tmp_path / "c", model=("anthropic", "claude-A"))
    db.request(SID, NOW - 10, "anthropic", "claude-A")
    assert read().kind == "warm"
    assert read(agent_model={"agent": "build", "model": "anthropic/claude-B"}).reason == \
        opencode.CHANGED
    # An agent with a null model falls through to the session's model.
    assert read(agent_model={"agent": "plan", "model": None}).kind == "warm"
    # A default agent's model that does not parse: unknown, never the next step.
    clock = read(agent_model={"agent": "build", "model": "no-slash"})
    assert clock.kind == "unknown" and "does not parse" in clock.reason
    # No agent_model reported: unknown.
    clock = read(agent_model=api.MISSING)
    assert clock.kind == "unknown" and clock.reason == opencode.NO_AGENT
    # No session model and no user model: unknown (no server default is read).
    db = oc_db(tmp_path / "d", model=None)
    db.request(SID, NOW - 10, "anthropic", "claude-A", user=False)
    assert read().kind == "unknown"
    # A session model that will not parse, and a latest user model that will not.
    db = oc_db(tmp_path / "e", raw_model="{not json")
    db.request(SID, NOW - 10, "anthropic", "claude-A")
    assert read().kind == "unknown"
    db = oc_db(tmp_path / "f", model=None)
    db.request(SID, NOW - 10, "anthropic", "claude-A")
    db.user(SID, NOW - 5, raw_model="anthropic/claude-A")
    assert read().kind == "unknown"
    # The latest user model, when the session has none, gives the next pair.
    db = oc_db(tmp_path / "g", model=None)
    db.request(SID, NOW - 10, "anthropic", "claude-A")
    assert read().kind == "warm"
    # A last assistant message with no providerID: unknown.
    db = oc_db(tmp_path / "h", model=("anthropic", "claude-A"))
    mid = db.message(SID, NOW - 10, {"role": "assistant", "agent": "build",
                                     "modelID": "claude-A"})
    db.part(SID, mid, NOW - 9, {"type": "step-start"})
    clock = read()
    assert clock.kind == "unknown" and clock.reason == opencode.NO_PAIR


def test_the_authorizer_refuses_account_and_credential(tmp_path):
    db = oc_db(tmp_path)
    conn = opencode.connect(db.path)
    try:
        for table in ("account", "credential"):
            with pytest.raises(sqlite3.DatabaseError):
                conn.execute(f"SELECT * FROM {table}").fetchall()
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("PRAGMA table_info(credential)").fetchall()
        assert conn.execute("SELECT id FROM session").fetchall() == [(SID,)]
    finally:
        conn.close()


def test_a_revert_pending_is_unknown_with_and_without_a_request(tmp_path):
    """The `revert` value is the one a real revert on OpenCode 1.18.34 wrote on
    2026-10-06 (two prompts on opencode/big-pickle in an isolated data dir, then
    `POST /session/:id/revert` at the second user message): `{"messageID": ...}`, the
    later messages left in the db until the next prompt.

    The same run checked whether a revert's cleanup lowers the session's totals: it
    does. After the revert the totals were unchanged (input 4,730, cache read 20,224);
    after the next prompt, which cleans up before it runs, input was 4,666, lower than
    before though that prompt's own request had been added. So the session row's totals
    are not a running maximum, and a fully reverted session could show low totals; the
    fresh check still needs `turned` false, which no session that has had a turn has."""
    revert = json.dumps({"messageID": "msg_000000000000000TestRevert"},
                        separators=(",", ":"))
    db = oc_db(tmp_path / "a", model=("anthropic", "claude-A"), revert=revert)
    db.request(SID, NOW - 10, "anthropic", "claude-A")
    clock = oc_read(db)
    assert clock.kind == "unknown" and clock.reason == opencode.REVERT
    db = fresh_session(tmp_path / "b", revert=revert)
    assert oc_read(db, turned=False).reason == opencode.REVERT

# ====================================================================== free models

def free_db(tmp_path, *, last=("opencode", "big-pickle"), age=600, model=...,
            **session) -> OcDb:
    db = oc_db(tmp_path, model=("opencode", "big-pickle") if model is ... else model,
               **session)
    if last is not None:
        db.request(SID, NOW - age, *last)
    return db


@pytest.mark.parametrize("case", [
    "last-big-pickle", "last-space-bunny", "change-onto-free", "no-request-turned",
    "no-request-fresh", "revert-by-session-row", "unshaped-row-by-session-row",
    "agent-no-source", "agent-missing-db", "agent-bad-db", "no-id-new-session",
    "no-id-no-new-session"])
def test_a_known_free_next_pair_is_no_cache_at_any_age(tmp_path, case):
    agent = BUILD_NULL
    source = ...
    sid = SID
    kw = {}
    if case == "last-big-pickle":
        db = free_db(tmp_path, age=40 * 86400)
    elif case == "last-space-bunny":
        db = free_db(tmp_path, last=("opencode", "space-bunny-free"))
    elif case == "change-onto-free":
        db = free_db(tmp_path, last=("anthropic", "claude-A"))
    elif case == "no-request-turned":
        db = free_db(tmp_path, last=None)
        kw["turned"] = True
    elif case == "no-request-fresh":
        db = free_db(tmp_path, last=None)
        kw["turned"] = False
    elif case == "revert-by-session-row":
        db = free_db(tmp_path, revert=json.dumps({"messageID": "msg_x"}))
    elif case == "unshaped-row-by-session-row":
        db = free_db(tmp_path)
        db.message(SID, NOW - 1, {"role": "tool"})
    else:
        db = free_db(tmp_path)
        agent = FREE_AGENT
        if case == "agent-no-source":
            source = None
        elif case == "agent-missing-db":
            source = str(tmp_path / "gone.db")
        elif case == "agent-bad-db":
            bad = tmp_path / "bad.db"
            bad.write_text("this is not sqlite")
            source = str(bad)
        elif case == "no-id-new-session":
            sid, kw["new_session"] = None, True
        elif case == "no-id-no-new-session":
            sid = None
    clock = oc_read(db, sid=sid, agent_model=agent, source=source, **kw)
    assert clock.kind == "no-cache", (case, clock)
    assert clock.key in ("opencode opencode/big-pickle (free)",)


@pytest.mark.parametrize("case", ["no-agent-model", "null-model-no-db", "bad-session-model",
                                  "user-model-revert", "user-model-unshaped", "no-table"])
def test_a_free_model_that_cannot_be_read_as_next_is_unknown(tmp_path, case):
    agent, source = BUILD_NULL, ...
    if case == "no-agent-model":
        db, agent = free_db(tmp_path), api.MISSING
    elif case == "null-model-no-db":
        db, source = free_db(tmp_path), None
    elif case == "bad-session-model":
        db = free_db(tmp_path, model=None, raw_model="{nope")
    elif case in ("user-model-revert", "user-model-unshaped"):
        db = free_db(tmp_path, model=None,
                     revert=json.dumps({"messageID": "m"}) if case == "user-model-revert"
                     else None)
        if case == "user-model-unshaped":
            db.message(SID, NOW - 1, {"role": "tool"})
    else:
        db = free_db(tmp_path)
        agent = FREE_AGENT
        table.TABLE_PATH.write_text("not = [toml")
    clock = oc_read(db, agent_model=agent, source=source)
    assert clock.kind == "unknown", (case, clock)


def test_a_paid_next_pair_after_a_free_one_reads_by_the_pair_rule(tmp_path):
    db = free_db(tmp_path, model=("anthropic", "claude-A"))
    clock = oc_read(db)
    assert clock.kind == "cold" and clock.reason == opencode.CHANGED
    db = free_db(tmp_path / "b", last=("opencode", "gpt-5.5"), model=("opencode", "gpt-5.5"))
    clock = oc_read(db)
    assert clock.kind == "no-cache" and clock.reason == opencode.no_row("opencode",
                                                                              "gpt-5.5")


def test_the_rule_reads_only_the_next_pair(tmp_path, monkeypatch):
    """Mutation guards: a free next pair never needs the last request, and an unknown
    next pair never falls through to the last one."""
    db = free_db(tmp_path, last=("opencode", "big-pickle"), model=None)
    db.user(SID, NOW - 1, model=("anthropic", "claude-A"))
    clock = oc_read(db)
    assert clock.kind == "cold", "the next pair is the latest user model, not the last"
    db = free_db(tmp_path / "b", last=("opencode", "big-pickle"))
    assert oc_read(db, agent_model=api.MISSING).kind == "unknown"


# ============================================================================ drift



def test_a_stamp_in_the_future_is_unknown(tmp_path):
    db = OcDb(tmp_path / "oc.db")
    db.session(SID, model=("opencode", "gpt-5.6-sol"))
    db.request(SID, NOW + 6, "openai", "gpt-5.6-sol")
    clock = oc_read(db, agent_model=BUILD_NULL)
    assert (clock.kind, clock.code) == ("unknown", "future-stamp")
    assert "stamped in the future" in clock.reason


def test_every_unknown_and_no_cache_carries_a_code(tmp_path):
    """A caller branches on the code, never on the reason's words."""
    db = oc_db(tmp_path, model=("deepseek", "deepseek-chat"))
    db.request(SID, NOW - 5, "deepseek", "deepseek-chat")
    assert oc_read(db).code == "no-row"
    assert oc_read(db, agent_model=api.MISSING).code == "no-pair"
    assert oc_read(db, source=None).code in ("no-source", "no-pair")
    assert oc_read(db, sid=SID_B, turned=False).code == "no-record"
    free = oc_db(tmp_path / "free")
    assert oc_read(free, agent_model=FREE_AGENT).code == "free"
