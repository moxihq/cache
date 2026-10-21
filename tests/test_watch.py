"""The watcher: the live list, the clock records it writes and removes, and when it reads."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import pytest
from conftest import assistant_line, write_cache_config, write_transcript

from moxi_cache import api, records, watch
from moxi_cache.clock import Clock

A = "aaaaaaaa-1111-4111-8111-111111111111"
B = "bbbbbbbb-2222-4222-8222-222222222222"
C = "cccccccc-3333-4333-8333-333333333333"


def entry(pid, session, kind="interactive", status="idle", **extra):
    out = {"pid": pid, "cwd": "/work", "kind": kind, "startedAt": 1, "sessionId": session,
           "name": "n", "status": status}
    out.update(extra)
    return out


class Fake:
    """A settable `claude agents --json`, a clock and a counting reader."""

    def __init__(self, doc=None, now=1_800_000_000.0):
        self.doc = [] if doc is None else doc
        self.calls = 0
        self.t = now
        self.reads: list[str] = []

    def agents(self):
        self.calls += 1
        if isinstance(self.doc, Exception):
            raise self.doc
        return self.doc

    def now(self):
        return self.t

    def read(self, harness, session, *, now):
        self.reads.append(session)
        return api.read(harness, session, now=now)


@pytest.fixture
def root(tmp_path) -> Path:
    """The projects root the watcher finds transcripts under (Claude's, by the
    environment the suite sets)."""
    path = tmp_path / "claude-config" / "projects"
    path.mkdir(parents=True)
    return path


@pytest.fixture
def sessions(tmp_path) -> Path:
    path = tmp_path / "claude-config" / "sessions"
    path.mkdir(parents=True)
    return path


def make(fake, sessions, **kw) -> watch.Watcher:
    return watch.Watcher(agents=fake.agents, read=fake.read, now=fake.now,
                         sessions=sessions, **kw)


def poke(sessions: Path) -> None:
    """Change the sessions folder, as Claude Code does at a status change, so the next tick
    asks for the list."""
    poke.n = getattr(poke, "n", 0) + 1
    path = sessions / "1.json"
    path.write_text(json.dumps({"n": poke.n}))
    os.utime(path, ns=(poke.n, poke.n))


def rec(session) -> dict | None:
    path = records.record_dir() / f"{session}.json"
    return json.loads(path.read_text()) if path.exists() else None


# --- the live list --------------------------------------------------------------------------

def test_parse_agents_keeps_interactive_sessions():
    doc = [entry(1, A), entry(2, B, kind="background"), entry(3, None), entry(4, ""),
           {"pid": 5, "sessionId": C}]
    assert watch.parse_agents(doc) == [watch.Live(1, A), watch.Live(5, C)]


@pytest.mark.parametrize("doc, words", [
    ({"sessions": []}, "a dict, not a list"),
    (["x"], "a list holding a str"),
    ([{"sessionId": A}], "no integer pid"),
    ([{"pid": True, "sessionId": A}], "no integer pid"),
    ([{"pid": 1, "sessionId": 7}], "sessionId is a int"),
    ([{"pid": 1, "sessionId": A, "kind": 3}], "kind is a int"),
])
def test_parse_agents_names_a_shape_it_does_not_know(doc, words):
    with pytest.raises(watch.ShapeError, match=words):
        watch.parse_agents(doc)


def test_sessions_dir_follows_claude_config_dir(tmp_path):
    assert watch.sessions_dir({"CLAUDE_CONFIG_DIR": "/c"}) == Path("/c/sessions")
    assert watch.sessions_dir({"HOME": "/h"}) == Path("/h/.claude/sessions")


def test_claude_binary_falls_back_to_the_installers_place(tmp_path):
    exe = tmp_path / ".local" / "bin" / "claude"
    exe.parent.mkdir(parents=True)
    env = {"PATH": str(tmp_path / "empty"), "HOME": str(tmp_path)}
    assert watch.claude_binary(env) is None
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert watch.claude_binary(env) == str(exe)


def test_run_agents_with_no_claude(monkeypatch):
    monkeypatch.setattr(watch, "claude_binary", lambda: None)
    with pytest.raises(watch.AgentsUnreadable, match="no claude"):
        watch.run_agents()


def test_run_agents_runs_claude_agents_json(tmp_path, monkeypatch):
    exe = tmp_path / "claude"
    exe.write_text('#!/bin/sh\n[ "$1 $2" = "agents --json" ] || exit 9\n'
                   f"echo '{json.dumps([entry(1, A)])}'\n")
    exe.chmod(0o755)
    monkeypatch.setattr(watch, "claude_binary", lambda: str(exe))
    assert watch.run_agents() == [entry(1, A)]
    exe.write_text("#!/bin/sh\necho broken >&2\nexit 3\n")
    with pytest.raises(watch.AgentsUnreadable, match="exited 3: broken"):
        watch.run_agents()
    exe.write_text("#!/bin/sh\necho not json\n")
    with pytest.raises(watch.AgentsUnreadable, match="printed no JSON"):
        watch.run_agents()


# --- records -------------------------------------------------------------------------------

def test_a_tick_writes_a_record_for_each_live_session(root, sessions):
    write_transcript(A, root=root, lines=[assistant_line(120, now=1_800_000_000.0, tier="1h")])
    fake = Fake([entry(11, A), entry(12, B)])
    live = make(fake, sessions).tick()
    assert live == [watch.Live(11, A), watch.Live(12, B)]
    a = rec(A)
    assert list(a) == list(records.FIELDS)
    assert a == {"v": 1, "session": A, "harness": "claude", "pid": 11, "kind": "warm",
                 "last_request": pytest.approx(1_800_000_000.0 - 120), "ttl": 3600,
                 "read_at": 1_800_000_000.0, "seen_at": 1_800_000_000.0, "code": None,
                 "warn_before": 900, "stale_after": 30}
    b = rec(B)
    assert (b["kind"], b["code"], b["last_request"], b["ttl"]) == ("unknown", "no-record",
                                                                   None, None)
    assert records.valid(a) and records.valid(b)


UNSAFE = ["../escape", "a/b", "..", "sub/../../up", "/abs"]


def test_a_session_id_the_reader_refuses_is_skipped_with_one_log_line(tmp_path, sessions,
                                                                      caplog):
    directory = tmp_path / "rt" / "cache" / "clock"
    fake = Fake([entry(11, A)] + [entry(20 + i, s) for i, s in enumerate(UNSAFE)])
    w = make(fake, sessions, directory=directory)
    before = {p for p in tmp_path.rglob("*")}
    with caplog.at_level(logging.WARNING, logger="moxi.cache"):
        live = w.tick()
        fake.t += 10
        w.tick()
    assert live == [watch.Live(11, A)]
    assert set(fake.reads) == {A}              # an unsafe id is never read either
    assert sorted(p.name for p in directory.iterdir()) == [f"{A}.json"]
    made = {p for p in tmp_path.rglob("*")} - before
    assert made <= {directory, directory.parent, directory.parent.parent,
                    directory / f"{A}.json"}
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 1
    assert all(repr(s) in lines[0] for s in UNSAFE)
    assert "no clock record" in lines[0]


@pytest.mark.parametrize("session", UNSAFE)
def test_write_record_refuses_an_id_that_cannot_name_a_file(tmp_path, session):
    directory = tmp_path / "clock"
    directory.mkdir()
    record = {"v": records.VERSION, "session": session}
    before = set(tmp_path.rglob("*"))
    with pytest.raises(ValueError):
        watch.write_record(directory, record)
    assert set(tmp_path.rglob("*")) == before


def test_seen_at_moves_every_tick_and_read_at_only_when_the_transcript_changes(root, sessions):
    path = write_transcript(A, root=root,
                            lines=[assistant_line(60, now=1_800_000_000.0, tier="1h")])
    fake = Fake([entry(11, A)])
    w = make(fake, sessions)
    w.tick()
    fake.t += 10
    w.tick()
    first = rec(A)
    assert first["seen_at"] == fake.t and first["read_at"] == fake.t - 10
    assert fake.reads == [A]
    with path.open("a") as fh:
        fh.write(json.dumps(assistant_line(0, now=fake.t, tier="1h")) + "\n")
    fake.t += 10
    w.tick()
    second = rec(A)
    assert fake.reads == [A, A]
    assert second["read_at"] == fake.t and second["last_request"] == pytest.approx(fake.t - 10)


def test_an_unknown_read_is_read_again_at_the_next_tick(root, sessions):
    fake = Fake([entry(11, A)])
    w = make(fake, sessions)
    w.tick()
    assert rec(A)["kind"] == "unknown"
    write_transcript(A, root=root, lines=[assistant_line(5, now=fake.t, tier="1h")])
    fake.t += 10
    w.tick()
    assert rec(A)["kind"] == "warm" and fake.reads == [A, A]


def test_a_fresh_session_with_no_transcript_is_read_once_a_transcript_appears(root, sessions):
    fake = Fake([entry(11, A)])
    fake_read = []

    def read(harness, session, *, now):
        fake_read.append(session)
        return Clock("fresh")
    w = watch.Watcher(agents=fake.agents, read=read, now=fake.now, sessions=sessions)
    w.tick()
    w.tick()
    assert fake_read == [A]
    write_transcript(A, root=root, lines=[assistant_line(5, now=fake.t, tier="1h")])
    w.tick()
    assert fake_read == [A, A]


def test_a_session_that_leaves_the_list_loses_its_record(root, sessions):
    fake = Fake([entry(11, A), entry(12, B)])
    w = make(fake, sessions)
    w.tick()
    assert rec(A) and rec(B)
    fake.doc = [entry(12, B)]
    for _ in range(watch.AGENTS_EVERY):
        w.tick()
    assert rec(A) is None and rec(B) is not None


def test_a_clear_moves_the_record_to_the_new_id(root, sessions):
    """After `/clear` Claude Code lists the same pid with a new id and rewrites the pid's
    file in the sessions folder; the next tick asks for the list again."""
    pidfile = sessions / "11.json"
    pidfile.write_text(json.dumps({"pid": 11, "sessionId": A}))
    fake = Fake([entry(11, A)])
    w = make(fake, sessions)
    w.tick()
    fake.doc = [entry(11, C)]
    pidfile.write_text(json.dumps({"pid": 11, "sessionId": C, "status": "idle"}))
    os.utime(pidfile, ns=(1, 1))
    w.tick()
    assert fake.calls == 2
    assert rec(A) is None and rec(C)["pid"] == 11


def test_the_list_is_asked_every_third_tick_or_when_the_sessions_folder_changes(sessions):
    fake = Fake([entry(11, A)])
    w = make(fake, sessions)
    for _ in range(4):
        w.tick()
    assert fake.calls == 2                     # ticks 0 and 3
    (sessions / "99.json").write_text("{}")
    w.tick()
    assert fake.calls == 3
    (sessions / "99.json").unlink()
    w.tick()
    assert fake.calls == 4


def test_a_restart_removes_records_of_sessions_no_longer_live(sessions):
    directory = records.record_dir()
    directory.mkdir(parents=True)
    (directory / f"{B}.json").write_text("{}")
    (directory / f".{A}.123.tmp").write_text("partial")
    (directory / "notes.txt").write_text("left alone")
    make(Fake([entry(11, A)]), sessions).tick()
    assert sorted(p.name for p in directory.iterdir()) == [f"{A}.json", "notes.txt"]


def test_one_record_per_session_when_two_pids_list_it(sessions):
    make(Fake([entry(11, A), entry(12, A)]), sessions).tick()
    assert rec(A)["pid"] == 11
    assert [p.name for p in records.record_dir().iterdir()] == [f"{A}.json"]


def test_a_shape_it_does_not_know_stops_its_writes(sessions, caplog):
    fake = Fake([entry(11, A)])
    w = make(fake, sessions)
    w.tick()
    before = rec(A)
    fake.doc = {"agents": []}
    fake.t += 10
    poke(sessions)
    with caplog.at_level(logging.WARNING, logger="moxi.cache"):
        assert w.tick() == []
        fake.t += 10
        assert w.tick() == []                  # asked again, the same shape
    assert fake.calls == 3
    assert rec(A) == before                    # left standing, so it shows dim
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 1 and "a dict, not a list" in lines[0]
    fake.doc = [entry(11, A)]
    fake.t += 10
    w.tick()
    assert rec(A)["seen_at"] == fake.t


def test_a_list_it_cannot_read_writes_nothing_and_asks_again(sessions, caplog):
    fake = Fake([entry(11, A)])
    w = make(fake, sessions)
    w.tick()
    before = rec(A)
    fake.doc = watch.AgentsUnreadable("`claude agents --json` exited 1")
    poke(sessions)
    with caplog.at_level(logging.WARNING, logger="moxi.cache"):
        fake.t += 10
        assert w.tick() == []
        fake.t += 10
        assert w.tick() == []
    assert fake.calls == 3 and rec(A) == before
    assert len(caplog.records) == 1
    fake.doc = [entry(11, A)]
    fake.t += 10
    w.tick()
    assert fake.calls == 4 and rec(A)["seen_at"] == fake.t


def test_no_list_ever_read_writes_nothing(sessions):
    fake = Fake(watch.AgentsUnreadable("no claude on PATH"))
    assert make(fake, sessions).tick() == []
    assert not records.record_dir().exists()


def test_a_reader_and_the_segment_agree_on_the_record(root, sessions):
    from moxi_cache import segment

    write_transcript(A, root=root, lines=[assistant_line(2800, now=1_800_000_000.0,
                                                          tier="1h")])
    fake = Fake([entry(11, A)])
    make(fake, sessions).tick()
    found = segment.segment(A, now=fake.t)
    assert (found.text, found.band, found.dim) == ("cache:46m40s", "near", False)
    assert segment.segment(A, now=fake.t + 31).dim


def test_every_record_carries_the_watchers_thresholds_at_every_tick(root, sessions):
    write_transcript(A, root=root, lines=[assistant_line(120, now=1_800_000_000.0, tier="1h")])
    fake = Fake([entry(11, A), entry(12, B)])
    w = make(fake, sessions)
    w.warn_before, w.stale_after = 1234, 45
    w.tick()
    for session in (A, B):
        assert (rec(session)["warn_before"], rec(session)["stale_after"]) == (1234, 45)
        assert w.current[session]["warn_before"] == 1234
    # a quiet tick (no read) rewrites them too
    fake.t += 5
    w.warn_before, w.stale_after = 60, 6
    w.tick()
    assert fake.reads.count(A) == 1
    for session in (A, B):
        assert (rec(session)["warn_before"], rec(session)["stale_after"]) == (60, 6)
        assert rec(session)["seen_at"] == fake.t


class Steps(threading.Event):
    """A stop event whose wait returns at once: it notes the wait, runs the next step
    (which may change the config), and sets itself once the steps have run out."""

    def __init__(self, steps):
        super().__init__()
        self.steps = list(steps)
        self.waits: list[float] = []

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.steps:
            self.steps.pop(0)()
        if not self.steps:
            self.set()
        return self.is_set()


def test_a_config_change_reaches_every_record_within_one_tick(own_roots, root, sessions):
    write_transcript(A, root=root, lines=[assistant_line(120, now=1_800_000_000.0, tier="1h")])
    fake = Fake([entry(11, A), entry(12, B)])
    w = make(fake, sessions)
    seen = []

    def note():
        seen.append({s: (rec(s)["warn_before"], rec(s)["stale_after"]) for s in (A, B)})

    def change(text):
        def step():
            note()
            write_cache_config(own_roots, text)
        return step
    write_cache_config(own_roots, "[watch]\ntick = 10\n")
    stop = Steps([change("[segment]\nwarn_before = 1800\n[watch]\ntick = 25\n"),
                  change("[segment]\nwarn_before = 120\n"),
                  note])
    assert watch.run(watcher=w, stop=stop) == 0
    assert seen == [
        {A: (900, 30), B: (900, 30)},
        {A: (1800, 75), B: (1800, 75)},     # both values, at the tick after the change
        {A: (120, 30), B: (120, 30)},       # back to the default tick
    ]
    # the tick that set stale_after is the tick the watcher waited
    assert stop.waits == [10, 25, 10]


def test_stale_after_is_three_effective_ticks(own_roots, sessions):
    write_cache_config(own_roots, "[watch]\ntick = 0\n")
    fake = Fake([entry(11, A)])
    w = make(fake, sessions)
    stop = Steps([])
    assert watch.run(watcher=w, stop=stop) == 0
    assert rec(A)["stale_after"] == 3 and stop.waits == [1]


def test_a_new_watcher_writes_the_defaults(sessions):
    fake = Fake([entry(11, A)])
    make(fake, sessions).tick()
    assert (rec(A)["warn_before"], rec(A)["stale_after"]) == (900, 30)


def test_write_record_replaces_whole(tmp_path):
    directory = tmp_path / "records"
    directory.mkdir()
    r = {"session": A, "v": 1}
    watch.write_record(directory, {"session": A, "v": 0})
    path = watch.write_record(directory, r)
    assert json.loads(path.read_text()) == r
    assert [p.name for p in directory.iterdir()] == [f"{A}.json"]


# --- running ---------------------------------------------------------------------------------

def test_run_ticks_until_stopped(sessions):
    stop = threading.Event()
    ticks = []

    class Once(watch.Watcher):
        def tick(self):
            ticks.append(1)
            stop.set()
            return []
    assert watch.run(watcher=Once(agents=lambda: [], sessions=sessions), stop=stop) == 0
    assert ticks == [1]


def test_run_keeps_going_after_a_failed_tick(own_roots, sessions, caplog):
    stop = threading.Event()
    count = []

    class Flaky(watch.Watcher):
        def tick(self):
            count.append(1)
            if len(count) == 1:
                raise RuntimeError("boom")
            stop.set()
            return []
    write_cache_config(own_roots, "[watch]\ntick = 1\n")
    with caplog.at_level(logging.WARNING, logger="moxi.cache"):
        assert watch.run(watcher=Flaky(agents=lambda: [], sessions=sessions), stop=stop) == 0
    assert len(count) == 2 and "boom" in caplog.text


def test_run_refuses_to_start_on_a_config_in_error(own_roots, capsys):
    write_cache_config(own_roots, "[watch]\ntick = 'ten'\n")
    assert watch.run(stop=threading.Event()) == 2
    assert "refuses the start" in capsys.readouterr().err


def test_a_tick_below_one_second_ticks_every_second(own_roots, sessions, caplog):
    from moxi.config import Config

    from moxi_cache.settings import MODULE, schema

    write_cache_config(own_roots, "[watch]\ntick = 0\n")
    config = Config(MODULE, schema())
    config.start()
    w = watch.Watcher(agents=lambda: [], sessions=sessions)
    with caplog.at_level(logging.WARNING, logger="moxi.cache"):
        assert watch._interval(config, w) == 1.0
    assert "watch.tick is 0" in caplog.text


def test_main_takes_no_arguments(capsys):
    assert watch.main(["--now"]) == 2
    assert watch.main(["--help"]) == 0
