"""The timer: subscriptions from modules and the config, checked at load; when it asks,
`requires_env`, the detached start, the timer log, `moxi cache timers` and doctor's lines."""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from conftest import write_cache_config

from moxi_cache import cli, timer, watch
from moxi_cache.timer import Refusal, Subscription, Timer

A = "aaaaaaaa-1111-4111-8111-111111111111"
B = "bbbbbbbb-2222-4222-8222-222222222222"
T0 = 1_800_000_000.0
TTL = 3600.0


@pytest.fixture
def exe(tmp_path) -> Path:
    """An executable file to name as run[0]."""
    path = tmp_path / "bin" / "verb"
    path.parent.mkdir()
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def sub(exe, **fields) -> dict:
    out = {"name": "s", "at": 0, "run": [str(exe), "--session", "{session}"]}
    out.update(fields)
    return out


def checked(raw) -> Subscription | Refusal:
    return timer.check(raw, source=timer.CONFIG, resolve=timer.resolve_config_argv0)


def record(session=A, pid=4242, *, last_request=T0, ttl=TTL, kind="warm") -> dict:
    return {"v": 1, "session": session, "harness": "claude", "pid": pid, "kind": kind,
            "last_request": last_request, "ttl": ttl, "read_at": T0, "seen_at": T0,
            "code": None}


# --- checking ---------------------------------------------------------------------------------

def test_a_config_subscription_checks_and_fills_its_slots(exe):
    s = checked(sub(exe, run=[str(exe), "x-{session}", "--pid={pid}", "{other}"],
                    until=60, every=120, requires_env={"V": "on", "W": "*"}))
    assert isinstance(s, Subscription)
    assert (s.at, s.until, s.every, s.argv0) == (0.0, 60.0, 120.0, str(exe))
    assert s.requires_env == (("V", "on"), ("W", "*"))
    assert s.argv(A, 7) == [str(exe), f"x-{A}", "--pid=7", "{other}"]


@pytest.mark.parametrize("change, words", [
    ({"run": ["~/bin/verb"]}, "starts with ~"),
    ({"run": ["bin/verb"]}, "not an absolute path"),
    ({"run": ["verb"]}, "not an absolute path"),
    ({"run": ["/nonexistent/verb"]}, "not an executable file"),
    ({"run": []}, "run is not a list"),
    ({"run": "/bin/true"}, "run is not a list"),
    ({"run": ["/bin/true", 3]}, "run is not a list"),
    ({"name": "a b"}, "name 'a b'"),
    ({"name": ""}, "name ''"),
    ({"at": True}, "at is True"),
    ({"at": "0"}, "at is '0'"),
    ({"at": float("nan")}, "at is nan"),
    ({"until": 0}, "until (0) is not after at (0)"),
    ({"until": -10}, "is not after at"),
    ({"every": 0}, "every (0) is not a positive"),
    ({"requires_env": ["V"]}, "requires_env is not a table"),
    ({"requires_env": {"1V": "on"}}, "not a variable name"),
    ({"requires_env": {"V": 1}}, "not a string"),
    ({"bogus": 1}, "fields a subscription does not take (bogus)"),
])
def test_a_config_subscription_is_refused(exe, change, words):
    s = checked(sub(exe, **change))
    assert isinstance(s, Refusal), s
    assert words in s.reason and s.source == "config"


def test_missing_fields_and_a_non_table_are_refused(exe):
    s = checked({"name": "x", "run": [str(exe)]})
    assert isinstance(s, Refusal) and "lacks at" in s.reason and s.name == "x"
    s = checked("a-string")
    assert isinstance(s, Refusal) and "not a table" in s.reason and s.name is None
    assert timer.from_config({"name": "x"})[0].reason == \
        "timer.subscriber is not a list of tables"


def test_a_file_that_is_not_executable_is_refused(exe):
    exe.chmod(0o644)
    s = checked(sub(exe))
    assert isinstance(s, Refusal) and "not an executable file" in s.reason
    s = checked(sub(exe, run=[str(exe.parent)]))
    assert isinstance(s, Refusal) and "not an executable file" in s.reason


def test_a_taken_name_is_refused_and_the_others_kept(exe):
    loaded = timer.load([sub(exe), sub(exe, at=5), sub(exe, name="t", run=["rel"])],
                        modules=[])
    assert [s.name for s in loaded.subscriptions] == ["s"]
    assert [r.name for r in loaded.refused] == ["s", "t"]
    assert "already taken by a subscription from config" in loaded.refused[0].reason


# --- modules ----------------------------------------------------------------------------------

class Script:
    def __init__(self, name, group="console_scripts"):
        self.name, self.group = name, group


class Dist:
    def __init__(self, *scripts):
        self.entry_points = [Script(n) for n in scripts] + [Script("cache", "moxi.modules")]


class EP:
    """A fake entry point in `moxi.cache.timers`."""

    def __init__(self, name, target=None, *, scripts=(), error=None):
        self.name, self.value, self.dist = name, f"pkg_{name}.timers", Dist(*scripts)
        self._target, self._error = target, error

    def load(self):
        if self._error:
            raise self._error
        return self._target


class Module:
    def __init__(self, timers):
        self.TIMERS = timers


@pytest.fixture
def scripts(tmp_path):
    """A scripts directory, and `find` resolving names in it as `moxi.paths.script` does."""
    where = tmp_path / "scripts"
    where.mkdir()

    def make(name, mode=0o755):
        path = where / name
        path.write_text("#!/bin/sh\n")
        path.chmod(mode)
        return path

    def find(name):
        path = where / name
        if not path.is_file():
            raise FileNotFoundError(f"{name} is not installed in this environment "
                                    f"(looked for {path})")
        return path

    return make, find


def test_a_modules_subscription_resolves_its_own_script(scripts):
    make, find = scripts
    path = make("moxi-lc-clear")
    ep = EP("lc", Module(({"name": "recycle", "at": 0, "every": 120,
                           "requires_env": {"RECYCLE": "on"},
                           "run": ["moxi-lc-clear", "--session", "{session}"]},)),
            scripts=["moxi-lc-clear"])
    (s,) = timer.from_modules([ep], find)
    assert isinstance(s, Subscription)
    assert (s.source, s.argv0, s.run[0]) == ("module lc", str(path), "moxi-lc-clear")
    assert s.argv(A, 1)[0] == str(path)


@pytest.mark.parametrize("argv0, scripts_of, words", [
    ("/abs/moxi-lc-clear", ["moxi-lc-clear"], "not a console script's bare name"),
    ("bin/moxi-lc-clear", ["moxi-lc-clear"], "not a console script's bare name"),
    ("moxi-other", ["moxi-lc-clear"], "not one of the module's own console scripts"),
    ("moxi-lc-gone", ["moxi-lc-gone"], "is not installed in this environment"),
    ("moxi-lc-plain", ["moxi-lc-plain"], "not an executable file"),
])
def test_a_modules_run0_is_refused(scripts, argv0, scripts_of, words):
    make, find = scripts
    make("moxi-lc-clear")
    make("moxi-other")
    make("moxi-lc-plain", 0o644)
    ep = EP("lc", Module(({"name": "r", "at": 0, "run": [argv0]},)), scripts=scripts_of)
    (s,) = timer.from_modules([ep], find)
    assert isinstance(s, Refusal) and words in s.reason, s


def test_an_entry_point_that_does_not_load_or_holds_no_timers(scripts):
    _, find = scripts
    out = timer.from_modules([EP("b", error=ImportError("no module")),
                              EP("a", Module("not a tuple"))], find)
    assert [(r.source, isinstance(r, Refusal)) for r in out] == [
        ("module a", True), ("module b", True)]
    assert "holds no TIMERS tuple" in out[0].reason
    assert "does not load (ImportError: no module)" in out[1].reason


def test_discovery_reads_the_entry_point_group_from_installed_metadata(tmp_path, monkeypatch,
                                                                       scripts):
    """A distribution on sys.path with an entry point in `moxi.cache.timers`: discovery
    finds it, loads its TIMERS, and holds run[0] to the distribution's own scripts."""
    make, find = scripts
    site = tmp_path / "site"
    info = site / "moxi_b11probe-0.1.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: moxi-b11probe\n"
                                   "Version: 0.1\n")
    (info / "entry_points.txt").write_text(
        "[console_scripts]\nmoxi-b11probe-run = b11probe_timers:main\n\n"
        "[moxi.cache.timers]\nb11probe = b11probe_timers\n")
    (site / "b11probe_timers.py").write_text(
        "TIMERS = ({'name': 'probe', 'at': -60, 'run': ['moxi-b11probe-run', '{session}']},\n"
        "          {'name': 'stranger', 'at': 0, 'run': ['moxi-cache-watch']})\n")
    monkeypatch.syspath_prepend(str(site))
    from importlib import metadata

    eps = [ep for ep in metadata.entry_points(group=timer.GROUP) if ep.name == "b11probe"]
    assert len(eps) == 1
    make("moxi-b11probe-run")
    make("moxi-cache-watch")
    probe, stranger = timer.from_modules(eps, find)
    assert isinstance(probe, Subscription) and probe.source == "module b11probe"
    assert probe.argv0.endswith("/scripts/moxi-b11probe-run")
    assert isinstance(stranger, Refusal)
    assert "not one of the module's own console scripts (moxi-b11probe-run)" in \
        stranger.reason
    # Resolved by moxi.paths.script, in this environment, which holds no such script.
    (unresolved, _) = timer.from_modules(eps)
    assert isinstance(unresolved, Refusal)
    assert "moxi-b11probe-run is not installed in this environment" in unresolved.reason


# --- when it asks -------------------------------------------------------------------------------

class Proc:
    def __init__(self, pid, code=None):
        self.pid, self.code = pid, code

    def poll(self):
        return self.code


class Spawner:
    def __init__(self):
        self.argvs: list[list[str]] = []
        self.procs: list[Proc] = []
        self.error: OSError | None = None

    def __call__(self, argv):
        if self.error:
            raise self.error
        self.argvs.append(argv)
        proc = Proc(1000 + len(self.procs))
        self.procs.append(proc)
        return proc


def make_timer(tmp_path, subs, *, environ=None, entries=None):
    spawner = Spawner()
    config = [entries if entries is not None else []]
    t = Timer(config_entries=lambda: config[0], modules=subs,
              environ=environ or (lambda pid: {}), spawn=spawner,
              log_file=tmp_path / "state" / "timer.jsonl")
    return t, spawner, config


def raw_sub(exe, **fields):
    return checked(sub(exe, **fields))


def asks(t, live, now):
    return [(e["subscriber"], e["session"]) for e in t.tick(live, now)]


def test_it_asks_once_on_the_first_tick_at_or_past_at(tmp_path, exe):
    t, spawner, _ = make_timer(tmp_path, [raw_sub(exe, name="r", at=0)])
    live = {A: record()}
    assert asks(t, live, T0 + TTL - 5) == []
    assert asks(t, live, T0 + TTL + 3) == [("r", A)]
    assert spawner.argvs == [[str(exe), "--session", A]]
    assert asks(t, live, T0 + TTL + 13) == []
    assert asks(t, live, T0 + TTL + 5000) == []


def test_every_repeats_from_the_last_ask_while_the_window_is_open(tmp_path, exe):
    t, _, _ = make_timer(tmp_path, [raw_sub(exe, name="r", at=0, every=120)])
    live = {A: record()}
    times = [T0 + TTL + d for d in (0, 10, 110, 120, 130, 240, 250)]
    got = [bool(asks(t, live, now)) for now in times]
    assert got == [True, False, False, True, False, True, False]


def test_until_closes_the_window(tmp_path, exe):
    t, _, _ = make_timer(tmp_path, [raw_sub(exe, name="kw", at=-900, until=0, every=120)])
    live = {A: record()}
    times = [T0 + TTL + d for d in (-901, -900, -780, -660, -120, -1, 0, 120)]
    got = [bool(asks(t, live, now)) for now in times]
    assert got == [False, True, True, True, True, False, False, False]


def test_a_moved_clock_opens_a_new_window(tmp_path, exe):
    t, _, _ = make_timer(tmp_path, [raw_sub(exe, name="r", at=0)])
    assert asks(t, {A: record()}, T0 + TTL) == [("r", A)]
    later = T0 + 4000
    assert asks(t, {A: record(last_request=later)}, later + 10) == []
    assert asks(t, {A: record(last_request=later)}, later + TTL) == [("r", A)]
    # A re-read that gives the same clock within a rounding error is the same clock.
    assert asks(t, {A: record(last_request=later + 0.0001)}, later + TTL + 10) == []


def test_a_session_that_leaves_the_list_is_forgotten(tmp_path, exe):
    t, _, _ = make_timer(tmp_path, [raw_sub(exe, name="r", at=0)])
    assert asks(t, {A: record(), B: record(B)}, T0 + TTL) == [("r", A), ("r", B)]
    assert asks(t, {B: record(B)}, T0 + TTL + 10) == []
    assert ("r", A) not in t.tracks
    assert asks(t, {A: record(), B: record(B)}, T0 + TTL + 20) == [("r", A)]


def test_a_tick_with_no_live_list_changes_nothing(tmp_path, exe):
    t, _, _ = make_timer(tmp_path, [raw_sub(exe, name="r", at=0)])
    assert asks(t, {A: record()}, T0 + TTL) == [("r", A)]
    assert asks(t, None, T0 + TTL + 10) == []
    assert ("r", A) in t.tracks
    assert asks(t, {A: record()}, T0 + TTL + 20) == []


def test_after_a_restart_it_asks_inside_a_window_and_not_for_a_closed_one(tmp_path, exe):
    subs = [raw_sub(exe, name="r", at=0), raw_sub(exe, name="kw", at=-900, until=0)]
    t, _, _ = make_timer(tmp_path, subs)
    assert asks(t, {A: record()}, T0 + TTL - 100) == [("kw", A)]
    fresh, _, _ = make_timer(tmp_path, subs)        # the watcher restarted
    assert asks(fresh, {A: record()}, T0 + TTL - 50) == [("kw", A)]
    again, _, _ = make_timer(tmp_path, subs)
    assert asks(again, {A: record()}, T0 + TTL + 30) == [("r", A)]


@pytest.mark.parametrize("rec", [record(last_request=None, kind="fresh"),
                                 record(ttl=None, kind="unknown"),
                                 record(last_request=None, ttl=None, kind="unknown")])
def test_a_session_with_no_clock_is_not_asked_about(tmp_path, exe, rec):
    t, _, _ = make_timer(tmp_path, [raw_sub(exe, name="r", at=-1e9)])
    assert asks(t, {A: rec}, T0 + TTL) == []


@pytest.mark.parametrize("want, env, met", [
    ({"V": "on"}, {"V": "on"}, True),
    ({"V": "on"}, {"V": "off"}, False),
    ({"V": "on"}, {}, False),
    ({"V": "*"}, {"V": "/some/dir"}, True),
    ({"V": "*"}, {"V": ""}, False),
    ({"V": "on", "W": "*"}, {"V": "on"}, False),
    ({"V": "on"}, None, False),
])
def test_requires_env_reads_the_sessions_process(tmp_path, exe, want, env, met):
    seen = []

    def environ(pid):
        seen.append(pid)
        return env

    t, _, _ = make_timer(tmp_path, [raw_sub(exe, name="r", at=0, requires_env=want)],
                         environ=environ)
    assert bool(asks(t, {A: record(pid=777)}, T0 + TTL)) is met
    assert seen == [777]


def test_requires_env_not_met_is_checked_again_at_the_next_tick(tmp_path, exe):
    env = {}
    t, _, _ = make_timer(tmp_path, [raw_sub(exe, name="r", at=0, requires_env={"V": "on"})],
                         environ=lambda pid: env)
    assert asks(t, {A: record()}, T0 + TTL) == []
    env["V"] = "on"
    assert asks(t, {A: record()}, T0 + TTL + 10) == [("r", A)]


def test_proc_environ_reads_a_process_and_none_for_a_gone_one(tmp_path):
    env = timer.proc_environ(os.getpid())
    assert env is not None and env.get("PATH") == os.environ.get("PATH")
    fake = tmp_path / "proc" / "12"
    fake.mkdir(parents=True)
    (fake / "environ").write_bytes(b"A=1\0B=x=y\0junk\0C=\0")
    assert timer.proc_environ(12, tmp_path / "proc") == {"A": "1", "B": "x=y", "C": ""}
    assert timer.proc_environ(13, tmp_path / "proc") is None


# --- the log ------------------------------------------------------------------------------------

def logged(t) -> list[dict]:
    return timer.read_log(t.log_file)


def test_one_line_per_ask_when_the_process_exits(tmp_path, exe):
    t, spawner, _ = make_timer(tmp_path, [raw_sub(exe, name="r", at=0)])
    t.tick({A: record(pid=55)}, T0 + TTL + 2)
    assert logged(t) == []                       # it does not wait
    spawner.procs[0].code = 3
    t.tick({A: record(pid=55)}, T0 + TTL + 12)
    (line,) = logged(t)
    assert line == {"v": 1, "at": T0 + TTL + 2, "subscriber": "r", "source": "config",
                    "session": A, "pid": 55, "age": TTL + 2, "last_request": T0,
                    "child": 1000, "exit": 3, "ended": T0 + TTL + 12, "error": None}


def test_a_start_that_fails_is_logged_at_once(tmp_path, exe):
    t, spawner, _ = make_timer(tmp_path, [raw_sub(exe, name="r", at=0)])
    spawner.error = PermissionError(13, "Permission denied")
    t.tick({A: record()}, T0 + TTL)
    (line,) = logged(t)
    assert line["child"] is None and line["exit"] is None
    assert line["error"] == f"could not start {exe}: Permission denied"


def test_close_logs_what_still_runs(tmp_path, exe):
    t, spawner, _ = make_timer(tmp_path, [raw_sub(exe, name="r", at=0)])
    t.tick({A: record()}, T0 + TTL)
    t.close(T0 + TTL + 1)
    (line,) = logged(t)
    assert line["exit"] is None and line["error"] == "still running when the watcher stopped"
    assert t.pending == []


def test_the_log_moves_aside_past_its_limit(tmp_path):
    path = tmp_path / "timer.jsonl"
    for i in range(5):
        timer.append_log(path, {"i": i}, limit=9)
    assert path.with_name("timer.jsonl.1").exists()
    assert [e["i"] for e in timer.read_log(path)] == [4]


def test_read_log_skips_a_cut_first_line_and_junk(tmp_path):
    path = tmp_path / "timer.jsonl"
    path.write_text('{"i": 1}\nnot json\n[1]\n{"i": 2}\n')
    assert timer.read_log(path) == [{"i": 1}, {"i": 2}]
    assert timer.read_log(path, tail=12) == [{"i": 2}]
    assert timer.read_log(tmp_path / "absent") == []


def test_the_log_is_under_caches_state_directory(own_roots):
    assert timer.log_path() == own_roots / "state" / timer.log_path().parent.parent.name \
        / "cache" / "timer.jsonl"


# --- a real start -------------------------------------------------------------------------------

def test_the_run_starts_detached_with_no_shell(tmp_path):
    out = tmp_path / "out.json"
    probe = tmp_path / "probe"
    probe.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "json.dump({'argv': sys.argv[1:], 'sid': os.getsid(0), 'pid': os.getpid(),\n"
        "           'stdin': os.readlink('/proc/self/fd/0'),\n"
        "           'stdout': os.readlink('/proc/self/fd/1')},\n"
        f"          open({str(out)!r}, 'w'))\n")
    probe.chmod(0o755)
    s = checked({"name": "probe", "at": 0, "run": [str(probe), "{session}", "$HOME", "a b",
                                                   "{pid}"]})
    t = Timer(config_entries=lambda: [], modules=[s], log_file=tmp_path / "timer.jsonl")
    (entry,) = t.tick({A: record(pid=31)}, T0 + TTL)
    deadline = time.time() + 10
    while not t.pending == [] and time.time() < deadline:
        time.sleep(0.05)
        t.reap()
    assert t.pending == []
    got = json.loads(out.read_text())
    assert got["argv"] == [A, "$HOME", "a b", "31"]
    assert got["sid"] == got["pid"] == entry["child"]       # a session of its own
    assert got["stdin"] == got["stdout"] == "/dev/null"
    (line,) = timer.read_log(tmp_path / "timer.jsonl")
    assert line["exit"] == 0 and line["child"] == entry["child"]


# --- loading: the same check at each tick as `timers` and doctor run ------------------------

def test_the_config_is_loaded_again_when_it_changes(tmp_path, exe, caplog):
    caplog.set_level(logging.INFO, logger="moxi.cache")
    t, _, config = make_timer(tmp_path, [], entries=[sub(exe, name="r"),
                                                     sub(exe, name="bad", run=["~/x"])])
    t.tick({}, T0)
    t.tick({}, T0 + 10)
    refusals = [r for r in caplog.messages if "refused subscription" in r]
    assert refusals == ["timer: refused subscription bad (config): run[0] '~/x' starts "
                        "with ~, which nothing expands: write the absolute path"]
    assert [s.name for s in t.loaded.subscriptions] == ["r"]
    t.tick({A: record()}, T0 + TTL)
    assert ("r", A) in t.tracks
    config[0] = [sub(exe, name="q")]
    t.tick({A: record()}, T0 + TTL + 10)
    assert [s.name for s in t.loaded.subscriptions] == ["q"]
    assert ("r", A) not in t.tracks and ("q", A) in t.tracks


def test_a_refusal_lasts_only_while_its_cause_does(tmp_path, caplog):
    """B11 verification [L2]: a run[0] missing at the first tick (a venv mid-rebuild) is
    refused, and taken up at the next tick once it is an executable file; refused again
    when it goes. Each refusal is logged once when it appears, not once per tick."""
    caplog.set_level(logging.INFO, logger="moxi.cache")
    late = tmp_path / "bin" / "late"
    t, spawner, _ = make_timer(tmp_path, [], entries=[sub(late, name="late", at=0,
                                                          every=60)])
    live = {A: record()}
    assert asks(t, live, T0 + TTL) == [] and asks(t, live, T0 + TTL + 1) == []
    assert [r.name for r in t.loaded.refused] == ["late"]
    late.parent.mkdir()
    late.write_text("#!/bin/sh\nexit 0\n")
    late.chmod(0o755)
    assert asks(t, live, T0 + TTL + 2) == [("late", A)]
    assert [s.name for s in t.loaded.subscriptions] == ["late"] and not t.loaded.refused
    late.chmod(0o644)
    assert asks(t, live, T0 + TTL + 100) == [] and asks(t, live, T0 + TTL + 101) == []
    assert [r.name for r in t.loaded.refused] == ["late"]
    refusals = [m for m in caplog.messages if "refused subscription late" in m]
    assert len(refusals) == 2 and all("not an executable file" in m for m in refusals)
    assert len(spawner.argvs) == 1


def test_a_module_installed_while_the_watcher_runs_is_taken_up(tmp_path, exe):
    """B11 verification [L2]: the modules' subscriptions are read at each tick, as
    `moxi cache timers` reads them, not once at the watcher's start."""
    installed: list = []
    mod = timer.check(sub(exe, name="m", at=0), source="module m",
                      resolve=lambda argv0: argv0)
    t, _, _ = make_timer(tmp_path, lambda: list(installed))
    live = {A: record()}
    assert asks(t, live, T0 + TTL) == []
    installed.append(mod)
    assert asks(t, live, T0 + TTL + 1) == [("m", A)]
    installed.clear()
    t.tick(live, T0 + TTL + 2)
    assert t.loaded == timer.Loaded() and ("m", A) not in t.tracks


def test_the_watcher_and_timers_judge_by_the_same_check(own_roots, tmp_path):
    """B11 verification [L2]'s probe: after each tick, the watcher's timer runs what
    `moxi cache timers` lists, refused or not, at the same config and files."""
    late = tmp_path / "late-bin" / "verb"
    write_cache_config(own_roots, f"""
[[timer.subscriber]]
name = "late"
at = 0
run = ["{late}", "{{session}}"]
""")
    t = Timer(config_entries=lambda: cli._config_entries()[0], spawn=Spawner(),
              log_file=tmp_path / "timer.jsonl")

    def both():
        view = cli.timers_view(T0)
        return ([s.name for s in t.loaded.subscriptions],
                [r.name for r in t.loaded.refused],
                [s["name"] for s in view["subscriptions"]],
                [r["name"] for r in view["refused"]])

    t.tick({}, T0)
    assert both() == ([], ["late"], [], ["late"])
    late.parent.mkdir()
    late.write_text("#!/bin/sh\nexit 0\n")
    late.chmod(0o755)
    t.tick({}, T0 + 1)
    assert both() == (["late"], [], ["late"], [])


def test_the_schema_takes_subscriber_tables(own_roots, exe):
    from moxi_cache.settings import current

    write_cache_config(own_roots, f"""
[[timer.subscriber]]
name = "on-expiry"
at = 0
every = 120
requires_env = {{ RECYCLE_ON_EXPIRY = "on" }}
run = ["{exe}", "clear-expired", "--session", "{{session}}"]
""")
    (entry,) = current()["timer.subscriber"]
    s = checked(entry)
    assert isinstance(s, Subscription)
    assert s.argv(A, 1) == [str(exe), "clear-expired", "--session", A]


# --- the watcher runs it ------------------------------------------------------------------------

def test_the_watcher_hands_its_records_to_the_timer(tmp_path):
    from test_watch import Fake, entry

    root = tmp_path / "claude-config" / "projects"
    root.mkdir(parents=True)
    fake = Fake([entry(11, A)])
    w = watch.Watcher(agents=fake.agents, read=fake.read, now=fake.now,
                      projects_root=lambda: str(root), directory=tmp_path / "clock",
                      sessions=tmp_path / "sessions")
    w.tick()
    assert set(w.current) == {A} and w.current[A]["pid"] == 11
    assert w.tick_at == fake.t
    fake.doc = watch.AgentsUnreadable("no claude")
    w._ask = True
    assert w.tick() == [] and w.current is None


def test_run_ticks_the_timer_after_the_watcher_and_closes_it(tmp_path):
    stop = threading.Event()
    calls = []

    class W:
        directory = tmp_path
        current = {A: record()}
        tick_at = T0

        def tick(self):
            calls.append("watch")

        def say(self, topic, message):
            if message:
                calls.append(message)

    class T:
        def tick(self, live, now):
            calls.append(("timer", live, now))
            stop.set()

        def close(self):
            calls.append("close")

    assert watch.run(watcher=W(), timer=T(), stop=stop) == 0
    assert calls == ["watch", ("timer", {A: record()}, T0), "close"]


def test_a_failing_timer_tick_is_logged_and_the_watcher_goes_on(tmp_path):
    stop = threading.Event()
    said = []

    class W:
        directory = tmp_path
        current = None
        tick_at = T0
        n = 0

        def tick(self):
            self.n += 1
            if self.n == 2:
                stop.set()

        def say(self, topic, message):
            said.append((topic, message))

    class T:
        closed = False

        def tick(self, live, now):
            raise ValueError("boom")

        def close(self):
            T.closed = True

    write_cache_config(Path(os.environ["MOXI_CACHE_HOME"]), "[watch]\ntick = 0\n")
    assert watch.run(watcher=W(), timer=T(), stop=stop) == 0
    assert ("timer-error", "the timer's tick failed (ValueError: boom)") in said
    assert T.closed


# --- moxi cache timers and doctor ---------------------------------------------------------------

def run_cli(capsys, *argv):
    code = cli.main(list(argv), prog="moxi cache")
    out = capsys.readouterr()
    return code, out.out


def put_record(rec):
    from moxi_cache import records

    directory = records.record_dir()
    directory.mkdir(parents=True, exist_ok=True)
    watch.write_record(directory, rec)


def test_timers_lists_subscriptions_refusals_sessions_and_next_asks(own_roots, exe, capsys):
    write_cache_config(own_roots, f"""
[[timer.subscriber]]
name = "on-expiry"
at = 0
every = 120
requires_env = {{ MOXI_B11_UNSET_VAR = "on" }}
run = ["{exe}", "clear-expired", "--session", "{{session}}"]

[[timer.subscriber]]
name = "before-expiry"
at = -900
until = 0
every = 120
run = ["{exe}", "keep-warm", "--session", "{{session}}"]

[[timer.subscriber]]
name = "tilde"
at = 0
run = ["~/bin/x"]
""")
    now = time.time()
    put_record(record(A, os.getpid(), last_request=now - 100))
    put_record(record(B, os.getpid(), last_request=now - TTL + 300))
    code, out = run_cli(capsys, "timers")
    assert code == 0
    assert "subscriptions: 2" in out
    assert "on-expiry (config): at +0s from expiry, every 120s, requires " \
           "MOXI_B11_UNSET_VAR=on" in out
    assert "refused: 1" in out and "tilde (config): run[0] '~/bin/x' starts with ~" in out
    assert "live sessions: 2" in out
    assert "on-expiry: not asked: the process lacks MOXI_B11_UNSET_VAR=on" in out
    assert "before-expiry: asks when the window opens, in 43m" in out
    assert "before-expiry: window open: asks at the watcher's next tick" in out

    code, out = run_cli(capsys, "timers", "--json")
    doc = json.loads(out)
    assert [s["name"] for s in doc["subscriptions"]] == ["on-expiry", "before-expiry"]
    assert doc["refused"][0]["name"] == "tilde"
    states = {(s["session"], a["subscriber"]): a["state"] for s in doc["sessions"]
              for a in s["asks"]}
    assert states == {(A, "on-expiry"): "env-not-met", (A, "before-expiry"): "waiting",
                      (B, "on-expiry"): "env-not-met", (B, "before-expiry"): "due"}


def test_timers_reads_the_log_for_the_last_ask(own_roots, exe):
    s = checked(sub(exe, name="kw", at=-900, until=0, every=120))
    once = checked(sub(exe, name="once", at=-900))
    now = T0 + TTL - 500
    rec = record()
    ask = {"subscriber": "kw", "session": A, "at": now - 30, "last_request": T0}
    assert timer.plan(s, rec, now, {}, [ask])["state"] == "repeating"
    assert timer.plan(s, rec, now, {}, [ask])["next_at"] == now + 90
    old = dict(ask, last_request=T0 - 5000)                 # an ask on an older clock
    assert timer.plan(s, rec, now, {}, [old])["state"] == "due"
    assert timer.plan(once, rec, now, {}, [dict(ask, subscriber="once")])["state"] == "asked"
    assert timer.plan(s, rec, T0 + TTL + 1, {}, [])["state"] == "closed"
    assert timer.plan(s, record(last_request=None), now, {}, [])["state"] == "no-clock"
    late = dict(ask, at=T0 + TTL - 60)
    assert timer.plan(s, rec, T0 + TTL - 30, {}, [late])["state"] == "asked"


def test_timers_with_no_watcher_and_no_subscriptions(capsys):
    code, out = run_cli(capsys, "timers")
    assert code == 0
    assert "subscriptions: 0 (none: nothing is asked)" in out
    assert "live sessions: none in the watcher's records" in out


def test_timers_with_a_config_in_error(own_roots, capsys):
    write_cache_config(own_roots, "[[timer.subscriber]\n")
    code, out = run_cli(capsys, "timers")
    assert code == 0 and out.startswith("cache's config cannot be read")


def test_doctor_lists_the_subscriptions_and_the_refused(own_roots, exe, capsys):
    write_cache_config(own_roots, f"""
[[timer.subscriber]]
name = "ok"
at = 0
run = ["{exe}"]

[[timer.subscriber]]
name = "rel"
at = 0
run = ["bin/verb"]
""")
    code, out = run_cli(capsys, "doctor")
    assert code == 0
    assert "timer: 1 subscription(s), 0 from modules, 1 from your config: ok" in out
    assert "timer: refused rel (config): run[0] 'bin/verb' is not an absolute path" in out
    assert "timer log: " in out and "(no ask yet)" in out
