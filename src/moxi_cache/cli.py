"""`moxi cache`: read a session's cache, show it on a status line, keep the watcher, and
check cache's own state.

    moxi cache read SESSION... [--harness claude|codex|opencode] [--source PATH] [--json]
    moxi cache segment SESSION [--color|--json]
    moxi cache watch
    moxi cache service install|remove|status
    moxi cache timers [--json]
    moxi cache doctor

`read` prints each session's kind, age and lifetime, or why it reads `no-cache` or
`unknown`, with the reason's code. It passes no evidence of the session's own, so a
session with no record reads `unknown`, and it opens the drift record read-only, so it
writes nothing. `--source` is where the record is read from: the Claude projects root
(default: sessions' root), the Codex sessions directory, or the OpenCode database.

`segment` prints a Claude session's status line segment from the watcher's clock record
(`cache:12m3s`), nothing when there is none. It installs the render audit hook before it
loads the segment module, so the render can read and do nothing else.

`watch` runs the watcher in the foreground; `service` installs it as a systemd user unit,
removes it, or says how it stands.

`timers` lists the timer's subscriptions (the modules' and your config's), the refused
ones and why, the live sessions in the watcher's records, and when each subscription
next asks about each session. It checks the subscriptions as the watcher does at its
load, in this environment, and reads the records and the timer log; it starts nothing.

`doctor` shows the config file, the lifetime table's load, the drift record and whether a
process is learning into it, the Claude projects root, whether a watcher is keeping
clock records, and the timer's subscriptions with the refused ones. It exits 1 when
something there stops reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence

from moxi.config import ConfigError

from . import api, drift, records, service, table
from . import timer as _timer
from .clock import COLD, WARM, Clock
from .settings import current


def main(argv: Sequence[str] | None = None, *, prog: str = "moxi cache") -> int:
    parser = argparse.ArgumentParser(
        prog=prog, description="Know when a session's prompt cache goes cold.")
    verbs = parser.add_subparsers(dest="verb", metavar="VERB")
    verbs.required = True

    read = verbs.add_parser("read", help="print each session's cache kind, age and lifetime",
                            description="Print each session's cache kind, age and lifetime, "
                                        "or why it cannot be read. Writes nothing.")
    read.add_argument("sessions", nargs="+", metavar="SESSION",
                      help="the session or thread id")
    read.add_argument("--harness", choices=api.HARNESSES, default="claude",
                      help="whose session it is (default: claude)")
    read.add_argument("--source", metavar="PATH",
                      help="where the record is: the Claude projects root, the Codex "
                           "sessions directory, or the OpenCode database")
    read.add_argument("--json", action="store_true", help="print one JSON object")

    seg = verbs.add_parser("segment", help="print a session's status line segment",
                           description="Print a Claude session's status line segment from "
                                       "the watcher's clock record; nothing when there is "
                                       "none. The render can read and do nothing else.")
    seg.add_argument("session", metavar="SESSION", help="the Claude session id")
    how = seg.add_mutually_exclusive_group()
    how.add_argument("--color", action="store_true",
                     help="in the band's colour, dim when the watcher has stopped")
    how.add_argument("--json", action="store_true", help="print one JSON object, or null")

    verbs.add_parser("watch", help="run the watcher in the foreground",
                     description="Keep a clock record for every live Claude session, in "
                                 "the foreground, until SIGTERM or Ctrl-C.")

    svc = verbs.add_parser("service", help="the watcher as a systemd user service",
                           description="Install, remove or show the watcher's systemd user "
                                       "unit, moxi-cache-watch.service.")
    svc.add_argument("action", choices=service.ACTIONS)

    tim = verbs.add_parser("timers", help="list the timer's subscriptions and next asks",
                           description="List the timer's subscriptions, the refused ones, "
                                       "the live sessions in the watcher's records, and "
                                       "when each subscription next asks about each. "
                                       "Starts nothing.")
    tim.add_argument("--json", action="store_true", help="print one JSON object")

    verbs.add_parser("doctor", help="check cache's config, table, drift record and watcher",
                     description="Check cache's config, lifetime table, drift record and "
                                 "watcher.")

    args = parser.parse_args(argv)
    if args.verb == "read":
        return _read(args)
    if args.verb == "segment":
        return _segment(args)
    if args.verb == "watch":
        from . import watch

        return watch.main([])
    if args.verb == "service":
        return service.run(args.action)
    if args.verb == "timers":
        return _timers(args)
    return _doctor()


# --- read -----------------------------------------------------------------------------

def _read(args) -> int:
    store = api.open_store(learn=False)
    results = []
    for session in args.sessions:
        kw = {"store": store}
        if args.harness == "claude":
            kw["projects_root"] = args.source
        else:
            kw["cache_source"] = args.source
        results.append((session, api.read(args.harness, session, **kw)))
    if args.json:
        print(json.dumps({"harness": args.harness,
                          "sessions": [{"session": s, **c.as_dict()} for s, c in results]},
                         indent=2))
    else:
        for session, clock in results:
            print(line(session, clock))
    return 0


def _duration(seconds: float) -> str:
    """`45s`, `12m3s`, `1h0m`, `2d3h`."""
    total = int(max(0.0, seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{secs}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h{minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours}h"


def line(session: str, clock: Clock) -> str:
    """One session's read, in words."""
    if clock.kind in (WARM, COLD) and clock.age_seconds is not None \
            and clock.ttl_seconds is not None:
        source = f", from the {clock.ttl_source}" if clock.ttl_source else ""
        key = f", {clock.key}" if clock.key else ""
        return (f"{session}: {clock.kind}, last request {_duration(clock.age_seconds)} ago, "
                f"lifetime {_duration(clock.ttl_seconds)}{source}{key}")
    words = f"{session}: {clock.kind}"
    if clock.code:
        words += f" [{clock.code}]"
    if clock.reason:
        words += f": {clock.reason}"
    if clock.age_seconds is not None:
        words += f" (last request {_duration(clock.age_seconds)} ago)"
    return words


# --- segment --------------------------------------------------------------------------

def _segment(args) -> int:
    """Render inside the audit hook. The hook stays for the rest of the process, which
    ends when the render is printed."""
    from . import render
    from .module import SEGMENT

    try:
        found = render.load_segment(SEGMENT).segment(args.session)
    except render.Refused as e:
        print(f"moxi cache segment: the render was refused: {e}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(None if found is None else found.as_dict()))
    elif found is not None:
        print(found.coloured() if args.color else found.text)
    return 0


# --- timers ---------------------------------------------------------------------------

def _config_entries() -> tuple[object, str | None]:
    """The user's subscription tables, and the config's error when it cannot be read."""
    try:
        return current()[_timer.CONFIG_KEY], None
    except ConfigError as e:
        return [], str(e)


def _live_records() -> list[dict]:
    found = []
    try:
        for path in sorted(records.record_dir().iterdir()):
            if path.suffix == ".json":
                record = records.load(path)
                if record is not None:
                    found.append(record)
    except OSError:
        pass
    return found


def _clock_time(t: float | None, now: float) -> str:
    if t is None:
        return "never"
    when = time.strftime("%H:%M:%S", time.localtime(t))
    delta = t - now
    if abs(delta) < 1:
        return f"now ({when})"
    return (f"in {_duration(delta)} ({when})" if delta > 0
            else f"{_duration(-delta)} ago ({when})")


def timers_view(now: float | None = None) -> dict:
    """What `moxi cache timers` shows, as one object."""
    now = time.time() if now is None else now
    entries, config_error = _config_entries()
    loaded = _timer.load(entries)
    logged = _timer.read_log(_timer.log_path())
    sessions = []
    for record in _live_records():
        environ = None
        if any(s.requires_env for s in loaded.subscriptions):
            environ = _timer.proc_environ(record["pid"])
        asks = []
        for sub in loaded.subscriptions:
            mine = [e for e in logged if e.get("subscriber") == sub.name
                    and e.get("session") == record["session"]]
            asks.append(_timer.plan(sub, record, now, environ, mine))
        age = (now - record["last_request"]) if record["last_request"] is not None else None
        sessions.append({"session": record["session"], "pid": record["pid"],
                         "kind": record["kind"], "age": age, "ttl": record["ttl"],
                         "seen_at": record["seen_at"], "asks": asks})
    return {"now": now, "config_error": config_error,
            "subscriptions": [s.as_dict() for s in loaded.subscriptions],
            "refused": [r.as_dict() for r in loaded.refused],
            "sessions": sessions, "log": str(_timer.log_path())}


def _sub_line(sub: dict) -> str:
    parts = [f"at {sub['at']:+g}s from expiry"]
    if sub["until"] is not None:
        parts.append(f"until {sub['until']:+g}s")
    if sub["every"] is not None:
        parts.append(f"every {sub['every']:g}s")
    if sub["requires_env"]:
        parts.append("requires " + ", ".join(
            f"{k}={v}" for k, v in sorted(sub["requires_env"].items())))
    return f"{sub['name']} ({sub['source']}): " + ", ".join(parts)


_STATES = {
    "no-clock": "no clock to count from (no request read yet)",
    "env-not-met": "not asked: the process lacks {missing}",
    "closed": "window closed {closes}",
    "waiting": "asks when the window opens, {next}",
    "due": "window open: asks at the watcher's next tick",
    "asked": "asked on this clock {last}; no more asks on it",
    "repeating": "asked {last}; asks again {next}",
}


def _timers(args) -> int:
    view = timers_view()
    if args.json:
        print(json.dumps(view, indent=2))
        return 0
    now = view["now"]
    if view["config_error"]:
        print(f"cache's config cannot be read, so your subscriptions are not listed: "
              f"{view['config_error']}")
    subs = view["subscriptions"]
    print(f"subscriptions: {len(subs)}" + ("" if subs else " (none: nothing is asked)"))
    for sub in subs:
        print(f"  {_sub_line(sub)}")
        print(f"      run {' '.join(sub['run'])}"
              + (f"  [{sub['argv0']}]" if sub["argv0"] != sub["run"][0] else ""))
    if view["refused"]:
        print(f"refused: {len(view['refused'])}")
        for ref in view["refused"]:
            print(f"  {ref['name'] or '(unnamed)'} ({ref['source']}): {ref['reason']}")
    sessions = view["sessions"]
    if not sessions:
        print("live sessions: none in the watcher's records (is the watcher running? "
              "moxi cache doctor)")
    else:
        print(f"live sessions: {len(sessions)}")
    for s in sessions:
        clock = (f"{s['kind']}, last request {_duration(s['age'])} ago, lifetime "
                 f"{_duration(s['ttl'])}" if s["age"] is not None and s["ttl"] is not None
                 else s["kind"])
        print(f"  {s['session']} (pid {s['pid']}): {clock}")
        for ask in s["asks"]:
            words = _STATES[ask["state"]].format(
                missing=", ".join(ask["missing_env"]),
                closes=_clock_time(ask["closes_at"], now),
                next=_clock_time(ask["next_at"], now),
                last=_clock_time(ask["last_ask"], now))
            print(f"      {ask['subscriber']}: {words}")
    print(f"log: {view['log']}")
    return 0


# --- doctor ---------------------------------------------------------------------------

def _doctor() -> int:
    bad = False
    print("cache doctor")
    try:
        settings = current()
        print(f"  config: {settings.file} "
              f"({'read' if settings.file.exists() else 'absent: every key at its default'})")
    except ConfigError as e:
        bad = True
        print(f"  config: cannot be read: {e}")

    loaded = None
    try:
        loaded = table.load_table()
        print(f"  lifetime table: {len(loaded.of(table.SHIPPED))} shipped rows, "
              f"{len(loaded.of(table.USER))} of yours")
    except table.TableError as e:
        bad = True
        print(f"  lifetime table: does not load, so every Codex and OpenCode read is "
              f"unknown: {e}")

    store = api.open_store(learn=False)
    state = store.describe(loaded)
    if state["unreadable"]:
        bad = True
        record = f"{state['unreadable']}; every read on a measured lifetime is unknown"
    elif state["missing"]:
        record = "absent: no lowering, each measured row counts its shipped number"
    elif state["rejected"]:
        bad = True
        record = (f"holds an entry that cannot be accepted ({'; '.join(state['rejected'])}); "
                  "every read on a measured lifetime is unknown")
    elif loaded is None:
        record = f"{state['entries']} entries, not judged (the table does not load)"
    else:
        record = (f"{len(state['in_force'])} lowering(s) in force"
                  + (f": {'; '.join(state['in_force'])}" if state["in_force"] else ""))
    print(f"  drift record: {state['path']}: {record}")
    holder = drift.learner_holds_lock()
    print("  learner: " + {True: "a process holds the learning lock",
                           False: "no process is learning now",
                           None: "no process has learned on this device"}[holder])

    try:
        print(f"  claude projects root: {api.projects_root(None)}")
    except ConfigError as e:
        bad = True
        print(f"  claude projects root: sessions' config cannot be read: {e}")

    print("  watcher: " + _watcher_line())
    unit = service.unit_path(os.environ)
    print(f"  service: {unit} " + ("(installed)" if unit.exists() else
                                   "(not installed: moxi cache service install)"))
    for line_ in _timer_lines():
        print(f"  {line_}")
    return 1 if bad else 0


def _timer_lines() -> list[str]:
    """The timer's subscriptions as the watcher would load them here, and the refused."""
    entries, config_error = _config_entries()
    loaded = _timer.load(entries)
    subs = loaded.subscriptions
    from_config = sum(1 for s in subs if s.source == _timer.CONFIG)
    out = [f"timer: {len(subs)} subscription(s), {len(subs) - from_config} from modules, "
           f"{from_config} from your config"
           + (" (your config cannot be read, so its tables are not counted)"
              if config_error else "")
           + (": " + ", ".join(s.name for s in subs) if subs else "")]
    for ref in loaded.refused:
        out.append(f"timer: refused {ref.label()}: {ref.reason}")
    path = _timer.log_path()
    out.append(f"timer log: {path} " + ("(present)" if path.exists() else "(no ask yet)"))
    return out


def _watcher_line(now: float | None = None) -> str:
    """How the watcher's records stand: none, fresh, or standing still. Each record is
    judged stale by its own `stale_after`, as the segment judges it."""
    directory = records.record_dir()
    found = []
    try:
        for path in directory.iterdir():
            if path.suffix == ".json":
                record = records.load(path)
                if record is not None:
                    found.append(record)
    except OSError:
        pass
    if not found:
        return (f"no clock records under {directory}: no watcher is running, so there is "
                "no status line segment and no timer")
    now = time.time() if now is None else now
    newest = max(r["seen_at"] for r in found)
    words = f"{len(found)} clock record(s) under {directory}, newest seen {now - newest:.0f}s ago"
    judged = [(r, records.thresholds(r)) for r in found]
    older = sum(1 for _, t in judged if t is None)
    stale = sum(1 for r, t in judged if t is not None and now - r["seen_at"] > t[1])
    notes = []
    if stale and stale == len(found) - older:
        notes.append("the watcher has stopped, so the segment shows dim and no timer runs")
    elif stale:
        notes.append(f"{stale} of them not rewritten within their own stale_after, so "
                     "their segment shows dim")
    if older:
        notes.append(f"{older} lack warn_before or stale_after (an older watcher wrote "
                     "them), so their segment shows nothing until the watcher rewrites them")
    return words + (": " + "; ".join(notes) if notes else "")


__all__ = ["main", "line"]
