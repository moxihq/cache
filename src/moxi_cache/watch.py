"""The watcher, `moxi-cache-watch`: it keeps a clock record for every live Claude session.

**RULE C5: the watcher writes what the clock read, for display and for the timer. No
verdict is made from its records; a caller that decides about a cache calls `api.read`.**
A record can lag the transcript by a tick, so it is not evidence.

At each tick (`watch.tick`, 10 s by default):

1. Every third tick, and whenever a file in Claude Code's sessions folder
   (`${CLAUDE_CONFIG_DIR:-~/.claude}/sessions`) is added, removed or rewritten, it runs
   `claude agents --json` for the live interactive sessions (pid and session id). Claude
   Code rewrites a session's file there at a `/clear`, a resume and each change of status,
   so a new id is seen within a tick. The list and the folder are Claude Code's own records
   of its sessions, read as files are read.
2. For each live session it finds the transcript through sessions and stats it. When its
   size or modification time changed, or the last read was `unknown`, it reads the clock
   (`api.read`).
3. It writes `RUNTIME/cache/clock/<session>.json` by atomic replace (`records`), with
   `seen_at` at this tick whether it read or not, and removes the record of a session that
   left the live list. Every record it writes carries `warn_before` (the effective
   `segment.warn_before`) and `stale_after` (three times the effective tick), both in
   seconds, from the config as read at this tick, so a config change reaches every record
   within one tick (RULE C6a). The tick that sets `stale_after` is the one the watcher
   then waits, so a fresh record never shows dim.
4. It runs the timer's work for the tick (`timer`) over the records it wrote.

A session id is used in a path only after the reader's own check (`records.record_path`):
an id it refuses, such as one holding `/` or `..`, gets no record and one log line.

A shape of `claude agents --json` it does not know stops its writes, with one log line
naming the shape, until the shape is one it knows again; the records then stand still and
show dim. When the list cannot be read at all (no `claude`, an exit, a timeout), the tick
writes nothing and the next tick asks again. Each log line is written once per change of
what it says. Codex and OpenCode sessions get no records.

The watcher runs in the foreground (`moxi cache watch`, or this script), or as a systemd
user service (`moxi cache service install`).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from moxi.config import Config, ConfigError
from moxi_sessions.api import claude as transcripts

from . import api, records
from . import timer as _timer
from .clock import UNKNOWN, Clock
from .settings import MODULE, default, schema

log = logging.getLogger("moxi.cache")

#: `claude agents --json` runs at least every this many ticks.
AGENTS_EVERY = 3
#: How long one `claude agents --json` may take.
AGENTS_TIMEOUT = 20
#: The kind of session the watcher keeps records for.
INTERACTIVE = "interactive"
#: The shortest tick the watcher keeps, whatever the config says.
MIN_TICK = 1


class ShapeError(Exception):
    """`claude agents --json` printed a shape the watcher does not know."""


class AgentsUnreadable(Exception):
    """`claude agents --json` could not be run or printed no JSON."""


@dataclass(frozen=True)
class Live:
    """One live interactive session, as `claude agents --json` lists it."""

    pid: int
    session: str


def _keys(obj: dict) -> str:
    return "[" + ", ".join(sorted(map(str, obj))) + "]"


def parse_agents(doc) -> list[Live]:
    """The live interactive sessions in `claude agents --json`'s output.

    The list is of objects, each with an integer `pid` and a string `sessionId`; one whose
    `kind` is present and not `interactive` is skipped, and so is one whose `sessionId` is
    null or empty. Anything else raises `ShapeError` naming the shape."""
    if not isinstance(doc, list):
        raise ShapeError(f"a {type(doc).__name__}, not a list")
    live: list[Live] = []
    for item in doc:
        if not isinstance(item, dict):
            raise ShapeError(f"a list holding a {type(item).__name__}, not objects")
        pid, session, kind = item.get("pid"), item.get("sessionId"), item.get("kind")
        if not isinstance(pid, int) or isinstance(pid, bool):
            raise ShapeError(f"an entry with keys {_keys(item)} and no integer pid")
        if kind is not None and not isinstance(kind, str):
            raise ShapeError(f"an entry whose kind is a {type(kind).__name__}")
        if session is not None and not isinstance(session, str):
            raise ShapeError(f"an entry whose sessionId is a {type(session).__name__}")
        if kind not in (None, INTERACTIVE) or not session:
            continue
        live.append(Live(pid, session))
    return live


def claude_binary(environ=None) -> str | None:
    """`claude` on PATH, else at the installer's place, `~/.local/bin/claude`. A user
    service's PATH often lacks the second."""
    environ = os.environ if environ is None else environ
    found = shutil.which("claude", path=environ.get("PATH"))
    if found:
        return found
    home = environ.get("HOME") or str(Path.home())
    fallback = Path(home) / ".local" / "bin" / "claude"
    return str(fallback) if os.access(fallback, os.X_OK) else None


def run_agents(timeout: float = AGENTS_TIMEOUT):
    """`claude agents --json`, parsed. Raises `AgentsUnreadable`."""
    exe = claude_binary()
    if exe is None:
        raise AgentsUnreadable("no claude on PATH or at ~/.local/bin/claude")
    try:
        proc = subprocess.run([exe, "agents", "--json"], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise AgentsUnreadable(f"`claude agents --json` took over {timeout:g}s") from None
    except OSError as e:
        raise AgentsUnreadable(f"`claude agents --json` could not run ({e})") from None
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        raise AgentsUnreadable(f"`claude agents --json` exited {proc.returncode}"
                               + (f": {detail[-1]}" if detail else ""))
    try:
        return json.loads(proc.stdout)
    except ValueError as e:
        raise AgentsUnreadable(f"`claude agents --json` printed no JSON ({e})") from None


def sessions_dir(environ=None) -> Path:
    """Claude Code's sessions folder, `${CLAUDE_CONFIG_DIR:-~/.claude}/sessions`."""
    environ = os.environ if environ is None else environ
    config = environ.get("CLAUDE_CONFIG_DIR")
    if config:
        return Path(config) / "sessions"
    home = environ.get("HOME") or str(Path.home())
    return Path(home) / ".claude" / "sessions"


def sessions_stamp(directory: Path):
    """What the sessions folder holds now: each `.json` file's name, size and
    modification time. None when the folder cannot be read."""
    try:
        with os.scandir(directory) as entries:
            found = []
            for entry in entries:
                if not entry.name.endswith(".json"):
                    continue
                try:
                    st = entry.stat()
                except OSError:
                    continue
                found.append((entry.name, st.st_size, st.st_mtime_ns))
    except OSError:
        return None
    return frozenset(found)


@dataclass
class Seen:
    """What the watcher holds of one live session between ticks."""

    pid: int
    path: Path | None = None
    stat: tuple | None = None
    clock: Clock | None = None
    read_at: float | None = None
    last_request: float | None = None


def make_record(session: str, seen: Seen, now: float, *, warn_before: float,
                stale_after: float) -> dict:
    clock = seen.clock
    return {"v": records.VERSION, "session": session, "harness": "claude", "pid": seen.pid,
            "kind": clock.kind, "last_request": seen.last_request,
            "ttl": clock.ttl_seconds, "read_at": seen.read_at, "seen_at": now,
            "code": clock.code, "warn_before": warn_before, "stale_after": stale_after}


def write_record(directory: Path, record: dict) -> Path:
    """Replace one record whole: a reader sees the old file or the new, never part.
    Raises `ValueError` for a session id the reader would refuse (`records.record_path`),
    so no id names a path outside `directory`."""
    path = records.record_path(record["session"], directory)
    if path is None:
        raise ValueError(f"session id {record['session']!r} cannot name a record file")
    tmp = directory / f".{record['session']}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh)
        fh.write("\n")
    os.replace(tmp, path)
    return path


class Watcher:
    """The watcher's state and its tick. Every outside effect can be replaced for tests:
    `agents` (the live list), `read` (the clock), `now`, `projects_root` (where
    transcripts are found), `directory` (where records go) and `sessions` (the folder
    whose changes ask for the list again).

    `warn_before` and `stale_after` are what the next tick writes into every record;
    `run` sets them from the config before each tick. They start at the keys' defaults."""

    def __init__(self, *, agents: Callable[[], object] = run_agents,
                 read: Callable[..., Clock] = api.read,
                 now: Callable[[], float] = time.time,
                 projects_root: Callable[[], str] | None = None,
                 directory: Path | None = None, sessions: Path | None = None):
        self.agents = agents
        self.read = read
        self.now = now
        self.projects_root = projects_root or (lambda: api.projects_root(None))
        self.directory = records.record_dir() if directory is None else Path(directory)
        self.sessions = sessions_dir() if sessions is None else Path(sessions)
        self.warn_before = default("segment.warn_before")
        self.stale_after = records.STALE_TICKS * max(MIN_TICK, default("watch.tick"))
        self.ticks = 0
        #: The records this tick wrote, by session id, for the timer; None when the tick
        #: wrote nothing because the live list could not be read or has a shape it does
        #: not know.
        self.current: dict[str, dict] | None = None
        self.tick_at: float | None = None
        self.live: list[Live] | None = None
        self.seen: dict[str, Seen] = {}
        self._stamp = None
        self._ask = True                  # ask for the list at the next tick
        self._halted = False              # the list's shape is not one the watcher knows
        self._said: dict[str, str] = {}

    # -- logging, once per change --------------------------------------------------------

    def say(self, topic: str, message: str | None) -> None:
        """Log `message` under `topic` once, until it changes; None clears the topic."""
        if message is None:
            self._said.pop(topic, None)
            return
        if self._said.get(topic) != message:
            self._said[topic] = message
            log.warning(message)

    # -- the tick ------------------------------------------------------------------------

    def tick(self) -> list[Live]:
        """One tick. Returns the live sessions it wrote records for."""
        now = self.tick_at = self.now()
        self.current = None
        stamp = sessions_stamp(self.sessions)
        if self._ask or self.ticks % AGENTS_EVERY == 0 or stamp != self._stamp:
            self._stamp = stamp
            if not self._refresh():
                self.ticks += 1
                return []
        self.ticks += 1
        if self._halted or self.live is None:
            return []
        self.directory.mkdir(parents=True, exist_ok=True)
        current: dict[str, dict] = {}
        for live in self.live:
            seen = self._observe(live, now)
            record = current[live.session] = make_record(
                live.session, seen, now, warn_before=self.warn_before,
                stale_after=self.stale_after)
            try:
                write_record(self.directory, record)
                self.say("write", None)
            except OSError as e:
                self.say("write", f"cannot write the clock record of {live.session}: {e}")
        self.current = current
        return list(self.live)

    def _refresh(self) -> bool:
        """Ask for the live list. False when this tick writes nothing."""
        try:
            listed = parse_agents(self.agents())
        except AgentsUnreadable as e:
            self._ask = True
            self.say("agents", f"{e}; no clock record is written until it can be read")
            return False
        except ShapeError as e:
            self._ask = True
            self._halted = True
            self.say("agents", f"`claude agents --json` printed {e}, a shape the watcher "
                               "does not know; it writes no clock record until it does")
            return False
        self._ask = False
        self._halted = False
        self.say("agents", None)
        unique: dict[str, Live] = {}
        unsafe: list[str] = []
        for live in listed:
            if records.record_path(live.session, self.directory) is None:
                unsafe.append(live.session)
                continue
            unique.setdefault(live.session, live)
        self.say("unsafe", None if not unsafe else
                 "`claude agents --json` listed session ids that cannot name a record file "
                 f"({', '.join(map(repr, sorted(set(unsafe))))}); they get no clock record")
        self.live = list(unique.values())
        for gone in set(self.seen) - set(unique):
            del self.seen[gone]
        self._remove_others(set(unique))
        return True

    def _remove_others(self, keep: set[str]) -> None:
        """Remove the record of every session not in `keep`, and stray temporary files."""
        try:
            entries = list(self.directory.iterdir())
        except FileNotFoundError:
            return
        except OSError as e:
            self.say("remove", f"cannot list {self.directory}: {e}")
            return
        for path in entries:
            stray = path.name.startswith(".") and path.name.endswith(".tmp")
            if not stray and (path.suffix != ".json" or path.stem in keep):
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError as e:
                self.say("remove", f"cannot remove {path}: {e}")

    def _observe(self, live: Live, now: float) -> Seen:
        """Find and stat one session's transcript; read its clock when it changed."""
        seen = self.seen.get(live.session)
        if seen is None:
            seen = self.seen[live.session] = Seen(live.pid)
        seen.pid = live.pid
        if seen.path is None or not seen.path.is_file():
            try:
                probe = transcripts.probe_root(self.projects_root(), live.session)
                seen.path = probe.path_or_none()
            except (ConfigError, OSError) as e:
                self.say("root", f"cannot find transcripts: {e}")
                seen.path = None
        stat = None
        if seen.path is not None:
            try:
                st = seen.path.stat()
                stat = (str(seen.path), st.st_size, st.st_mtime_ns)
            except OSError:
                stat = None
        if seen.clock is None or stat != seen.stat or seen.clock.kind == UNKNOWN:
            clock = self.read("claude", live.session, now=now)
            seen.clock, seen.stat, seen.read_at = clock, stat, now
            seen.last_request = (now - clock.age_seconds
                                 if clock.age_seconds is not None else None)
        return seen


# --- running it ---------------------------------------------------------------------------

def _interval(config: Config, watcher: Watcher) -> int:
    """The effective tick: `watch.tick`, at least `MIN_TICK`."""
    tick = config.use("watch.tick")
    if tick < MIN_TICK:
        watcher.say("tick", f"watch.tick is {tick}; the watcher ticks every {MIN_TICK}s")
        return MIN_TICK
    watcher.say("tick", None)
    return tick


def _thresholds(config: Config, watcher: Watcher) -> int:
    """Read the config once for this tick: set what the tick writes into every record
    (RULE C6a) and return the tick it then waits, the same read that set `stale_after`."""
    interval = _interval(config, watcher)
    watcher.warn_before = config.use("segment.warn_before")
    watcher.stale_after = records.STALE_TICKS * interval
    return interval


def run(*, watcher: Watcher | None = None, stop: threading.Event | None = None,
        config: Config | None = None, timer: _timer.Timer | None = None) -> int:
    """Tick until `stop` is set (SIGTERM or SIGINT in the foreground), with the timer's
    work after each tick. Returns the exit code: 0 on a stop, 2 when the config refuses
    the start."""
    config = config or Config(MODULE, schema(),
                              report=lambda e: log.warning("cache's config: %s", e))
    try:
        config.start()
    except ConfigError as e:
        print(f"moxi-cache-watch: cache's config refuses the start: {e}", file=sys.stderr)
        return 2
    watcher = watcher or Watcher()
    timer = timer or _timer.Timer(config_entries=lambda: config.use(_timer.CONFIG_KEY))
    stop = stop or threading.Event()
    log.info("watching live Claude sessions; records under %s", watcher.directory)
    try:
        while not stop.is_set():
            interval = _thresholds(config, watcher)
            try:
                watcher.tick()
                watcher.say("tick-error", None)
            except Exception as e:                                # noqa: BLE001
                watcher.current = None
                watcher.say("tick-error", f"a tick failed ({type(e).__name__}: {e})")
            try:
                timer.tick(watcher.current, watcher.tick_at)
                watcher.say("timer-error", None)
            except Exception as e:                                # noqa: BLE001
                watcher.say("timer-error", f"the timer's tick failed "
                                           f"({type(e).__name__}: {e})")
            stop.wait(interval)
    finally:
        timer.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """`moxi-cache-watch`: the watcher in the foreground, until SIGTERM or Ctrl-C."""
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__.strip().splitlines()[0])
        print("usage: moxi-cache-watch   (no arguments; stop it with SIGTERM or Ctrl-C)")
        return 0
    if argv:
        print(f"moxi-cache-watch: takes no arguments (got {' '.join(argv)})", file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="moxi-cache-watch: %(message)s")
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    return run(stop=stop)


__all__ = ["Watcher", "Live", "Seen", "ShapeError", "AgentsUnreadable", "parse_agents",
           "run_agents", "claude_binary", "sessions_dir", "sessions_stamp", "make_record",
           "write_record", "run", "main", "AGENTS_EVERY", "AGENTS_TIMEOUT"]
