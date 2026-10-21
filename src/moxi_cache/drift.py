"""The drift record: the lowerings learned for measured lifetimes, and the store that
owns it.

**RULE: cache owns the files it learns into. One process at a time learns; every other
process reads.** The record is `STATE/cache/drift.json`, `{"entries": [...]}`, one entry
per (harness, provider, model): the lifetime it was lowered to, the observation that
lowered it, and the row's `seconds` and `checked` it was lowered from. STATE, not DATA: it
describes this box's route. The lifetime table is code and changes only by a commit; this
file is what reads have learned since.

`Store(learn=True)` takes an exclusive lock on `STATE/cache/drift.lock` for the life of
the process, loads the record once, and writes a lowering by atomic replace before the
read that found it returns. It keeps the newest turn judged per thread, so each read
judges only turns it has not judged before. A second process asking to learn gets a
read-only store and one log line. A read-only store creates nothing and writes nothing:
it reads the record again whenever the file changes, and a lowering one of its reads
observes counts in that read alone.

What each state of the file reads as, for the models of a measured row:

* missing: no lowering; each model counts the row's shipped number;
* there but unreadable, not JSON, or not `{"entries": [list]}`: `unknown`;
* an entry whose row's `seconds` or `checked` no longer matches the row, or whose model
  is on no measured row: ignored, and dropped at the next write; a new row starts from its
  own number;
* an entry the store does not recognise, or whose lifetime is not an integer between the
  row's floor and its shipped number: the whole record is held as unreadable, so every
  read on a measured row is `unknown` until the entry is fixed or the file removed. A bad
  entry never raises a lifetime, and never silently falls back to the longer shipped
  number;
* a lowering the store could not write: that model is `unknown` until a write succeeds.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

from moxi.paths import Layout

from . import clock as _clock
from .clock import Clock, unknown

log = logging.getLogger("moxi.cache")

MODULE = "cache"
RECORD_NAME = "drift.json"
LOCK_NAME = "drift.lock"

UNREADABLE = "the cache drift record could not be read"
UNWRITTEN = "the cache drift record could not be written"

#: The keys of an entry, and their types.
ENTRY_KEYS = {"harness": str, "provider": str, "model": str, "lifetime": int,
              "row_seconds": int, "row_checked": str}


def record_path(env=None) -> Path:
    """STATE/cache/drift.json."""
    return Layout(MODULE).state_dir(env) / RECORD_NAME


def lock_path(env=None) -> Path:
    """STATE/cache/drift.lock."""
    return Layout(MODULE).state_dir(env) / LOCK_NAME


# ------------------------------------------------------------------ the view a read takes

class DriftView:
    """The drift record as a reader takes it: the lowerings in force, and why a measured
    row's model cannot be read (the record unreadable, an entry rejected, a lowering not
    written)."""

    def __init__(self, entries=None, blocked_all=None, blocked=None):
        self.entries = dict(entries or {})
        self.blocked_all = blocked_all
        self.blocked = dict(blocked or {})

    @classmethod
    def of(cls, value) -> DriftView:
        if isinstance(value, DriftView):
            return value
        return cls(entries=value or {})

    def blocked_reason(self, key) -> str | None:
        return self.blocked_all or self.blocked.get(key)

    def lifetime(self, key, row) -> tuple:
        """(lifetime in force, when it was lowered or None) for one model."""
        entry = self.entries.get(key)
        if (isinstance(entry, dict) and entry.get("row_seconds") == row.seconds
                and entry.get("row_checked") == row.checked
                and isinstance(entry.get("lifetime"), int)
                and row.floor <= entry["lifetime"] <= row.seconds):
            return entry["lifetime"], entry.get("at_iso")
        return row.seconds, None


# ------------------------------------------------------------------ the record on disk

@dataclass
class Record:
    """What was read from the file, before it is judged against the table."""
    entries: list = field(default_factory=list)
    unreadable: str | None = None        # why the file as a whole could not be read
    missing: bool = False


def load(path) -> Record:
    path = Path(path)
    try:
        text = path.read_text()
    except FileNotFoundError:
        return Record(missing=True)
    except OSError as e:
        return Record(unreadable=f"{UNREADABLE} ({e.strerror or e})")
    try:
        doc = json.loads(text)
    except ValueError:
        return Record(unreadable=f"{UNREADABLE} (it is not JSON)")
    if not isinstance(doc, dict) or not isinstance(doc.get("entries"), list):
        return Record(unreadable=f"{UNREADABLE} (it has no list of entries)")
    return Record(entries=list(doc["entries"]))


def _key(entry) -> tuple:
    return (entry["harness"], entry["provider"], entry["model"])


def describe(entry) -> str:
    if isinstance(entry, dict):
        return (f"{entry.get('harness')} {entry.get('provider')}/{entry.get('model')} "
                f"at {entry.get('lifetime')!r} seconds")
    return repr(entry)[:80]


def judge(record: Record, table) -> tuple:
    """(DriftView, kept entries, rejected entries) for `record` against the table.

    Kept entries are those in force; stale ones are left out of both lists, so the next
    write drops them."""
    if record.unreadable:
        return DriftView(blocked_all=record.unreadable), [], []
    kept, rejected = [], []
    in_force = {}
    for entry in record.entries:
        if not isinstance(entry, dict) or any(
                not isinstance(entry.get(k), t) or isinstance(entry.get(k), bool)
                for k, t in ENTRY_KEYS.items()):
            rejected.append(entry)
            continue
        row = table.lookup(entry["harness"], entry["provider"], entry["model"])
        if row is None or not row.measured:
            continue                                     # stale: no measured row now
        if entry["row_seconds"] != row.seconds or entry["row_checked"] != row.checked:
            continue                                     # stale: the row changed
        if not row.floor <= entry["lifetime"] <= row.seconds:
            rejected.append(entry)
            continue
        if _key(entry) in in_force:
            rejected.append(entry)                       # two entries for one model
            continue
        in_force[_key(entry)] = entry
        kept.append(entry)
    if rejected:
        names = "; ".join(describe(e) for e in rejected)
        return (DriftView(blocked_all=f"the cache drift record has an entry that cannot be "
                                      f"accepted ({names})"), kept, rejected)
    return DriftView(entries=in_force), kept, []


def write(path, entries) -> None:
    """The whole record, atomically. Raises OSError."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".drift.")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"entries": entries}, f, indent=1, sort_keys=True)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def entry_of(observation: dict) -> dict:
    """The record's entry for a lowering a read observed."""
    keep = ("harness", "provider", "model", "lifetime", "from", "gap", "thread",
            "response_id", "at", "at_iso", "prev_at_iso", "cached", "input",
            "row_seconds", "row_checked")
    return {k: observation.get(k) for k in keep}


# ------------------------------------------------------------------ the store

class Store:
    """The drift record behind `read`: learning (the one writer) or read-only.

    `learning` says which it is. A learning store holds the lock until `close()` or the
    end of the process. `view(table)` is what a read takes; `judged(thread)` the newest
    turn judged on a thread; `take(clock, table)` settles a read's drift results."""

    def __init__(self, *, learn: bool, env=None):
        self.path = record_path(env)
        self.lock_path = lock_path(env)
        self.learning = False
        self._lock_fd: int | None = None
        self._mutex = threading.RLock()
        self._judged: dict = {}
        self._unwritten: set = set()
        self._stamp = None
        self.record = Record(missing=True)
        if learn:
            self.learning = self._take_lock()
        self.record = load(self.path)
        self._stamp = _stamp(self.path)

    def _take_lock(self) -> bool:
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as e:
            log.warning("cache: could not open %s (%s); this process reads the drift "
                        "record and does not learn", self.lock_path, e.strerror or e)
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            log.warning("cache: another process is learning into %s; this process reads "
                        "it and does not learn", self.path)
            return False
        self._lock_fd = fd
        return True

    def close(self) -> None:
        """Release the lock, if this store holds it. The store reads only afterwards."""
        with self._mutex:
            if self._lock_fd is not None:
                try:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                finally:
                    os.close(self._lock_fd)
                    self._lock_fd = None
            self.learning = False

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _refresh(self) -> None:
        """A read-only store reads the record again when the file changed."""
        if self.learning:
            return
        stamp = _stamp(self.path)
        if stamp != self._stamp:
            self.record = load(self.path)
            self._stamp = stamp

    def view(self, table) -> DriftView:
        """The lowerings in force, with every model a read must hold: the whole record
        when it is unreadable or holds an entry that cannot be accepted, and each model
        whose lowering is not yet written. A learning store retries a failed write here."""
        with self._mutex:
            self._refresh()
            if self.learning and self._unwritten:
                self._write()
            view, _kept, _rejected = judge(self.record, table)
            view.blocked.update({k: UNWRITTEN for k in self._unwritten})
            return view

    def judged(self, thread) -> str | None:
        with self._mutex:
            return self._judged.get(thread)

    def take(self, clock: Clock, table) -> Clock:
        """A Codex read's drift results. A learning store keeps the newest judged turn and
        writes a lowering before the read returns; a lowering it cannot write turns the
        read `unknown`. A read-only store keeps nothing."""
        if not self.learning:
            return clock
        if clock.judged:
            thread, turn = clock.judged
            with self._mutex:
                self._judged[thread] = turn
        lowered = clock.lowering
        if not lowered:
            return clock
        key = (lowered["harness"], lowered["provider"], lowered["model"])
        with self._mutex:
            _view, kept, _rejected = judge(self.record, table)
            current = next((e for e in kept if _key(e) == key), None)
            if current is not None and current["lifetime"] <= lowered["lifetime"]:
                return clock            # another read already lowered it as far
            entries = [e for e in kept if _key(e) != key]
            entries.append(entry_of(lowered))
            self.record = Record(entries=entries)
            if not self._write():
                self._unwritten.add(key)
                return unknown(_clock.DRIFT_HELD, UNWRITTEN)
        return clock

    def _write(self) -> bool:
        """Write the record from memory. Under the mutex."""
        try:
            write(self.path, self.record.entries)
        except OSError as e:
            log.warning("cache: could not write %s (%s); the lowered models read unknown "
                        "until a write succeeds", self.path, e.strerror or e)
            return False
        self._unwritten.clear()
        return True

    def describe(self, table=None) -> dict:
        """What the store holds, for `doctor`."""
        with self._mutex:
            self._refresh()
            out = {"path": str(self.path), "learning": self.learning,
                   "missing": self.record.missing, "unreadable": self.record.unreadable,
                   "entries": len(self.record.entries), "rejected": [],
                   "unwritten": [list(k) for k in sorted(self._unwritten)]}
            if table is not None and not self.record.unreadable:
                _view, kept, rejected = judge(self.record, table)
                out["in_force"] = [describe(e) for e in kept]
                out["rejected"] = [describe(e) for e in rejected]
            return out


def _stamp(path: Path):
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns, st.st_ino)


def open_store(*, learn: bool) -> Store:
    """The drift store. `learn=True` asks to be the one learner; when another process
    holds the lock, the store is read-only (`Store.learning` says which)."""
    return Store(learn=learn)


def learner_holds_lock(env=None) -> bool | None:
    """Whether some process holds the learning lock now: True, False, or None when there
    is no lock file. Opens the file read-only and creates nothing."""
    path = lock_path(env)
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        return None
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


__all__ = ["UNREADABLE", "UNWRITTEN", "DriftView", "Record", "Store", "load", "judge",
           "write", "entry_of", "describe", "open_store", "record_path", "lock_path",
           "learner_holds_lock"]
