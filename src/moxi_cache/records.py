"""The clock record: what the watcher last read of one live session's clock.

The watcher keeps one file per live Claude session, `RUNTIME/cache/clock/<session>.json`,
replaced whole at every tick that finds the session live:

    {"v": 1, "session": ..., "harness": "claude", "pid": ..., "kind": ...,
     "last_request": <epoch seconds or null>, "ttl": <seconds or null>,
     "read_at": <epoch seconds>, "seen_at": <epoch seconds>, "code": <code or null>,
     "warn_before": <seconds>, "stale_after": <seconds>}

`kind`, `ttl` and `code` are the clock's read (`api.read`) at `read_at`, and
`last_request` is when the session's last request ended by that read. `seen_at` is the
newest tick that found the session live: it moves at every tick, read or not, so a
record whose `seen_at` stands still means the watcher has stopped, not that the session
is idle. The file goes when the session leaves Claude Code's live list.

**A clock record carries every value a reader needs to render or judge it, so
no reader takes a record's band or staleness from cache's config.** `warn_before` is the
effective `segment.warn_before`: the band turns from warm to near at `ttl - warn_before`.
`stale_after` is three times the watcher's effective tick (`watch.tick`, at least one
second): a record whose `seen_at` is older than that shows dim. The watcher writes both at
every tick from its config as it read it at that tick, so a config change reaches every
record within one tick. A record without them, one an older watcher wrote, renders
nothing until the next tick rewrites it (`thresholds` gives None); it is still a record,
so the timer and `moxi cache timers` read it.

The format is a contract: a shell status line reads it with one `jq` and no config. A
record is for display and for the timer; it can lag the transcript by a tick, so nothing
decides about a cache from it. A caller that decides calls `api.read`.

This module imports only the standard library and the core, so the status line segment
can read records without loading the clock.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from moxi.paths import Layout

#: The record format's version.
VERSION = 1
#: The two fields a reader renders and judges a record by.
THRESHOLDS = ("warn_before", "stale_after")
#: Every field a record holds, in the order the watcher writes them.
FIELDS = ("v", "session", "harness", "pid", "kind", "last_request", "ttl", "read_at",
          "seen_at", "code") + THRESHOLDS
#: The fields every record holds; a record an older watcher wrote, before records
#: carried them, lacks `THRESHOLDS`, and is still a record.
REQUIRED = FIELDS[:-len(THRESHOLDS)]
#: `stale_after` is this many of the watcher's ticks.
STALE_TICKS = 3
#: The directory under cache's runtime directory that holds the records.
DIR_NAME = "clock"

#: A session id that can name a record file: no separator, no leading dot.
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")


def record_dir(env=None) -> Path:
    """`RUNTIME/cache/clock`, where the records are."""
    return Layout("cache").runtime_dir(env) / DIR_NAME


def record_path(session_id, directory: Path | None = None) -> Path | None:
    """The record file of one session, or None when the id cannot name a file."""
    if not isinstance(session_id, str) or not SAFE_ID_RE.fullmatch(session_id):
        return None
    return (record_dir() if directory is None else Path(directory)) / f"{session_id}.json"


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def valid(record) -> bool:
    """Whether `record` holds to the format. The two `THRESHOLDS` are not asked for here:
    `thresholds` reads them."""
    if not isinstance(record, dict) or record.get("v") != VERSION:
        return False
    if any(name not in record for name in REQUIRED):
        return False
    return (isinstance(record["session"], str) and isinstance(record["harness"], str)
            and isinstance(record["pid"], int) and not isinstance(record["pid"], bool)
            and isinstance(record["kind"], str)
            and (record["last_request"] is None or _number(record["last_request"]))
            and (record["ttl"] is None or _number(record["ttl"]))
            and _number(record["read_at"]) and _number(record["seen_at"])
            and (record["code"] is None or isinstance(record["code"], str)))


def thresholds(record: dict) -> tuple[float, float] | None:
    """The record's own `(warn_before, stale_after)`, or None when it lacks either or
    holds one that is not a finite number. A reader renders nothing for a record without
    them, as for a missing record."""
    found = tuple(record.get(name) for name in THRESHOLDS)
    if all(_number(v) and math.isfinite(v) for v in found):
        return found
    return None


def load(path: Path) -> dict | None:
    """One record, or None when it is absent, unreadable or not in the format."""
    try:
        with open(path, "rb") as fh:
            record = json.loads(fh.read())
    except (OSError, ValueError):
        return None
    return record if valid(record) else None


__all__ = ["VERSION", "FIELDS", "REQUIRED", "THRESHOLDS", "STALE_TICKS", "DIR_NAME",
           "SAFE_ID_RE", "record_dir", "record_path", "valid", "thresholds", "load"]
