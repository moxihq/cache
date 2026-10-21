"""The cache lifetime table: how long a Codex or OpenCode prompt cache counts as warm.

The shipped table is data in the package, `lifetimes.toml`, and changes only by a commit.
Each row is keyed on harness, provider and the exact model ids it covers, and carries the
source that promises its number. The loader refuses any row it cannot vouch for, and a
table it refuses does not load at all: every Codex and OpenCode read is then `unknown`.
Claude reads do not use the table.

Three kinds of row:

* a documented row: `seconds` from the provider's documentation (`source`), with
  `evidence` or a `ruling` where the source does not name the route in use;
* a measured row: `seconds` from a ruled measurement, `floor` the provider's documented
  minimum, `measured = true`. A learning drift store may lower it when a read sees a
  cache go cold sooner (`codex`, the drift scan), never below `floor`;
* a free row: `charge = "free"` and no `seconds`. A read whose next request runs on one
  of its models is `no-cache`, since a cold landing there costs nothing.

Two rows that cover one (harness, provider, model) are refused, `"*"` covering every model
of its provider: a row decides who can be reached, so an overlap is never settled by row
order.

**The user's own rows.** cache's config may add `[[lifetime]]` rows in the same schema,
checked as shipped rows are. A user row adds a model and never overrides one: a user row
that overlaps a shipped row or another user row is refused, and so is any row the loader
cannot vouch for. A refused user row refuses the table, as a refused shipped row does, so
a typo never silently changes who can be reached; `moxi cache doctor` names it.
"""

from __future__ import annotations

import datetime
import os
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path

#: The shipped table. A test points this at a table of its own.
TABLE_PATH = Path(__file__).with_name("lifetimes.toml")

HARNESSES = ("codex", "opencode")
ANCHORS = ("last-use", "request-start")
FREE = "free"
WILDCARD = "*"

SHIPPED, USER = "shipped", "user"

#: The keys a row may carry. Anything else is refused: a key the loader does not read
#: could be a qualification the row's author meant to be honoured.
ROW_KEYS = {"harness", "provider", "models", "seconds", "anchor", "source", "checked",
            "evidence", "ruling", "note", "floor", "measured", "charge"}

#: Why every Codex and OpenCode read is `unknown` when the table does not load.
NOT_LOADED = "the cache lifetime table could not be read"


class TableError(ValueError):
    """The table, or one of its rows, is not one the loader can vouch for."""


@dataclass(frozen=True)
class Row:
    harness: str
    provider: str
    models: tuple
    seconds: int | None
    anchor: str | None
    source: str
    checked: str
    evidence: str
    ruling: str
    note: str
    floor: int | None = None
    measured: bool = False
    charge: str | None = None
    origin: str = SHIPPED

    @property
    def free(self) -> bool:
        return self.charge == FREE

    @property
    def wildcard(self) -> bool:
        return self.models == (WILDCARD,)

    def covers(self, model: str) -> bool:
        return self.wildcard or model in self.models


class Table:
    """The rows that loaded, looked up by (harness, provider, model)."""

    def __init__(self, rows):
        self.rows = tuple(rows)

    def lookup(self, harness: str, provider: str, model: str) -> Row | None:
        for row in self.rows:
            if row.harness == harness and row.provider == provider and row.covers(model):
                return row
        return None

    def is_free(self, harness: str, provider: str, model: str) -> bool:
        row = self.lookup(harness, provider, model)
        return row is not None and row.free

    def measured_rows(self) -> tuple:
        return tuple(r for r in self.rows if r.measured)

    def of(self, origin: str) -> tuple:
        return tuple(r for r in self.rows if r.origin == origin)


def _text(where: str, raw: dict, key: str, *, required: bool = False) -> str:
    value = raw.get(key, "")
    if not isinstance(value, str):
        raise TableError(f"{where}: `{key}` is not a string")
    if required and not value.strip():
        raise TableError(f"{where}: no `{key}`")
    return value


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def parse_row(raw, n: int, *, origin: str = SHIPPED) -> Row:
    """One `[[lifetime]]` table -> a Row, or TableError naming what is wrong."""
    where = f"row {n}" if origin == SHIPPED else f"your config's lifetime row {n}"
    if not isinstance(raw, dict):
        raise TableError(f"{where} is not a table")
    unknown = set(raw) - ROW_KEYS
    if unknown:
        raise TableError(f"{where}: keys the table does not read: "
                         f"{', '.join(sorted(unknown))}")
    harness = raw.get("harness")
    if harness not in HARNESSES:
        raise TableError(f"{where}: unknown harness {harness!r}")
    provider = _text(where, raw, "provider", required=True)
    models = raw.get("models")
    if (not isinstance(models, list) or not models
            or not all(isinstance(m, str) and m.strip() for m in models)):
        raise TableError(f"{where}: `models` is not a list of model ids ({models!r})")
    if WILDCARD in models and len(models) > 1:
        raise TableError(f"{where}: `\"*\"` beside other model ids")
    if len(set(models)) != len(models):
        raise TableError(f"{where}: a model id is listed twice")
    source = _text(where, raw, "source")
    ruling = _text(where, raw, "ruling")
    evidence = _text(where, raw, "evidence")
    note = _text(where, raw, "note")
    if not source.strip() and not ruling.strip():
        raise TableError(f"{where}: no `source` and no `ruling`")
    checked = raw.get("checked")
    if not isinstance(checked, datetime.date) or isinstance(checked, datetime.datetime):
        raise TableError(f"{where}: `checked` is not a date")
    has_seconds, has_charge = "seconds" in raw, "charge" in raw
    if has_seconds == has_charge:
        raise TableError(f"{where}: needs exactly one of `seconds` and `charge`")
    floor = raw.get("floor")
    measured = raw.get("measured", False)
    if not isinstance(measured, bool):
        raise TableError(f"{where}: `measured` is not true or false")
    anchor = raw.get("anchor")
    if has_charge:
        if raw.get("charge") != FREE:
            raise TableError(f"{where}: `charge` is not \"free\" ({raw.get('charge')!r})")
        if "anchor" in raw or "floor" in raw or "measured" in raw:
            raise TableError(f"{where}: a free row has no `anchor`, `floor` or `measured`")
        if models == [WILDCARD]:
            raise TableError(f"{where}: a free row lists exact model ids, never \"*\"")
        if not ruling.strip():
            raise TableError(f"{where}: a free row needs the `ruling` that made it free")
        seconds = None
    else:
        seconds = raw.get("seconds")
        if not _positive_int(seconds):
            raise TableError(f"{where}: `seconds` is not a positive integer ({seconds!r})")
        if anchor not in ANCHORS:
            raise TableError(f"{where}: unknown anchor {anchor!r}")
        if measured:
            if not ruling.strip():
                raise TableError(f"{where}: a measured row needs a `ruling`")
            if not evidence.strip():
                raise TableError(f"{where}: a measured row needs its `evidence`")
            if not _positive_int(floor) or floor > seconds:
                raise TableError(f"{where}: a measured row's `floor` must be a positive "
                                 f"integer at or below its `seconds` ({floor!r})")
        elif "floor" in raw:
            raise TableError(f"{where}: `floor` only on a measured row")
    return Row(harness=harness, provider=provider, models=tuple(models), seconds=seconds,
               anchor=anchor, source=source, checked=checked.isoformat(),
               evidence=evidence, ruling=ruling, note=note,
               floor=floor if measured else None, measured=measured,
               charge=raw.get("charge"), origin=origin)


def _refuse_overlaps(rows) -> None:
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a.harness != b.harness or a.provider != b.provider:
                continue
            if a.wildcard or b.wildcard or set(a.models) & set(b.models):
                both = sorted(set(a.models) & set(b.models)) or [WILDCARD]
                which = ("your config's row overlaps a shipped row: "
                         if SHIPPED in (a.origin, b.origin) and USER in (a.origin, b.origin)
                         else "your config's rows overlap: " if a.origin == USER else "")
                raise TableError(f"{which}two rows cover {a.harness} {a.provider}/"
                                 f"{', '.join(both)}")


def parse_table(text: str) -> Table:
    """The whole shipped table, or TableError. One bad row refuses the table."""
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise TableError(f"the table is not TOML ({e})") from e
    extra = set(doc) - {"lifetime"}
    if extra:
        raise TableError(f"the table has keys it does not read: {', '.join(sorted(extra))}")
    raws = doc.get("lifetime")
    if not isinstance(raws, list) or not raws:
        raise TableError("the table has no rows")
    rows = [parse_row(raw, n) for n, raw in enumerate(raws, 1)]
    _refuse_overlaps(rows)
    return Table(rows)


def with_user_rows(shipped: Table, raws) -> Table:
    """The shipped table with the user's rows added, or TableError."""
    if not raws:
        return shipped
    if not isinstance(raws, list):
        raise TableError("your config's `lifetime` is not a list of [[lifetime]] tables")
    user = [parse_row(raw, n, origin=USER) for n, raw in enumerate(raws, 1)]
    rows = [*shipped.rows, *user]
    _refuse_overlaps(rows)
    return Table(rows)


_cache: dict = {}
_cache_lock = threading.Lock()


def load_shipped(path=None) -> Table:
    """The shipped table at `path` (default `TABLE_PATH`), or TableError. Kept per
    process while the file is unchanged."""
    path = Path(path or TABLE_PATH)
    try:
        st = os.stat(path)
    except OSError as e:
        raise TableError(f"{path} could not be read ({e.strerror or e})") from e
    key = (str(path), st.st_mtime_ns, st.st_size)
    with _cache_lock:
        kept = _cache.get(str(path))
    if kept is not None and kept[0] == key:
        if isinstance(kept[1], TableError):
            raise kept[1]
        return kept[1]
    try:
        result = parse_table(path.read_text(encoding="utf-8"))
    except OSError as e:
        result = TableError(f"{path} could not be read ({e.strerror or e})")
    except TableError as e:
        result = e
    with _cache_lock:
        _cache[str(path)] = (key, result)
    if isinstance(result, TableError):
        raise result
    return result


def load_table(path=None) -> Table:
    """The table in force: the shipped rows and the user's own, or TableError. A config
    file that cannot be read is a TableError too, naming the file."""
    from moxi.config import ConfigError

    from .settings import current

    shipped = load_shipped(path)
    try:
        raws = current()["lifetime"]
    except ConfigError as e:
        raise TableError(f"cache's config could not be read ({e})") from e
    return with_user_rows(shipped, raws)


__all__ = ["TABLE_PATH", "HARNESSES", "ANCHORS", "FREE", "WILDCARD", "NOT_LOADED",
           "SHIPPED", "USER", "TableError", "Row", "Table", "parse_row", "parse_table",
           "with_user_rows", "load_shipped", "load_table"]
