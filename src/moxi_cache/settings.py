"""cache's config, read through `moxi.config` at each use.

Keys: `claude.ttl_seconds`, Claude's lifetime when no request read names a cache tier;
`lifetime`, the user's own `[[lifetime]]` rows (see `table`); `watch.tick`, the watcher's
tick; `segment.warn_before`, where the status line segment turns from warm to near; and
`timer.subscriber`, the user's own `[[timer.subscriber]]` tables (see `timer`).

The watcher writes `segment.warn_before` and three times `watch.tick` into every clock
record (RULE C6a, `records`), so the segment and doctor's watcher line read them from the
record, not from here.
"""

from __future__ import annotations

from moxi.config import Schema, Settings, load

MODULE = "cache"
SCHEMA_RESOURCE = "settings.toml"

_schema: Schema | None = None


def schema() -> Schema:
    """The module's schema, parsed once per process."""
    global _schema
    if _schema is None:
        _schema = Schema.from_resource(__package__, SCHEMA_RESOURCE)
    return _schema


def current() -> Settings:
    """The settings now. Raises `moxi.config.ConfigError` on a file that cannot be read."""
    return load(MODULE, schema())


def default(key: str):
    """A key's declared default, for a reader that shows status while the file is in error."""
    return schema().keys[key].default
