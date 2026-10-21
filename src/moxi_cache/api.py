"""cache's interface 1, the module other Moxi modules reach (`moxi_cache.api`).

cache says what a session's cache is; the caller decides what to do about it. A caller
that has cache as an optional module reaches this module only through
`moxi.modules.load("cache", interface=1).api`.

* `read(harness, session_id, ...)`: one session's `Clock`. Never raises.
* `KINDS`: `warm`, `cold`, `fresh`, `no-cache`, `unknown`.
* `CODES`: every reason code a read can carry; every `no-cache` and `unknown` carries one.
* `MISSING`: an `agent_model` the caller did not pass.
* `open_store(learn=...)`: the drift record behind measured lifetimes (`drift.Store`).
* `projects_root(environ)`: sessions' rule for Claude Code's projects root in an
  environment, re-exported, so a caller states a session's root without reaching sessions.
* `Clock`: what `read` gives back.
* `segment(session_id, now=None)` and `Segment`: a Claude session's status line segment,
  from the watcher's clock record (`moxi_cache.segment`). Imported on first use, so loading
  this module does not load the segment module, which a renderer loads inside its audit
  hook.
"""

from __future__ import annotations

import time

from moxi_sessions.api import projects_root as _sessions_projects_root

from . import claude, codex, opencode
from . import clock as _clock
from . import table as _table
from .clock import CODES, KINDS, MISSING, NO_CACHE, Clock, unknown
from .drift import Store, open_store

#: The harnesses `read` has a clock for.
HARNESSES = (claude.HARNESS, codex.HARNESS, opencode.HARNESS)


def read(harness, session_id, *, now=None, projects_root=None, turned=True,
         cache_source=None, new_session=False, agent_model=MISSING, bound_at=None,
         store: Store | None = None) -> Clock:
    """One session's cache: its kind, its age, the lifetime it counts, and why.

    `harness` is `claude`, `codex` or `opencode`; any other reads `no-cache`, code
    `unknown-harness`. `session_id` is the harness's id for the session (a Claude session,
    a Codex thread, an OpenCode session).

    The session's own evidence for `fresh`: `projects_root` is the Claude projects root the
    session itself stated (`projects_root(environ)` over the environment its launcher
    stamped), `turned` whether a prompt or a turn end has been seen in the session, and
    `new_session` the caller's word that its next delivery starts a new session (with an
    empty `session_id`). For a Claude session that stated no root, the root is sessions'
    (config `claude.projects`). A session id taken from someone's record of a session it
    ran is passed with `projects_root=None` and `turned=True`, so a missing transcript
    reads `unknown`.

    For Codex and OpenCode: `cache_source` is the root the launcher reported (the Codex
    sessions directory, the OpenCode database), `agent_model` the OpenCode launcher's
    report of the default agent and its model, `bound_at` when the session registered with
    the caller, and `store` the drift record (`open_store`); with no store, a Codex model
    on a measured row reads `unknown`. Claude ignores `cache_source`.

    `turned` defaults to True, `new_session` to False and `agent_model` to not reported,
    so a caller that passes no evidence gets `unknown` for a missing record. An exception
    inside the read is `unknown`, code `read-failed`.
    """
    try:
        return _read(harness, session_id, now=time.time() if now is None else now,
                     projects_root=projects_root, turned=turned, cache_source=cache_source,
                     new_session=new_session, agent_model=agent_model, bound_at=bound_at,
                     store=store)
    except Exception as e:                                         # noqa: BLE001
        return unknown(_clock.READ_FAILED,
                       f"the cache read failed ({type(e).__name__}: {e})")


def _read(harness, session_id, *, now, projects_root, turned, cache_source, new_session,
          agent_model, bound_at, store) -> Clock:
    if harness == claude.HARNESS:
        return claude.read(session_id, now=now, projects_root=projects_root, turned=turned,
                           new_session=new_session)
    if harness == codex.HARNESS:
        if store is None:
            drift, judged = MISSING, None
        else:
            try:
                table = _table.load_table()
            except _table.TableError as e:
                return unknown(_clock.TABLE_NOT_LOADED, f"{_table.NOT_LOADED} ({e})")
            drift = store.view(table)
            judged = store.judged(session_id) if session_id else None
        result = codex.read(session_id, cache_source=cache_source, turned=turned,
                            new_session=new_session, now=now, drift=drift,
                            bound_at=bound_at, judged=judged)
        if store is not None:
            result = store.take(result, table)
        return result
    if harness == opencode.HARNESS:
        return opencode.read(session_id, cache_source=cache_source, turned=turned,
                             new_session=new_session, agent_model=agent_model, now=now)
    return Clock(NO_CACHE, reason=f"cache has no clock for {harness or 'unknown'} sessions",
                 code=_clock.UNKNOWN_HARNESS)


def __getattr__(name):
    if name in ("segment", "Segment"):
        from . import segment as _segment

        return getattr(_segment, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def projects_root(environ=None) -> str:
    """Claude Code's projects root, `${CLAUDE_CONFIG_DIR:-~/.claude}/projects`, by the
    environment `environ` (a session's own); with None, the root for a session that stated
    none. sessions' rule, re-exported."""
    return str(_sessions_projects_root(environ))


__all__ = ["read", "KINDS", "CODES", "MISSING", "open_store", "projects_root", "Clock",
           "HARNESSES", "segment", "Segment"]
