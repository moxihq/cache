"""The announcement and interface 1, as other modules reach them through `moxi.modules`."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from moxi import modules
from moxi.config import Schema
from moxi_cache import api, claude, clock, module
from moxi_sessions import api as sessions_api

SRC = Path(api.__file__).parent


def test_the_announcement_offers_interface_1():
    assert module.INTERFACES == frozenset({1})
    assert module.API == "moxi_cache.api"
    assert module.CLI == "moxi_cache.cli:main"
    assert module.SETTINGS == "settings.toml"
    assert module.OPTIONAL == {} and module.GATED == {} and module.INTERNAL_ENV == {}


def test_load_finds_the_module_and_its_api():
    modules.reset()
    found = modules.load("cache", interface=1)
    assert found.ok, found
    assert found.api is api
    modules.reset()
    assert not modules.load("cache", interface=2).ok


def test_interface_1_holds_its_names():
    assert {"read", "KINDS", "CODES", "MISSING", "open_store", "projects_root"} <= set(
        api.__all__)
    assert api.KINDS == ("warm", "cold", "fresh", "no-cache", "unknown")
    assert repr(api.MISSING) == "MISSING"


def test_the_read_signature_is_fixed():
    sig = inspect.signature(api.read)
    assert list(sig.parameters) == [
        "harness", "session_id", "now", "projects_root", "turned", "cache_source",
        "new_session", "agent_model", "bound_at", "store"]
    params = sig.parameters
    assert all(params[n].kind is inspect.Parameter.KEYWORD_ONLY
               for n in list(params)[2:])
    assert (params["turned"].default, params["new_session"].default,
            params["agent_model"].default, params["store"].default) == (
        True, False, api.MISSING, None)
    assert list(inspect.signature(api.open_store).parameters) == ["learn"]


def test_projects_root_is_sessions_rule_re_exported(tmp_path):
    env = {"HOME": "/home/someone"}
    assert api.projects_root(env) == "/home/someone/.claude/projects"
    env = {"HOME": "/home/someone", "CLAUDE_CONFIG_DIR": "~/elsewhere"}
    assert api.projects_root(env) == "/home/someone/elsewhere/projects"
    assert api.projects_root(env) == str(sessions_api.projects_root(env))
    assert api.projects_root(None) == str(tmp_path / "claude-config" / "projects")


def test_an_unknown_harness_reads_no_cache():
    for harness in ("claude-shim", "speech", "", None):
        got = api.read(harness, "aaaaaaaa-1111-4111-8111-111111111111")
        assert (got.kind, got.code) == ("no-cache", "unknown-harness"), harness
        assert got.reason == f"cache has no clock for {harness or 'unknown'} sessions"


def test_read_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("inside")

    for name in ("claude", "codex", "opencode"):
        target = getattr(api, name)
        monkeypatch.setattr(target, "read", boom)
        got = api.read(name, "x")
        assert (got.kind, got.code) == ("unknown", "read-failed")
        assert got.reason == "the cache read failed (RuntimeError: inside)"


def test_every_code_the_readers_give_is_in_codes():
    """A code is given only by name from `clock`; each such name is in CODES."""
    used = set()
    for path in SRC.glob("*.py"):
        used |= set(re.findall(r"_clock\.([A-Z_]+)\b", path.read_text()))
    used -= {"CLOCK_BLOCK", "CLOCK_SCAN_LIMIT"}
    assert used, "the readers name their codes through `clock`"
    for name in used:
        assert getattr(clock, name) in api.CODES, name


def test_a_clock_is_json(tmp_path):
    got = api.read("claude", "", new_session=True)
    doc = json.loads(json.dumps(got.as_dict()))
    assert doc["kind"] == "fresh" and set(doc) >= {"code", "ttl_source", "lowering"}


def test_the_schema_holds_the_first_releases_keys():
    schema = Schema.from_resource("moxi_cache", "settings.toml")
    assert list(schema.keys) == ["claude.ttl_seconds", "lifetime", "watch.tick",
                                 "segment.warn_before", "timer.subscriber"]
    subscribers = schema.keys["timer.subscriber"]
    assert (subscribers.type, subscribers.default, subscribers.read) == ("list", [], "use")
    ttl = schema.keys["claude.ttl_seconds"]
    assert (ttl.type, ttl.default, ttl.read) == ("int", 3600, "use")
    rows = schema.keys["lifetime"]
    assert (rows.type, rows.default, rows.read) == ("list", [], "use")
    tick = schema.keys["watch.tick"]
    assert (tick.type, tick.default, tick.read) == ("int", 10, "use")
    warn = schema.keys["segment.warn_before"]
    assert (warn.type, warn.default, warn.read) == ("int", 900, "use")


def test_the_tiers_are_one_hour_and_five_minutes():
    assert claude.TIER_1H == ("ephemeral_1h_input_tokens", 3600)
    assert claude.TIER_5M == ("ephemeral_5m_input_tokens", 300)
