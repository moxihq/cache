"""The Claude clock: the transcript's last request, what each ending of the read means,
the tier read past the clock line, and the evidence for `fresh`.

Every read here is against tmp paths. `api.read` is interface 1; `claude.last_request`
is the reader beneath it.
"""

from __future__ import annotations

import pytest
from conftest import (HOUR, assistant_line, user_line, write_cache_config,
                      write_sessions_config, write_transcript)

from moxi_cache import api, claude, clock

SESS_A = "aaaaaaaa-1111-4111-8111-111111111111"
SESS_B = "bbbbbbbb-2222-4222-8222-222222222222"


def read(session_id, root=None, **kw):
    """A read with the caller naming the root as the session's own (`projects_root`),
    which with `turned` left True is a reader's root: a missing file is `unknown`."""
    if root is not None:
        kw.setdefault("projects_root", str(root))
    return api.read("claude", session_id, **kw)


# ============================================================ the clock and its kinds

def test_a_warm_and_a_cold_clock(tmp_path):
    write_transcript(SESS_A, 60, root=tmp_path)
    write_transcript(SESS_B, 2 * HOUR, root=tmp_path)
    warm = read(SESS_A, tmp_path)
    cold = read(SESS_B, tmp_path)
    assert warm.kind == "warm" and 55 <= warm.age_seconds <= 120
    assert cold.kind == "cold" and cold.age_seconds >= 2 * HOUR
    assert warm.code is None and cold.code is None


def test_fresh_on_each_kind_of_evidence(tmp_path):
    """(1) a transcript read whole with no line that meets the rule; (2) the caller's
    word that its next delivery starts a new session, with no id; (3) no transcript under
    the session's own readable root with no prompt seen."""
    write_transcript(SESS_A, None, root=tmp_path)
    assert read(SESS_A, tmp_path).kind == "fresh"
    write_transcript(SESS_A, None, root=tmp_path, lines=[])       # empty file: read whole
    assert read(SESS_A, tmp_path).kind == "fresh"
    for empty in ("", None):
        assert api.read("claude", empty, new_session=True).kind == "fresh"
    assert read(SESS_B, tmp_path, turned=False).kind == "fresh"


@pytest.mark.parametrize("empty", ["", None])
def test_an_empty_id_without_the_callers_word_is_unknown(empty):
    clock = api.read("claude", empty, new_session=False, projects_root="/tmp",
                     turned=False)
    assert (clock.kind, clock.code) == ("unknown", "no-session-yet")
    assert clock.reason == "no session id reported yet"


def test_new_session_with_an_id_reads_the_transcript(tmp_path):
    """The caller's word is about an empty id only: a session id is read by its
    transcript."""
    write_transcript(SESS_A, 2 * HOUR, root=tmp_path)
    assert read(SESS_A, tmp_path, new_session=True).kind == "cold"


@pytest.mark.parametrize("age,kind", [(60, "warm"), (3 * HOUR, "cold")])
def test_an_id_from_someones_record_is_read_from_its_transcript(tmp_path, age, kind):
    """An id taken from a kind's evidence (a record of a session someone ran) is passed
    with `projects_root=None` and `turned=True`: read under the reader's root."""
    write_sessions_config(tmp_path, f'[claude]\nprojects = "{tmp_path}"\n')
    write_transcript(SESS_B, age, root=tmp_path)
    clock = api.read("claude", SESS_B, projects_root=None, turned=True)
    assert clock.kind == kind and clock.age_seconds >= age - 1


def test_an_id_from_someones_record_with_no_transcript_is_unknown(tmp_path):
    write_sessions_config(tmp_path, f'[claude]\nprojects = "{tmp_path}"\n')
    clock = api.read("claude", SESS_B, projects_root=None, turned=True)
    assert (clock.kind, clock.code) == ("unknown", "no-record")


@pytest.mark.parametrize("case", ["no root reported", "root missing", "root unreadable",
                                  "turned"])
def test_no_transcript_without_the_sessions_own_evidence_is_unknown(tmp_path, case):
    root = tmp_path / "projects"
    root.mkdir()
    kw = dict(projects_root=str(root), turned=False)
    if case == "no root reported":
        write_sessions_config(tmp_path, f'[claude]\nprojects = "{root}"\n')
        kw = dict(turned=False)
    elif case == "root missing":
        kw["projects_root"] = str(tmp_path / "nowhere")
    elif case == "root unreadable":
        root.chmod(0)
    elif case == "turned":
        kw["turned"] = True
    try:
        clock = api.read("claude", SESS_B, **kw)
    finally:
        root.chmod(0o700)
    assert (clock.kind, clock.code) == ("unknown", "no-record"), case
    assert f"no transcript for session {SESS_B} under" in clock.reason
    assert "nothing shows it has made no request" in clock.reason


def test_the_default_for_turned_holds_a_missing_transcript(tmp_path):
    """A caller that passes no evidence gets unknown."""
    assert read(SESS_B, tmp_path).kind == "unknown"


def test_a_session_that_stated_no_root_reads_under_sessions_root(tmp_path):
    """The reader's root is sessions' (config `claude.projects`, else the rule over the
    reader's environment); cache has no root setting of its own."""
    default = tmp_path / "claude-config" / "projects"
    write_transcript(SESS_A, 60, root=default)
    assert api.read("claude", SESS_A).kind == "warm"
    other = tmp_path / "elsewhere"
    write_transcript(SESS_A, 2 * HOUR, root=other)
    write_sessions_config(tmp_path, f'[claude]\nprojects = "{other}"\n')
    assert api.read("claude", SESS_A).kind == "cold"


def test_a_sessions_config_that_cannot_be_read_is_unknown(tmp_path):
    write_sessions_config(tmp_path, "[claude\n")
    clock = api.read("claude", SESS_A)
    assert (clock.kind, clock.code) == ("unknown", "config-error")


def test_synthetic_sidechain_and_usage_less_lines_are_skipped(tmp_path):
    real = assistant_line(2 * HOUR)
    newer = [assistant_line(10, message={"model": "<synthetic>", "usage": {}}),
             assistant_line(10, isSidechain=True),
             assistant_line(10, message={"model": "claude-model-x"}),     # no usage
             {"type": "user", "timestamp": "2026-10-05T00:00:00Z"}]
    write_transcript(SESS_A, root=tmp_path, lines=[real, *newer])
    assert read(SESS_A, tmp_path).kind == "cold", \
        "the clock is the main conversation's last request"


def test_a_final_line_longer_than_one_block(tmp_path):
    big = assistant_line(30, pad="x" * (3 * clock.CLOCK_BLOCK))
    write_transcript(SESS_A, root=tmp_path, lines=[assistant_line(2 * HOUR), big])
    assert read(SESS_A, tmp_path).kind == "warm"
    tail = {"type": "user", "pad": "y" * (3 * clock.CLOCK_BLOCK)}
    write_transcript(SESS_B, root=tmp_path, lines=[assistant_line(30), tail])
    assert read(SESS_B, tmp_path).kind == "warm"


# ============================================================ the endings of the read

def test_a_read_that_ends_at_the_limit_is_unknown_never_fresh(tmp_path, monkeypatch):
    """A file whose only request is in its first block reads by its clock when the read
    reaches it, and `unknown` when the limit stops it first: a long tail with no request
    is not evidence of no request."""
    tail = [{"type": "user", "pad": "x" * 1000} for _ in range(400)]       # ~400 KB
    path = write_transcript(SESS_A, root=tmp_path, lines=[assistant_line(2 * HOUR), *tail])
    assert read(SESS_A, tmp_path).kind == "cold"
    monkeypatch.setattr(clock, "CLOCK_SCAN_LIMIT", 2 * clock.CLOCK_BLOCK)
    got = read(SESS_A, tmp_path)
    assert (got.kind, got.code) == ("unknown", "scan-limit")
    assert "no request found in the last" in got.reason
    assert claude.last_request(path).outcome == "limit"


def test_a_read_to_the_start_with_no_request_is_fresh(tmp_path):
    path = write_transcript(SESS_A, root=tmp_path,
                            lines=[user_line(), {"type": "system", "subtype": "x"}])
    assert claude.last_request(path).outcome == "none"
    assert read(SESS_A, tmp_path).kind == "fresh"


@pytest.mark.parametrize("lines", [
    ["{ this will not parse", {"type": "user", "timestamp": "2026-10-05T00:00:00Z"}],
    [{"type": "assistant", "timestamp": "2026-10-05T00:00:00Z", "message": "text"}],
    ['["a", "list"]'],
], ids=["bad json", "message not an object", "not an object"])
def test_a_read_to_the_start_past_a_line_that_will_not_parse_is_unknown(tmp_path, lines):
    """No qualifying line in a file with lines that cannot be read is not evidence of no
    request, so it is unknown, never fresh."""
    write_transcript(SESS_A, root=tmp_path, lines=lines)
    got = read(SESS_A, tmp_path)
    assert (got.kind, got.code) == ("unknown", "malformed")
    assert "lines that will not parse and no request" in got.reason


def test_an_unterminated_last_line_is_judged_like_any_other(tmp_path):
    """sessions hands the unterminated last line with its flag; the clock judges it. One
    that meets the rule is the clock; one that will not parse is a line that will not
    parse, and an older request behind it is the clock."""
    write_transcript(SESS_A, root=tmp_path,
                     lines=[assistant_line(2 * HOUR), assistant_line(60)],
                     newline_at_end=False)
    assert read(SESS_A, tmp_path).kind == "warm"
    write_transcript(SESS_B, root=tmp_path,
                     lines=[assistant_line(2 * HOUR), '{"type": "assist'],
                     newline_at_end=False)
    assert read(SESS_B, tmp_path).kind == "cold"
    write_transcript(SESS_B, root=tmp_path, lines=[user_line(), '{"type": "assist'],
                     newline_at_end=False)
    assert read(SESS_B, tmp_path).code == "malformed"


@pytest.mark.parametrize("stamp", [None, "", "not a time", 1759622400],
                         ids=["no stamp", "blank", "garbled", "a number"])
def test_a_last_request_with_no_readable_stamp_is_unknown(tmp_path, stamp):
    """An older stamped request behind it does not stand in for it: unknown, not cold
    and not warm."""
    last = assistant_line(10)
    if stamp is None:
        del last["timestamp"]
    else:
        last["timestamp"] = stamp
    write_transcript(SESS_A, root=tmp_path, lines=[assistant_line(2 * HOUR), last])
    got = read(SESS_A, tmp_path)
    assert (got.kind, got.code) == ("unknown", "no-stamp")
    assert got.reason == "the transcript's last request has no readable time"


@pytest.mark.parametrize("sid", ["*", "?" * 36, "[ab]*", "../proj/" + SESS_A, ".hidden",
                                 "proj/" + SESS_A],
                         ids=["star", "question marks", "class", "parent", "dot", "slash"])
def test_a_session_id_that_cannot_be_looked_up_is_unknown(tmp_path, sid):
    """An id with a glob character, a path separator or a leading dot is never searched
    for. Without the guard `*` would glob `*/*.jsonl` and read the newest transcript on
    the box, here a warm one, as this session's."""
    write_transcript(SESS_A, 30, root=tmp_path)
    got = read(sid, tmp_path)
    assert (got.kind, got.code) == ("unknown", "bad-id")
    assert got.reason == f"the session id {sid!r} is not one that can be looked up"
    reported = read(sid, tmp_path, turned=False)
    assert reported.kind == "unknown", "and never fresh on the session's own word"


def test_an_unreadable_file_is_unknown(tmp_path):
    path = write_transcript(SESS_A, 30, root=tmp_path)
    path.chmod(0)
    try:
        got = read(SESS_A, tmp_path)
    finally:
        path.chmod(0o600)
    assert (got.kind, got.code) == ("unknown", "unreadable")
    assert "could not be read" in got.reason


def test_a_stamp_in_the_future_is_unknown(tmp_path):
    write_transcript(SESS_A, -6, root=tmp_path)
    got = read(SESS_A, tmp_path)
    assert (got.kind, got.code) == ("unknown", "future-stamp")
    assert "stamped in the future" in got.reason
    write_transcript(SESS_A, -4, root=tmp_path)
    got = read(SESS_A, tmp_path)
    assert (got.kind, got.age_seconds, got.ttl_seconds) == ("warm", 0.0, 3600)


# ============================================================ the lifetime: the tier

@pytest.mark.parametrize("tier,ttl", [("1h", 3600), ("5m", 300), ("both", 300)])
def test_the_clock_lines_tier_is_the_lifetime(tmp_path, tier, ttl):
    write_transcript(SESS_A, root=tmp_path, lines=[assistant_line(ttl - 30, tier=tier)])
    got = read(SESS_A, tmp_path)
    assert (got.kind, got.ttl_seconds, got.ttl_source) == ("warm", ttl, "tier")
    write_transcript(SESS_A, root=tmp_path, lines=[assistant_line(ttl, tier=tier)])
    assert read(SESS_A, tmp_path).kind == "cold"


def test_a_five_minute_user_reads_cold_past_five_minutes(tmp_path):
    """The reason for the tier rule: on the five-minute tier a 10-minute-old request is
    cold, where the old constant hour read it warm for 55 more minutes."""
    write_transcript(SESS_A, root=tmp_path, lines=[assistant_line(600, tier="5m")])
    got = read(SESS_A, tmp_path)
    assert (got.kind, got.ttl_seconds) == ("cold", 300)


@pytest.mark.parametrize("untiered", ["none", None], ids=["wrote-nothing", "no-field"])
def test_a_clock_line_with_no_tier_reads_on_past_it_to_one(tmp_path, untiered):
    """A request that only read the cache wrote no tier; it carries the tier of the write
    it read, so the read goes on past the clock line to the newest tiered line."""
    lines = [assistant_line(4000, tier="1h"), assistant_line(3000, tier="5m"),
             user_line(), assistant_line(100, tier=untiered),
             assistant_line(90, tier="1h", isSidechain=True)]
    path = write_transcript(SESS_A, root=tmp_path, lines=lines)
    got = read(SESS_A, tmp_path)
    assert (got.ttl_seconds, got.ttl_source) == (300, "tier"), \
        "the newest tiered line at or before the clock line; a sidechain's does not count"
    assert 95 <= got.age_seconds <= 160, "the clock is still the newest request"
    assert got.kind == "warm" and claude.last_request(path).tier == 300


def test_lines_after_the_clock_line_name_no_tier(tmp_path):
    """Only lines at or before the clock line count: a newer sidechain or synthetic line
    that names a tier is not one."""
    lines = [assistant_line(400, tier="1h"),
             assistant_line(10, tier="5m", isSidechain=True)]
    write_transcript(SESS_A, root=tmp_path, lines=lines)
    got = read(SESS_A, tmp_path)
    assert (got.kind, got.ttl_seconds) == ("warm", 3600)


def test_no_tiered_line_to_the_start_uses_the_config(tmp_path, own_roots):
    write_transcript(SESS_A, root=tmp_path, lines=[assistant_line(2000, tier="none"),
                                                   assistant_line(1000)])
    got = read(SESS_A, tmp_path)
    assert (got.kind, got.ttl_seconds, got.ttl_source) == ("warm", 3600, "config")
    write_cache_config(own_roots, "[claude]\nttl_seconds = 900\n")
    got = read(SESS_A, tmp_path)
    assert (got.kind, got.ttl_seconds, got.ttl_source) == ("cold", 900, "config")


def test_the_tier_read_that_ends_at_the_limit_uses_the_config(tmp_path, monkeypatch):
    """The tier read is the same backward read, within the same limit: past the clock
    line it may stop at the limit with no tiered line, and the config's lifetime applies."""
    tail = [{"type": "user", "pad": "x" * 1000} for _ in range(400)]
    lines = [assistant_line(4000, tier="5m"), *tail, assistant_line(1000, tier="none")]
    path = write_transcript(SESS_A, root=tmp_path, lines=lines)
    assert read(SESS_A, tmp_path).ttl_seconds == 300
    monkeypatch.setattr(clock, "CLOCK_SCAN_LIMIT", 2 * clock.CLOCK_BLOCK)
    found = claude.last_request(path)
    assert (found.outcome, found.tier, found.tier_ended) == ("clock", None, "limit")
    got = read(SESS_A, tmp_path)
    assert (got.kind, got.ttl_seconds, got.ttl_source) == ("warm", 3600, "config")


def test_a_config_that_cannot_be_read_holds_only_the_reads_that_need_it(tmp_path,
                                                                      own_roots):
    write_cache_config(own_roots, "[claude\n")
    write_transcript(SESS_A, root=tmp_path, lines=[assistant_line(100, tier="1h")])
    assert read(SESS_A, tmp_path).kind == "warm"
    write_transcript(SESS_B, root=tmp_path, lines=[assistant_line(100)])
    got = read(SESS_B, tmp_path)
    assert (got.kind, got.code) == ("unknown", "config-error")


def test_tier_of_reads_counts_above_zero():
    assert claude.tier_of({"usage": {"cache_creation": {
        "ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0}}}) is None
    assert claude.tier_of({"usage": {"cache_creation": {
        "ephemeral_1h_input_tokens": 3}}}) == 3600
    assert claude.tier_of({"usage": {"cache_creation": {
        "ephemeral_1h_input_tokens": True}}}) is None
    assert claude.tier_of({"usage": {}}) is None


# ============================================================ never raises

def test_the_read_never_raises(monkeypatch):
    def boom(*a, **k):
        raise ZeroDivisionError("division by zero")

    monkeypatch.setattr(claude.transcripts, "probe_root", boom)
    got = api.read("claude", SESS_A, projects_root="/tmp", turned=False)
    assert (got.kind, got.code) == ("unknown", "read-failed")
    assert "ZeroDivisionError" in got.reason
