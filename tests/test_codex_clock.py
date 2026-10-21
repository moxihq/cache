"""The Codex clock: a thread's rollout read by the evidence rule, the two pairs, `fresh`,
and drift on a measured row.

Fixture rollouts are built in Codex 0.153.0's shapes; every read is against tmp paths.
The pure reads call `codex.read` with the drift view a test builds, as the store would
hand it; `test_drift.py` covers the store itself, and `api.read` the store's seam.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from moxi_cache import api, codex, drift, opencode, table
from moxi_cache import clock as _clock
from moxi_sessions.api import claude as transcripts

NOW = 1_791_300_000.0                    # a fixed "now" for the pure reads
TID = "0190c000-0000-7000-8000-00000000000a"
TID_B = "0190c000-0000-7000-8000-00000000000b"
OTHER = "0190c000-0000-7000-8000-0000000009ff"

FULL_LIMIT = _clock.CLOCK_SCAN_LIMIT


def measured_seconds() -> int:
    """The measured codex row's number, as the table in force gives it."""
    row = table.load_table().lookup("codex", "openai", "gpt-5.6-sol")
    return row.seconds


def measured_models() -> list:
    return [m for m in ("gpt-5.6-sol", "gpt-6-astra")
            if table.load_table().lookup("codex", "openai", m).measured]


# =================================================================== rollout fixtures

def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def meta_payload(tid=TID, **over) -> dict:
    payload = {"session_id": tid, "id": tid, "timestamp": iso(NOW - 86400),
               "cwd": "/tmp/work", "originator": "moxi-test", "cli_version": "0.153.0",
               "source": "vscode", "thread_source": "user", "model_provider": "openai",
               "base_instructions": {"text": "You are Codex."},
               "history_mode": "paginated", "multi_agent_version": 1,
               "context_window": 258400}
    payload.update(over)
    return {k: v for k, v in payload.items() if v is not ...}


class Rollout:
    """A rollout, line by line, in Codex 0.153.0's shapes."""

    def __init__(self, tid=TID, **meta):
        self.tid = tid
        self.lines = [{"timestamp": iso(NOW - 86400), "ordinal": 0, "type": "session_meta",
                       "payload": meta_payload(tid, **meta)}]
        self.raw_tail = None

    def add(self, kind, t, payload):
        self.lines.append({"timestamp": iso(t), "ordinal": len(self.lines), "type": kind,
                           "payload": payload})
        return self

    def event(self, sub, t, **payload):
        return self.add("event_msg", t, {"type": sub, **payload})

    def item(self, sub, t, **payload):
        return self.add("response_item", t, {"type": sub, **payload})

    def start(self, turn, t):
        return self.event("task_started", t, turn_id=turn, started_at=int(t))

    def context(self, turn, t, model="gpt-5.6-terra"):
        return self.add("turn_context", t, {"turn_id": turn, "model": model,
                                            "cwd": "/tmp/work"})

    def tsa(self, t, model="gpt-5.6-terra", provider="openai", thread=...):
        thread = self.tid if thread is ... else thread
        payload = {"type": "thread_settings_applied",
                   "thread_settings": {"model": model, "model_provider_id": provider}}
        if thread is not None:
            payload["thread_id"] = thread
        return self.add("event_msg", t, payload)

    def record(self, turn, t, *, inp=20_000, cached=19_500, session=..., thread=...,
               rid=None):
        return self.add("token_usage_record", t, {
            "thread_id": self.tid if thread is ... else thread, "turn_id": turn,
            "session_id": self.tid if session is ... else session,
            "response_id": rid or f"resp_{turn}_{int(t)}",
            "usage": {"input_tokens": inp, "cached_input_tokens": cached,
                      "output_tokens": 5, "total_tokens": inp + 5}})

    def turn(self, turn, t, *, model="gpt-5.6-terra", length=5.0, inp=20_000,
             cached=19_500, records=1):
        """A whole turn: start, context, the user message, `records` requests, done."""
        self.start(turn, t).context(turn, t + 0.01, model)
        self.event("item_completed", t + 0.02, item={"type": "UserMessage", "id": "u"})
        self.item("message", t + 0.02, role="user", content=[])
        for n in range(records):
            stamp = t + length * (n + 1) / records
            self.item("reasoning", stamp - 0.01, summary=[])
            self.record(turn, stamp, inp=inp, cached=cached if n == 0 else inp - 100)
        self.item("message", t + length + 0.01, role="assistant", content=[])
        return self.event("task_complete", t + length + 0.02, turn_id=turn)

    def text(self) -> str:
        body = "".join(json.dumps(line) + "\n" for line in self.lines)
        if self.raw_tail is not None:
            body += self.raw_tail
        return body

    def write(self, root: Path, *, suffix="", name_tid=None, day="2026/10/05",
              zst=False) -> Path:
        directory = root / day
        directory.mkdir(parents=True, exist_ok=True)
        name = f"rollout-2026-10-05T15-55-21-{name_tid or self.tid}{suffix}.jsonl"
        if zst:
            name += ".zst"
        path = directory / name
        path.write_text(self.text())
        return path


def sessions_root(tmp_path) -> Path:
    root = tmp_path / "codex-home" / "sessions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def codex_read(root, tid=TID, *, turned=True, new_session=False, now=NOW, drift=None,
               bound_at=0.0, judged=None):
    return codex.read(tid, cache_source=str(root) if root else None, turned=turned,
                      new_session=new_session, now=now,
                      drift={} if drift is None else drift, bound_at=bound_at,
                      judged=judged)


def one_turn(age, model="gpt-5.6-terra", **kw) -> Rollout:
    """A thread whose one turn's request ended `age` seconds before NOW."""
    return Rollout(**kw).turn("t1", NOW - age - 5, model=model)


# ================================================================== the codex reader


@pytest.mark.parametrize("model", ["gpt-5.6-terra", "gpt-5.6-luna"])
def test_t1_the_30_minute_rows(tmp_path, model):
    root = sessions_root(tmp_path)
    one_turn(1799, model).write(root)
    clock = codex_read(root)
    assert (clock.kind, clock.ttl_seconds, clock.key) == ("warm", 1800,
                                                          f"codex openai/{model}")
    one_turn(1800, model).write(root)
    assert codex_read(root).kind == "cold"


def test_t1_each_model_on_the_measured_row(tmp_path):
    seconds = measured_seconds()
    for model in measured_models():
        root = sessions_root(tmp_path / model)
        one_turn(seconds - 1, model).write(root)
        clock = codex_read(root)
        assert (clock.kind, clock.ttl_seconds) == ("warm", seconds)
        assert clock.key == f"codex openai/{model} (measured on this box)"
        one_turn(seconds, model).write(root)
        assert codex_read(root).kind == "cold"


@pytest.mark.parametrize("model,provider", [("gpt-5.5", "openai"),
                                            ("gpt-5.6-terra", "azure")])
def test_t2_no_row_reads_no_cache_with_the_reason(tmp_path, model, provider):
    """A model with no lifetime row has no clock to count by: it reads `no-cache`, so
    mail goes live."""
    root = sessions_root(tmp_path)
    one_turn(10, model, model_provider=provider).write(root)
    clock = codex_read(root)
    assert clock.kind == "no-cache" and clock.ttl_seconds is None
    assert clock.reason == _clock.no_row(provider, model)
    assert clock.age_seconds == pytest.approx(10, abs=0.1)


def test_t3_a_turn_context_naming_another_model_after_the_record_is_cold(tmp_path):
    root = sessions_root(tmp_path)
    r = one_turn(60)
    r.start("t2", NOW - 30).context("t2", NOW - 29, "gpt-5.6-luna")
    r.write(root)
    clock = codex_read(root)
    assert clock.kind == "cold" and clock.reason == codex.CHANGED


def test_t3_a_settings_line_naming_another_provider_after_the_record_is_cold(tmp_path):
    root = sessions_root(tmp_path)
    r = one_turn(60)
    r.tsa(NOW - 20, model="gpt-5.6-terra", provider="azure")
    r.write(root)
    assert codex_read(root).reason == codex.CHANGED


def test_t3_with_no_settings_line_the_provider_is_line_ones(tmp_path):
    root = sessions_root(tmp_path)
    one_turn(60).write(root)
    assert codex_read(root).kind == "warm"
    one_turn(60, model_provider=None).write(root)
    clock = codex_read(root)
    assert clock.kind == "unknown" and clock.reason == codex.NO_PAIR


@pytest.mark.parametrize("thread", [OTHER, None], ids=["another-thread", "no-thread"])
def test_t3_a_settings_line_of_another_thread_is_unknown(tmp_path, thread):
    root = sessions_root(tmp_path)
    r = Rollout().tsa(NOW - 200, thread=thread)
    r.turn("t1", NOW - 100)
    r.write(root)
    clock = codex_read(root)
    assert clock.kind == "unknown" and clock.reason == codex.OTHER_THREAD


@pytest.mark.parametrize("line", ["context", "settings"])
def test_t3_a_next_model_that_cannot_be_read_is_unknown(tmp_path, line):
    root = sessions_root(tmp_path)
    r = one_turn(60)
    if line == "context":
        r.add("turn_context", NOW - 30, {"turn_id": "t2", "model": None})
    else:
        r.add("event_msg", NOW - 30, {"type": "thread_settings_applied", "thread_id": TID,
                                      "thread_settings": {"model_provider_id": "openai"}})
    r.write(root)
    clock = codex_read(root)
    assert clock.kind == "unknown" and clock.reason == codex.NO_PAIR


def test_t35_a_scan_that_reaches_the_limit_stops_judging_there(tmp_path, monkeypatch):
    """R_prev's provider is line 1's only when the scan reached line 1: a settings line
    could sit in what the limit left unread."""
    gap = measured_seconds() - 1200
    r = Rollout().start("t0", A - 9000).context("t0", A - 8999, "gpt-5.6-sol")
    for _n in range(300):
        r.item("reasoning", A - 8000, summary=["z" * 400])
    r.record("t0", A - 7000, inp=48_210, cached=48_000)
    r.start("t1", A - 30).context("t1", A - 29, "gpt-5.6-sol")
    r.record("t1", A, inp=48_210, cached=48_000)
    r.start("t2", A + gap).context("t2", A + gap + 0.01, "gpt-5.6-sol")
    r.tsa(A + gap + 0.02, model="gpt-5.6-sol")
    r.record("t2", A + gap + 5, inp=48_700, cached=7_424)
    monkeypatch.setattr(_clock, "CLOCK_SCAN_LIMIT", 64 * 1024)
    clock = drift_read(tmp_path, r)
    assert clock.kind == "warm" and clock.lowering is None
    monkeypatch.setattr(_clock, "CLOCK_SCAN_LIMIT", FULL_LIMIT)
    assert drift_read(tmp_path / "whole", r).lowering is not None


def test_t3_a_record_whose_turn_has_no_turn_context_is_unknown(tmp_path):
    root = sessions_root(tmp_path)
    r = Rollout().turn("t0", NOW - 500)          # an earlier turn_context in the file
    r.start("t1", NOW - 100).record("t1", NOW - 90)
    r.write(root)
    clock = codex_read(root)
    assert clock.kind == "unknown" and clock.reason == codex.NO_PAIR


def test_t3_a_scan_that_reaches_the_limit_before_its_evidence_is_unknown(tmp_path,
                                                                         monkeypatch):
    root = sessions_root(tmp_path)
    r = Rollout().start("t1", NOW - 100).context("t1", NOW - 99)
    for n in range(400):
        r.item("reasoning", NOW - 98 + n * 0.01, summary=["x" * 200])
    r.record("t1", NOW - 60)
    r.write(root)
    assert codex_read(root).kind == "warm"
    monkeypatch.setattr(_clock, "CLOCK_SCAN_LIMIT", 4096)
    clock = codex.read(TID, cache_source=str(root), turned=True, new_session=False, now=NOW, drift={})
    assert clock.kind == "unknown" and "MB" in clock.reason


def test_t4_a_record_of_another_session_does_not_move_the_clock(tmp_path):
    root = sessions_root(tmp_path)
    r = one_turn(600)
    r.record("t1", NOW - 10, session=f"guardian:{TID}")
    r.write(root)
    clock = codex_read(root)
    assert clock.age_seconds == pytest.approx(600, abs=0.1)


def test_t5_a_compactions_record_moves_the_clock(tmp_path):
    root = sessions_root(tmp_path)
    r = one_turn(1000)
    r.start("t2", NOW - 100).context("t2", NOW - 99)
    r.add("compacted", NOW - 98, {"message": "summary"})
    r.record("t2", NOW - 50, inp=13_000, cached=12_000)
    r.write(root)
    assert codex_read(root).age_seconds == pytest.approx(50, abs=0.1)


def test_t6_an_older_codex_with_token_count_and_no_record_is_unknown(tmp_path):
    root = sessions_root(tmp_path)
    r = Rollout().start("t1", NOW - 100).context("t1", NOW - 99)
    r.event("token_count", NOW - 90, info={"total_token_usage": {}})
    r.write(root)
    clock = codex_read(root, turned=False)
    assert clock.kind == "unknown" and clock.reason == codex.OLDER_CODEX


def fresh_rollout() -> Rollout:
    """Every kind a new thread writes before its first response."""
    r = Rollout().start("t1", NOW - 10)
    r.item("message", NOW - 9.9, role="developer", content=[])
    r.item("message", NOW - 9.8, role="user", content=[])
    r.add("world_state", NOW - 9.7, {"full": True, "state": {}})
    r.context("t1", NOW - 9.6)
    r.event("item_completed", NOW - 9.5, item={"type": "UserMessage", "id": "u"})
    r.event("user_message", NOW - 9.5, message="hi")
    return r.tsa(NOW - 9.4)


def test_t7_a_whitelisted_rollout_of_the_launchers_own_thread_is_fresh(tmp_path):
    root = sessions_root(tmp_path)
    fresh_rollout().write(root)
    assert codex_read(root, turned=False).kind == "fresh"
    clock = codex_read(root, turned=True)
    assert clock.kind == "unknown" and clock.reason == codex.NO_REQUEST


@pytest.mark.parametrize("extra", [
    ("response_item", {"type": "message", "role": "assistant", "content": []}),
    ("response_item", {"type": "reasoning", "summary": []}),
    ("event_msg", {"type": "item_completed", "item": {"type": "AgentMessage"}}),
    ("event_msg", {"type": "task_complete", "turn_id": "t1"}),
    ("event_msg", {"type": "turn_aborted", "turn_id": "t1"}),
    ("event_msg", {"type": "token_count", "info": None}),
    ("compacted", {"message": "x"}),
    ("token_usage_record", {"thread_id": TID, "session_id": f"guardian:{TID}",
                            "turn_id": "t1", "usage": {}}),
], ids=["assistant", "reasoning", "agent-item", "task-complete", "turn-aborted",
        "token-count", "compacted", "other-record"])
def test_t7_one_line_off_the_fresh_whitelist_is_unknown(tmp_path, extra):
    root = sessions_root(tmp_path)
    fresh_rollout().add(extra[0], NOW - 5, extra[1]).write(root)
    clock = codex_read(root, turned=False)
    assert clock.kind == "unknown", extra


def test_t7_a_cli_version_the_reader_was_not_checked_against_is_unknown(tmp_path):
    root = sessions_root(tmp_path)
    r = fresh_rollout()
    r.lines[0]["payload"]["cli_version"] = "0.154.0"
    r.write(root)
    assert codex_read(root, turned=False).kind == "unknown"
    # ... but a record of the thread's own reads by its record, whatever the version.
    r.record("t1", NOW - 3).write(root)
    assert codex_read(root, turned=False).kind == "warm"


@pytest.mark.parametrize("where", ["before", "after", "none"])
@pytest.mark.parametrize("bad", [
    "not json {",
    json.dumps({"timestamp": iso(NOW - 5), "type": "inter_agent_communication",
                "payload": {}}),
    json.dumps({"timestamp": iso(NOW - 5), "type": "event_msg",
                "payload": {"type": "brand_new_event"}}),
    json.dumps({"timestamp": iso(NOW - 5), "type": "response_item",
                "payload": {"type": "brand_new_item"}}),
    json.dumps({"type": "turn_context", "payload": {}}),
], ids=["unparseable", "top-kind", "event-kind", "item-kind", "no-timestamp"])
def test_t7a_a_line_the_reader_does_not_recognise_is_unknown(tmp_path, bad, where):
    root = sessions_root(tmp_path)
    r = Rollout()
    if where == "before":
        r.turn("t1", NOW - 100)
    path = r.write(root)
    text = path.read_text() + bad + "\n"
    if where == "after":
        text = Rollout().turn("t1", NOW - 100).text() + bad + "\n"
    path.write_text(text)
    for turned in (True, False):
        clock = codex_read(root, turned=turned)
        assert clock.kind == "unknown", (where, turned)


def test_t7a_a_last_line_being_written_is_skipped_but_a_whole_one_is_judged(tmp_path):
    root = sessions_root(tmp_path)
    r = one_turn(60)
    r.raw_tail = '{"timestamp": "2026-10-05T20:55:2'          # no newline, no parse
    r.write(root)
    assert codex_read(root).kind == "warm"
    r.raw_tail = json.dumps({"timestamp": iso(NOW - 1), "type": "realtime_item",
                             "payload": {}})                # no newline, parses
    r.write(root)
    assert codex_read(root).kind == "unknown"


def test_t7a_the_lists_are_pinned():
    assert codex.TOP_KINDS == ("session_meta", "turn_context", "world_state",
                                     "token_usage_record", "compacted", "event_msg",
                                     "response_item")
    assert set(codex.EVENT_KINDS) == {
        "item_completed", "token_count", "thread_goal_updated", "thread_rolled_back",
        "turn_aborted", "task_started", "task_complete", "thread_settings_applied",
        "user_message", "agent_message", "agent_reasoning", "agent_reasoning_raw_content",
        "entered_review_mode", "exited_review_mode", "patch_apply_end",
        "context_compacted", "mcp_tool_call_end", "web_search_end",
        "image_generation_end", "sub_agent_activity"}
    assert set(codex.RESPONSE_KINDS) == {
        "message", "agent_message", "reasoning", "local_shell_call", "function_call",
        "tool_search_call", "function_call_output", "tool_search_output",
        "custom_tool_call", "custom_tool_call_output", "web_search_call",
        "image_generation_call", "compaction", "context_compaction"}
    assert set(codex.META_KEYS) == {
        "session_id", "id", "timestamp", "cwd", "originator", "cli_version", "source",
        "thread_source", "model_provider", "base_instructions", "history_mode",
        "multi_agent_version", "context_window", "git"}
    assert codex.CHECKED_VERSIONS == ("0.153.0",)
    assert opencode.CHECKED_VERSIONS == ("1.18.34",)


def test_t8_no_rollout_file(tmp_path):
    root = sessions_root(tmp_path)
    assert codex_read(root, turned=False).kind == "fresh"
    assert codex_read(root, turned=True).kind == "unknown"
    assert codex_read(tmp_path / "missing", turned=False).kind == "unknown"
    assert codex_read(None, turned=False).reason == codex.NO_ROOT
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        assert codex_read(locked, turned=False).kind == "unknown"
    finally:
        locked.chmod(0o700)


def test_t8a_finding_the_rollout(tmp_path):
    root = sessions_root(tmp_path)
    # A file for another thread whose id shares a prefix is not a candidate.
    Rollout(TID_B).turn("t1", NOW - 100).write(root)
    assert codex_read(root, turned=False).kind == "fresh"
    one_turn(60).write(root)
    assert codex_read(root).kind == "warm"


@pytest.mark.parametrize("turned", [True, False])
@pytest.mark.parametrize("which", ["old", "new"])
def test_t8a_a_reverted_thread_with_two_files_is_unknown(tmp_path, which, turned):
    root = sessions_root(tmp_path)
    old = Rollout() if which == "new" else one_turn(60)
    old.write(root)
    # A revert segment: same thread id, a `history_base` into the old file, and a
    # suffixed name (`revert_thread.rs`).
    new = Rollout(history_base={"thread_id": TID, "end_ordinal_exclusive": 3,
                                "end_byte_offset": 4096})
    if which == "new":
        new.turn("t1", NOW - 100)
    new.write(root, suffix="_0190c000-1111-7000-8000-00000000000a", day="2026/10/06")
    clock = codex_read(root, turned=turned)
    assert clock.kind == "unknown" and clock.reason == codex.SPLIT


def test_t8a_a_jsonl_beside_its_zst_is_unknown(tmp_path):
    root = sessions_root(tmp_path)
    one_turn(60).write(root)
    one_turn(60).write(root, zst=True)
    assert codex_read(root).reason == codex.SPLIT


def test_t9_new_session_with_no_id_is_fresh_and_only_then(tmp_path):
    root = sessions_root(tmp_path)
    assert codex_read(root, tid=None, new_session=True).kind == "fresh"
    assert codex_read(root, tid=None, new_session=False).kind == "unknown"
    one_turn(4000).write(root)
    assert codex_read(root, new_session=True).kind == "cold", \
        "ignored once an id is reported"


def test_t10_a_lone_zst_is_cold_with_no_age(tmp_path):
    root = sessions_root(tmp_path)
    one_turn(60).write(root, zst=True)
    clock = codex_read(root)
    assert (clock.kind, clock.age_seconds, clock.ttl_seconds) == ("cold", None, None)
    assert clock.reason == codex.COMPRESSED


def test_t11_unreadable_and_scan_limit_and_no_stamp(tmp_path, monkeypatch):
    root = sessions_root(tmp_path)
    path = one_turn(60).write(root)
    path.chmod(0)
    try:
        clock = codex_read(root)
        assert clock.kind == "unknown" and "could not be read" in clock.reason
    finally:
        path.chmod(0o600)
    r = Rollout().start("t1", NOW - 100)
    for _n in range(300):
        r.item("reasoning", NOW - 99, summary=["y" * 200])
    r.write(root)
    monkeypatch.setattr(_clock, "CLOCK_SCAN_LIMIT", 4096)
    clock = codex.read(TID, cache_source=str(root), turned=True, new_session=False, now=NOW, drift={})
    assert clock.kind == "unknown" and "no request found" in clock.reason
    monkeypatch.setattr(_clock, "CLOCK_SCAN_LIMIT", FULL_LIMIT)
    r = one_turn(60)
    record = [ln for ln in r.lines if ln["type"] == "token_usage_record"][-1]
    record["timestamp"] = "not a time"
    r.write(root)
    assert codex_read(root).reason == codex.NO_STAMP


def test_t12_a_stamp_in_the_future_is_unknown(tmp_path):
    root = sessions_root(tmp_path)
    Rollout().turn("t1", NOW + 1, length=5).write(root)        # ends 6 s ahead
    clock = codex_read(root)
    assert clock.kind == "unknown" and "stamped in the future" in clock.reason
    Rollout().turn("t1", NOW - 1, length=5).write(root)        # ends 4 s ahead
    clock = codex_read(root)
    assert (clock.kind, clock.age_seconds) == ("warm", 0.0)



# ============================================================ the whole-history check

@pytest.mark.parametrize("change", [
    {"history_base": {"thread_id": TID, "end_ordinal_exclusive": 3,
                      "end_byte_offset": 4096}},
    {"forked_from_id": OTHER},
    {"forked_from_ordinal_exclusive": 4}, {"parent_thread_id": OTHER},
    {"subagent_history_start_ordinal": 2}, {"brand_new_key": 1},
    {"id": OTHER}, {"session_id": OTHER}, {"history_mode": "lineage"},
], ids=["history_base", "forked_from_id", "forked_from_ordinal", "parent_thread",
        "subagent_start", "unknown-key", "other-id", "other-session", "history-mode"])
def test_t31_a_session_meta_that_is_not_the_whole_history_is_unknown(tmp_path, change):
    root = sessions_root(tmp_path)
    one_turn(60, **change).write(root)
    for turned in (True, False):
        clock = codex_read(root, turned=turned)
        assert clock.kind == "unknown" and clock.reason == codex.NOT_WHOLE
    fresh = fresh_rollout()
    fresh.lines[0]["payload"].update(change)
    fresh.write(root)
    assert codex_read(root, turned=False).kind == "unknown"


def test_t31_a_second_session_meta_and_a_line_one_that_is_not_one(tmp_path):
    root = sessions_root(tmp_path)
    r = one_turn(60)
    r.lines.insert(3, dict(r.lines[0]))
    r.write(root)
    assert codex_read(root).reason == codex.NOT_WHOLE
    r = one_turn(60)
    r.lines.insert(0, {"timestamp": iso(NOW - 90000), "type": "world_state",
                       "payload": {}})
    r.write(root)
    for turned in (True, False):
        assert codex_read(root, turned=turned).reason == codex.NOT_WHOLE



A = NOW - 20_000                          # R_prev, the previous turn's request


def drifting(gap, *, cached=7_424, inp=48_210, model="gpt-5.6-sol", prev_model=None,
             breaker=None, where=None, settings_between=False, second_cold=False,
             request_length=5.0) -> Rollout:
    """Turn t1 ends with R_prev at A; turn t2 starts `gap` seconds later and its first
    request R comes back with `cached` tokens."""
    r = Rollout().start("t1", A - 30).context("t1", A - 29, prev_model or model)
    if breaker and where == "earlier":
        r.add(*breaker(A - 20))
    r.record("t1", A, inp=inp, cached=inp - 200)
    r.event("task_complete", A + 0.1, turn_id="t1")
    if breaker and where == "between":
        r.add(*breaker(A + 10))
    if settings_between:
        r.tsa(A + gap - 1, model=model)
    r.start("t2", A + gap).context("t2", A + gap + 0.01, model)
    if breaker and where == "this-turn":
        r.add(*breaker(A + gap + 0.02))
    if second_cold:
        r.record("t2", A + gap + 2, inp=inp + 500, cached=inp)
        r.record("t2", A + gap + request_length, inp=inp + 900, cached=cached)
    else:
        r.record("t2", A + gap + request_length, inp=inp + 500, cached=cached)
    return r


BREAKERS = {
    "compacted": lambda t: ("compacted", t, {"message": "summary"}),
    "context_compacted": lambda t: ("event_msg", t, {"type": "context_compacted"}),
    "rolled_back": lambda t: ("event_msg", t, {"type": "thread_rolled_back",
                                               "num_turns": 1}),
    "turn_aborted": lambda t: ("event_msg", t, {"type": "turn_aborted", "turn_id": "t1"}),
}


def drift_read(tmp_path, rollout, *, drift=None, bound_at=0.0, judged=None, now=None):
    root = sessions_root(tmp_path)
    rollout.write(root)
    stamp = [ln for ln in rollout.lines if ln["type"] == "token_usage_record"][-1]
    end = transcripts.parse_timestamp(stamp["timestamp"])
    return codex_read(root, now=end + 60 if now is None else now, drift=drift,
                      bound_at=bound_at, judged=judged)


def test_t35_a_cold_landing_lowers_and_the_same_read_uses_it(tmp_path):
    seconds = measured_seconds()
    gap = seconds - 1200
    clock = drift_read(tmp_path, drifting(gap))
    lowered = codex.lower_to(gap, 1800)
    assert lowered == max(1800, int(gap * 0.8) // 60 * 60)
    assert clock.lowering["lifetime"] == lowered and clock.ttl_seconds == lowered
    assert clock.lowering["gap"] == pytest.approx(gap)
    assert clock.lowering["cached"] == 7_424 and clock.lowering["input"] == 48_210
    assert clock.key.startswith(f"codex openai/gpt-5.6-sol (measured on this box, lowered "
                                f"from {_clock.span(seconds)} on ")
    assert clock.lowered_from == seconds
    late = drift_read(tmp_path / "late", drifting(gap), now=A + gap + 5 + lowered)
    assert late.kind == "cold", "the verdict of the read that found it uses the new number"


@pytest.mark.parametrize("case", ["warm", "under-floor", "at-lifetime", "not-first",
                                  "other-pair", "settings-between", "registered-after",
                                  "small-prompt", "no-bound-at"])
def test_t35_each_of_these_lowers_nothing(tmp_path, case):
    seconds = measured_seconds()
    gap, kw, read_kw = seconds - 1200, {}, {}
    if case == "warm":
        kw["cached"] = 48_000
    elif case == "under-floor":
        gap = 1700
    elif case == "at-lifetime":
        gap = seconds
    elif case == "not-first":
        kw["second_cold"] = True
    elif case == "other-pair":
        kw["prev_model"] = "gpt-5.6-terra"
    elif case == "settings-between":
        kw["settings_between"] = True
    elif case == "registered-after":
        read_kw["bound_at"] = A + 1
    elif case == "small-prompt":
        kw["inp"] = 16_000
    elif case == "no-bound-at":
        read_kw["bound_at"] = None
    clock = drift_read(tmp_path, drifting(gap, **kw), **read_kw)
    assert clock.lowering is None, case
    assert clock.ttl_seconds == seconds


@pytest.mark.parametrize("breaker", sorted(BREAKERS))
@pytest.mark.parametrize("where", ["earlier", "between", "this-turn"])
def test_t35_a_turn_that_is_not_a_normal_continuation_lowers_nothing(tmp_path, breaker,
                                                                     where):
    clock = drift_read(tmp_path, drifting(measured_seconds() - 1200,
                                          breaker=BREAKERS[breaker], where=where))
    assert clock.lowering is None


def test_t35_a_row_not_marked_measured_is_never_lowered(tmp_path):
    clock = drift_read(tmp_path, drifting(1700, model="gpt-5.6-terra"))
    assert clock.lowering is None and clock.ttl_seconds == 1800 and clock.kind == "warm"


def test_t35_a_cold_landing_under_the_floor_is_a_journal_line_only(tmp_path):
    clock = drift_read(tmp_path, drifting(1500))
    assert clock.lowering is None and clock.ttl_seconds == measured_seconds()
    assert len(clock.notes) == 1 and "under the documented 30m" in clock.notes[0]


def entry(lifetime, *, seconds=None, checked=None, model="gpt-5.6-sol", **extra):
    row = table.load_table().lookup("codex", "openai", model)
    return {"harness": "codex", "provider": "openai", "model": model,
            "lifetime": lifetime, "row_seconds": row.seconds if seconds is None else seconds,
            "row_checked": row.checked if checked is None else checked, **extra}


def view_of(*entries):
    view, _kept, _rejected = drift.judge(drift.Record(entries=list(entries)),
                                               table.load_table())
    return view


def test_t35_a_longer_miss_never_raises_and_a_shorter_one_lowers_to_the_floor(tmp_path):
    seconds = measured_seconds()
    in_force = codex.lower_to(seconds - 1200, 1800)
    view = view_of(entry(in_force))
    clock = drift_read(tmp_path / "a", drifting(in_force + 300), drift=view)
    assert clock.lowering is None and clock.ttl_seconds == in_force
    clock = drift_read(tmp_path / "b", drifting(3000), drift=view)
    assert clock.lowering["lifetime"] == 2400 and clock.ttl_seconds == 2400
    clock = drift_read(tmp_path / "c", drifting(2000), drift=view)
    assert clock.lowering["lifetime"] == 1800, "never below the floor"


def test_t35_the_gap_is_measured_from_the_turns_start(tmp_path):
    gap = measured_seconds() - 1200
    clock = drift_read(tmp_path, drifting(gap, request_length=600))
    assert clock.lowering["gap"] == pytest.approx(gap)
    assert clock.lowering["lifetime"] == codex.lower_to(gap, 1800)


def test_t35_only_turns_newer_than_the_judged_one_are_judged(tmp_path):
    r = drifting(measured_seconds() - 1200)
    assert drift_read(tmp_path / "a", r).judged == (TID, "t2")
    assert drift_read(tmp_path / "b", r, judged="t2").lowering is None
    assert drift_read(tmp_path / "c", r, judged="t1").lowering is not None


# ===================================================== the drift record as read (T36)

def test_t36_a_read_with_no_drift_store_reads_a_measured_row_unknown(tmp_path):
    root = sessions_root(tmp_path)
    one_turn(60, "gpt-5.6-sol").write(root)
    clock = api.read("codex", TID, cache_source=str(root), now=NOW)
    assert clock.kind == "unknown" and "drift record" in clock.reason
    one_turn(60, "gpt-5.6-terra").write(root)
    assert api.read("codex", TID, cache_source=str(root), now=NOW).kind == "warm"


def test_t36_a_stale_entry_is_ignored(tmp_path):
    root = sessions_root(tmp_path)
    one_turn(2000, "gpt-5.6-sol").write(root)
    for stale in (entry(1800, seconds=999_999), entry(1800, checked="2020-01-01")):
        view, kept, rejected = drift.judge(drift.Record(entries=[stale]),
                                                 table.load_table())
        assert kept == [] and rejected == []
        assert codex_read(root, drift=view).ttl_seconds == measured_seconds()


@pytest.mark.parametrize("bad", ["unrecognised", "above", "below", "not-int", "bool"])
def test_t36_a_rejected_entry_holds_every_measured_read(tmp_path, bad):
    """The fail-closed reading: a rejected entry is read like an unreadable record,
    never as the longer shipped number."""
    seconds = measured_seconds()
    e = {"unrecognised": {"what": "is this"}, "above": entry(seconds + 60),
         "below": entry(1799), "not-int": entry(2400.5), "bool": entry(True)}[bad]
    view = view_of(e)
    root = sessions_root(tmp_path)
    one_turn(60, "gpt-5.6-sol").write(root)
    clock = codex_read(root, drift=view)
    assert clock.kind == "unknown" and "cannot be accepted" in clock.reason
    one_turn(60, "gpt-5.6-terra").write(root)
    assert codex_read(root, drift=view).kind == "warm", "the 30-minute rows read as before"


def test_t36_an_unreadable_record_holds_the_measured_rows_only(tmp_path):
    path = tmp_path / "cache-drift.json"
    path.write_text("{not json")
    record = drift.load(path)
    view, _k, _r = drift.judge(record, table.load_table())
    root = sessions_root(tmp_path)
    one_turn(60, "gpt-5.6-sol").write(root)
    clock = codex_read(root, drift=view)
    assert clock.kind == "unknown" and clock.reason.startswith(drift.UNREADABLE)
    one_turn(60, "gpt-5.6-terra").write(root)
    assert codex_read(root, drift=view).kind == "warm"
    assert drift.load(tmp_path / "missing.json").entries == []


def test_t36_a_read_writes_nothing(tmp_path):
    clock = drift_read(tmp_path, drifting(measured_seconds() - 1200))
    assert clock.lowering is not None
    assert not drift.record_path().exists(), "a read writes no record"
