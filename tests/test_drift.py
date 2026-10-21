"""The drift store: one process learns into `STATE/cache/drift.json`, every other reads.

A learning store holds `drift.lock`, writes a lowering by atomic replace before the read
that found it returns, and keeps the judged turn per thread; a lowering it cannot write
reads `unknown` until a write succeeds. A read-only store creates and writes nothing.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

import pytest
from test_codex_clock import (TID, drifting, measured_seconds, one_turn,
                              sessions_root)

from moxi_cache import api, codex, drift, table
from moxi_sessions.api import claude as transcripts


def write_rollout(tmp_path, rollout):
    root = sessions_root(tmp_path)
    rollout.write(root)
    stamp = [ln for ln in rollout.lines if ln["type"] == "token_usage_record"][-1]
    end = transcripts.parse_timestamp(stamp["timestamp"])
    return root, end + 60


def read(root, now, store, **kw):
    return api.read("codex", TID, cache_source=str(root), now=now, bound_at=0.0,
                    store=store, **kw)


@pytest.fixture
def learner():
    store = api.open_store(learn=True)
    yield store
    store.close()


def test_a_learning_store_holds_the_lock_and_a_second_one_reads(learner, caplog):
    assert learner.learning
    assert drift.lock_path().is_file()
    assert drift.learner_holds_lock() is True
    with caplog.at_level(logging.WARNING, logger="moxi.cache"):
        second = api.open_store(learn=True)
    assert not second.learning
    assert sum("another process is learning" in r.message for r in caplog.records) == 1
    learner.close()
    assert drift.learner_holds_lock() is False
    third = api.open_store(learn=True)
    try:
        assert third.learning, "the lock is free once the learner closes"
    finally:
        third.close()


def test_a_second_process_asking_to_learn_gets_a_read_only_store(learner):
    code = ("from moxi_cache import api\n"
            "s = api.open_store(learn=True)\n"
            "print('learning' if s.learning else 'reading')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(os.environ), timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "reading"
    assert "another process is learning" in out.stderr


def test_a_learning_read_writes_the_lowering_before_it_returns(tmp_path, learner):
    seconds = measured_seconds()
    gap = seconds - 1200
    root, now = write_rollout(tmp_path, drifting(gap))
    clock = read(root, now, learner)
    lowered = codex.lower_to(gap, 1800)
    assert clock.lowering["lifetime"] == lowered and clock.ttl_seconds == lowered
    doc = json.loads(drift.record_path().read_text())
    assert [e["lifetime"] for e in doc["entries"]] == [lowered]
    entry = doc["entries"][0]
    assert (entry["harness"], entry["provider"], entry["model"]) == ("codex", "openai",
                                                                     "gpt-5.6-sol")
    row = table.load_table().lookup("codex", "openai", "gpt-5.6-sol")
    assert (entry["row_seconds"], entry["row_checked"]) == (row.seconds, row.checked)
    assert oct(drift.record_path().stat().st_mode & 0o777) == "0o600"


def test_the_judged_turn_lives_in_the_store(tmp_path, learner):
    root, now = write_rollout(tmp_path, drifting(measured_seconds() - 1200))
    first = read(root, now, learner)
    assert first.lowering is not None and first.judged == (TID, "t2")
    assert learner.judged(TID) == "t2"
    again = read(root, now, learner)
    assert again.lowering is None, "a turn already judged is not judged again"
    assert again.ttl_seconds == first.ttl_seconds, "the written lowering is in force"


def test_a_read_only_store_writes_nothing_and_creates_nothing(tmp_path, own_roots):
    store = api.open_store(learn=False)
    assert not store.learning
    root, now = write_rollout(tmp_path, drifting(measured_seconds() - 1200))
    clock = read(root, now, store)
    assert clock.lowering is not None, "the read observes the lowering ..."
    assert clock.ttl_seconds == clock.lowering["lifetime"], "... and counts it this once"
    assert not drift.record_path().parent.exists(), "no directory is made"
    assert drift.learner_holds_lock() is None


def test_a_read_only_store_reads_what_the_learner_wrote(tmp_path, learner):
    reader = api.open_store(learn=False)
    root, now = write_rollout(tmp_path, drifting(measured_seconds() - 1200))
    first = read(root, now, learner)
    lowered = first.lowering["lifetime"]
    late = tmp_path / "late"
    root2, _ = write_rollout(late, one_turn(60, "gpt-5.6-sol"))
    clock = api.read("codex", TID, cache_source=str(root2),
                     now=1_791_300_000.0, store=reader)
    assert clock.ttl_seconds == lowered
    assert "lowered from" in clock.key


def test_a_lowering_that_cannot_be_written_reads_unknown_until_one_is(tmp_path, learner):
    state = drift.record_path().parent
    root, now = write_rollout(tmp_path, drifting(measured_seconds() - 1200))
    state.chmod(0o500)
    try:
        clock = read(root, now, learner)
        assert (clock.kind, clock.code) == ("unknown", "drift-held")
        assert clock.reason == drift.UNWRITTEN
        held = api.read("codex", TID, cache_source=str(root), now=now, store=learner)
        assert held.code == "drift-held", "the model is held while the write fails"
    finally:
        state.chmod(0o700)
    retried = api.read("codex", TID, cache_source=str(root), now=now, store=learner)
    assert retried.kind in ("warm", "cold") and retried.code is None
    assert drift.record_path().is_file(), "the next read retried the write"


def test_a_record_with_an_entry_that_cannot_be_accepted_holds_measured_reads(tmp_path):
    path = drift.record_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"entries": [{"what": "is this"}]}))
    store = api.open_store(learn=False)
    root, now = write_rollout(tmp_path, one_turn(60, "gpt-5.6-sol"))
    clock = api.read("codex", TID, cache_source=str(root), now=now, store=store)
    assert (clock.kind, clock.code) == ("unknown", "drift-held")
    assert "cannot be accepted" in clock.reason
    root2, _ = write_rollout(tmp_path / "terra", one_turn(60, "gpt-5.6-terra"))
    assert api.read("codex", TID, cache_source=str(root2), now=now,
                    store=store).kind == "warm", "the documented rows read as before"


def test_an_unreadable_record_holds_measured_reads(tmp_path):
    path = drift.record_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    store = api.open_store(learn=False)
    root, now = write_rollout(tmp_path, one_turn(60, "gpt-5.6-sol"))
    clock = api.read("codex", TID, cache_source=str(root), now=now, store=store)
    assert (clock.kind, clock.code) == ("unknown", "drift-held")
    assert clock.reason.startswith(drift.UNREADABLE)


def test_no_store_reads_a_measured_row_unknown(tmp_path):
    root, now = write_rollout(tmp_path, one_turn(60, "gpt-5.6-sol"))
    clock = api.read("codex", TID, cache_source=str(root), now=now)
    assert (clock.kind, clock.code) == ("unknown", "no-drift-record")
    with api.open_store(learn=False) as store:
        assert api.read("codex", TID, cache_source=str(root), now=now,
                        store=store).kind == "warm"


def test_the_store_lives_under_cache_state(own_roots):
    path = drift.record_path()
    assert path.name == "drift.json" and drift.lock_path().name == "drift.lock"
    assert str(path).startswith(str(own_roots / "state"))
    assert path.parent.name == "cache"
