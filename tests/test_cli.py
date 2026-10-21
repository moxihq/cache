"""`moxi cache read`, `moxi cache doctor`, and the verbs' wiring (`segment`'s render is in
test_render, the watcher in test_watch, the service in test_service)."""

from __future__ import annotations

import json

import pytest
from conftest import assistant_line, write_cache_config, write_transcript

from moxi import cli as umbrella
from moxi_cache import cli, drift

SID = "aaaaaaaa-1111-4111-8111-111111111111"
SID_B = "bbbbbbbb-2222-4222-8222-222222222222"


def run(capsys, *argv):
    code = cli.main(list(argv), prog="moxi cache")
    out = capsys.readouterr()
    return code, out.out, out.err


def test_read_prints_each_sessions_kind_age_and_lifetime(tmp_path, capsys):
    write_transcript(SID, root=tmp_path, lines=[assistant_line(125, tier="5m")])
    code, out, _ = run(capsys, "read", SID, SID_B, "--source", str(tmp_path))
    assert code == 0
    first, second = out.splitlines()
    assert first.startswith(f"{SID}: warm, last request 2m")
    assert first.endswith("lifetime 5m0s, from the tier")
    assert second.startswith(f"{SID_B}: unknown [no-record]: no transcript for session")


def test_read_json(tmp_path, capsys):
    write_transcript(SID, root=tmp_path, lines=[assistant_line(4000, tier="1h")])
    code, out, _ = run(capsys, "read", SID, "--source", str(tmp_path), "--json")
    doc = json.loads(out)
    assert code == 0 and doc["harness"] == "claude"
    (one,) = doc["sessions"]
    assert (one["session"], one["kind"], one["ttl_seconds"], one["ttl_source"]) == (
        SID, "cold", 3600, "tier")


def test_read_uses_sessions_root_by_default(tmp_path, capsys):
    write_transcript(SID, 30, root=tmp_path / "claude-config" / "projects")
    code, out, _ = run(capsys, "read", SID)
    assert code == 0 and ": warm," in out


def test_read_a_codex_thread_needs_its_source(tmp_path, capsys):
    code, out, _ = run(capsys, "read", "0190c000-0000-7000-8000-00000000000a",
                       "--harness", "codex")
    assert code == 0 and "[no-source]" in out
    code, out, _ = run(capsys, "read", "0190c000-0000-7000-8000-00000000000a",
                       "--harness", "codex", "--source", str(tmp_path / "missing"))
    assert "[source-unreadable]" in out


def test_read_writes_nothing(tmp_path, capsys, own_roots):
    write_transcript(SID, 30, root=tmp_path)
    run(capsys, "read", SID, "--source", str(tmp_path))
    assert not drift.record_path().parent.exists()
    assert list(own_roots.iterdir()) == []


def test_read_through_the_umbrella(tmp_path, capsys):
    write_transcript(SID, 30, root=tmp_path)
    assert umbrella.main(["cache", "read", SID, "--source", str(tmp_path)]) == 0
    assert ": warm," in capsys.readouterr().out


def test_help_names_the_umbrella(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"], prog="moxi cache")
    assert "usage: moxi cache" in capsys.readouterr().out


def test_doctor_on_a_clean_box(capsys):
    code, out, _ = run(capsys, "doctor")
    assert code == 0
    assert "lifetime table: 5 shipped rows, 0 of yours" in out
    assert "absent: every key at its default" in out
    assert "drift record:" in out and "absent: no lowering" in out
    assert "learner: no process has learned on this device" in out
    assert "no watcher is running" in out
    assert "service:" in out and "not installed: moxi cache service install" in out


def test_doctor_names_a_learner_and_the_lowerings(capsys):
    store = drift.open_store(learn=True)
    try:
        code, out, _ = run(capsys, "doctor")
    finally:
        store.close()
    assert code == 0 and "a process holds the learning lock" in out
    assert "absent: no lowering" in out


@pytest.mark.parametrize("damage", ["config", "table-row", "drift"])
def test_doctor_exits_1_on_what_stops_reads(capsys, own_roots, damage):
    if damage == "config":
        write_cache_config(own_roots, "[claude\n")
    elif damage == "table-row":
        write_cache_config(own_roots, '[[lifetime]]\nharness = "claude"\n')
    else:
        path = drift.record_path()
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
    code, out, _ = run(capsys, "doctor")
    assert code == 1
    expected = {"config": "config: cannot be read",
                "table-row": "lifetime table: does not load",
                "drift": "the cache drift record could not be read"}[damage]
    assert expected in out


def _record(seen_at: float, session: str = SID, **thresholds) -> None:
    from moxi_cache import records

    directory = records.record_dir()
    directory.mkdir(parents=True, exist_ok=True)
    rec = {"v": 1, "session": session, "harness": "claude", "pid": 1, "kind": "warm",
           "last_request": seen_at - 60, "ttl": 3600, "read_at": seen_at, "seen_at": seen_at,
           "code": None, "warn_before": 900, "stale_after": 30}
    rec.update(thresholds)
    for name, value in thresholds.items():
        if value is None:
            del rec[name]
    (directory / f"{session}.json").write_text(json.dumps(rec))


def test_doctor_reads_how_the_watchers_records_stand(capsys):
    import time

    _record(time.time() - 2)
    code, out, _ = run(capsys, "doctor")
    assert code == 0 and "1 clock record(s) under" in out and "has stopped" not in out
    _record(time.time() - 600)
    code, out, _ = run(capsys, "doctor")
    assert "the watcher has stopped" in out


def test_doctor_judges_each_record_by_its_own_stale_after(own_roots, capsys):
    import time

    # the config's tick would call a 60 s old record fresh (or stale); the record decides
    write_cache_config(own_roots, "[watch]\ntick = 100\n")
    _record(time.time() - 60, stale_after=45)
    assert "the watcher has stopped" in run(capsys, "doctor")[1]
    write_cache_config(own_roots, "[watch]\ntick = 1\n")
    _record(time.time() - 60, stale_after=300)
    assert "has stopped" not in run(capsys, "doctor")[1]


def test_doctor_names_records_that_are_stale_beside_fresh_ones(capsys):
    import time

    _record(time.time() - 60, stale_after=30)
    _record(time.time() - 60, SID_B, stale_after=300)
    out = run(capsys, "doctor")[1]
    assert "1 of them not rewritten within their own stale_after" in out
    assert "has stopped" not in out


def test_doctor_names_records_an_older_watcher_wrote(capsys):
    import time

    _record(time.time() - 2, warn_before=None, stale_after=None)
    out = run(capsys, "doctor")[1]
    assert "1 clock record(s) under" in out and "has stopped" not in out
    assert "1 lack warn_before or stale_after (an older watcher wrote them)" in out
    _record(time.time() - 600, SID_B, stale_after=30)
    out = run(capsys, "doctor")[1]
    assert "the watcher has stopped" in out and "1 lack warn_before" in out


def test_service_and_watch_are_wired(monkeypatch, capsys):
    from moxi_cache import service, watch

    seen = []
    monkeypatch.setattr(service, "run", lambda action: seen.append(action) or 0)
    monkeypatch.setattr(watch, "main", lambda argv: seen.append(("watch", argv)) or 0)
    assert run(capsys, "service", "status")[0] == 0
    assert run(capsys, "watch")[0] == 0
    assert seen == ["status", ("watch", [])]
    with pytest.raises(SystemExit):
        cli.main(["service", "start"], prog="moxi cache")
