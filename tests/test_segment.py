"""The status line segment and the clock record format it reads."""

from __future__ import annotations

import json

import pytest
from conftest import write_cache_config

from moxi_cache import api, records, segment

SID = "aaaaaaaa-1111-4111-8111-111111111111"
NOW = 1_800_000_000.0


def record(*, kind="warm", age=60.0, ttl=3600, seen_ago=0.0, session=SID, **extra) -> dict:
    out = {"v": 1, "session": session, "harness": "claude", "pid": 4242, "kind": kind,
           "last_request": None if age is None else NOW - age, "ttl": ttl,
           "read_at": NOW - 5, "seen_at": NOW - seen_ago, "code": None,
           "warn_before": 900, "stale_after": 30}
    out.update(extra)
    return out


def put(rec: dict) -> None:
    directory = records.record_dir()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{rec['session']}.json").write_text(json.dumps(rec))


def seg(rec, *, now=NOW):
    return segment.from_record(rec, now)


# --- the record format -----------------------------------------------------------------

def test_a_record_holds_every_field_in_order():
    assert records.FIELDS == ("v", "session", "harness", "pid", "kind", "last_request",
                              "ttl", "read_at", "seen_at", "code", "warn_before",
                              "stale_after")
    assert records.valid(record())
    assert records.valid(record(kind="fresh", age=None, ttl=None))
    assert records.thresholds(record()) == (900, 30)


@pytest.mark.parametrize("missing", ["warn_before", "stale_after"])
def test_a_record_from_an_older_watcher_is_a_record_without_thresholds(missing):
    rec = record()
    del rec[missing]
    assert records.valid(rec)
    assert records.thresholds(rec) is None


@pytest.mark.parametrize("change", [
    {"warn_before": "900"}, {"stale_after": None}, {"stale_after": True},
    {"warn_before": float("inf")}, {"stale_after": float("nan")},
])
def test_thresholds_that_are_not_finite_numbers_are_none(change):
    assert records.thresholds(record(**change)) is None


@pytest.mark.parametrize("change", [
    {"v": 2}, {"pid": "4242"}, {"pid": True}, {"last_request": "soon"}, {"ttl": "1h"},
    {"seen_at": None}, {"read_at": None}, {"code": 3}, {"session": None},
])
def test_a_record_out_of_format_is_not_valid(change):
    assert not records.valid(record(**change))


def test_a_record_missing_a_field_is_not_valid():
    rec = record()
    del rec["code"]
    assert not records.valid(rec)


def test_the_records_live_under_the_runtime_directory(own_roots):
    assert records.record_dir() == own_roots / "runtime" / "cache" / "clock"
    assert records.record_path(SID) == own_roots / "runtime" / "cache" / "clock" / f"{SID}.json"


@pytest.mark.parametrize("bad", ["", "../x", "a/b", ".hidden", None, 7, "x" * 200])
def test_an_id_that_cannot_name_a_file_has_no_record(bad):
    assert records.record_path(bad) is None


def test_load_refuses_what_is_not_a_record(tmp_path):
    path = tmp_path / "r.json"
    assert records.load(path) is None
    path.write_text("{not json")
    assert records.load(path) is None
    path.write_text(json.dumps([1, 2]))
    assert records.load(path) is None
    path.write_text(json.dumps(record()))
    assert records.load(path) == record()


# --- bands, text and dimming -------------------------------------------------------------

@pytest.mark.parametrize("age, band, text", [
    (0, "warm", "cache:0m0s"),
    (45, "warm", "cache:0m45s"),
    (723, "warm", "cache:12m3s"),
    (2699, "warm", "cache:44m59s"),
    (2700, "near", "cache:45m0s"),
    (3599, "near", "cache:59m59s"),
    (3600, "cold", "cache:1h0m"),
    (3840, "cold", "cache:1h4m"),
    (93_900, "cold", "cache:26h5m"),
])
def test_bands_and_text_on_the_hour_tier(age, band, text):
    found = seg(record(age=age, kind="cold" if age >= 3600 else "warm"))
    assert (found.band, found.text) == (band, text)
    assert found.age_s == age and found.ttl_s == 3600


def test_the_age_counts_on_from_the_record_past_its_kind():
    found = seg(record(kind="warm", age=4000))
    assert found.band == "cold" and found.text == "cache:1h6m"


def test_a_five_minute_lifetime_is_near_from_the_start():
    assert seg(record(age=10, ttl=300)).band == "near"
    assert seg(record(age=300, ttl=300)).band == "cold"


def test_the_records_own_warn_before_moves_near():
    assert seg(record(age=3000, warn_before=300)).band == "warm"
    assert seg(record(age=3300, warn_before=300)).band == "near"
    assert seg(record(age=599, warn_before=3000)).band == "warm"
    assert seg(record(age=600, warn_before=3000)).band == "near"


def test_a_future_last_request_counts_as_age_zero():
    assert seg(record(age=-30)).text == "cache:0m0s"


def test_dim_past_the_records_own_stale_after():
    assert not seg(record(seen_ago=30)).dim
    assert seg(record(seen_ago=30.5)).dim
    assert seg(record(seen_ago=7, stale_after=6)).dim
    assert not seg(record(seen_ago=299, stale_after=300)).dim
    assert seg(record(seen_ago=301, stale_after=300)).dim


@pytest.mark.parametrize("missing", ["warn_before", "stale_after"])
def test_no_segment_for_a_record_lacking_a_threshold(missing):
    rec = record()
    del rec[missing]
    assert seg(rec) is None


def test_an_idle_session_stays_bright_while_its_age_counts():
    # read long ago, seen at this tick: bright, and the age is from the last request
    found = seg(record(age=1800, seen_ago=1, read_at=NOW - 1800))
    assert not found.dim and found.text == "cache:30m0s"


@pytest.mark.parametrize("kind", ["fresh", "unknown", "no-cache"])
def test_no_segment_for_a_kind_with_no_age(kind):
    assert seg(record(kind=kind)) is None


@pytest.mark.parametrize("change", [{"age": None}, {"ttl": None}])
def test_no_segment_without_a_last_request_or_a_lifetime(change):
    assert seg(record(**change)) is None


def test_no_segment_for_a_record_out_of_format():
    assert seg(record(v=2)) is None
    assert seg({"kind": "warm"}) is None


def test_coloured_text():
    assert seg(record(age=60)).coloured() == "\033[32mcache:1m0s\033[0m"
    assert seg(record(age=2800)).coloured() == "\033[33mcache:46m40s\033[0m"
    assert seg(record(age=3700, kind="cold")).coloured() == "\033[31mcache:1h1m\033[0m"
    assert seg(record(age=60, seen_ago=100)).coloured() == "\033[2m\033[32mcache:1m0s\033[0m"


def test_as_dict():
    assert seg(record(age=60)).as_dict() == {"text": "cache:1m0s", "band": "warm",
                                             "age_s": 60.0, "ttl_s": 3600.0, "dim": False}


# --- segment(): one record, read from disk ------------------------------------------------

def test_segment_reads_the_sessions_record():
    put(record(age=125))
    found = segment.segment(SID, now=NOW)
    assert found.text == "cache:2m5s" and found.band == "warm"


def test_segment_is_none_with_no_record():
    assert segment.segment(SID, now=NOW) is None


def test_segment_is_none_for_a_record_naming_another_session():
    other = "bbbbbbbb-2222-4222-8222-222222222222"
    rec = record(session=other)
    directory = records.record_dir()
    directory.mkdir(parents=True)
    (directory / f"{SID}.json").write_text(json.dumps(rec))
    assert segment.segment(SID, now=NOW) is None


def test_segment_reads_the_records_thresholds_and_no_config(own_roots):
    # the config says otherwise; the record's own values decide the band and the dim
    write_cache_config(own_roots, "[segment]\nwarn_before = 300\n[watch]\ntick = 20\n")
    put(record(age=3000, seen_ago=50, warn_before=3599, stale_after=300))
    found = segment.segment(SID, now=NOW)
    assert found.band == "near" and not found.dim
    put(record(age=3000, seen_ago=50, warn_before=60, stale_after=45))
    found = segment.segment(SID, now=NOW)
    assert found.band == "warm" and found.dim


def test_segment_is_none_for_a_record_an_older_watcher_wrote(own_roots):
    rec = record(age=125)
    del rec["warn_before"], rec["stale_after"]
    put(rec)
    assert segment.segment(SID, now=NOW) is None


def test_a_config_in_error_does_not_touch_the_segment(own_roots):
    put(record(age=3000))
    write_cache_config(own_roots, "[segment\n")
    assert segment.segment(SID, now=NOW).band == "near"


def test_the_segment_module_reads_no_config():
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(segment))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {f"{n.module or ''}:{a.name}" for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not [m for m in imported if "settings" in m or "config" in m], imported


def test_api_re_exports_the_segment():
    assert api.segment is segment.segment
    assert api.Segment is segment.Segment
    with pytest.raises(AttributeError):
        api.no_such_name  # noqa: B018
