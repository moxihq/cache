"""The render audit hook and `moxi cache segment`. An audit hook cannot be removed, so every
test that installs one runs in a child process."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import moxi_cache
from moxi_cache import records, render

SID = "aaaaaaaa-1111-4111-8111-111111111111"


def child(code: str, *args, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code), *args],
                          capture_output=True, text=True, timeout=60, cwd=cwd,
                          env=dict(os.environ))


def put_record(age=125.0, seen_ago=0.0, kind="warm", **thresholds):
    now = time.time()
    directory = records.record_dir()
    directory.mkdir(parents=True, exist_ok=True)
    rec = {"v": 1, "session": SID, "harness": "claude", "pid": 1, "kind": kind,
           "last_request": now - age, "ttl": 3600, "read_at": now, "seen_at": now - seen_ago,
           "code": None, "warn_before": 900, "stale_after": 30}
    rec.update(thresholds)
    for name, value in thresholds.items():
        if value is None:
            del rec[name]
    (directory / f"{SID}.json").write_text(json.dumps(rec))


def test_refused_is_a_permission_error():
    assert issubclass(render.Refused, PermissionError)
    assert issubclass(render.Refused, OSError)


def test_the_hook_refuses_effects_and_passes_reads(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "r.txt").write_text("read me")
    probe = r'''
        import os, socket, sqlite3, subprocess, sys, pathlib, tempfile
        from moxi_cache import render
        render.install()
        assert sys.dont_write_bytecode
        d = pathlib.Path(sys.argv[1])

        def t(name, f):
            try:
                f()
                print(name, "allowed")
            except render.Refused:
                print(name, "refused")

        t("read", lambda: open(d / "r.txt").read())
        t("read bytes", lambda: (d / "r.txt").read_bytes())
        t("os.open read", lambda: os.close(os.open(d / "r.txt", os.O_RDONLY)))
        t("stat", lambda: os.stat(d / "r.txt"))
        t("listdir", lambda: os.listdir(d))
        t("open w", lambda: open(d / "w.txt", "w"))
        t("open a", lambda: open(d / "r.txt", "a"))
        t("open r+", lambda: open(d / "r.txt", "r+"))
        t("os.open creat", lambda: os.open(d / "c.txt", os.O_CREAT | os.O_WRONLY))
        t("touch", lambda: (d / "t.txt").touch())
        t("replace", lambda: os.replace(d / "r.txt", d / "r2.txt"))
        t("unlink", lambda: os.unlink(d / "r.txt"))
        t("mkdir", lambda: os.mkdir(d / "sub"))
        t("chmod", lambda: os.chmod(d / "r.txt", 0o600))
        t("utime", lambda: os.utime(d / "r.txt"))
        t("symlink", lambda: os.symlink(d / "r.txt", d / "l"))
        t("mkstemp", lambda: tempfile.mkstemp(dir=d))
        t("subprocess", lambda: subprocess.run(["true"]))
        t("system", lambda: os.system("true"))
        t("kill", lambda: os.kill(os.getpid(), 0))
        t("connect", lambda: socket.socket(socket.AF_UNIX).connect(str(d / "sock")))
        t("bind", lambda: socket.socket(socket.AF_UNIX).bind(str(d / "sock")))
        t("sqlite rw", lambda: sqlite3.connect(str(d / "x.db")))
        t("sqlite memory", lambda: sqlite3.connect(":memory:"))
        t("sqlite uri rw", lambda: sqlite3.connect(f"file:{d}/x.db?mode=rwc", uri=True))
        t("sqlite ro", lambda: sqlite3.connect(f"file:{d}/r.txt?mode=ro", uri=True).close())
    '''
    proc = child(probe, str(work))
    assert proc.returncode == 0, proc.stderr
    said = dict(line.rsplit(" ", 1) for line in proc.stdout.splitlines())
    allowed = {"read", "read bytes", "os.open read", "stat", "listdir", "sqlite ro"}
    assert {k for k, v in said.items() if v == "allowed"} == allowed
    assert len(said) == 26
    assert sorted(p.name for p in work.iterdir()) == ["r.txt"]


def test_an_import_with_no_bytecode_goes_on_under_the_hook(tmp_path):
    """The hook alone, with bytecode writes left on: the import tries to write each
    module's bytecode, the hook refuses with a PermissionError, and the import goes on
    without it, as it does in a read-only directory."""
    pkg = tmp_path / "pkg"
    shutil.copytree(Path(moxi_cache.__file__).parent, pkg / "moxi_cache",
                    ignore=shutil.ignore_patterns("__pycache__"))
    put_record(age=2900)
    probe = r'''
        import sys
        sys.path.insert(0, sys.argv[1])
        import moxi_cache.render as render
        assert render.__file__.startswith(sys.argv[1])
        refused = []

        def hook(event, args):
            try:
                render._hook(event, args)
            except render.Refused:
                refused.append(event)
                raise
        sys.addaudithook(hook)
        assert not sys.dont_write_bytecode
        import moxi_cache.segment as segment
        found = segment.segment(sys.argv[2])
        print(found.text, found.band, len(refused))
    '''
    proc = child(probe, str(pkg), SID)
    assert proc.returncode == 0, proc.stderr
    text, band, refusals = proc.stdout.split()
    assert (text[:9], band) == ("cache:48m", "near") and int(refusals) >= 1
    assert not list((pkg / "moxi_cache").glob("__pycache__/segment.*"))


def test_a_hook_raising_another_error_breaks_that_import(tmp_path):
    """Why a refusal is a PermissionError: the import machinery skips a bytecode write
    that fails with an OSError and no other."""
    pkg = tmp_path / "pkg"
    shutil.copytree(Path(moxi_cache.__file__).parent, pkg / "moxi_cache",
                    ignore=shutil.ignore_patterns("__pycache__"))
    probe = r'''
        import sys
        sys.path.insert(0, sys.argv[1])
        import moxi_cache.render as render

        def hook(event, args):
            try:
                render._hook(event, args)
            except render.Refused as e:
                raise RuntimeError(str(e)) from None
        sys.addaudithook(hook)
        try:
            import moxi_cache.segment
        except RuntimeError:
            print("import failed")
    '''
    proc = child(probe, str(pkg))
    assert proc.stdout.strip() == "import failed", proc.stderr


def test_loading_the_api_and_the_cli_does_not_load_the_segment():
    proc = child('''
        import sys
        import moxi_cache.api, moxi_cache.cli
        from moxi import modules
        assert modules.load("cache").state == "present"
        print("moxi_cache.segment" in sys.modules)
    ''')
    assert proc.stdout.strip() == "False", proc.stderr


def run_segment(*args) -> subprocess.CompletedProcess:
    code = '''
        import sys
        from moxi import cli
        sys.exit(cli.main(["cache", "segment", *sys.argv[1:]]))
    '''
    return child(code, *args)


def test_moxi_cache_segment_prints_the_text():
    put_record(age=125)
    proc = run_segment(SID)
    assert proc.returncode == 0 and proc.stdout == "cache:2m5s\n", proc.stderr


def test_moxi_cache_segment_colour_and_dim():
    put_record(age=3700, kind="cold", seen_ago=100)
    proc = run_segment(SID, "--color")
    assert proc.stdout == "\033[2m\033[31mcache:1h1m\033[0m\n", proc.stderr


def test_moxi_cache_segment_json():
    put_record(age=60)
    doc = json.loads(run_segment(SID, "--json").stdout)
    assert doc["text"] == "cache:1m0s" and doc["band"] == "warm" and doc["dim"] is False
    assert set(doc) == {"text", "band", "age_s", "ttl_s", "dim"}


def test_moxi_cache_segment_bands_and_dims_by_the_records_own_values():
    put_record(age=125, seen_ago=100, warn_before=3500, stale_after=600)
    proc = run_segment(SID, "--color")
    assert proc.stdout == "\033[33mcache:2m5s\033[0m\n", proc.stderr
    put_record(age=3000, seen_ago=10, warn_before=60, stale_after=5)
    proc = run_segment(SID, "--color")
    assert proc.stdout == "\033[2m\033[32mcache:50m0s\033[0m\n", proc.stderr


@pytest.mark.parametrize("missing", ["warn_before", "stale_after"])
def test_moxi_cache_segment_prints_nothing_for_a_record_lacking_a_threshold(missing):
    put_record(age=125, **{missing: None})
    proc = run_segment(SID)
    assert (proc.returncode, proc.stdout, proc.stderr) == (0, "", "")
    assert run_segment(SID, "--json").stdout == "null\n"


def test_moxi_cache_segment_with_no_record_prints_nothing():
    proc = run_segment(SID)
    assert (proc.returncode, proc.stdout, proc.stderr) == (0, "", "")
    assert run_segment(SID, "--json").stdout == "null\n"


def test_moxi_cache_segment_refuses_a_segment_module_that_writes(tmp_path):
    (tmp_path / "evil_segment.py").write_text(textwrap.dedent(f'''
        open({str(tmp_path / "written")!r}, "w").write("x")

        def segment(session_id, now=None):
            return None
    '''))
    proc = child('''
        import sys
        sys.path.insert(0, sys.argv[1])
        import moxi_cache.module
        moxi_cache.module.SEGMENT = "evil_segment"
        from moxi_cache import cli
        sys.exit(cli.main(["segment", "x"], prog="moxi cache"))
    ''', str(tmp_path))
    assert proc.returncode == 1
    assert "the render was refused" in proc.stderr and "for writing" in proc.stderr
    assert not (tmp_path / "written").exists()


def test_the_announcement_names_the_segment_module():
    from moxi import modules

    assert modules.load("cache").segment == "moxi_cache.segment"


@pytest.mark.parametrize("args", [["--color", "--json"]])
def test_colour_and_json_are_one_or_the_other(args):
    proc = run_segment(SID, *args)
    assert proc.returncode == 2
