"""`moxi cache service`: the watcher's systemd user unit. systemctl is never run."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from moxi_cache import service


class Systemctl:
    """Records each systemctl call and answers with `rc`."""

    def __init__(self, rc=0, out=""):
        self.calls: list[list[str]] = []
        self.rc, self.out = rc, out

    def __call__(self, argv, **kw):
        self.calls.append(argv[1:])
        return subprocess.CompletedProcess(argv, self.rc, self.out, "it broke" if self.rc else "")


@pytest.fixture
def env(tmp_path) -> dict:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    systemctl = bin_dir / "systemctl"
    systemctl.write_text("#!/bin/sh\nexit 0\n")
    systemctl.chmod(0o755)
    return {"PATH": str(bin_dir), "HOME": str(tmp_path / "home"),
            "XDG_CONFIG_HOME": str(tmp_path / "xdg"), "USER": "someone"}


WATCHER = Path("/opt/env/bin/moxi-cache-watch")


def run(action, env, systemctl, *, found=WATCHER, platform="linux"):
    said: list[str] = []

    def find(name):
        assert name == "moxi-cache-watch"
        if found is None:
            raise FileNotFoundError(f"{name} is not installed in this environment")
        return found
    code = service.run(action, env=env, say=said.append, run=systemctl, platform=platform,
                       find_script=find)
    return code, "\n".join(said)


def test_install_writes_the_unit_with_this_environments_watcher(env):
    systemctl = Systemctl()
    code, out = run("install", env, systemctl)
    assert code == 0
    unit = Path(env["XDG_CONFIG_HOME"]) / "systemd" / "user" / "moxi-cache-watch.service"
    text = unit.read_text()
    assert f"ExecStart={WATCHER}\n" in text and "WantedBy=default.target" in text
    assert oct(unit.stat().st_mode & 0o777) == "0o644"
    assert systemctl.calls == [["--user", "daemon-reload"],
                               ["--user", "enable", "--now", "moxi-cache-watch.service"]]
    assert "loginctl enable-linger someone" in out
    assert "NOTE" not in out


def test_the_unit_ends_the_watcher_only_on_a_stop():
    """A stop or restart ends the watcher's process, not the timer's asks in its cgroup
    (B11 verification [L1]): the unit sets KillMode=process, in [Service]."""
    text = service.render_unit(WATCHER)
    lines = text.splitlines()
    assert lines.count("KillMode=process") == 1
    assert lines.index("[Service]") < lines.index("KillMode=process") \
        < lines.index("[Install]")
    assert [line for line in lines if line.startswith("KillMode=")] == ["KillMode=process"]


def test_install_names_variables_the_service_will_not_see(env):
    env["MOXI_CACHE_HOME"] = "/elsewhere"
    env["CLAUDE_CONFIG_DIR"] = "/claude"
    _, out = run("install", env, Systemctl())
    assert "NOTE" in out and "MOXI_CACHE_HOME" in out and "CLAUDE_CONFIG_DIR" in out


def test_install_refuses_without_the_watcher_script(env):
    code, out = run("install", env, Systemctl(), found=None)
    assert code == service.EXIT_REFUSED and "moxi install cache" in out
    assert not (Path(env["XDG_CONFIG_HOME"]) / "systemd").exists()


def test_install_refuses_off_linux_and_without_systemctl(env):
    assert run("install", env, Systemctl(), platform="darwin")[0] == service.EXIT_REFUSED
    env["PATH"] = "/nonexistent"
    code, out = run("install", env, Systemctl())
    assert code == service.EXIT_REFUSED and "no systemctl" in out


def test_install_reports_a_failed_systemctl(env):
    code, out = run("install", env, Systemctl(rc=1))
    assert code == service.EXIT_FAILED and "daemon-reload failed (exit 1): it broke" in out


def test_remove(env):
    run("install", env, Systemctl())
    systemctl = Systemctl()
    code, out = run("remove", env, systemctl)
    assert code == 0
    assert systemctl.calls == [["--user", "disable", "--now", "moxi-cache-watch.service"],
                               ["--user", "daemon-reload"]]
    assert not service.unit_path(env).exists()
    code, out = run("remove", env, Systemctl())
    assert code == 0 and "not installed" in out


def test_status_reads_and_changes_nothing(env):
    code, out = run("status", env, Systemctl(out="inactive"))
    assert code == 0 and "not installed" in out and "is-active   inactive" in out
    run("install", env, Systemctl())
    systemctl = Systemctl(out="active")
    code, out = run("status", env, systemctl)
    assert f"exec      {WATCHER}" in out
    assert systemctl.calls == [["--user", "is-enabled", "moxi-cache-watch.service"],
                               ["--user", "is-active", "moxi-cache-watch.service"]]


def test_the_unit_folder_without_xdg_config_home(tmp_path):
    assert service.unit_dir({"HOME": "/h"}) == Path("/h/.config/systemd/user")
