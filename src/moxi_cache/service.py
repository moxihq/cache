"""`moxi cache service install|remove|status`: the watcher as a systemd user service.

Linux and systemd only in this release.

`install` writes `moxi-cache-watch.service` into the systemd user unit folder
(`$XDG_CONFIG_HOME/systemd/user`, else `~/.config/systemd/user`, which is where systemd
looks), with `ExecStart=` the `moxi-cache-watch` of the Python environment this `moxi`
runs in (`moxi.paths.script`), never one found on PATH. Then it runs `systemctl --user
daemon-reload` and `enable --now`. It never enables lingering: lingering is a login
policy for the whole account, so `install` prints the `loginctl` command and leaves the
choice with the person.

`remove` disables and stops the unit, removes the file and reloads. `status` reads the
file, the `ExecStart=` it holds, and what `systemctl --user` says about the unit; it
changes nothing.

The service runs with the user manager's environment, not the installing shell's, so a
variable that moves cache's or sessions' files or Claude Code's folder in this shell is
named at install: the service will not see it.

**A stop or restart ends the watcher only.** The unit sets `KillMode=process`: on `stop`,
`restart`, or a restart after a crash, systemd signals the watcher's own process and
leaves the rest of the unit's cgroup alone. The timer's asks run there, each started
detached in a session of its own (`timer.spawn`), and they outlive the watcher, as a
detached start means. systemd's default, `control-group`, would kill every running ask
with the watcher. At the next start systemd logs any ask still running as a left-over
process, which is harmless.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from moxi.paths import Layout, script

ACTIONS = ("install", "remove", "status")

#: The unit's file name.
UNIT = "moxi-cache-watch.service"
#: The console script the unit runs.
SCRIPT = "moxi-cache-watch"

#: Exit codes: 0 done, 1 systemctl failed, 2 refused before anything ran.
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2

LINGER = (
    "The watcher now starts when you log in and stops when your last session ends. To "
    "keep it running while you are logged out, enable lingering for your user:\n"
    "    loginctl enable-linger {user}\n"
    "Lingering starts your user services at boot and keeps them after logout. It is a "
    "login policy for your whole account, so cache does not turn it on for you.")


def shell_only() -> tuple[str, ...]:
    """The variables that move what the watcher reads or writes, which a service started
    by the user manager does not see when they are set only in this shell."""
    return (*Layout("cache").locators(), *Layout("sessions").locators(), "CLAUDE_CONFIG_DIR")


def unit_dir(env) -> Path:
    base = env.get("XDG_CONFIG_HOME") or os.path.join(
        env.get("HOME") or os.path.expanduser("~"), ".config")
    return Path(base) / "systemd" / "user"


def unit_path(env) -> Path:
    return unit_dir(env) / UNIT


def render_unit(watcher: Path) -> str:
    return (
        "[Unit]\n"
        "Description=Moxi cache: the watcher that keeps clock records of live Claude "
        "sessions\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"ExecStart={watcher}\n"
        "Restart=on-failure\n"
        # A stop or restart ends the watcher only, never the asks it started (module doc).
        "KillMode=process\n"
        "RestartSec=5\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n")


class Service:
    """One run of the verb. `run` has `subprocess.run`'s shape and `script` the shape of
    `moxi.paths.script`; both can be replaced for tests."""

    def __init__(self, env=None, say=print, run=None, platform=None, find_script=None):
        self.env = os.environ if env is None else env
        self.say = say
        self.run = run or subprocess.run
        self.platform = platform or sys.platform
        self.find_script = find_script or script

    def systemctl(self, *args) -> subprocess.CompletedProcess:
        binary = shutil.which("systemctl", path=self.env.get("PATH"))
        return self.run([binary, "--user", *args], capture_output=True, text=True,
                        env=dict(self.env))

    def _refuse_unless_systemd(self) -> int | None:
        if not self.platform.startswith("linux"):
            self.say(f"moxi cache service: this is {self.platform}. The service is a "
                     "systemd user unit on Linux; run `moxi cache watch` in the foreground "
                     "here.")
            return EXIT_REFUSED
        if shutil.which("systemctl", path=self.env.get("PATH")) is None:
            self.say("moxi cache service: no systemctl on PATH, so there is no systemd user "
                     "manager to run the watcher. Run `moxi cache watch` in the foreground.")
            return EXIT_REFUSED
        return None

    def _failed(self, step: str, proc) -> int:
        detail = (proc.stderr or proc.stdout or "").strip()
        self.say(f"moxi cache service: systemctl --user {step} failed "
                 f"(exit {proc.returncode})" + (f": {detail}" if detail else ""))
        return EXIT_FAILED

    def install(self) -> int:
        refused = self._refuse_unless_systemd()
        if refused is not None:
            return refused
        try:
            watcher = self.find_script(SCRIPT)
        except FileNotFoundError as e:
            self.say(f"moxi cache service: {e}. Reinstall cache in this environment "
                     "(moxi install cache).")
            return EXIT_REFUSED
        path = unit_path(self.env)
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        path.write_text(render_unit(Path(watcher)))
        os.chmod(path, 0o644)
        self.say(f"{'rewrote' if existed else 'wrote'} {path}")
        self.say(f"  ExecStart={watcher}")
        for step in (("daemon-reload",), ("enable", "--now", UNIT)):
            proc = self.systemctl(*step)
            if proc.returncode != 0:
                return self._failed(" ".join(step), proc)
            self.say(f"systemctl --user {' '.join(step)}: done")
        moved = [v for v in shell_only() if self.env.get(v)]
        if moved:
            self.say("NOTE: set in this shell and not seen by the service, which runs with "
                     f"your user manager's environment: {', '.join(moved)}. Put them in the "
                     f"unit with `systemctl --user edit {UNIT}` if the watcher needs them.")
        user = self.env.get("USER") or self.env.get("LOGNAME") or "$USER"
        self.say(LINGER.format(user=user))
        return EXIT_OK

    def remove(self) -> int:
        refused = self._refuse_unless_systemd()
        if refused is not None:
            return refused
        path = unit_path(self.env)
        if not path.exists():
            self.say(f"moxi cache service: not installed ({path} does not exist)")
            return EXIT_OK
        proc = self.systemctl("disable", "--now", UNIT)
        if proc.returncode != 0:
            return self._failed(f"disable --now {UNIT}", proc)
        self.say(f"systemctl --user disable --now {UNIT}: done")
        path.unlink()
        self.say(f"removed {path}")
        proc = self.systemctl("daemon-reload")
        if proc.returncode != 0:
            return self._failed("daemon-reload", proc)
        self.say("systemctl --user daemon-reload: done")
        return EXIT_OK

    def status(self) -> int:
        """Read-only. Exit 0 whatever it finds; the lines say what that is."""
        path = unit_path(self.env)
        if not path.exists():
            self.say(f"unit        not installed ({path} does not exist)")
        else:
            exec_start = next((line.partition("=")[2] for line in
                               path.read_text().splitlines()
                               if line.startswith("ExecStart=")), "(none)")
            self.say(f"unit        {path}")
            self.say(f"  exec      {exec_start}")
        if self._refuse_unless_systemd() is not None:
            return EXIT_OK
        for question in ("is-enabled", "is-active"):
            proc = self.systemctl(question, UNIT)
            answer = (proc.stdout or proc.stderr or "").strip() or f"exit {proc.returncode}"
            self.say(f"{question:<10}  {answer}")
        return EXIT_OK


def run(action: str, *, env=None, say=print, run=None, platform=None,
        find_script=None) -> int:
    svc = Service(env=env, say=say, run=run, platform=platform, find_script=find_script)
    return getattr(svc, action)()


__all__ = ["ACTIONS", "UNIT", "SCRIPT", "Service", "run", "render_unit", "unit_path",
           "unit_dir", "shell_only", "LINGER"]
