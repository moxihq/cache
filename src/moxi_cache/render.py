"""The render audit hook: a process that renders a status line segment shows status and
can do nothing else.

`install()` sets `sys.dont_write_bytecode` and adds an audit hook (`sys.addaudithook`)
that refuses, for the rest of the process:

* an open for writing: a mode with `w`, `a`, `x` or `+`, or flags holding `O_WRONLY`,
  `O_RDWR`, `O_CREAT`, `O_APPEND` or `O_TRUNC`;
* a rename or replace, a removal, a directory made, a link, and a change of a file's
  size, mode, owner, times or attributes;
* a process started or forked, and a signal sent;
* a socket connected, bound or sent on;
* a C library loaded through `ctypes`;
* `sqlite3.connect`, unless its database is a `file:` URI whose query says `mode=ro`:
  SQLite opens its file in C, so no `open` event shows a write it makes.

A refusal raises `Refused`, a `PermissionError`, so code that copes with a read-only file
system copes with it: an import that would write its module's bytecode cache goes on
without it, as it does in a read-only directory. An audit hook cannot be removed, so it
binds everything the process does after `install()`; the renderer installs it before it
loads the segment module, and exits when the render is done.

What it does not cover:

* an effect made below Python by a C extension other than SQLite;
* standard library calls that raise no audit event, so the hook never sees them:
  `os.mkfifo` and `os.mknod` make a file, `_posixshmem.shm_open` with `O_CREAT` makes a
  shared memory object under `/dev/shm`, and `signal.pidfd_send_signal` sends a signal;
* a `sqlite3.connect` without `uri=True` given a name that starts `file:`: the event
  carries only that name, which passes the `mode=ro` test, and SQLite then makes a file
  of that literal name;
* a read another process takes as a signal, such as opening a FIFO a writer waits on.

Review covers those.
"""

from __future__ import annotations

import importlib
import os
import sys
from urllib.parse import parse_qs, urlsplit

#: Open flags that write.
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
#: Open mode letters that write.
WRITE_MODES = frozenset("wax+")

#: Audit events refused outright.
REFUSED_EVENTS = frozenset({
    # files and directories
    "os.rename", "os.remove", "os.rmdir", "shutil.rmtree", "os.mkdir", "os.symlink",
    "os.link", "os.truncate", "os.chmod", "os.chown", "os.chflags", "os.utime",
    "os.setxattr", "os.removexattr", "tempfile.mkstemp", "tempfile.mkdtemp",
    # processes and signals
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork",
    "os.forkpty", "pty.spawn", "os.startfile", "os.kill", "os.killpg", "signal.pthread_kill",
    # sockets
    "socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg",
    # native code
    "ctypes.dlopen",
})


class Refused(PermissionError):
    """What the render audit hook raises: an effect a status line render may not make."""


_installed = False


def _read_only_sqlite(database) -> bool:
    if not isinstance(database, str) or not database.startswith("file:"):
        return False
    modes = parse_qs(urlsplit(database).query).get("mode", [])
    return modes == ["ro"]


def _hook(event: str, args: tuple) -> None:
    if event == "open":
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if isinstance(mode, str) and WRITE_MODES & set(mode):
            raise Refused(f"a status line render may not open {path!r} for writing")
        if isinstance(flags, int) and flags & WRITE_FLAGS:
            raise Refused(f"a status line render may not open {path!r} for writing")
    elif event in REFUSED_EVENTS:
        raise Refused(f"a status line render may not make {event}")
    elif event == "sqlite3.connect":
        database = args[0] if args else None
        if not _read_only_sqlite(database):
            raise Refused(f"a status line render may open {database!r} only as a read-only "
                          "file: URI (mode=ro)")


def install() -> None:
    """Stop bytecode writes and add the hook, once per process. It cannot be undone."""
    global _installed
    sys.dont_write_bytecode = True
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True


def installed() -> bool:
    return _installed


def load_segment(name: str):
    """Install the hook, then import the segment module `name` (an announcement's
    `SEGMENT`) and return it."""
    install()
    return importlib.import_module(name)


__all__ = ["Refused", "install", "installed", "load_segment", "REFUSED_EVENTS",
           "WRITE_FLAGS", "WRITE_MODES"]
