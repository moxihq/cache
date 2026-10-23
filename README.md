# moxi-cache

Moxi's cache module tells you when an agent session's prompt cache goes cold. A read gives
a session's cache one of five kinds, `warm`, `cold`, `fresh`, `no-cache` or `unknown`,
with its age and the lifetime it counts, and a reason code whenever it gives `no-cache` or
`unknown`. It reads Claude Code sessions through the sessions module's transcript reader,
Codex threads from their rollout files, and OpenCode sessions from OpenCode's database,
opened read-only. Codex and OpenCode lifetimes come from a lifetime table shipped with the
module, keyed on harness, provider and model, which you can extend in cache's config. A
watcher keeps a clock record for every live Claude Code session, a status line segment
shows a session's cache age from that record, and a timer runs the commands that modules
or your config subscribe, at set times before or after a session's cache expires. cache
says what a session's cache is and leaves what to do about it to the caller. Other Moxi
modules import it through `moxi_cache.api` (comms, with its `cache` extra, holds a send
while the recipient's cache is cold), and it adds these commands:

- `moxi cache read SESSION... [--harness claude|codex|opencode] [--source PATH] [--json]`
  prints each session's kind, age and lifetime, or why it reads `no-cache` or `unknown`.
  It writes nothing.
- `moxi cache segment SESSION [--color|--json]` prints a Claude Code session's status
  line segment from the watcher's record, such as `cache:12m3s`, and nothing when there
  is no record.
- `moxi cache watch` runs the watcher in the foreground. `moxi cache service
  install|remove|status` runs it as a systemd user service instead.
- `moxi cache timers [--json]` lists the timer's subscriptions, the refused ones and why,
  and when each subscription next asks about each live session.
- `moxi cache doctor` checks the config, the lifetime table, the drift record and the
  watcher, and exits 1 when something there stops reads.

Its settings are `claude.ttl_seconds`, `lifetime` (your own lifetime table rows),
`watch.tick`, `segment.warn_before` and `timer.subscriber`; `moxi config cache` shows
them.

## Install

Linux, Python 3.11 or later, and [uv](https://docs.astral.sh/uv/).

```sh
uv tool install moxi
moxi install cache
```

Before the modules are on PyPI, install the core from git and give `moxi install` a git
source for sessions and for cache, in that order:

```sh
uv tool install git+https://github.com/moxihq/moxi
moxi install cache \
  --from 'moxi-sessions @ git+https://github.com/moxihq/sessions' \
  --from 'moxi-cache @ git+https://github.com/moxihq/cache'
```

The front door's README, at [moxihq/moxi](https://github.com/moxihq/moxi), covers the
`moxi` command and the other modules.

## Dependencies

`moxi`, Moxi's core, and `moxi-sessions`. cache has no optional extras. The watcher lists
live sessions with Claude Code's `claude agents --json`, so it needs `claude` on PATH or
at `~/.local/bin/claude`, and `moxi cache service` needs a systemd user manager.

## License

MIT. See [LICENSE](LICENSE).
