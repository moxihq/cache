"""The announcement of Moxi's cache module, which moxi.modules reads."""

INTERFACES = frozenset({1})
API = "moxi_cache.api"
CLI = "moxi_cache.cli:main"
SUMMARY = "Know when a session's prompt cache goes cold, and act before it does."
OPTIONAL = {}
GATED = {}
INTERNAL_ENV = {}
SETTINGS = "settings.toml"
SETUP = None
SEGMENT = "moxi_cache.segment"
