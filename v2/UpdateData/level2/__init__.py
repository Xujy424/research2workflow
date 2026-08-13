"""Data loading package public interface.

Configuration attributes are loaded on demand to avoid a circular import:
UpdateData.config needs CIFSLoader, while callers historically import
configuration values from this package.
"""

from .cifsLoader import CIFSLoader

_CONFIG_EXPORTS = {
    "CIFTABLE_PATTERNS",
    "L2DATA_PATH",
    "cifs",
    "get_jy_conn",
    "get_str_engine",
    "get_zyyx_conn",
}


def __getattr__(name: str):
    if name in _CONFIG_EXPORTS:
        from .. import config

        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CIFSLoader",
    "CIFTABLE_PATTERNS",
    "cifs",
    "L2DATA_PATH",
    "get_jy_conn",
    "get_str_engine",
    "get_zyyx_conn",
]
