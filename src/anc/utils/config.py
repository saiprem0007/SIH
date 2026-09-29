"""Recursive dot-notation config wrapper for ANC-SIH YAML configs."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterator

import yaml


class Config:
    """Recursive dot-notation wrapper around a nested dictionary.

    Nested ``dict`` values are automatically converted to :class:`Config`
    objects so that deep access like ``cfg.audio.sample_rate`` works
    out of the box.

    Attributes:
        _data: The underlying flat mapping at this level of the config tree.

    Example::

        cfg = load_config("configs/base.yaml")
        sr = cfg.audio.sample_rate        # dot access
        sr = cfg["audio"]["sample_rate"]  # dict-style access
        sr = cfg.audio.get("sample_rate", 16000)  # with default
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialise from a (possibly nested) dictionary.

        Args:
            data: Mapping to wrap.  Nested ``dict`` values are recursively
                  converted to :class:`Config` instances.
        """
        object.__setattr__(self, "_data", {})
        for key, value in data.items():
            if isinstance(value, dict):
                object.__getattribute__(self, "_data")[key] = Config(value)
            else:
                object.__getattribute__(self, "_data")[key] = value

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, key: str) -> Any:
        """Support ``cfg.key`` access.

        Args:
            key: Attribute name to look up.

        Returns:
            The value stored under *key*.

        Raises:
            AttributeError: If *key* is not present in the config.
        """
        data = object.__getattribute__(self, "_data")
        if key in data:
            return data[key]
        raise AttributeError(
            f"Config has no attribute '{key}'. "
            f"Available keys: {list(data.keys())}"
        )

    def __setattr__(self, key: str, value: Any) -> None:
        """Support ``cfg.key = value`` assignment.

        Args:
            key:   Attribute name.
            value: Value to store.  Dicts are wrapped in a new :class:`Config`.
        """
        data = object.__getattribute__(self, "_data")
        data[key] = Config(value) if isinstance(value, dict) else value

    # ------------------------------------------------------------------
    # Mapping-style access
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        """Support ``cfg['key']`` access.

        Args:
            key: Key to retrieve.

        Returns:
            Value stored at *key*.

        Raises:
            KeyError: If *key* is absent.
        """
        data = object.__getattribute__(self, "_data")
        if key in data:
            return data[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Support ``cfg['key'] = value`` assignment.

        Args:
            key:   Key to set.
            value: Value.  Dicts are wrapped automatically.
        """
        data = object.__getattribute__(self, "_data")
        data[key] = Config(value) if isinstance(value, dict) else value

    def __contains__(self, key: object) -> bool:
        """Support ``'key' in cfg``.

        Args:
            key: Key to test.

        Returns:
            ``True`` if *key* exists at this config level.
        """
        data = object.__getattribute__(self, "_data")
        return key in data

    def __iter__(self) -> Iterator[str]:
        """Iterate over top-level keys.

        Yields:
            Each key in the underlying data dictionary.
        """
        data = object.__getattribute__(self, "_data")
        return iter(data)

    def __len__(self) -> int:
        """Return the number of top-level keys.

        Returns:
            Integer count of keys.
        """
        data = object.__getattribute__(self, "_data")
        return len(data)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key* or *default* if absent.

        Args:
            key:     Key to look up.
            default: Fallback value (default ``None``).

        Returns:
            Stored value or *default*.
        """
        data = object.__getattribute__(self, "_data")
        return data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert back to a plain Python dictionary.

        Returns:
            Deep ``dict`` representation of this config tree.
        """
        data = object.__getattribute__(self, "_data")
        result: dict[str, Any] = {}
        for key, value in data.items():
            result[key] = value.to_dict() if isinstance(value, Config) else value
        return result

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Pretty-print the config as YAML.

        Returns:
            YAML-formatted string of this config level and all children.
        """
        return yaml.dump(
            self.to_dict(),
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    def __eq__(self, other: object) -> bool:
        """Equality based on underlying dict contents.

        Args:
            other: Object to compare with.

        Returns:
            ``True`` if both configs have identical data trees.
        """
        if isinstance(other, Config):
            return self.to_dict() == other.to_dict()
        return NotImplemented


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> Config:
    """Load a YAML file and return it wrapped in a :class:`Config`.

    Args:
        path: Filesystem path to a ``.yaml`` / ``.yml`` config file.

    Returns:
        :class:`Config` object containing all keys from the YAML file.

    Raises:
        FileNotFoundError: If *path* does not exist.
        yaml.YAMLError:    If the file cannot be parsed.

    Example::

        cfg = load_config("configs/base.yaml")
        print(cfg.audio.sample_rate)  # 16000
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    return Config(raw)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *overrides* into *base* (non-destructive).

    Args:
        base:      The base dictionary to merge into.
        overrides: Overriding values.  Keys present here overwrite *base*.

    Returns:
        A new merged dictionary.  Neither *base* nor *overrides* is mutated.
    """
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def merge_configs(base: Config, overrides: dict[str, Any]) -> Config:
    """Deep-merge an *overrides* dict into a :class:`Config`.

    Nested keys in *overrides* are merged recursively rather than
    replacing whole sub-trees, so you can supply partial overrides
    (e.g. just ``{"audio": {"sample_rate": 8000}}``).

    Args:
        base:      The base :class:`Config` to extend.
        overrides: A plain Python dict with values that override *base*.

    Returns:
        A new :class:`Config` containing the merged configuration.
        *base* is not mutated.

    Example::

        cfg = load_config("configs/base.yaml")
        cfg = merge_configs(cfg, {"audio": {"sample_rate": 8000}})
    """
    merged = _deep_merge(base.to_dict(), overrides)
    return Config(merged)
