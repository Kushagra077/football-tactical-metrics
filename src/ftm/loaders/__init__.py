"""Provider adapters.

Each loader knows one provider's native format (via kloppy or raw files)
and nothing else. Its job: read a match, apply the coordinate + orientation
transforms, and return a DataFrame that passes ``ftm.schema.validate()``.

Every provider produces the same canonical tracking frame, so nothing
downstream of the loaders needs to know where the data came from.

Public surface
--------------
BaseLoader          Abstract contract: ``load(match_id) -> DataFrame``.
get_loader(name)    Factory: "metrica" / "skillcorner" -> loader instance.
MetricaLoader       Metrica Sample Data (complete, 25 Hz).
SkillCornerLoader   SkillCorner Open Data (broadcast, ~10 Hz, players drop).
"""

from __future__ import annotations

__all__ = [
    "BaseLoader",
    "get_loader",
    "MetricaLoader",
    "SkillCornerLoader",
]


def get_loader(name: str) -> BaseLoader:
    """Return a loader instance for a provider name.

    Parameters
    ----------
    name:
        Case-insensitive. Accept at least {"metrica", "skillcorner"};
        raise ``KeyError`` (with the list of known names) on anything
        else. Keeping this factory the only construction path means the
        pipeline, cache builder and dashboard all resolve providers the
        same way.
    """
    registry: dict[str, type[BaseLoader]] = {
        "metrica": MetricaLoader,
        "skillcorner": SkillCornerLoader,
    }
    key = name.strip().lower() if isinstance(name, str) else name
    if key not in registry:
        raise KeyError(f"Unknown loader {name!r}; valid names: {sorted(registry)}")
    return registry[key]()


# Re-exports. Import here (not at top) to keep the module importable even
# while the concrete loaders are still stubs during early development.
from ftm.loaders.base import BaseLoader  # noqa: E402
from ftm.loaders.metrica import MetricaLoader  # noqa: E402
from ftm.loaders.skillcorner import SkillCornerLoader  # noqa: E402
