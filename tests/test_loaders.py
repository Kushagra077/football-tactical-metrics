"""Loader tests that do NOT hit the network.

The real ``metrica.load`` / ``skillcorner.load`` need downloads, so they
live behind a marker (``@pytest.mark.integration``) and are skipped in
the default CI run. What we can test cheaply:

* the factory
* the abstraction boundary (a fake loader's output flows through every
  metric unchanged)
"""

from __future__ import annotations


def test_get_loader_returns_right_types():
    """``get_loader("metrica")`` -> ``MetricaLoader``; ``"SkillCorner"``
    (any case) -> ``SkillCornerLoader``; unknown name -> ``KeyError``
    listing the valid names."""
    raise NotImplementedError


def test_fake_loader_output_passes_schema_and_every_metric():
    """Build a synthetic canonical frame (via ``conftest``), run it
    through ``kinematics`` + ``shape`` + ``pressing`` + ``space`` +
    ``physical`` with no code under ``ftm/metrics`` knowing which
    'provider' it came from. This is the executable form of the
    source-agnostic rule."""
    raise NotImplementedError


def test_metrics_package_never_imports_kloppy_or_loaders():
    """Static check: import every module under ``ftm.metrics`` and assert
    ``kloppy`` and ``ftm.loaders`` are absent from ``sys.modules`` attrs
    of those modules (or grep their source). Fails loudly if the
    abstraction leaks."""
    raise NotImplementedError


# @pytest.mark.integration
def test_metrica_game3_loads_and_validates():
    """(integration, network) ``MetricaLoader().load("3")`` passes
    ``validate``, has 22 players + 1 ball, ``x_pitch`` spanning ~+/-52.5,
    and ``frame_rate`` read from metadata == 25."""
    raise NotImplementedError


# @pytest.mark.integration
def test_skillcorner_match_loads_validates_and_has_coverage_gaps():
    """(integration, network) A SkillCorner match passes ``validate`` and
    at least one player has coverage < 100% in a period (proves broadcast
    drop-out is preserved, not forward-filled)."""
    raise NotImplementedError
