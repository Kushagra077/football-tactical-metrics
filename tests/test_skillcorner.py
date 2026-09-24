"""SkillCorner loader tests (network ones are integration-marked)."""

from __future__ import annotations


# @pytest.mark.integration
def test_skillcorner_match_loads_validates_and_has_coverage_gaps():
    """(integration, network) A SkillCorner match passes ``validate`` and
    at least one player has coverage < 100% in a period (proves broadcast
    drop-out is preserved, not forward-filled)."""
    raise NotImplementedError
