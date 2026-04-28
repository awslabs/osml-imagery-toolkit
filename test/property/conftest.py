#  Copyright 2023-2024 Amazon.com, Inc. or its affiliates.

"""Shared fixtures and pytest configuration for property-based tests.

This module provides:
- Hypothesis profiles for dev (fast) and ci (thorough) runs
- pbt_settings: shared hypothesis settings for all property tests
- pytest marker registration for 'property' tests

Usage:
    # Fast local development (default):
    pytest -m property

    # Full CI run:
    HYPOTHESIS_PROFILE=ci pytest -m property
"""

import os

from hypothesis import Phase, settings


def pytest_configure(config):
    """Register the 'property' marker for property-based tests."""
    config.addinivalue_line(
        "markers",
        "property: mark test as a property-based test (run with pytest -m property)",
    )


# ---------------------------------------------------------------------------
# Hypothesis profiles
# ---------------------------------------------------------------------------

# CI profile: thorough coverage
settings.register_profile(
    "ci",
    max_examples=100,
    deadline=None,
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
    suppress_health_check=[],
)

# Dev profile: fast iteration
settings.register_profile(
    "dev",
    max_examples=10,
    deadline=None,
    phases=[Phase.explicit, Phase.reuse, Phase.generate],  # skip shrink
    suppress_health_check=[],
)

settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


# Shared settings object — inherits max_examples and phases from the active profile.
# All property test files should import and use this instead of defining their own.
pbt_settings = settings(deadline=None)
