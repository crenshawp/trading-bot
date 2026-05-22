"""Smoke tests that verify the audit harness itself works.

These exist only so `pytest` has something to run during Phase 1.1a.
They will be deleted or absorbed once real tests arrive in 1.1b/1.1c.
"""

import trading_bot


def test_package_imports() -> None:
    assert trading_bot.__version__ == "0.1.0"


def test_harness_is_alive() -> None:
    assert 1 + 1 == 2
