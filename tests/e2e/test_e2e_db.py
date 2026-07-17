"""Unit coverage for the shared e2e DB cleanup guard.

No `e2e` marker, so these run without a proxy or a DB. They lock in that the
session-finish truncate of the spend-log table fires only on an explicit opt-in
plus an actual test run: neither condition alone, nor any opt-in value other than
"1", may arm the destructive path.
"""

import pytest

from e2e_db import run_spend_log_cleanup


class _Spy:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


@pytest.mark.parametrize(
    ("opt_in", "e2e_test_ran", "should_truncate"),
    [
        ("1", True, True),
        ("1", False, False),
        (None, True, False),
        (None, False, False),
        ("0", True, False),
        ("true", True, False),
        ("", True, False),
    ],
)
def test_cleanup_gates_on_opt_in_and_test_run(
    opt_in: str | None, e2e_test_ran: bool, should_truncate: bool
) -> None:
    spy = _Spy()
    ran = run_spend_log_cleanup(opt_in=opt_in, e2e_test_ran=e2e_test_ran, truncate=spy)
    assert ran is should_truncate
    assert spy.calls == (1 if should_truncate else 0)


def test_cleanup_swallows_truncate_failure() -> None:
    def boom() -> None:
        raise RuntimeError("db unreachable")

    assert run_spend_log_cleanup(opt_in="1", e2e_test_ran=True, truncate=boom) is True
