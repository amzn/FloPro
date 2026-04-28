"""Problem format constants shared across coordinator implementations.

Controls the sign of the dual (price) update in ADMM:
  minimization → prices penalise excess  (sign = +1)
  maximization → prices reward scarcity  (sign = −1)
"""

from typing import Literal, Tuple

VALID_PROBLEM_FORMATS: Tuple[str, ...] = ("minimization", "maximization")
ProblemFormat = Literal["minimization", "maximization"]


def price_sign(problem_format: str) -> int:
    """Return +1 for minimization, −1 for maximization.

    Raises ``ValueError`` for unrecognised formats.
    """
    if problem_format not in VALID_PROBLEM_FORMATS:
        raise ValueError(
            f"problem_format must be one of {VALID_PROBLEM_FORMATS}, got {problem_format!r}"
        )
    return 1 if problem_format == "minimization" else -1
