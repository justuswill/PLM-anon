"""Mean and two-sided 95% Student-t interval over seeds."""

from __future__ import annotations

from statistics import fmean, stdev

# Two-sided 95% Student-t quantiles by degrees of freedom.
T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def spread(values: list[float]) -> dict:
    if len(values) < 2:
        return {"mean": values[0], "std": 0.0, "ci95": 0.0}
    deviation = stdev(values)
    return {"mean": fmean(values), "std": deviation, "ci95": T_95[len(values) - 1] * deviation / len(values) ** 0.5}
