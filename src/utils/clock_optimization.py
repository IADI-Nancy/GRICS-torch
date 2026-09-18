"""Bounded scalar finite-difference gradient descent for expensive reconstructions."""

from dataclasses import dataclass
import math


@dataclass
class ClockOptimizationResult:
    correction_seconds: float
    objective: float
    iterations: int
    stop_reason: str
    evaluations: list[dict]
    accepted_corrections: list[float]


def optimize_clock_correction(objective, initial=0.0, *, bounds=(-1.0, 1.0),
                              difference_step=0.05, learning_rate=0.1,
                              max_step=0.2, tolerance=0.001, max_iterations=8,
                              max_backtracks=8):
    """Minimize an objective using finite differences and backtracking.

    Every objective evaluation must use the same slices and reconstruction
    settings. Bounds must already respect physiological recording coverage.
    Probe results are cached; the best evaluated point is always retained,
    including probes, which helps with discontinuous motion-state binning.
    This is a local search, not a guarantee of a global optimum.
    """
    lower, upper = map(float, bounds)
    if not all(math.isfinite(v) for v in (lower, upper, initial)) or lower >= upper:
        raise ValueError('Clock bounds must be finite and strictly increasing.')
    if not lower <= initial <= upper:
        raise ValueError('Initial clock correction must lie within the search bounds.')
    for name, value in [('difference_step', difference_step), ('learning_rate', learning_rate),
                        ('max_step', max_step), ('tolerance', tolerance)]:
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive.')
    for name, value in [('max_iterations', max_iterations), ('max_backtracks', max_backtracks)]:
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    cache = {}
    evaluations = []

    def evaluate(x):
        x = float(x)
        if x not in cache:
            value = float(objective(x))
            if not math.isfinite(value):
                raise ValueError(f'Clock objective is nonfinite at {x:g} seconds.')
            cache[x] = value
            evaluations.append({'correction_seconds': x, 'objective': value})
        return cache[x]

    x = float(initial)
    value = evaluate(x)
    accepted = [x]
    reason = 'maximum iterations'
    for iteration in range(1, max_iterations + 1):
        lo, hi = max(lower, x - difference_step), min(upper, x + difference_step)
        gradient = (evaluate(hi) - evaluate(lo)) / (hi - lo)
        step = max(-max_step, min(max_step, -learning_rate * gradient))
        for _ in range(max_backtracks):
            candidate = max(lower, min(upper, x + step))
            if abs(candidate - x) < tolerance:
                break
            if evaluate(candidate) < value:
                break
            step *= 0.5
        best = min(cache, key=cache.get)
        if cache[best] >= value:
            reason = 'no improving step or finite-difference probe'
            break
        movement = abs(best - x)
        x, value = best, cache[best]
        accepted.append(x)
        if movement < tolerance:
            reason = 'correction tolerance'
            break
    return ClockOptimizationResult(x, value, iteration, reason, evaluations, accepted)
