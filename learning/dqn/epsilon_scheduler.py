from __future__ import annotations


class EpsilonScheduler:
    def __init__(
        self,
        start_epsilon: float = 1.0,
        end_epsilon: float = 0.05,
        decay_steps: int = 100_000,
    ):
        self.start_epsilon = float(start_epsilon)
        self.end_epsilon = float(end_epsilon)
        self.decay_steps = max(1, int(decay_steps))

    def value(self, step: int) -> float:
        step = max(0, int(step))
        fraction = min(step / self.decay_steps, 1.0)
        return self.start_epsilon + fraction * (self.end_epsilon - self.start_epsilon)