from dataclasses import InitVar, dataclass, field

import numpy as np


@dataclass
class SpiralData:
    """Data generation"""

    rng: InitVar[np.random.Generator]
    n_points: int
    n_laps: int
    noise: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):
        self.index = np.arange(2 * self.n_points)
        theta = np.linspace(0, self.n_laps * 2 * np.pi, self.n_points)
        r = theta / self.n_laps  # just setting it proportional to theta
        x1 = r * np.cos(theta)
        y1 = r * np.sin(theta)

        x2 = r * np.cos(theta + np.pi)
        y2 = r * np.sin(theta + np.pi)

        X = np.vstack(
            [
                np.stack([x1, y1], axis=1),
                np.stack([x2, y2], axis=1),
            ]
        )

        epsilon = rng.normal(0, self.noise, size=X.shape)

        X += epsilon

        y = np.concatenate(
            [
                np.zeros(self.n_points, dtype=int),
                np.ones(self.n_points, dtype=int),
            ]
        )

        self.x = X
        self.y = y

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices]
