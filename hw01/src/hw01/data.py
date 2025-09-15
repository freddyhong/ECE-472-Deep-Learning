from dataclasses import InitVar, dataclass, field

import numpy as np

from .model import BasisExpansionModel


@dataclass
class Data:
    """Handles generation of synthetic data for basis expansion regression on sine data."""

    model: BasisExpansionModel
    rng: InitVar[np.random.Generator]
    num_features: int
    num_samples: int
    sigma: float
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    index: np.ndarray = field(init=False)

    def __post_init__(self, rng: np.random.Generator):
        """Generate synthetic data based on the model."""
        self.index = np.arange(self.num_samples)
        self.x = rng.uniform(0.0, 1.0, size=(self.num_samples, self.num_features))
        epsilon = rng.normal(
            0.0, self.sigma, size=(self.num_samples, self.num_features)
        )
        self.y = np.sin(2 * np.pi * self.x) + epsilon

    def get_batch(
        self, rng: np.random.Generator, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select random subset of examples for training batch."""
        choices = rng.choice(self.index, size=batch_size)

        return self.x[choices], self.y[choices].flatten()
