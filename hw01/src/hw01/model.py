from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


@dataclass
class BasisExpansionModel:
    """Represents a simple basis expansion model."""

    weights: np.ndarray
    sigma: np.ndarray
    mu: np.ndarray
    bias: float


@dataclass
class TrueSineModel:
    sigma: float

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * x)


class NNXBasisExpansionModel(nnx.Module):
    """A Flax NNX module for a basis expansion model"""

    def __init__(self, *, rngs: nnx.Rngs, num_basis: int):
        self.num_basis = num_basis
        key = rngs.params()
        mu_key, w_key = jax.random.split(
            key
        )  # splitting key to assign different key value for mu and w
        self.mu = nnx.Param(jax.random.uniform(mu_key, (num_basis, 1)))
        self.sigma = nnx.Param(jnp.full((num_basis, 1), 0.1))
        self.w = nnx.Param(jax.random.normal(w_key, (num_basis, 1)))
        self.b = nnx.Param(jnp.zeros((1, 1)))

    def __call__(self, x: jax.Array) -> jax.Array:
        mu = jax.nn.sigmoid(self.mu.value)  # setting mu value within [0,1] range
        diff = x - mu.T
        phi = jnp.exp(-(diff**2) / (self.sigma.value.T**2))
        y_hat = phi @ self.w.value + self.b.value
        return jnp.squeeze(y_hat)

    @property
    def model(self) -> BasisExpansionModel:
        """Returns the underlying simple linear model."""
        return BasisExpansionModel(
            weights=np.array(self.w.value).reshape([self.num_basis]),
            bias=np.array(self.b.value).squeeze(),
            sigma=np.array(self.sigma.value).reshape(self.num_basis),
            mu=np.array(self.mu.value).reshape(self.num_basis),
        )
