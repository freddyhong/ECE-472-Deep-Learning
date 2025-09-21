import jax
import numpy as np
import optax
import structlog
from flax import nnx

from .config import load_settings
from .data import SpiralData
from .logging import configure_logging
from .model import NNXMLP
from .plotting import plot_spiral
from .training import train


def main() -> None:
    """CLI entry point."""
    settings = load_settings()
    configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())

    key = jax.random.PRNGKey(settings.random_seed)
    data_key, model_key = jax.random.split(key)
    np_rng = np.random.default_rng(np.array(data_key))

    data = SpiralData(
        rng=np_rng,
        n_points=settings.data.n_points,
        n_laps=settings.data.n_laps,
        noise=settings.data.noise,
    )

    model = NNXMLP(
        rngs=nnx.Rngs(params=model_key),
        num_input=settings.model.num_input,
        num_output=settings.model.num_output,
        hidden_layer_width=settings.model.hidden_layer_width,
        num_hidden_layers=settings.model.num_hidden_layers,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adam(settings.training.learning_rate),
        wrt=nnx.Param,
    )

    train(model, optimizer, data, settings.training, np_rng)

    plot_spiral(model, data, settings.plotting)
