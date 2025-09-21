import jax.numpy as jnp
import numpy as np
import optax
import structlog
from flax import nnx
from tqdm import trange

from .config import TrainingSettings
from .data import SpiralData
from .model import NNXMLP

log = structlog.get_logger()


@nnx.jit
def train_step(model: NNXMLP, optimizer: nnx.Optimizer, x: jnp.ndarray, y: jnp.ndarray):
    """Performs a single training step."""

    def loss_fn(m: NNXMLP):
        logits = m(x)
        yb = jnp.asarray(y, jnp.float32).reshape(-1, 1)
        bce = optax.sigmoid_binary_cross_entropy(logits, yb).mean()
        return bce

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def train(
    model: NNXMLP,
    optimizer: nnx.Optimizer,
    data: SpiralData,
    settings: TrainingSettings,
    np_rng: np.random.Generator,
) -> None:
    """Train the model using SGD."""
    log.info("Starting training", **settings.model_dump())
    bar = trange(settings.num_iters)
    for i in bar:
        x_np, y_np = data.get_batch(np_rng, settings.batch_size)
        x, y = jnp.asarray(x_np), jnp.asarray(y_np)

        loss = train_step(model, optimizer, x, y)
        if i % 10 == 0:
            log.info(f"Training Loss @ {i}: {loss:.6f}")
        bar.set_description(f"Loss @ {i} => {loss:.6f}")
        bar.refresh()
    log.info("Training finished")
