import optax
import structlog
from flax import nnx
import jax
import pathlib as Path

from .config import load_settings
from .logging import configure_logging 
from .model import Classifier
from .training import train, test_evaluation
from .data import load_CIFAR10

def main() -> None:
    settings = load_settings()
    log_file = configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())
    print(f"Logs are being saved to {log_file}")

    ds_train, ds_val, ds_test = load_CIFAR10(
        batch_size=settings.data.batch_size,
        validation_size=settings.data.validation_size,
    )
    rngs = nnx.Rngs(params=settings.random_seed, dropout=settings.random_seed + 1)
    model = Classifier(
        rngs=rngs,
        input_depth=settings.model.input_depth,
        layer_depths=settings.model.layer_depths,
        blocks_per_stage=settings.model.blocks_per_stage,
        layer_kernel_sizes=[(3,3)] * len(settings.model.layer_depths), 
        num_classes=settings.model.num_classes,
        dropout=settings.model.dropout,
    )

    params = nnx.state(model, nnx.Param) 
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(params)) 
    log.info("Model created", num_params=num_params)

    # LR schedule and optimizer
    train_size      = 50000  
    steps_per_epoch = train_size // settings.data.batch_size
    decay_steps     = settings.training.num_epochs * steps_per_epoch
    lr_schedular = optax.schedules.cosine_decay_schedule(
        init_value=settings.training.learning_rate,
        decay_steps=decay_steps,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=lr_schedular,
            weight_decay=settings.training.weight_decay,
        ),
        wrt=nnx.Param,
    )

    _ = train(model, optimizer, ds_train, ds_val, rngs)

    # Test
    log.info("Starting final testing...")
    test_loss, test_acc = test_evaluation(model, ds_test, rngs)
    log.info("Final Test Results", test_loss=test_loss, test_acc=test_acc)
