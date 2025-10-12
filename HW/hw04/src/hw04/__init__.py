import optax
import structlog
from flax import nnx
import jax
import pathlib as Path
import orbax.checkpoint as ocp

from .config import load_settings
from .logging import configure_logging 
from .model import Classifier
from .training import train, test_evaluation, load_checkpoint
from .data import load_CIFAR10, load_CIFAR100

def main() -> None:
    settings = load_settings()
    log_file = configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())
    print(f"Logs are being saved to {log_file}")

    if settings.data.cifar100:
        ds_train, ds_val, ds_test = load_CIFAR100(
        batch_size=settings.data.batch_size,
        validation_size=settings.data.validation_size,
    )
    else:
        ds_train, ds_val, ds_test = load_CIFAR10(
            batch_size=settings.data.batch_size,
            validation_size=settings.data.validation_size,
        )
    rngs = nnx.Rngs(params=settings.random_seed)
    model = Classifier(
        rngs=rngs,
        input_depth=settings.model.input_depth,
        layer_depths=settings.model.layer_depths,
        blocks_per_stage=settings.model.blocks_per_stage,
        layer_kernel_sizes=[(3,3)] * len(settings.model.layer_depths), 
        num_classes=settings.model.num_classes,
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

    if settings.training.final_test:
        log.info("Starting final testing...")

        model_kwargs = dict(
            input_depth=settings.model.input_depth,
            layer_depths=settings.model.layer_depths,
            blocks_per_stage=settings.model.blocks_per_stage,
            layer_kernel_sizes=[(3, 3)] * len(settings.model.layer_depths),
            num_classes=settings.model.num_classes,
        )

        model = load_checkpoint(
            Classifier,
            model_kwargs
        )

        test_loss, test_acc = test_evaluation(model, ds_test, rngs)
        log.info("Final Test Results", test_loss=test_loss, test_acc=test_acc)
    else:
        _ = train(model, optimizer, ds_train, ds_val, rngs)


    
