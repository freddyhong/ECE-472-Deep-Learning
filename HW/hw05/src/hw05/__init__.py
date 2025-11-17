import optax
import structlog
from flax import nnx
import jax
import jax.numpy as jnp
import pathlib as Path
import orbax.checkpoint as ocp

from .config import load_settings
from .logging import configure_logging 
from .model import Conv1DTextClassifier
from .training import train, test_evaluation, load_checkpoint
from .data import load_AG_News, tokenize_datasets

def main() -> None:
    settings = load_settings()
    log_file = configure_logging()
    log = structlog.get_logger()
    log.info("Settings loaded", settings=settings.model_dump())
    print(f"Logs are being saved to {log_file}")


    raw_ds = load_AG_News(
        val_per=settings.data.validation_percentage
    )

    rngs = nnx.Rngs(params=settings.random_seed, dropout=settings.random_seed + 1)
    tokenized, tokenizer = tokenize_datasets(
        raw_ds,
        model_name="distilbert-base-uncased",
        max_length=settings.data.max_length,
    )
    def make_batches(ds_split, batch_size):
        for i in range(0, len(ds_split), batch_size):
            batch = ds_split[i : i + batch_size]
            yield {
                "input_ids": jnp.array(batch["input_ids"]),
                "label": jnp.array(batch["label"]),
            }

    train_batches = list(make_batches(tokenized["train"], settings.data.batch_size))
    val_batches   = list(make_batches(tokenized["val"], settings.data.batch_size))
    test_batches  = list(make_batches(tokenized["test"], settings.data.batch_size))

    model = Conv1DTextClassifier(
        vocab_size=tokenizer.vocab_size,
        num_classes=settings.model.num_classes,
        embed_dim=settings.model.embed_dim,
        conv_features=settings.model.conv_features,
        kernel_size=settings.model.kernel_size,
        dropout_rate=settings.model.dropout_rate,
        rngs=rngs,
    )

    params = nnx.state(model, nnx.Param)
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    log.info("Model created", num_params=num_params)

    steps_per_epoch = len(train_batches)
    decay_steps = settings.training.num_epochs * steps_per_epoch

    lr_schedule = optax.schedules.cosine_decay_schedule(
        init_value=settings.training.learning_rate,
        decay_steps=decay_steps,
    )

    optimizer = nnx.Optimizer(
        model,
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=settings.training.weight_decay,
        ),
        wrt=nnx.Param,
    )

    if settings.training.final_test:
        log.info("Starting final testing...")

        model_kwargs = dict(
            vocab_size=tokenizer.vocab_size,
            num_classes=settings.model.num_classes,
            embed_dim=settings.model.embed_dim,
            conv_features=settings.model.conv_features,
            kernel_size=settings.model.kernel_size,
            dropout_rate=settings.model.dropout_rate,
        )

        model = load_checkpoint(Conv1DTextClassifier, model_kwargs)
        test_loss, test_acc = test_evaluation(model, test_batches, rngs)
        log.info("Final Test Results", test_loss=test_loss, test_acc=test_acc)
    else:
        _ = train(model, optimizer, train_batches, val_batches, rngs)