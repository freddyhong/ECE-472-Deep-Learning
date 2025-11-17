# src/hw05/training.py
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import matplotlib.pyplot as plt
import structlog
import orbax.checkpoint as ocp
from pathlib import Path
import os

from .config import load_settings

log = structlog.get_logger()

# Loss & accuracy

def cross_entropy_loss(logits, labels, num_classes, smoothing=0.1):
    one_hot = jax.nn.one_hot(labels, num_classes)
    smoothed = one_hot * (1 - smoothing) + smoothing / num_classes
    loss = optax.softmax_cross_entropy(logits, smoothed).mean()
    return loss

def compute_accuracy(logits, labels):
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels)


# Core loss_fn
def _loss_fn(model, batch, rngs, train):
    input_ids = batch["input_ids"]
    labels = batch["label"]
    settings = load_settings()
    logits = model(input_ids, train=train)
    loss = cross_entropy_loss(logits, labels, settings.model.num_classes)
    return loss, logits


# Train & Eval Steps
@nnx.jit
def train_step(model, optimizer, batch, rngs):
    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch, rngs, True)
    optimizer.update(model, grads)
    return loss, logits


@nnx.jit(static_argnames=("train",))
def eval_step(model, batch, rngs, train):
    loss, logits = _loss_fn(model, batch, rngs, train)
    return loss, logits


# Epoch Evaluation

def evaluate(model, dataset, rngs):
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    for batch in dataset:
        loss, logits = eval_step(model, batch, rngs, train=False)
        labels = batch["label"]
        acc = compute_accuracy(logits, labels)
        total_loss += float(loss)
        total_acc += float(acc)
        n_batches += 1
    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


# Main Training Loop
def train(model, optimizer, train_ds, val_ds, rngs):
    settings = load_settings()
    output_dir = settings.plotting.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = settings.training.checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ocp.test_utils.erase_and_create_empty(checkpoint_dir)

    log_interval = settings.training.log_interval
    num_epochs = settings.training.num_epochs

    metrics_history = {
        "train_steps": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_steps": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    global_step = 0
    best_val_acc = 0.0
    best_ckpt_path = None

    for epoch in range(1, num_epochs + 1):
        for i, batch in enumerate(train_ds):
            loss, logits = train_step(model, optimizer, batch, rngs)

            if global_step % log_interval == 0:
                labels = batch["label"]
                acc = compute_accuracy(logits, labels)
                metrics_history["train_steps"].append(global_step)
                metrics_history["train_loss"].append(float(loss))
                metrics_history["train_accuracy"].append(float(acc))
                log.info(
                    "train_iter",
                    epoch=epoch,
                    step=global_step,
                    loss=float(loss),
                    acc=float(acc),
                )
            global_step += 1

        # Evaluate
        val_loss, val_acc = evaluate(model, val_ds, rngs)
        if epoch % 5 == 0 or epoch == num_epochs:
            checkpointer = ocp.StandardCheckpointer()
            _, state = nnx.split(model)
            ckpt_path = checkpoint_dir / f"state_epoch_{epoch}"
            param_state = nnx.state(model, nnx.Param)
            checkpointer.save(ckpt_path, param_state)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_ckpt_path = checkpoint_dir / f"best_state_epoch_{epoch}"
                checkpointer.save(best_ckpt_path, param_state)

        metrics_history["val_steps"].append(global_step)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["val_accuracy"].append(val_acc)
        log.info(
            "epoch_summary",
            epoch=epoch,
            step=global_step,
            val_loss=val_loss,
            val_acc=val_acc,
        )

    # Plotting metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.set_title("Loss vs. Steps")
    ax2.set_title("Accuracy vs. Steps")

    ax1.plot(metrics_history["train_steps"], metrics_history["train_loss"], label="train")
    ax2.plot(metrics_history["train_steps"], metrics_history["train_accuracy"], label="train")

    ax1.plot(metrics_history["val_steps"], metrics_history["val_loss"], label="val", marker='o', linestyle='--')
    ax2.plot(metrics_history["val_steps"], metrics_history["val_accuracy"], label="val", marker='o', linestyle='--')

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")

    ax1.legend(); ax2.legend()
    ax1.grid(True, linestyle='--', linewidth=0.5)
    ax2.grid(True, linestyle='--', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_dir / "final_training_metrics.png", dpi=settings.plotting.dpi)
    plt.close(fig)

    log.info("Training complete.")
    return metrics_history


# Test Evaluation
def test_evaluation(model, test_ds, rngs):
    test_loss, test_acc = evaluate(model, test_ds, rngs)
    log.info("test_results", test_loss=test_loss, test_acc=test_acc)
    return test_loss, test_acc


def load_checkpoint(model_cls, model_kwargs: dict):
    settings = load_settings()
    checkpoint_dir = settings.training.checkpoint_dir.resolve()
    checkpointer = ocp.StandardCheckpointer()

    ckpts = sorted(checkpoint_dir.glob("best_state_epoch_*"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    latest_ckpt = ckpts[-1]
    print(f"Loading checkpoint from: {latest_ckpt}")

    rngs = nnx.Rngs(0)
    model = model_cls(**model_kwargs, rngs=rngs)

    param_ref = nnx.state(model, nnx.Param)
    restored_params = checkpointer.restore(latest_ckpt, param_ref)

    nnx.update(model, restored_params)

    print("Checkpoint successfully restored!")
    return model
