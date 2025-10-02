import jax
import jax.numpy as jnp
import optax
from flax import nnx
import matplotlib.pyplot as plt
import structlog

from .config import load_settings

log = structlog.get_logger()

def cross_entropy_loss(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(
        logits, labels
    ).mean()


def compute_accuracy(logits, labels):
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels)


def _loss_fn(model, batch, rngs):
    images, labels = batch
    labels = labels.ravel()
    logits = model(images, rngs=rngs)
    loss = cross_entropy_loss(logits, labels)
    return loss, logits


@nnx.jit
def train_step(model, optimizer, batch, rngs):
    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model, batch, rngs)
    optimizer.update(model, grads)
    return loss, logits


@nnx.jit
def eval_step(model, batch, rngs):
    loss, logits = _loss_fn(model, batch, rngs)
    return loss, logits

def evaluate(model, dataset, rngs):
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    for batch in dataset.as_numpy_iterator():
        loss, logits = eval_step(model, batch, rngs)
        _, labels = batch
        labels = labels.ravel()
        acc = compute_accuracy(logits, labels)
        loss = float(loss)
        acc = float(acc)
        total_loss += loss
        total_acc += float(acc)
        n_batches += 1
    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)

def train(model, optimizer, train_ds, val_ds, rngs):
    settings = load_settings()
    output_dir = settings.plotting.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

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

    for epoch in range(1, num_epochs + 1):

        for batch in train_ds.as_numpy_iterator():
            loss, logits = train_step(model, optimizer, batch, rngs)

            if global_step % log_interval == 0:
                images, labels = batch
                labels = labels.ravel()
                acc = compute_accuracy(logits, labels)
                
                loss_val = float(loss)
                acc_val = float(acc)

                metrics_history["train_steps"].append(global_step)
                metrics_history["train_loss"].append(loss_val)
                metrics_history["train_accuracy"].append(acc_val)
                
                log.info(
                    "train_iter",
                    epoch=epoch,
                    step=global_step,
                    loss=loss_val,
                    acc=acc_val,
                )
            global_step += 1

   
        val_loss, val_acc = evaluate(model, val_ds, rngs)
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
       
    log.info("Training finished. Generating final plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.set_title("Loss vs. Steps")
    ax2.set_title("Accuracy vs. Steps")

    ax1.plot(metrics_history["train_steps"], metrics_history["train_loss"], label="train", alpha=0.7)
    ax2.plot(metrics_history["train_steps"], metrics_history["train_accuracy"], label="train", alpha=0.7)

    ax1.plot(metrics_history["val_steps"], metrics_history["val_loss"], label="val", marker='o', linestyle='--')
    ax2.plot(metrics_history["val_steps"], metrics_history["val_accuracy"], label="val", marker='o', linestyle='--')

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_dir / "final_training_metrics.png", dpi=settings.plotting.dpi)
    plt.close(fig)

    return metrics_history

def test_evaluation(model, test_ds, rngs):
    settings = load_settings()
    output_dir = settings.plotting.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    test_loss, test_acc = evaluate(model, test_ds, rngs)
    log.info("test_results", test_loss=test_loss, test_acc=test_acc)

    test_batch = next(test_ds.as_numpy_iterator())
    _, logits = eval_step(model, test_batch, rngs)
    images, labels = test_batch
    preds = jnp.argmax(logits, axis=-1)

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(images[i, ..., 0])
        ax.set_title(f"pred={int(preds[i])}, label={int(labels[i])}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "test_predictions.png", dpi=settings.plotting.dpi)
    plt.close(fig)

    return test_loss, test_acc
