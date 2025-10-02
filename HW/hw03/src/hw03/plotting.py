import matplotlib
import matplotlib.pyplot as plt
import structlog

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)

def plot_losses(train_losses, val_losses, settings):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", alpha=0.7)
    steps_per_epoch = 50000 // settings.training.batch_size
    val_x = [i * steps_per_epoch for i in range(1, len(val_losses)+1)]
    plt.plot(val_x, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)

    settings.plotting.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.plotting.output_dir / "Loss Graph.pdf"
    plt.savefig(output_path)
    log.info("Saved Loss Graph plot", path=str(output_path))
