import matplotlib
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import numpy as np
import structlog

from .config import PlottingSettings
from .data import Data
from .model import TrueSineModel, NNXBasisExpansionModel

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)


def compare_models(
    true_model: TrueSineModel,
    est_model: NNXBasisExpansionModel,
    data: Data,
    settings: PlottingSettings,
):
    """Plots true sine function vs estimated regression model."""
    xs = np.linspace(0, 1, 200)

    # true sine
    ys_true = true_model(xs)

    # estimated model
    ys_est = est_model(jnp.asarray(xs[:, None]))

    fig, ax = plt.subplots(1, 1, figsize=settings.figsize, dpi=settings.dpi)
    ax.set_title("Noisy data, true sine, and estimated fit")
    ax.plot(xs, ys_true, label="True sine")
    ax.plot(xs, ys_est, label="Estimated fit")
    ax.plot(data.x.squeeze(), data.y, "o", label="Noisy samples", alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "compare.pdf"
    plt.savefig(output_path)
    log.info("Saved comparison plot", path=str(output_path))


def plot_basis_functions(model: NNXBasisExpansionModel, settings: PlottingSettings):
    xs = np.linspace(0, 1, 200)
    phi_matrix = []
    fig, ax = plt.subplots(figsize=settings.figsize, dpi=settings.dpi)
    mu_sig = jax.nn.sigmoid(model.mu.value)
    for mu, sigma in zip(mu_sig, model.sigma.value):
        mu = float(mu.item())
        phi = np.exp(-((xs - mu) ** 2) / (sigma**2))
        ax.plot(xs, phi, label=f"Basis function with mu = {mu:.2f}")
        phi_matrix.append(phi)

    ax.set_title("Gaussian basis functions")
    ax.set_xlabel("x")
    ax.set_ylabel("Φ(x)")
    ax.legend(
        title="Basis functions",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
    )

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(settings.output_dir / "basis_functions.pdf", bbox_inches="tight")
