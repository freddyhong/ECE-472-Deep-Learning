import matplotlib
import matplotlib.pyplot as plt
import jax
import numpy as np
import structlog
from sklearn.inspection import DecisionBoundaryDisplay

from .config import PlottingSettings
from .data import SpiralData
from .model import NNXMLP

log = structlog.get_logger()

font = {
    # "family": "Adobe Caslon Pro",
    "size": 10,
}

matplotlib.style.use("classic")
matplotlib.rc("font", **font)


def plot_spiral(model: NNXMLP, data: SpiralData, settings: PlottingSettings):
    X, y = data.x, data.y

    # Creating a mesh grid to plot on
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 600),
        np.linspace(y_min, y_max, 600),
    )

    # Get model predictions
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Get the raw logit output
    logits = model(grid_points)

    # Changing to probability with sigmoid
    preds = (jax.nn.sigmoid(logits) > 0.5).astype(int)
    response = np.array(preds).reshape(xx.shape)

    disp = DecisionBoundaryDisplay(
        xx0=xx,
        xx1=yy,
        response=response,
    )
    disp.plot(ax=plt.gca(), cmap=plt.cm.coolwarm, alpha=0.6)

    plt.contour(xx, yy, response, levels=[0.5], colors="k", linewidths=1, alpha=0.6)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")

    plt.title("MLP Decision Boundary on Spirals")
    plt.xlabel("X")
    plt.ylabel("Y")

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / "decision_boundary.pdf"
    plt.savefig(output_path)
    log.info("Saved decision boundary plot", path=str(output_path))
