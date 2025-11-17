from pathlib import Path
from typing import Tuple
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ModelSettings(BaseModel): 
    num_classes: int = 4
    embed_dim: int = 128
    conv_features: int = 64
    kernel_size: int = 5
    dropout_rate: float = 0.3


class DataSettings(BaseModel):
    validation_percentage: float = 0.1
    batch_size: int = 256
    max_length: int = 128


class TrainingSettings(BaseModel):
    """Settings for model training."""
    num_epochs: int = 7
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4                   # L2 penalty
    log_interval: int = 100  
    final_test: bool = True
    checkpoint_dir: Path = Path("hw05/checkpoints")                 


class PlottingSettings(BaseModel):
    """Settings for logging and saving artifacts."""
    output_dir: Path = Path("hw05/artifacts")
    figsize: Tuple[int, int] = (5, 3)
    dpi: int = 200


class AppSettings(BaseSettings):
    """Main application settings."""
    debug: bool = False
    random_seed: int = 31415

    model: ModelSettings = ModelSettings()
    data: DataSettings = DataSettings()
    training: TrainingSettings = TrainingSettings()
    plotting: PlottingSettings = PlottingSettings()


def load_settings() -> AppSettings:
    """Load application settings."""
    return AppSettings()
