from pathlib import Path
from typing import Tuple
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class ModelSettings(BaseModel):
    input_depth: int = 3                        
    layer_depths: list[int] = [32, 64, 128, 256]     
    blocks_per_stage: list[int] = [3, 4, 6, 3]     
    num_classes: int = 100


class DataSettings(BaseModel):
    batch_size: int = 64
    validation_size: int = 5000
    shuffle_buffer: int = 10000
    cifar100: bool = True


class TrainingSettings(BaseModel):
    """Settings for model training."""
    num_epochs: int = 40
    learning_rate: float = 0.001
    weight_decay: float = 1e-4                   # L2 penalty
    log_interval: int = 100  
    final_test: bool = True
    checkpoint_dir: Path = Path("hw04/checkpoints")                 


class PlottingSettings(BaseModel):
    """Settings for logging and saving artifacts."""
    output_dir: Path = Path("hw04/artifacts")
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
