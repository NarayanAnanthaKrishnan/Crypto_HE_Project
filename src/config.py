"""
Configuration Module for FHE ML Project

Centralizes all configuration parameters for reproducibility
and easy experimentation.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import yaml


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_path: str = "data/diabetes.csv"
    test_size: float = 0.2
    random_state: int = 42
    drop_columns: List[str] = field(default_factory=lambda: ["Id", "id", "index"])
    target_column: str = "Outcome"
    standardize: bool = True


@dataclass
class ModelConfig:
    """Configuration for model training."""
    logreg_max_iter: int = 1000
    logreg_solver: str = "lbfgs"
    rf_n_estimators: int = 200
    rf_random_state: int = 42
    rf_n_jobs: int = -1


@dataclass
class FHEConfig:
    """Configuration for FHE parameters."""
    preset: str = "medium"
    custom_n: Optional[int] = None
    custom_qi_sizes: Optional[List[int]] = None
    custom_scale: Optional[int] = None
    
    # Predefined presets
    PRESETS: Dict[str, Dict] = field(default_factory=lambda: {
        "small": {"n": 8192, "qi_sizes": [60, 40, 60], "scale": 2**30},
        "medium": {"n": 16384, "qi_sizes": [60, 40, 40, 60], "scale": 2**30},
        "large": {"n": 32768, "qi_sizes": [60, 40, 40, 40, 60], "scale": 2**40},
    })
    
    def get_params(self) -> Dict[str, Any]:
        """Get the active FHE parameters."""
        if self.custom_n is not None:
            return {
                "n": self.custom_n,
                "qi_sizes": self.custom_qi_sizes or [60, 40, 40, 60],
                "scale": self.custom_scale or 2**30,
            }
        return self.PRESETS[self.preset]


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    n_fhe_samples: int = 50
    random_state: int = 42
    parameter_sweep_presets: List[str] = field(default_factory=lambda: ["small", "medium", "large"])
    verbose: bool = True
    save_figures: bool = True
    figures_dir: str = "figures"
    results_dir: str = "results"


@dataclass
class ProjectConfig:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fhe: FHEConfig = field(default_factory=FHEConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        config_dict = {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "fhe": {k: v for k, v in self.fhe.__dict__.items() if k != "PRESETS"},
            "experiment": self.experiment.__dict__,
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ProjectConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            fhe=FHEConfig(**config_dict.get("fhe", {})),
            experiment=ExperimentConfig(**config_dict.get("experiment", {})),
        )


# Default configuration instance
DEFAULT_CONFIG = ProjectConfig()


def get_default_config() -> ProjectConfig:
    """Get a fresh copy of the default configuration."""
    return ProjectConfig()


def create_experiment_config(
    preset: str = "medium",
    n_samples: int = 50,
    save_figures: bool = True
) -> ProjectConfig:
    """
    Create a configuration for a specific experiment.
    
    Args:
        preset: FHE parameter preset
        n_samples: Number of samples for FHE evaluation
        save_figures: Whether to save figures
        
    Returns:
        Configured ProjectConfig
    """
    config = ProjectConfig()
    config.fhe.preset = preset
    config.experiment.n_fhe_samples = n_samples
    config.experiment.save_figures = save_figures
    return config