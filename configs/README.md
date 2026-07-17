# Configuration Management

This directory contains YAML files specifying all hyperparameters, data configurations, and generation templates for the Fingerprint pipeline.

## Configuration Files

- [datasets.yaml](file:///d:/Fingerprint/configs/datasets.yaml): Configures directory paths, dataset splitting thresholds, and preprocessing settings.
- [models.yaml](file:///d:/Fingerprint/configs/models.yaml): Specifies the feature extraction techniques (stylometrics vs embeddings) and settings for training the classical classifiers, neural sequence classifiers, and open-set anomaly detectors.
- [generation.yaml](file:///d:/Fingerprint/configs/generation.yaml): Configurations for invoking LLM APIs to generate dataset samples.

## Loading Configurations

In python modules, you can load these configurations using `pyyaml`:

```python
import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
```
