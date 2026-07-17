#!/usr/bin/env python3
"""
Main entry point and orchestrator CLI for the Fingerprint research project.
Wires preprocessing, feature engineering, model training, and evaluation steps together.
"""

import argparse
import os
import yaml
from typing import Dict, Any

# Package Imports
from preprocessing import TextCleaner, TextSplitter
from feature_engineering import StylometricExtractor, TransformerEmbedder
from models import ClassicalMLClassifier, TransformerClassifier, OpenSetDetector
from evaluation import Evaluator, ResultVisualizer


def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """
    Utility helper to load YAML configurations.

    Args:
        filepath: Absolute or relative path to the YAML file.

    Returns:
        Configuration dictionary.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_pipeline(configs: Dict[str, Dict[str, Any]], stage: str) -> None:
    """
    Orchestrates execution of individual stages or the entire pipeline.

    Args:
        configs: Combined dictionary of datasets, models, and generation configs.
        stage: Specific pipeline stage name or "all" to run the entire flow.
    """
    dataset_conf = configs["datasets"]
    model_conf = configs["models"]

    print(f"[*] Starting Fingerprint pipeline (Target Stage: {stage})...")

    # 1. Preprocessing Stage
    cleaner = TextCleaner(dataset_conf)
    splitter = TextSplitter(dataset_conf)

    if stage in ("preprocessing", "all"):
        print("[+] Phase 1: Running text preprocessing (cleaning & chunking)...")
        # TODO: Load raw text files from dataset_conf["data_paths"]["raw_dir"]
        # TODO: Apply cleaner.clean_dataframe and splitter.process_dataset
        # TODO: Serialize outputs to dataset_conf["data_paths"]["cleaned_dir"]

    # 2. Feature Engineering Stage
    stylometry = StylometricExtractor(model_conf)
    embedder = TransformerEmbedder(model_conf)

    if stage in ("feature_engineering", "all"):
        print("[+] Phase 2: Running feature extraction...")
        # TODO: Load cleaned chunks from cleaned_dir
        # TODO: Extract stylometric features and transformer embeddings
        # TODO: Save feature matrices to features_dir

    # 3. Model Training & Validation Stage
    classifier = ClassicalMLClassifier(model_conf)
    transformer = TransformerClassifier(model_conf)
    open_set = OpenSetDetector(model_conf)

    if stage in ("modeling", "all"):
        print("[+] Phase 3: Fitting attribution classifiers and open-set detectors...")
        # TODO: Load master training datasets from master_dir
        # TODO: Fit classifiers on known generators
        # TODO: Fit open_set anomaly detector on known generator representations
        # TODO: Save trained model checkpoits to models directory

    # 4. Evaluation Stage
    evaluator = Evaluator(model_conf)
    visualizer = ResultVisualizer(model_conf)

    if stage in ("evaluation", "all"):
        print("[+] Phase 4: Scoring models and generating visual reports...")
        # TODO: Perform predictions on validation/test subsets
        # TODO: Run evaluator.evaluate_closed_set and evaluate_open_set
        # TODO: Invoke visualizer.plot_confusion_matrix, plot_roc_curve, and plot_tsne
        # TODO: Write final markdown summary report to reports directory


def main() -> None:
    """Parses CLI args and runs the appropriate pipeline flow."""
    parser = argparse.ArgumentParser(
        description="Fingerprint: LLM Authorship Attribution & Stylometric Fingerprinting Orchestrator"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Path to directory containing configs (default: configs)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["preprocessing", "feature_engineering", "modeling", "evaluation", "all"],
        help="Pipeline stage to execute (default: all)",
    )

    args = parser.parse_args()

    # Load all configuration files
    try:
        configs = {
            "datasets": load_yaml_config(os.path.join(args.config_dir, "datasets.yaml")),
            "models": load_yaml_config(os.path.join(args.config_dir, "models.yaml")),
            "generation": load_yaml_config(os.path.join(args.config_dir, "generation.yaml")),
        }
    except Exception as e:
        print(f"[!] Error loading configurations: {e}")
        return

    # Run selected pipeline stage
    try:
        run_pipeline(configs, args.stage)
        print("[*] Pipeline execution finished successfully.")
    except Exception as e:
        print(f"[!] Pipeline execution failed: {e}")


if __name__ == "__main__":
    main()
