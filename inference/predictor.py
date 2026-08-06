import os
import time
import numpy as np
from typing import Dict, Any

from .model_loader import load_stacking_model, load_artifact
from .preprocessing import preprocess_text
from .feature_pipeline import FeaturePipeline
from .fusion_pipeline import FusionPipeline
from .utils import setup_logger, format_class_name

logger = setup_logger("Predictor")


class LLMFingerprintPredictor:
    """
    Production-quality LLM Fingerprinting Predictor class.
    Executes the exact same pipeline used during training in the research notebook.
    """

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            if os.path.exists("Colab/models"):
                base_dir = "Colab"
            else:
                base_dir = "."

        self.base_dir = base_dir
        logger.info(f"Initializing LLMFingerprintPredictor with base directory: {base_dir}")

        features_dir = os.path.join(base_dir, "features")
        fusion_dir = os.path.join(base_dir, "fusion")
        scalers_dir = os.path.join(base_dir, "scalers")
        model_path = os.path.join(base_dir, "models", "stacking", "Hybrid_Stacked_Ensemble.pkl")

        logger.info("Loading pre-trained vectorizers, scalers, and dimensionality reducers...")

        # Vectorizers
        tfidf_vec = load_artifact(os.path.join(features_dir, "tfidf_vectorizer.pkl"))
        char_vec = load_artifact(os.path.join(features_dir, "char_vectorizer.pkl"))

        # Dimensionality reducers (fusion stage)
        tfidf_svd = load_artifact(os.path.join(fusion_dir, "tfidf_svd.pkl"))
        char_svd = load_artifact(os.path.join(fusion_dir, "char_svd.pkl"))
        embed_pca = load_artifact(os.path.join(fusion_dir, "embed_pca.pkl"))

        # Scalers (Cell 52 / Cell 60 — built by setup_scalers.py)
        style_scaler = load_artifact(os.path.join(scalers_dir, "style_scaler.pkl"))
        embed_scaler = load_artifact(os.path.join(scalers_dir, "embed_scaler.pkl"))
        embed_pca95 = load_artifact(os.path.join(scalers_dir, "embed_pca95.pkl"))

        # Ensemble model
        self.stack_model = load_stacking_model(model_path)

        # Initialize sub-pipelines
        self.feature_pipeline = FeaturePipeline(
            tfidf_vectorizer=tfidf_vec,
            char_vectorizer=char_vec,
            style_scaler=style_scaler,
            embed_scaler=embed_scaler,
            embed_pca_95=embed_pca95,
            embedding_model_name="all-MiniLM-L6-v2"
        )
        self.fusion_pipeline = FusionPipeline(
            tfidf_svd=tfidf_svd,
            char_svd=char_svd,
            embed_pca=embed_pca
        )

        self.classes_ = self.stack_model.classes_
        logger.info(f"Predictor ready. Supported classes: {list(self.classes_)}")

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Executes the full inference pipeline on the input text.

        Args:
            text: Raw AI-generated text to fingerprint.

        Returns:
            Dict containing:
                - predicted_model: str (Display name of predicted LLM)
                - raw_label: str (Internal label from training)
                - confidence: float (Percentage, 0-100)
                - probabilities: Dict[str, float] (Probability for each class, %)
                - processing_time: float (Inference duration in seconds)
        """
        start_time = time.time()

        if text is None or not isinstance(text, str) or not text.strip():
            raise ValueError("Input text cannot be empty or blank.")

        cleaned_text = preprocess_text(text)

        # Steps 1-4: Feature Extraction (TF-IDF, Char, Stylometric, Embeddings)
        raw_features = self.feature_pipeline.extract_features(cleaned_text)

        # Steps 5-8: Dimensionality Reduction + Equal Weighting + Fusion (773 dims)
        fusion_vector = self.fusion_pipeline.transform(raw_features)

        # Steps 9-11: Stacked Ensemble Prediction + Probabilities
        raw_prediction = self.stack_model.predict(fusion_vector)[0]
        proba_array = self.stack_model.predict_proba(fusion_vector)[0]

        confidence = float(np.max(proba_array) * 100.0)
        proc_time = float(time.time() - start_time)

        class_probs_raw = dict(zip(self.classes_, proba_array))

        display_probabilities = {
            "Gemma":   float(class_probs_raw.get("gemma2", 0.0) * 100.0),
            "Llama":   float(class_probs_raw.get("llama3", 0.0) * 100.0),
            "Mistral": float(class_probs_raw.get("mistral", 0.0) * 100.0),
            "Phi":     float(class_probs_raw.get("phi3", 0.0) * 100.0),
            "Qwen":    float(class_probs_raw.get("qwen_tiny", 0.0) * 100.0)
        }

        predicted_display = format_class_name(raw_prediction)

        return {
            "predicted_model": predicted_display,
            "raw_label": raw_prediction,
            "confidence": confidence,
            "probabilities": display_probabilities,
            "processing_time": proc_time
        }
