import numpy as np
from typing import Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from .preprocessing import extract_stylometric_features


class FeaturePipeline:
    """
    Handles feature extraction from raw text using pre-trained vectorizers and embedding models.
    Mirrors the exact pipeline used in the research notebook (Cell 36 for embeddings,
    Cell 52 for stylometric, Cell 60 for fusion dimensionality reduction).
    """
    def __init__(
        self,
        tfidf_vectorizer: Any,
        char_vectorizer: Any,
        style_scaler: StandardScaler,
        embed_scaler: StandardScaler,
        embed_pca_95: PCA,
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        self.tfidf_vectorizer = tfidf_vectorizer
        self.char_vectorizer = char_vectorizer
        self.style_scaler = style_scaler
        self.embed_scaler = embed_scaler
        self.embed_pca_95 = embed_pca_95
        self._embed_model_name = embedding_model_name
        self._embed_model = None

    @property
    def embed_model(self) -> SentenceTransformer:
        """Lazy load sentence transformer embedding model to optimize startup time."""
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(self._embed_model_name)
        return self._embed_model

    def extract_features(self, text: str) -> Dict[str, np.ndarray]:
        """
        Extracts all feature representations for a single text:
        - TF-IDF sparse matrix (10k vocab)
        - Char N-Gram sparse matrix (15k 3-5 grams)
        - 23 Stylometric features (scaled)
        - 244-dim sentence embeddings (scaled + PCA(95%))
        """
        # 1. TF-IDF Features (sparse)
        tfidf_feat = self.tfidf_vectorizer.transform([text])

        # 2. Character N-Gram Features (sparse)
        char_feat = self.char_vectorizer.transform([text])

        # 3. Stylometric Features (23 dims), with StandardScaler applied
        style_raw = np.array([extract_stylometric_features(text)], dtype=np.float64)
        style_feat = self.style_scaler.transform(style_raw)

        # 4. Sentence Embeddings: encode (384) -> StandardScaler -> PCA(0.95, 244 dims)
        embed_raw = self.embed_model.encode([text], convert_to_numpy=True)
        embed_scaled = self.embed_scaler.transform(embed_raw)
        embed_feat = self.embed_pca_95.transform(embed_scaled)

        return {
            "tfidf": tfidf_feat,
            "char": char_feat,
            "style": style_feat,
            "embed": embed_feat
        }
