import numpy as np
from typing import Dict, Any


class FusionPipeline:
    """
    Handles dimensionality reduction via saved TruncatedSVD and PCA,
    equal weighting (×0.25) and horizontal concatenation to produce the
    773-dimensional hybrid feature vector expected by the Stacked Ensemble.

    Pipeline (matching Cell 60 of research notebook):
        TF-IDF (sparse)  -> TruncatedSVD(300) -> ×0.25 -> 300 dims
        Char (sparse)    -> TruncatedSVD(300) -> ×0.25 -> 300 dims
        Style (23 scaled)                     -> ×0.25 ->  23 dims
        Embed (244 scaled+pca95) -> PCA(150)  -> ×0.25 -> 150 dims
        Total: 300 + 300 + 23 + 150 = 773
    """
    def __init__(
        self,
        tfidf_svd: Any,
        char_svd: Any,
        embed_pca: Any
    ):
        self.tfidf_svd = tfidf_svd
        self.char_svd = char_svd
        self.embed_pca = embed_pca

    def transform(self, raw_features: Dict[str, Any]) -> np.ndarray:
        """
        Transforms pre-extracted features into the 773-dim fusion vector.
        """
        # 1. TF-IDF SVD Reduction (300 dims) + 0.25 Weighting
        tfidf_red = self.tfidf_svd.transform(raw_features["tfidf"]) * 0.25

        # 2. Character N-Gram SVD Reduction (300 dims) + 0.25 Weighting
        char_red = self.char_svd.transform(raw_features["char"]) * 0.25

        # 3. Stylometric (23 dims, already scaled) + 0.25 Weighting
        style_scaled = raw_features["style"] * 0.25

        # 4. Semantic Embedding PCA Reduction (150 dims) + 0.25 Weighting
        embed_red = self.embed_pca.transform(raw_features["embed"]) * 0.25

        # 5. Horizontal Concatenation -> 773 dims
        fusion_matrix = np.hstack([
            tfidf_red,
            char_red,
            style_scaled,
            embed_red
        ])

        return fusion_matrix
