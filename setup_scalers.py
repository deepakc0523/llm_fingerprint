"""
One-time setup script: Builds and saves scalers and embedding PCA(0.95)
from stored training data artifacts.

Run once from the Fingerprint project root:
    python setup_scalers.py

This must be run once before using the inference pipeline for the first time.
"""
import os
import re
import string
import numpy as np
import pandas as pd
import joblib
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.sparse import load_npz

BASE = "Colab"
FEATURES = os.path.join(BASE, "features")
ARTIFACTS = os.path.join(BASE, "scalers")

os.makedirs(ARTIFACTS, exist_ok=True)


# ============================================================
# 23-Feature Stylometric Extractor (matches Cell 52)
# ============================================================

def extract_23_stylometric(text: str) -> list:
    if not isinstance(text, str):
        text = ""
    words = re.findall(r"\b\w+\b", text.lower())
    sentences = re.split(r"[.!?]+", text)
    characters = len(text)
    word_count = len(words)
    sentence_count = max(1, len([s for s in sentences if s.strip()]))
    unique_words = len(set(words))
    avg_word_length = np.mean([len(w) for w in words]) if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count
    ttr = unique_words / word_count if word_count else 0
    hapax = len([w for w, c in Counter(words).items() if c == 1]) / word_count if word_count else 0
    uppercase_ratio = sum(c.isupper() for c in text) / max(characters, 1)
    digit_ratio = sum(c.isdigit() for c in text) / max(characters, 1)
    whitespace_ratio = sum(c.isspace() for c in text) / max(characters, 1)
    punctuation_ratio = sum(c in string.punctuation for c in text) / max(characters, 1)
    comma = text.count(",")
    period = text.count(".")
    semicolon = text.count(";")
    colon = text.count(":")
    question = text.count("?")
    exclamation = text.count("!")
    quotes = text.count('"')
    dash = text.count("-")
    parenthesis = text.count("(") + text.count(")")
    long_words = sum(len(w) >= 7 for w in words)
    short_words = sum(len(w) <= 3 for w in words)
    lexical_density = long_words / word_count if word_count else 0
    short_word_ratio = short_words / word_count if word_count else 0
    return [characters, word_count, sentence_count, unique_words, avg_word_length,
            avg_sentence_length, ttr, hapax, uppercase_ratio, digit_ratio,
            whitespace_ratio, punctuation_ratio, comma, period, semicolon, colon,
            question, exclamation, quotes, dash, parenthesis, lexical_density, short_word_ratio]


# ============================================================
# Load fingerprint training datasets (use model_name for labeling)
# ============================================================

print("Loading fingerprint datasets...")

model_dirs = {
    "gemma2": "data/synthetic/gemma2/generated.parquet",
    "llama3": "data/synthetic/llama3/generated.parquet",
    "mistral": "data/synthetic/mistral/generated.parquet",
    "phi3": "data/synthetic/phi3/generated.parquet",
    "qwen_tiny": "data/synthetic/qwen_tiny/generated.parquet"
}

dfs = []
for label, path in model_dirs.items():
    if os.path.exists(path):
        df = pd.read_parquet(path)
        df["model_label"] = label
        dfs.append(df[["generated_text", "model_label"]])
    else:
        print(f"  WARNING: {path} not found")

fingerprint_df = pd.concat(dfs, ignore_index=True)
print(f"Total fingerprint samples: {len(fingerprint_df)}")

# ============================================================
# Build Style Scaler (fit on raw 23-feature stylometric values)
# ============================================================

print("\nExtracting 23-feature stylometric arrays...")
X_style_raw = np.array(
    fingerprint_df["generated_text"].apply(extract_23_stylometric).tolist(),
    dtype=np.float64
)
print("Raw style shape:", X_style_raw.shape)

print("Fitting StandardScaler for stylometric features...")
style_scaler = StandardScaler()
style_scaler.fit(X_style_raw)

# ============================================================
# Build Embed Scaler and PCA(0.95)
# ============================================================

print("\nLoading fingerprint embeddings (all-MiniLM-L6-v2, 384 dims)...")
embed_fp = np.load(os.path.join(FEATURES, "fingerprint_embeddings.npy"))
print("Embeddings shape:", embed_fp.shape)

print("Fitting StandardScaler for embeddings...")
embed_scaler = StandardScaler()
embed_scaler.fit(embed_fp)

X_embed_scaled = embed_scaler.transform(embed_fp)

print("Fitting PCA(0.95) on scaled embeddings...")
embed_pca95 = PCA(n_components=0.95, random_state=42)
embed_pca95.fit(X_embed_scaled)

X_embed_244 = embed_pca95.transform(X_embed_scaled)
print(f"PCA(0.95) output dims: {X_embed_244.shape[1]}")

# ============================================================
# Verify match against saved fusion matrix
# ============================================================

print("\nVerifying reconstruction matches saved fingerprint_fusion.pkl...")

embed_pca = joblib.load(os.path.join(BASE, "fusion", "embed_pca.pkl"))
tfidf_svd = joblib.load(os.path.join(BASE, "fusion", "tfidf_svd.pkl"))
char_svd = joblib.load(os.path.join(BASE, "fusion", "char_svd.pkl"))

tfidf_fp = load_npz(os.path.join(FEATURES, "fingerprint_tfidf.npz"))
char_fp = load_npz(os.path.join(FEATURES, "fingerprint_char_ngrams.npz"))

X_fusion_saved = joblib.load(os.path.join(BASE, "fusion", "fingerprint_fusion.pkl"))

X_tfidf = tfidf_svd.transform(tfidf_fp) * 0.25
X_char = char_svd.transform(char_fp) * 0.25

# Only reconstruct from the 5238 fingerprint samples (may be in same order)
n_verify = X_fusion_saved.shape[0]
X_style_scaled = style_scaler.transform(X_style_raw[:n_verify]) * 0.25
X_embed_150 = embed_pca.transform(X_embed_244[:n_verify]) * 0.25

X_fused_check = np.hstack([X_tfidf, X_char, X_style_scaled, X_embed_150])
print(f"Reconstructed shape: {X_fused_check.shape}")
print(f"Saved shape: {X_fusion_saved.shape}")
print(f"Shapes match: {X_fused_check.shape == X_fusion_saved.shape}")
print(f"Values match (atol=1e-4): {np.allclose(X_fused_check, X_fusion_saved, atol=1e-4)}")

# ============================================================
# Save
# ============================================================

print("\nSaving artifacts to", ARTIFACTS)
joblib.dump(style_scaler, os.path.join(ARTIFACTS, "style_scaler.pkl"))
joblib.dump(embed_scaler, os.path.join(ARTIFACTS, "embed_scaler.pkl"))
joblib.dump(embed_pca95, os.path.join(ARTIFACTS, "embed_pca95.pkl"))

print("  style_scaler.pkl  -> 23 stylometric feature StandardScaler")
print("  embed_scaler.pkl  -> 384-dim embedding StandardScaler")
print("  embed_pca95.pkl   -> PCA(0.95) reducing 384->244 dims")
print("\nSetup complete.")
