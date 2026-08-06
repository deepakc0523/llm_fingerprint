import joblib
import numpy as np
import os
import re
import string
from sentence_transformers import SentenceTransformer

tfidf_vec = joblib.load('Colab/features/tfidf_vectorizer.pkl')
char_vec = joblib.load('Colab/features/char_vectorizer.pkl')
tfidf_svd = joblib.load('Colab/fusion/tfidf_svd.pkl')
char_svd = joblib.load('Colab/fusion/char_svd.pkl')
embed_pca = joblib.load('Colab/fusion/embed_pca.pkl')

embed_model = SentenceTransformer('all-MiniLM-L6-v2')
stack_model = joblib.load('Colab/models/stacking/Hybrid_Stacked_Ensemble.pkl')

sample_text = "This is a sample generated text from LLM to test the end to end inference pipeline."

def stylometric_features(text):
    text = str(text)
    words = text.split()
    chars = len(text)
    return [
        len(words),
        chars,
        np.mean([len(w) for w in words]) if words else 0,
        len(set(words))/len(words) if words else 0,
        sum(c.isupper() for c in text),
        sum(c.isdigit() for c in text),
        sum(c in string.punctuation for c in text),
        text.count(','),
        text.count('.'),
        text.count('?'),
        text.count('!'),
        len(re.findall(r'\n', text)),
        text.count('"'),
        len(re.findall(r'\s', text))
    ]

X_tfidf = tfidf_vec.transform([sample_text])
X_char = char_vec.transform([sample_text])
X_style = np.array([stylometric_features(sample_text)])
X_embed = embed_model.encode([sample_text], convert_to_numpy=True)

X_tfidf_red = tfidf_svd.transform(X_tfidf) * 0.25
X_char_red = char_svd.transform(X_char) * 0.25
X_style_scaled = X_style * 0.25
X_embed_red = embed_pca.transform(X_embed) * 0.25

X_fusion = np.hstack([X_tfidf_red, X_char_red, X_style_scaled, X_embed_red])

pred = stack_model.predict(X_fusion)[0]
probs = stack_model.predict_proba(X_fusion)[0]

print('Prediction:', pred)
print('Classes:', stack_model.classes_)
print('Probabilities:', dict(zip(stack_model.classes_, probs)))
