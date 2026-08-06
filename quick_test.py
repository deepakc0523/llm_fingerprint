import joblib
import numpy as np
import re
import string

tfidf_vec = joblib.load('Colab/features/tfidf_vectorizer.pkl')
char_vec = joblib.load('Colab/features/char_vectorizer.pkl')
tfidf_svd = joblib.load('Colab/fusion/tfidf_svd.pkl')
char_svd = joblib.load('Colab/fusion/char_svd.pkl')
embed_pca = joblib.load('Colab/fusion/embed_pca.pkl')
stack_model = joblib.load('Colab/models/stacking/Hybrid_Stacked_Ensemble.pkl')

text = "Sample text for testing."

def stylometric_features(t):
    words = str(t).split()
    return [
        len(words),
        len(str(t)),
        np.mean([len(w) for w in words]) if words else 0,
        len(set(words))/len(words) if words else 0,
        sum(c.isupper() for c in t),
        sum(c.isdigit() for c in t),
        sum(c in string.punctuation for c in t),
        t.count(','),
        t.count('.'),
        t.count('?'),
        t.count('!'),
        len(re.findall(r'\n', t)),
        t.count('"'),
        len(re.findall(r'\s', t))
    ]

X_tfidf = tfidf_svd.transform(tfidf_vec.transform([text])) * 0.25
X_char = char_svd.transform(char_vec.transform([text])) * 0.25
X_style = np.array([stylometric_features(text)]) * 0.25
X_embed = embed_pca.transform(np.zeros((1, embed_pca.n_features_in_))) * 0.25

X_fusion = np.hstack([X_tfidf, X_char, X_style, X_embed])
print('Shape:', X_fusion.shape)
print('Predict:', stack_model.predict(X_fusion)[0])
print('Probs:', dict(zip(stack_model.classes_, stack_model.predict_proba(X_fusion)[0])))
