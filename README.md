# LLM Fingerprinting System

Production-quality inference pipeline and Streamlit application converted from the completed LLM Fingerprinting research notebook (`notebooks/Machine_learning.ipynb`).

## System Architecture

```
Input Text
    │
    ▼
Fingerprint Preserving Preprocessing
    │
    ├───────────────────────┬───────────────────────┬───────────────────────┐
    ▼                       ▼                       ▼                       ▼
TF-IDF Transform       Char N-Gram Transform    Stylometric Features    Sentence Embeddings
(10k Vocabulary)       (15k 3-5 N-Grams)        (14 Features)           (all-MiniLM-L6-v2)
    │                       │                       │                       │
    ▼                       ▼                       │                       ▼
TruncatedSVD (300D)    TruncatedSVD (300D)          │                   PCA (150D)
    │                       │                       │                       │
    └───────────────────────┴───────────┬───────────┴───────────────────────┘
                                        ▼
                               Scaled Feature Fusion
                                    (773 Dims)
                                        │
                                        ▼
                             Hybrid Stacked Ensemble
                           (LR + Calibrated SVM + RF + XGB)
                                        │
                                        ▼
                                 Class Prediction
                             Confidence & Probabilities
```

## Directory Structure

```
Fingerprint/
├── inference/
│   ├── predictor.py        # Core Predictor class
│   ├── preprocessing.py    # Text & stylometric preprocessing
│   ├── feature_pipeline.py # Feature extraction pipeline
│   ├── fusion_pipeline.py  # SVD/PCA reduction & fusion
│   ├── model_loader.py     # Model loading & XGBoost patch
│   └── utils.py            # Logging & label formatting
├── app/
│   └── streamlit_app.py    # Streamlit dashboard
├── notebooks/
│   └── Machine_learning.ipynb # Original research notebook
├── Colab/ (or models/, features/, fusion/) # Pre-trained research artifacts
├── requirements.txt
└── README.md
```

## Usage Instructions

### Python API

```python
from inference.predictor import LLMFingerprintPredictor

predictor = LLMFingerprintPredictor()

text = "In this paper, we explore the capabilities of large language models..."
result = predictor.predict(text)

print(f"Predicted Model: {result['predicted_model']}")
print(f"Confidence: {result['confidence']:.2f}%")
print(f"Probabilities: {result['probabilities']}")
print(f"Inference Time: {result['processing_time']:.2f}s")
```

### Streamlit Web App

To launch the Streamlit web dashboard:

```bash
streamlit run app/streamlit_app.py
```

## Supported Target Models

1. **Gemma** (`gemma2`)
2. **Llama** (`llama3`)
3. **Mistral** (`mistral`)
4. **Phi** (`phi3`)
5. **Qwen** (`qwen_tiny`)
