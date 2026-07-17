# Authorship Attribution & Open-set Models

This directory defines model wrappers, training scripts, and evaluation endpoints for known source classification and unknown generator detection.

## Modules

- [classical_ml.py](file:///d:/Fingerprint/models/classical_ml.py): Scikit-learn/XGBoost classification models fitted on stylometric tabular data.
- [transformer_classifier.py](file:///d:/Fingerprint/models/transformer_classifier.py): Fine-tuning framework for sequence classification transformers (e.g. RoBERTa).
- [open_set_detector.py](file:///d:/Fingerprint/models/open_set_detector.py): Multi-class attribution + out-of-distribution (OOD) modeling to verify whether a piece of text was created by an unknown LLM.
