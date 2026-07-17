# Data Registry & Stages

This directory follows a stage-by-stage data transformation pattern for reproducibility in ML/NLP research pipelines.

## Subdirectories

- [raw/](file:///d:/Fingerprint/data/raw): Initial dataset sources (e.g. human-written corpuses).
- [prefixes/](file:///d:/Fingerprint/data/prefixes): Prompts or text prefixes used for initiating generation.
- [generated/](file:///d:/Fingerprint/data/generated): Raw generated text responses straight from the language models.
- [cleaned/](file:///d:/Fingerprint/data/cleaned): Normalized texts with HTML tags, formatting artifacts, and metadata stripped.
- [features/](file:///d:/Fingerprint/data/features): Serialized computational features (stylometrics, tokenizer states, embedding matrices).
- [master/](file:///d:/Fingerprint/data/master): Combined, aligned and split training, validation, and test datasets.
