# Data Preprocessing

This directory contains modules to clean and split/chunk raw text datasets before feature extraction and modeling.

## Modules

- [cleaner.py](file:///d:/Fingerprint/preprocessing/cleaner.py): Logic to remove noise, HTML tags, normalize unicode characters, and handle capitalization.
- [splitter.py](file:///d:/Fingerprint/preprocessing/splitter.py): Mechanics to break large text files into smaller chunks of words/sentences with sliding window overlaps to maintain consistent context sizes for authorship detection.
