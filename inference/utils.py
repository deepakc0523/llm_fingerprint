import logging

def setup_logger(name: str = "llm_fingerprint") -> logging.Logger:
    """Configures and returns standard logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

LABEL_MAPPING = {
    "gemma2": "Gemma",
    "llama3": "Llama",
    "mistral": "Mistral",
    "phi3": "Phi",
    "qwen_tiny": "Qwen"
}

def format_class_name(label: str) -> str:
    """Formats internal model labels into clean display names."""
    return LABEL_MAPPING.get(label.strip(), label.title())
