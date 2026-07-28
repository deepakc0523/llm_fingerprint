"""Transformer Baseline — DistilBERT Fine-Tuning Pipeline.

Project : Fingerprint — LLM Fingerprinting Framework
Stage   : Transformer Baseline
Description:
    Fine-tunes DistilBERT for multi-class LLM source classification.
    Provides a deep learning baseline for comparison with classical ML models.
    Uses the HuggingFace Transformers library.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TransformerTrainer:
    """DistilBERT fine-tuning trainer for LLM Fingerprinting classification.

    Wraps HuggingFace Trainer API with a simplified interface that
    matches the classical ML workflow used in this project.

    Attributes:
        cfg: Transformer sub-config dict from training.yaml.
        model_name: HuggingFace model identifier.
        num_labels: Number of target classes.
        label_names: List of class name strings.
        tokenizer: Loaded HuggingFace tokenizer.
        model: Loaded HuggingFace sequence classification model.
        trainer: HuggingFace Trainer instance (populated after build_trainer).
        train_time_: Wall-clock training time in seconds.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        num_labels: int,
        label_names: List[str],
    ) -> None:
        """Initialise TransformerTrainer.

        Args:
            cfg: The ``transformer`` section from training.yaml.
            num_labels: Number of target LLM classes.
            label_names: Ordered list of class name strings.
        """
        self.cfg = cfg
        self.model_name: str = cfg.get("model_name", "distilbert-base-uncased")
        self.num_labels: int = num_labels
        self.label_names: List[str] = label_names
        self.max_length: int = int(cfg.get("max_length", 256))
        self.batch_size: int = int(cfg.get("batch_size", 16))
        self.num_epochs: int = int(cfg.get("num_epochs", 3))
        self.learning_rate: float = float(cfg.get("learning_rate", 2e-5))
        self.weight_decay: float = float(cfg.get("weight_decay", 0.01))
        self.warmup_ratio: float = float(cfg.get("warmup_ratio", 0.1))
        self.output_dir: str = cfg.get("output_dir", "models/distilbert_fingerprint")
        self.device: str = cfg.get("device", "cpu")
        self.tokenizer: Optional[Any] = None
        self.model: Optional[Any] = None
        self.trainer: Optional[Any] = None
        self.train_time_: float = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def load_tokenizer(self) -> Any:
        """Load the HuggingFace tokenizer.

        Returns:
            Loaded tokenizer instance.
        """
        from transformers import AutoTokenizer

        logger.info("Loading tokenizer: %s ...", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info("Tokenizer loaded.")
        return self.tokenizer

    def load_model(self) -> Any:
        """Load the HuggingFace sequence classification model.

        Returns:
            Loaded DistilBERT for sequence classification.
        """
        from transformers import AutoModelForSequenceClassification

        logger.info(
            "Loading model: %s  (num_labels=%d) ...",
            self.model_name, self.num_labels,
        )
        id2label = {i: name for i, name in enumerate(self.label_names)}
        label2id = {name: i for i, name in enumerate(self.label_names)}

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        logger.info("Model loaded.")
        return self.model

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize(self, texts: List[str], labels: Optional[List[int]] = None) -> Any:
        """Tokenise a list of text strings into a HuggingFace Dataset.

        Args:
            texts: List of raw text strings.
            labels: Optional list of integer class labels.

        Returns:
            HuggingFace Dataset object with tokenised inputs.
        """
        from datasets import Dataset as HFDataset

        logger.info("Tokenising %d texts (max_length=%d) ...", len(texts), self.max_length)

        data_dict = {"text": texts}
        if labels is not None:
            data_dict["label"] = labels

        dataset = HFDataset.from_dict(data_dict)

        def _tokenize_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )

        tokenized = dataset.map(
            _tokenize_fn,
            batched=True,
            remove_columns=["text"],
        )
        return tokenized

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def build_trainer(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
    ) -> Any:
        """Build a HuggingFace Trainer with this project's config.

        Args:
            train_dataset: Tokenised HuggingFace training Dataset.
            eval_dataset: Optional tokenised evaluation Dataset.

        Returns:
            Configured HuggingFace Trainer instance.
        """
        from transformers import TrainingArguments, Trainer
        import evaluate

        logger.info("Building HuggingFace Trainer ...")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size * 2,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            eval_strategy="epoch" if eval_dataset is not None else "no",
            save_strategy="epoch",
            load_best_model_at_end=eval_dataset is not None,
            metric_for_best_model="f1",
            logging_steps=50,
            report_to="none",
            no_cuda=(self.device == "cpu"),
        )

        metric_fn = evaluate.load("f1")

        def compute_metrics(eval_pred: Any) -> Dict[str, float]:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric_fn.compute(
                predictions=predictions,
                references=labels,
                average="macro",
            )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        return self.trainer

    def train(self) -> Any:
        """Execute the training loop.

        Returns:
            HuggingFace TrainOutput object.

        Raises:
            RuntimeError: If trainer has not been built.
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not built. Call build_trainer() first.")
        logger.info("Starting DistilBERT training ...")
        t0 = time.perf_counter()
        result = self.trainer.train()
        self.train_time_ = time.perf_counter() - t0
        logger.info("Training complete in %.1fs", self.train_time_)
        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def predict(self, test_dataset: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions on a tokenised test dataset.

        Args:
            test_dataset: Tokenised HuggingFace Dataset with a ``label`` column.

        Returns:
            Tuple of (predicted_integer_labels, true_integer_labels).
        """
        if self.trainer is None:
            raise RuntimeError("Trainer not built. Call build_trainer() first.")
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=-1)
        y_true = predictions.label_ids
        return y_pred, y_true

    def save(self, out_dir: Optional[str] = None) -> None:
        """Save the fine-tuned model and tokenizer.

        Args:
            out_dir: Directory to save to (defaults to self.output_dir).
        """
        save_path = out_dir or self.output_dir
        Path(save_path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info("Transformer model saved → %s", save_path)
