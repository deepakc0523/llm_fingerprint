"""
src.generation.prompt_formatter
==============================
Implements prompt formatting strategies for LLM text completion/generation.
Supports: 'raw', 'chat', 'instruction', and 'completion'.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Union
from transformers import PreTrainedTokenizer

_log = logging.getLogger(__name__)


class PromptFormatter:
    """Formats dataset prefixes to prepare them for causal LLM generation."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        mode: str = "raw",
        instruction_template: str = "Continue writing the following text naturally, maintaining the tone, style, and vocabulary:\n\n{prefix}",
        plain_instruction_format: str = "Instruction: Continue the following text naturally.\nText: {prefix}\nCompletion:",
        completion_format: str = "{prefix}",
    ) -> None:
        """
        Parameters
        ----------
        tokenizer:
            The model's pretrained tokenizer.
        mode:
            Formatting mode/strategy:
            - 'raw': Pass the prefix text directly without modifications.
            - 'chat': Wrap the prefix in an instruction and use tokenizer.apply_chat_template.
            - 'instruction': Standard plain-text instruction format (no chat template).
            - 'completion': Configurable completion format.
        instruction_template:
            The instruction text wrapped for 'chat' mode.
        plain_instruction_format:
            The plain text format used in 'instruction' mode.
        completion_format:
            The format template used in 'completion' mode.
        """
        self.tokenizer = tokenizer
        self.mode = mode.lower()
        self.instruction_template = instruction_template
        self.plain_instruction_format = plain_instruction_format
        self.completion_format = completion_format

        supported_modes = ("raw", "chat", "instruction", "completion")
        if self.mode not in supported_modes:
            _log.warning(
                "Unknown prompt formatting mode '%s'. Defaulting to 'raw'. Supported: %s",
                mode,
                supported_modes,
            )
            self.mode = "raw"

        _log.info("PromptFormatter initialized with mode: %s", self.mode)

    def format(self, prefix: str) -> str:
        """
        Formats a single prefix text based on the selected mode.

        Parameters
        ----------
        prefix:
            The raw human-written prefix.

        Returns
        -------
        str
            The formatted prompt to feed into the model.
        """
        if self.mode == "raw":
            return prefix

        if self.mode == "completion":
            return self.completion_format.format(prefix=prefix)

        if self.mode == "instruction":
            return self.plain_instruction_format.format(prefix=prefix)

        # 'chat' mode: Wrap in instruction format using tokenizer chat template
        user_message = self.instruction_template.format(prefix=prefix)
        messages = [
            {"role": "user", "content": user_message}
        ]

        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(formatted, str):
                return formatted
            return str(formatted)
        except Exception as e:
            _log.warning(
                "Failed to apply chat template: %s. Falling back to plain-text instruction format.",
                e,
            )
            # Fallback basic instruction format
            return f"[Instruction] {user_message}\n[Response] "

    def format_batch(self, prefixes: List[str]) -> List[str]:
        """
        Formats a list of prefixes.

        Parameters
        ----------
        prefixes:
            List of prefix strings.

        Returns
        -------
        List[str]
            List of formatted prompt strings.
        """
        return [self.format(p) for p in prefixes]
