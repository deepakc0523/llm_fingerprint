"""
Generation package for querying LLMs to construct attribution datasets.
"""

from .generator import LLMGenerator
from .prompt_templates import PromptManager

__all__ = ["LLMGenerator", "PromptManager"]
