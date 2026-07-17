"""
Prompt template manager for authorship dataset generation.
"""

from typing import Dict, List, Optional
import os


class PromptManager:
    """Manages prompt loading, templating, and formatting for systematic dataset collection."""

    def __init__(self, templates_dir: Optional[str] = None) -> None:
        """
        Initializes the PromptManager with a directory where template files are stored.

        Args:
            templates_dir: Path to directory containing raw text files with prompt templates.
        """
        self.templates_dir = templates_dir
        # TODO: Scan templates directory and preload any available files
        self.templates: Dict[str, str] = {
            "default_essay": "Write an essay about {topic} in {style} style.",
            "code_generation": "Write a python function to {task}.",
        }

    def load_template(self, template_name: str) -> str:
        """
        Loads a prompt template by name.

        Args:
            template_name: The file name or key of the template.

        Returns:
            The raw prompt template string.
        """
        # TODO: Add file-system fallback if the template name is not in memory
        return self.templates.get(template_name, "")

    def format_prompt(self, template_name: str, variables: Dict[str, str]) -> str:
        """
        Formats a template with keyword variables.

        Args:
            template_name: Name of the template to retrieve.
            variables: Key-value pairs to fill placeholders in the template.

        Returns:
            Formatted prompt string.
        """
        template = self.load_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found.")
        # TODO: Handle missing keys gracefully and output a clear error log
        return template.format(**variables)

    def list_templates(self) -> List[str]:
        """
        Lists all registered/loaded prompt templates.

        Returns:
            A list of template names.
        """
        return list(self.templates.keys())
