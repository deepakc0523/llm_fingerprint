# Dataset Generation

This directory contains code and modules to interface with local or API-based Large Language Models (LLMs) to generate authorship datasets.

## Modules

- [generator.py](file:///d:/Fingerprint/generation/generator.py): Standard interface/wrapper to call various LLM APIs (OpenAI, Anthropic, Gemini) or local models (via Hugging Face or Ollama) with temperature and max token configurations.
- [prompt_templates.py](file:///d:/Fingerprint/generation/prompt_templates.py): Manages formatting of prompts, instruction variation, and multi-turn configurations to produce diverse datasets.
