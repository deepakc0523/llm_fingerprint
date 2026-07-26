"""Publication-Quality Visualization Engine for LLM Fingerprinting EDA Layer."""

from collections import Counter
import logging
from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.eda.schema import EDAConfig

logger = logging.getLogger(__name__)

# Set publication style defaults
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})


class EDAVisualizer:
    """Renders publication-quality charts and saves PNG figures to target directory."""

    def __init__(self, config: EDAConfig):
        """Initialize visualizer with configuration."""
        self.config = config

    def generate_all_figures(self, df: pd.DataFrame, corr_df: pd.DataFrame) -> List[Path]:
        """Generate and save all 9 publication-grade figures.

        Args:
            df: Merged dataset pandas DataFrame.
            corr_df: Correlation matrix DataFrame.

        Returns:
            List of generated figure file paths.
        """
        fig_dir = self.config.figures_dir
        fig_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Generating publication-quality figures in %s...", fig_dir)

        generated_paths = []

        # 1. Class Distribution
        p1 = fig_dir / "class_distribution.png"
        self._plot_class_distribution(df, p1)
        generated_paths.append(p1)

        # 2. Character Length Histogram
        p2 = fig_dir / "char_length_histogram.png"
        self._plot_histogram(df, "gen_char_len", "Character Length Distribution (Generated Text)", "Character Count", p2)
        generated_paths.append(p2)

        # 3. Word Length Histogram
        p3 = fig_dir / "word_length_histogram.png"
        self._plot_histogram(df, "gen_word_count", "Word Count Distribution (Generated Text)", "Word Count", p3)
        generated_paths.append(p3)

        # 4. Response Length Histogram (Combined)
        p4 = fig_dir / "response_length_histogram.png"
        self._plot_histogram(df, "response_char_len", "Combined Response Length Distribution", "Total Character Count", p4)
        generated_paths.append(p4)

        # 5. Vocabulary Frequency (Top 20 Tokens)
        p5 = fig_dir / "vocabulary_frequency.png"
        self._plot_vocab_frequency(df, p5)
        generated_paths.append(p5)

        # 6. Word Cloud Visualization
        p6 = fig_dir / "word_cloud.png"
        self._plot_word_cloud(df, p6)
        generated_paths.append(p6)

        # 7. Length Boxplots by Model
        p7 = fig_dir / "length_boxplots.png"
        self._plot_boxplots(df, p7)
        generated_paths.append(p7)

        # 8. Length Violin Plots by Model
        p8 = fig_dir / "length_violinplots.png"
        self._plot_violinplots(df, p8)
        generated_paths.append(p8)

        # 9. Correlation Heatmap
        p9 = fig_dir / "correlation_heatmap.png"
        self._plot_correlation_heatmap(corr_df, p9)
        generated_paths.append(p9)

        logger.info("Successfully generated %d figures.", len(generated_paths))
        return generated_paths

    def _plot_class_distribution(self, df: pd.DataFrame, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        label_col = self.config.model_label_col
        counts = df[label_col].value_counts().reset_index()
        counts.columns = ["Model", "Count"]

        palette = sns.color_palette("deep", len(counts))
        bars = ax.bar(counts["Model"], counts["Count"], color=palette, edgecolor="black", linewidth=0.8)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:,}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title("Target Model Class Distribution", pad=12, fontweight="bold")
        ax.set_xlabel("Model Label", labelpad=8)
        ax.set_ylabel("Sample Count", labelpad=8)
        ax.set_ylim(0, max(counts["Count"]) * 1.12)
        sns.despine()
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_histogram(self, df: pd.DataFrame, col: str, title: str, xlabel: str, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
        sns.histplot(df[col], kde=True, ax=ax, color="#2B6CB0", edgecolor="black", linewidth=0.5)

        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color="#C53030", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color="#2F855A", linestyle="-.", linewidth=1.5, label=f"Median: {median_val:.1f}")

        ax.set_title(title, pad=12, fontweight="bold")
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel("Frequency", labelpad=8)
        ax.legend(fontsize=9)
        sns.despine()
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_vocab_frequency(self, df: pd.DataFrame, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        text_col = self.config.generated_text_col
        all_words = []
        for t in df[text_col].astype(str):
            all_words.extend(t.split())

        counter = Counter(all_words)
        most_common = counter.most_common(20)
        words, counts = zip(*most_common) if most_common else ([], [])

        y_pos = np.arange(len(words))
        ax.barh(y_pos, counts, align='center', color="#3182CE", edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([repr(w) for w in words], fontsize=9)
        ax.invert_yaxis()  # top-down
        ax.set_xlabel("Frequency Count", labelpad=8)
        ax.set_title("Top 20 Most Frequent Raw Tokens", pad=12, fontweight="bold")
        sns.despine()
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_word_cloud(self, df: pd.DataFrame, out_path: Path) -> None:
        """Pure Matplotlib scatter/text cloud visualization as robust fallback."""
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        text_col = self.config.generated_text_col
        all_words = []
        for t in df[text_col].astype(str):
            all_words.extend(t.split())

        counter = Counter(all_words)
        top50 = counter.most_common(50)

        # Plot words cleanly in a grid font-sized scatter cloud
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 6)
        ax.axis("off")

        np.random.seed(42)
        cols, rows = 10, 5
        max_freq = top50[0][1] if top50 else 1

        for idx, (word, freq) in enumerate(top50):
            r = idx // cols
            c = idx % cols
            x = c + np.random.uniform(-0.1, 0.1)
            y = 5 - r + np.random.uniform(-0.1, 0.1)
            font_size = 8 + int((freq / max_freq) * 16)
            color = plt.cm.tab20(idx % 20)
            ax.text(x, y, word, fontsize=font_size, color=color, ha='center', va='center', fontweight='bold', alpha=0.9)

        ax.set_title("Corpus Top 50 Token Cloud", pad=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_boxplots(self, df: pd.DataFrame, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        label_col = self.config.model_label_col
        sns.boxplot(data=df, x=label_col, y="gen_char_len", ax=ax, palette="Set2")
        ax.set_title("Generated Character Length Boxplot by LLM", pad=12, fontweight="bold")
        ax.set_xlabel("Model Label", labelpad=8)
        ax.set_ylabel("Character Length", labelpad=8)
        sns.despine()
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_violinplots(self, df: pd.DataFrame, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
        label_col = self.config.model_label_col
        sns.violinplot(data=df, x=label_col, y="gen_char_len", ax=ax, palette="Pastel1", inner="quartile")
        ax.set_title("Generated Character Length Violin Plot by LLM", pad=12, fontweight="bold")
        ax.set_xlabel("Model Label", labelpad=8)
        ax.set_ylabel("Character Length Density", labelpad=8)
        sns.despine()
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    def _plot_correlation_heatmap(self, corr_df: pd.DataFrame, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=300)
        if not corr_df.empty:
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="Blues", ax=ax, cbar=True, linewidths=0.5)
            ax.set_title("Quantitative Length Features Correlation Heatmap", pad=12, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No numerical correlation data available", ha='center', va='center')
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
