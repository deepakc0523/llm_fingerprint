"""Publication-quality PDF Report Generator module for LLM Fingerprinting Research Project."""

from datetime import datetime
import io
import logging
from pathlib import Path
from typing import Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from src.dataset.schema import MergeConfig, ValidationResult

logger = logging.getLogger(__name__)


class MergeReportGenerator:
    """Generates a publication-grade PDF report summarizing dataset merge, validation, and splits."""

    def __init__(self, config: MergeConfig):
        """Initialize report generator.

        Args:
            config: Clean MergeConfig instance.
        """
        self.config = config

    def generate_report(
        self,
        val_result: ValidationResult,
        stats: Dict[str, Any],
        splits: Dict[str, pd.DataFrame],
    ) -> Path:
        """Create merge_report.pdf inside target output directory.

        Args:
            val_result: Result object from DatasetValidator.
            stats: Dictionary from DatasetStatisticsGenerator.
            splits: Dictionary of split DataFrames from DatasetSplitter.

        Returns:
            Path to generated PDF report.
        """
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = output_dir / "merge_report.pdf"

        logger.info("Generating publication-grade PDF merge report at %s...", pdf_path)

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()

        # Custom Palette
        PRIMARY_COLOR = colors.HexColor("#1A365D")  # Deep Navy
        SECONDARY_COLOR = colors.HexColor("#2B6CB0") # Slate Blue
        ACCENT_COLOR = colors.HexColor("#C53030")    # Crimson Accent
        BG_LIGHT = colors.HexColor("#EDF2F7")

        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=PRIMARY_COLOR,
            spaceAfter=6,
        )

        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=15,
        )

        h2_style = ParagraphStyle(
            "Heading2Custom",
            parent=styles["Heading2"],
            fontSize=14,
            leading=18,
            textColor=SECONDARY_COLOR,
            spaceBefore=12,
            spaceAfter=8,
        )

        body_style = ParagraphStyle(
            "BodyCustom",
            parent=styles["BodyText"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#2D3748"),
        )

        story = []

        # 1. Header & Title
        story.append(Paragraph(self.config.report_title, title_style))
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sub_text = (
            f"<b>Author:</b> {self.config.report_author} | "
            f"<b>Version:</b> {self.config.project_version} | "
            f"<b>Timestamp:</b> {now_str}"
        )
        story.append(Paragraph(sub_text, subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY_COLOR, spaceAfter=15))

        # 2. Executive Dataset Summary
        story.append(Paragraph("1. Executive Dataset Summary", h2_style))

        summary_table_data = [
            ["Metric", "Value"],
            ["Total Merged Samples", f"{stats['total_samples']:,}"],
            ["Included Models", f"{stats['number_of_models']}"],
            ["Valid Datasets", f"{len(val_result.valid_datasets)}"],
            ["Excluded Datasets", f"{len(val_result.invalid_datasets)}"],
            ["Split Ratios (Train/Val/Test)", f"{int(self.config.train_ratio*100)}% / {int(self.config.val_ratio*100)}% / {int(self.config.test_ratio*100)}%"],
            ["Random Shuffle Seed", f"{self.config.random_seed}"],
        ]

        t_summary = Table(summary_table_data, colWidths=[200, 300])
        t_summary.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), PRIMARY_COLOR),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BACKGROUND', (0, 1), (-1, -1), BG_LIGHT),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ]))
        story.append(t_summary)
        story.append(Spacer(1, 12))

        # 3. Validation Results Section
        story.append(Paragraph("2. Validation Scan Results", h2_style))

        val_rows = [["Model Directory", "Status", "Records", "Details / Error Message"]]
        for d in val_result.valid_datasets:
            val_rows.append([d.dataset_name, "PASSED", f"{d.total_records:,}", "All schema checks satisfied."])
        for d in val_result.invalid_datasets:
            val_rows.append([d.dataset_name, "EXCLUDED", "0", d.error_message or "Validation error."])

        t_val = Table(val_rows, colWidths=[110, 65, 65, 260])
        t_val.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), SECONDARY_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ]))
        story.append(t_val)
        story.append(Spacer(1, 12))

        # 4. Class Distribution Chart
        story.append(Paragraph("3. Class Distribution & Split Breakdown", h2_style))
        chart_buffer = self._generate_distribution_chart(stats, splits)
        chart_img = Image(chart_buffer, width=500, height=220)
        story.append(chart_img)
        story.append(Spacer(1, 12))

        # 5. Dataset Statistics & Length Metrics
        story.append(Paragraph("4. Character & Token Metrics", h2_style))
        resp_stats = stats.get("response_length_char_statistics", {})
        combined = resp_stats.get("combined_response", {})
        prefix_s = resp_stats.get("prefix_length", {})
        comp_s = resp_stats.get("completion_length", {})

        metrics_data = [
            ["Metric Type", "Mean", "Median", "Min", "Max", "Std Dev"],
            ["Human Prefix Char Length", f"{prefix_s.get('mean', 0)}", f"{prefix_s.get('median', 0)}", f"{prefix_s.get('min', 0)}", f"{prefix_s.get('max', 0)}", f"{prefix_s.get('std', 0)}"],
            ["Generated Text Char Length", f"{comp_s.get('mean', 0)}", f"{comp_s.get('median', 0)}", f"{comp_s.get('min', 0)}", f"{comp_s.get('max', 0)}", f"{comp_s.get('std', 0)}"],
            ["Total Combined Response Char", f"{combined.get('mean', 0)}", f"{combined.get('median', 0)}", f"{combined.get('min', 0)}", f"{combined.get('max', 0)}", f"{combined.get('std', 0)}"],
        ]

        t_metrics = Table(metrics_data, colWidths=[180, 60, 60, 60, 60, 80])
        t_metrics.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ]))
        story.append(t_metrics)
        story.append(Spacer(1, 15))

        # Footer note
        story.append(Paragraph("<i>This report was automatically generated by the LLM Fingerprinting Research Dataset Management Layer.</i>", body_style))

        # Build PDF document
        doc.build(story)
        logger.info("PDF report successfully written to %s", pdf_path)
        return pdf_path

    def _generate_distribution_chart(
        self, stats: Dict[str, Any], splits: Dict[str, pd.DataFrame]
    ) -> io.BytesIO:
        """Render a bar chart of class distributions across splits using Matplotlib."""
        fig, ax = plt.subplots(figsize=(8, 3.5), dpi=300)

        model_counts = stats.get("samples_per_model", {})
        models = sorted(list(model_counts.keys()))

        train_counts = [splits["train"]["model_label"].value_counts().get(m, 0) for m in models]
        val_counts = [splits["validation"]["model_label"].value_counts().get(m, 0) for m in models]
        test_counts = [splits["test"]["model_label"].value_counts().get(m, 0) for m in models]

        x = range(len(models))
        width = 0.25

        ax.bar([i - width for i in x], train_counts, width=width, label="Train (70%)", color="#2B6CB0")
        ax.bar(x, val_counts, width=width, label="Val (15%)", color="#4299E1")
        ax.bar([i + width for i in x], test_counts, width=width, label="Test (15%)", color="#90CDF4")

        ax.set_title("Sample Distribution by Model Class across Stratified Splits", fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Model Name / Label", fontsize=9, labelpad=5)
        ax.set_ylabel("Number of Samples", fontsize=9)
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf
