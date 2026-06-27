"""doc-schema-extractor: Template-guided PDF/XLSX extraction pipeline."""

from .logging_utils import setup_logging
setup_logging()

from .extractor import Extractor
from .models import ExtractionResult, Template

__all__ = ["Extractor", "ExtractionResult", "Template"]
__version__ = "0.2.2"
