"""leakcheck: audit a labelled dataset for duplicates and train/test leakage."""

from .core import AuditResult, audit

__all__ = ["AuditResult", "audit"]
__version__ = "0.1.0"
