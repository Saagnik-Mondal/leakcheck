"""leakcheck: audit a labelled dataset for duplicates and train/test leakage."""

from .core import AuditResult, audit
from .split import SplitResult, safe_split

__all__ = ["AuditResult", "SplitResult", "audit", "safe_split"]
__version__ = "0.1.0"
