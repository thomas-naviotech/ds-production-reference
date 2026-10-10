"""Custom exception hierarchy.

One base class, one meaning per subclass — callers catch what they can
handle and let the rest surface with context instead of a bare traceback.
"""


# Catch TitanicProdError to handle any expected failure; let bugs crash loudly.
class TitanicProdError(Exception):
    """Base class for every expected failure in this repo."""


class ConfigError(TitanicProdError):
    """Bad config file: missing, unreadable, or missing required keys."""


class DataError(TitanicProdError):
    """Bad input data: missing file, missing columns, failed validation."""


class TrainingError(TitanicProdError):
    """Anything that fails inside the training pipeline."""


class ServingError(TitanicProdError):
    """Model bundle missing/corrupt, or a request that cannot be scored."""
