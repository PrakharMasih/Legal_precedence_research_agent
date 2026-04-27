class CaseyError(Exception):
    """Base exception for application-specific errors."""


class IngestionError(CaseyError):
    """Raised when corpus ingestion fails for a document or run."""


class RetrievalError(CaseyError):
    """Raised when retrieval fails."""


class LLMUnavailableError(CaseyError):
    """Raised when the configured LLM provider cannot be reached."""


class CorpusNotIndexedError(CaseyError):
    """Raised when retrieval is requested before ingestion has completed."""


class ValidationError(CaseyError):
    """Raised for domain validation failures outside FastAPI request validation."""
