class LexiError(Exception):
    """Base exception for application-specific errors."""


class IngestionError(LexiError):
    """Raised when corpus ingestion fails for a document or run."""


class RetrievalError(LexiError):
    """Raised when retrieval fails."""


class LLMUnavailableError(LexiError):
    """Raised when the configured LLM provider cannot be reached."""


class CorpusNotIndexedError(LexiError):
    """Raised when retrieval is requested before ingestion has completed."""


class ValidationError(LexiError):
    """Raised for domain validation failures outside FastAPI request validation."""
