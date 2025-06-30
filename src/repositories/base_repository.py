"""Base repository interface for database operations."""

from abc import ABC
from typing import Generic, TypeVar

from src.database import DatabaseManager

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Abstract base class for repository pattern implementation.

    This class provides a common interface for all repository implementations
    and handles database manager dependency injection.

    Parameters
    ----------
    db_manager : DatabaseManager
        Database manager instance for handling database connections and sessions.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize repository with database manager.

        Parameters
        ----------
        db_manager : DatabaseManager
            Database manager instance for database operations.
        """
        self.db_manager = db_manager
