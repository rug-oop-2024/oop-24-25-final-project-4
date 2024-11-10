from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """Exception raised when a given path is not found."""

    def __init__(self, path: str) -> None:
        """
        Initializes the NotFoundError with a specific path.

        Args:
            path (str): The path that was not found.

        Example:
            raise NotFoundError("/path/to/file")
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    Abstract base class defining the interface for data storage systems.
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """
    Local file system-based storage implementation of the Storage interface.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initializes LocalStorage with a specified base path.

        Args:
            base_path (str): Base directory for storage.
            Defaults to "./assets".
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to the local file system.

        Args:
            data (bytes): The data to save.
            key (str): The key representing the storage path.
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from the local file system.

        Args:
            key (str): The key representing the storage path.

        Returns:
            bytes: The loaded data.

        Raises:
            NotFoundError: If the path does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete a file from the local file system.

        Args:
            key (str): The key representing the storage path to delete.

        Raises:
            NotFoundError: If the path does not exist.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        List all files under a given path (prefix).

        Args:
            prefix (str): The directory to list. Defaults to the base path.

        Returns:
            List[str]: List of file paths relative to the base directory.

        Raises:
            NotFoundError: If the prefix path does not exist.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path)
                for p in keys if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None | Exception:
        """
        Assert that the given path exists. Raise NotFoundError if it doesn't.

        Args:
            path (str): Path to check for existence.

        Raises:
            NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Join the base path with the given relative path,
        ensuring OS compatibility.

        Args:
            path (str): The relative path to join with the base path.

        Returns:
            str: The full, OS-compatible path.
        """
        return os.path.normpath(os.path.join(self._base_path, path))
