import json
from typing import Tuple, List, Union
import os

from autoop.core.storage import Storage


class Database:
    """
    A  database class for managing data retrival and storage.

    This class interacts with a storage backend and provides methods
    to set, get, delete, and list data stored in collections.

    Attributes:
        _storage (Storage): The storage object to persist data.
        _data (dict): In-memory dictionary that holds collections and their data.

    Methods:
        set(collection: str, id: str, entry: dict) -> dict:
            Stores an entry in a specified collection with a given id.

        get(collection: str, id: str) -> Union[dict, None]:
            Retrieves an entry from a specified collection by id.

        delete(collection: str, id: str) -> None:
            Deletes an entry from a specified collection by id.

        list(collection: str) -> List[Tuple[str, dict]]:
            Lists all entries in a specified collection.

        refresh() -> None:
            Refreshes the in-memory data by loading it from storage.
    """

    def __init__(self, storage: Storage) -> None:
        """
        Initializes the Database with a storage backend and loads data.

        Args:
            storage (Storage): The storage backend to use for saving and loading data.
        """
        self._storage = storage
        self._data = {}
        self._load()

    def set(self, collection: str, id: str, entry: dict) -> dict:
        """Set a key in the database
        Args:
            collection (str): The collection to store the data in
            id (str): The id of the data
            entry (dict): The data to store
        Returns:
            dict: The data that was stored
        """
        assert isinstance(entry, dict), "Data must be a dictionary"
        assert isinstance(collection, str), "Collection must be a string"
        assert isinstance(id, str), "ID must be a string"
        if not self._data.get(collection, None):
            self._data[collection] = {}
        self._data[collection][id] = entry
        self._persist()
        return entry

    def get(self, collection: str, id: str) -> Union[dict, None]:
        """Get a key from the database
        Args:
            collection (str): The collection to get the data from
            id (str): The id of the data
        Returns:
            Union[dict, None]: The data that was stored,
            or None if it doesn't exist
        """
        if not self._data.get(collection, None):
            return None
        return self._data[collection].get(id, None)

    def delete(self, collection: str, id: str) -> None:
        """Delete a key from the database
        Args:
            collection (str): The collection to delete the data from
            id (str): The id of the data
        Returns:
            None
        """
        if not self._data.get(collection, None):
            return
        if self._data[collection].get(id, None):
            del self._data[collection][id]
        self._persist()

    def list(self, collection: str) -> List[Tuple[str, dict]]:
        """Lists all data in a collection
        Args:
            collection (str): The collection to list the data from
        Returns:
            List[Tuple[str, dict]]: A list of tuples containing
            the id and data for each item in the collection
        """
        if not self._data.get(collection, None):
            return []
        return [(id, data) for id, data in self._data[collection].items()]

    def refresh(self) -> None:
        """Refresh the database by loading the data from storage"""
        self._load()

    def _persist(self) -> None:
        """Persist the data to storage"""
        for collection, data in self._data.items():
            if not data:
                continue
            for id, item in data.items():
                self._storage.save(
                    json.dumps(item).encode(), f"{collection}{os.sep}{id}"
                )

        # for things that were deleted, we need to remove them from the storage
        keys = self._storage.list("")
        for key in keys:
            collection, id = key.split(os.sep)[-2:]
            if not self._data.get(collection, id):
                self._storage.delete(f"{collection}{os.sep}{id}")

    def _load(self) -> None:
        """Load the data from storage"""
        self._data = {}
        for key in self._storage.list(""):
            collection, id = key.split(os.sep)[-2:]
            data = self._storage.load(f"{collection}{os.sep}{id}")
            # Ensure the collection exists in the dictionary
            if collection not in self._data:
                self._data[collection] = {}
            self._data[collection][id] = json.loads(data.decode())
