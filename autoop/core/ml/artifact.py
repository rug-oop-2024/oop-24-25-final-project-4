from pydantic import BaseModel, Field

from abc import ABC

import base64


class Artifact(ABC, BaseModel):
    """
    Represents an artifact associated with the machine learning model.
    This class stores the metadata and raw data associated with the artifact.

    Attributes:
        name (str): The name of the artifact.
        asset_path (str): The file path where the artifact is stored.
        version (str): The version of the artifact.
        data (bytes): The raw binary data of the artifact.
        metadata (dict): Additional metadata associated with the artifact.
        type (str): The type of artifact (e.g., "model", "dataset").
        tags (list[str]): A list of tags associated with the artifact.
    """

    name: str
    asset_path: str
    version: str
    data: bytes
    metadata: dict = Field(
        default_factory=lambda: {"description": "No description provided"}
    )
    type: str
    tags: list[str] = Field(default_factory=list)

    def encode_id(artifact_id: str) -> str:
        """
        Encodes an artifact ID into a base64 string.

        Args:
            artifact_id (str): The artifact ID to encode.

        Returns:
            str: The base64-encoded artifact ID.
        """
        encoded_id = base64.b64encode(artifact_id.encode()).decode("utf-8")
        return encoded_id.rstrip("=")

    def decode_id(encoded_id: str) -> str:
        """
        Decodes a base64-encoded artifact ID back to its original form.

        Args:
            encoded_id (str): The base64-encoded artifact ID.

        Returns:
            str: The decoded artifact ID.
        """
        padding = "=" * (4 - len(encoded_id) % 4)
        return base64.b64decode(encoded_id + padding).decode("utf-8")

    @property
    def id(self) -> str:
        """
        Generates a unique identifier for the artifact
        based on its asset path and version.

        Returns:
            str: The unique identifier of the artifact.
        """
        asset_path_encoded = Artifact.encode_id(self.asset_path)
        version = (self.version.replace(",", "_")
                   .replace(":", "_")
                   .replace("=", "_"))
        return f"{asset_path_encoded}_{version}"

    def read(self) -> bytes:
        """
        Returns the raw data of the artifact.

        Returns:
            bytes: The raw data stored in the artifact.
        """
        return self.data
