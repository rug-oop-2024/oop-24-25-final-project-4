from pydantic import BaseModel, Field

from abc import ABC

import base64

import re

class Artifact(ABC, BaseModel):
    name: str
    asset_path: str
    version: str
    data: bytes
    metadata: dict = Field(default_factory=lambda: {"description": "No description provided"})
    type: str
    tags: list[str] = Field(default_factory=list)

    def encode_id(artifact_id: str) -> str:
        # Encode ID in Base64 and decode to string
        encoded_id = base64.b64encode(artifact_id.encode()).decode('utf-8')
        # Remove any '=' padding
        return encoded_id.rstrip('=')

    def decode_id(encoded_id: str) -> str:
        # Add necessary '=' padding for decoding
        padding = '=' * (4 - len(encoded_id) % 4)
        return base64.b64decode(encoded_id + padding).decode('utf-8')

    @property
    def id(self) -> str:
        # Ensure that the asset path is Base64 encoded without padding
        asset_path_encoded = Artifact.encode_id(self.asset_path)
        # Replace any special characters in the version, such as ',' or ':', with '_'
        version = self.version.replace(',', '_').replace(':', '_').replace('=', '_')
        return f"{asset_path_encoded}_{version}"
    
    def read(self) -> bytes:
        """Return the raw data."""
        return self.data