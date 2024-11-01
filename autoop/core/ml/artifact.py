from pydantic import BaseModel, Field

from abc import ABC

import base64

class Artifact(ABC, BaseModel):
    asset_path: str
    version: str
    data: bytes
    metadata: dict = Field(default_factory=lambda: {"description": "No description provided"})
    type: str
    tags: list[str] = Field(default_factory=list)

    @property
    def id(self) -> str:
        asset_path_encoded = base64.urlsafe_b64encode(self.asset_path.encode()).decode('utf-8')
        return f"{asset_path_encoded}:{self.version}"
    
    def read(self) -> bytes:
        """Return the raw data."""
        return self.data