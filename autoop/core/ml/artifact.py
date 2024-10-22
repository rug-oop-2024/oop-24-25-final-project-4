from pydantic import BaseModel, Field

from abc import ABC

import base64

class Artifact(ABC, BaseModel):
    asset_path: str
    version: str
    data: bytes
    metadata: dict
    type: str
    tags: list[str]

    @property
    def id(self) -> str:
        asset_path_encoded = base64.urlsafe_b64encode(self.asset_path.encode()).decode('utf-8')
        return f"{asset_path_encoded}:{self.version}"