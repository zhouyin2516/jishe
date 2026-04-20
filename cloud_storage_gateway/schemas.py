from pydantic import BaseModel
from enum import Enum
from typing import List


class ActionEnum(str, Enum):
    upload = "upload"
    download = "download"


class PresignedUrlRequest(BaseModel):
    group_id: str
    file_key: str
    action: ActionEnum


class PresignedUrlResponse(BaseModel):
    url: str


class BasicObjectRequest(BaseModel):
    group_id: str
    file_key: str


class ObjectMetaResponse(BaseModel):
    content_length: int
    etag: str


class ModelFileInfo(BaseModel):
    file_key: str
    size_bytes: int
    last_modified: str


class ListModelsResponse(BaseModel):
    group_id: str
    models: List[ModelFileInfo]
