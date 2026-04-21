from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from schemas import (
    PresignedUrlRequest, PresignedUrlResponse, ActionEnum,
    BasicObjectRequest, ObjectMetaResponse,
    ListModelsResponse, ModelFileInfo
)

from dependencies import verify_internal_token
from services import obs_service

# All routes in this router MUST pass the internal token verification
router = APIRouter(
    prefix="/api/internal/storage",
    tags=["Internal Storage Gateway"],
    dependencies=[Depends(verify_internal_token)]
)

@router.post("/generate_presigned_url", response_model=PresignedUrlResponse)
async def generate_presigned_url(req: PresignedUrlRequest):
    """
    Dynamically generate a highly restricted Pre-signed URL for client direct HTTP payload delivery/fetching.
    Returns:
        JSON with the 'url' property containing the pre-signed OBS URL.
    """
    try:
        url = obs_service.generate_presigned_url(
            action=req.action.value,
            group_id=req.group_id,
            file_key=req.file_key,
            content_type=req.content_type,
            expires=req.expires
        )


        # 自动生成的 curl 示例，为了防止 Windows 终端换行截断，建议用户配合该命令使用
        curl_cmd = None
        if req.action == ActionEnum.upload:
            # 基础上传命令
            header_str = f' -H "Content-Type: {req.content_type}"' if req.content_type else ""
            curl_cmd = f'curl -X PUT -T "YOUR_LOCAL_FILE_PATH"{header_str} "{url}"'
            
        return PresignedUrlResponse(url=url, curl_command=curl_cmd)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pre-signed URL Generation failed: {str(e)}"
        )

@router.post("/delete_object", response_model=dict)
async def delete_object(req: BasicObjectRequest):
    """
    Delete a large file via OBS SDK using server authoritative credentials.
    """
    try:
        success = obs_service.delete_object(req.group_id, req.file_key)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/object_meta", response_model=ObjectMetaResponse)
async def object_meta(group_id: str, file_key: str):
    """
    Retrieve metadata (ETag, Size) for an object.
    """
    try:
        meta = obs_service.get_object_metadata(group_id, file_key)
        return ObjectMetaResponse(**meta)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

@router.get("/list_groups", response_model=List[str])
async def list_groups():
    """
    List all top-level group directories.
    """
    try:
        groups = obs_service.list_groups()
        return groups
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/list_models", response_model=ListModelsResponse)
async def list_models(group_id: str):
    """
    List all model files within a specific group.
    """
    try:
        models = obs_service.list_models(group_id)
        # map to ModelFileInfo list
        model_infos = [ModelFileInfo(**m) for m in models]
        return ListModelsResponse(group_id=group_id, models=model_infos)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
