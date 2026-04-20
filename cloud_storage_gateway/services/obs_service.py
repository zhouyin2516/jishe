# coding: utf-8
from obs import ObsClient
from config import settings
from services.iam_service import get_server_credentials
import logging

logger = logging.getLogger(__name__)

def get_obs_client() -> ObsClient:
    """
    Constructs and returns an OBS Client using the server's authoritative credentials.
    """
    ak, sk, token = get_server_credentials()
    # ESDK-OBS-Python supports security_token for temporary credentials (e.g. from ECS metadata)
    return ObsClient(
        access_key_id=ak,
        secret_access_key=sk,
        security_token=token,
        server=settings.OBS_ENDPOINT
    )

def generate_presigned_url(action: str, group_id: str, file_key: str) -> str:
    """
    Generate a pre-signed URL for direct upload or download by the client.
    action: "upload" -> PUT, "download" -> GET
    """
    object_key = f"{group_id}/models/{file_key}"
    http_method = "PUT" if action == "upload" else "GET"
    
    # 过期时间硬编码设置为 3600 秒 (1小时)，以保证巨型模型的长时间断点续传
    expires = 3600
    
    obsClient = get_obs_client()
    try:
        res = obsClient.createSignedUrl(
            method=http_method, 
            bucketName=settings.OBS_BUCKET, 
            objectKey=object_key, 
            expires=expires
        )
        if hasattr(res, "signedUrl") and res.signedUrl:
            logger.info(f"Generated {http_method} Pre-signed URL for {object_key}")
            return res.signedUrl
        else:
            errMsg = getattr(res, "errorMessage", "Unknown OBS SDK URL creation error")
            logger.error(f"Failed to generate pre-signed URL for {object_key}: {errMsg}")
            raise Exception(f"OBS Signed URL Generation Failed: {errMsg}")
    except Exception as e:
        logger.error(f"Exception during URL generation: {e}")
        raise
    finally:
        obsClient.close()

def delete_object(group_id: str, file_key: str) -> bool:
    """
    Physically delete a large file from OBS.
    """
    object_key = f"{group_id}/models/{file_key}"
    obsClient = get_obs_client()
    try:
        resp = obsClient.deleteObject(bucketName=settings.OBS_BUCKET, objectKey=object_key)
        if resp.status < 300:
            logger.info(f"Successfully deleted object: {object_key}")
            return True
        else:
            logger.error(f"Failed to delete object {object_key}: {resp.errorMessage}")
            raise Exception(f"OBS Delete Failed: {resp.errorMessage}")
    except Exception as e:
        logger.error(f"Exception deleting file: {e}")
        raise
    finally:
        obsClient.close()

def get_object_metadata(group_id: str, file_key: str) -> dict:
    """
    Fetch metadata (ContentLength, ETag) for integrity checking.
    """
    object_key = f"{group_id}/models/{file_key}"
    obsClient = get_obs_client()
    try:
        resp = obsClient.getObjectMetadata(bucketName=settings.OBS_BUCKET, objectKey=object_key)
        if resp.status < 300:
            return {
                "content_length": int(resp.header.get("content-length", 0)),
                "etag": resp.header.get("etag", "").strip('"')
            }
        else:
            logger.error(f"Failed to get metadata for {object_key}: {resp.errorMessage}")
            raise Exception(f"OBS Metadata retrieval failed: {resp.errorMessage}")
    except Exception as e:
        logger.error(f"Exception getting file meta: {e}")
        raise
    finally:
        obsClient.close()

def list_groups() -> list:
    """
    List top-level directories (group_ids) in the bucket using delimiter='/'.
    """
    obsClient = get_obs_client()
    try:
        resp = obsClient.listObjects(bucketName=settings.OBS_BUCKET, delimiter='/', prefix='')
        if resp.status < 300:
            groups = []
            if resp.body.commonPrefixes:
                for prefix_dict in resp.body.commonPrefixes:
                    prefix = prefix_dict.get('prefix', '')
                    if prefix.endswith('/'):
                        prefix = prefix[:-1]
                    if prefix:
                        groups.append(prefix)
            return groups
        else:
            raise Exception(f"OBS List Groups Failed: {resp.errorMessage}")
    except Exception as e:
        logger.error(f"Exception listing groups: {e}")
        raise
    finally:
        obsClient.close()

def list_models(group_id: str) -> list:
    """
    List all model files within a specific group_id. Handles pagination.
    """
    prefix = f"{group_id}/models/"
    models = []
    marker = None
    
    obsClient = get_obs_client()
    try:
        while True:
            resp = obsClient.listObjects(
                bucketName=settings.OBS_BUCKET, 
                prefix=prefix, 
                marker=marker,
                max_keys=1000
            )
            if resp.status < 300:
                for content in resp.body.contents:
                    if content.key == prefix:
                        continue
                        
                    file_key = content.key.replace(prefix, "")
                    models.append({
                        "file_key": file_key,
                        "size_bytes": content.size,
                        "last_modified": content.lastModified
                    })
                
                if resp.body.is_truncated:
                    marker = resp.body.next_marker
                else:
                    break
            else:
                raise Exception(f"OBS List Models Failed: {resp.errorMessage}")
        return models
    except Exception as e:
        logger.error(f"Exception listing models: {e}")
        raise
    finally:
        obsClient.close()
