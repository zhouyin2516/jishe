import json
import logging
import requests
from config import settings
from typing import Tuple, Dict, Any

from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.auth.credentials import GlobalCredentials
from huaweicloudsdkiam.v3.region.iam_region import IamRegion
from huaweicloudsdkiam.v3 import IamClient
# Removing unused/invalid model imports due to SDK version differences
# Since IAM SDK can be complex for STS, we also provide a fallback or standard IAM API call approach for STS

logger = logging.getLogger(__name__)

def get_server_credentials() -> Tuple[str, str, str | None]:
    """
    Obtain the server's credentials.
    If USE_ECS_AGENCY is True, fetches AK, SK, and Token from ECS metadata.
    If USE_ECS_AGENCY is False, uses the provided DEV_AK and DEV_SK.
    Returns: (AK, SK, SecurityToken)
    """
    if not settings.USE_ECS_AGENCY:
        logger.info("Using local mock DEV_AK/DEV_SK credentials for IAM/OBS.")
        return settings.DEV_AK, settings.DEV_SK, None
    
    # Fetch from ECS Metadata
    try:
        url = "http://169.254.169.254/openstack/latest/securitykey"
        # Usually returns JSON with {"credential": {"access": "...", "secret": "...", "securitytoken": "...", "expires_at": "..."}}
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        credential = data.get("credential", {})
        ak = credential.get("access")
        sk = credential.get("secret")
        token = credential.get("securitytoken")
        if not ak or not sk:
            raise ValueError("Failed to parse AK/SK from ECS Metadata.")
        return ak, sk, token
    except Exception as e:
        logger.error(f"Failed to fetch ECS agency credentials: {e}")
        raise

def generate_policy(group_id: str, file_key: str, action: str) -> str:
    """
    Generate a precise IAM Policy for STS token.
    action: 'upload' -> obs:object:PutObject
    action: 'download' -> obs:object:GetObject
    Resource pattern: OBS:*:*:object:<bucket_name>/<group_id>/models/<file_key>
    """
    obs_action = "obs:object:PutObject" if action == "upload" else "obs:object:GetObject"
    resource_arn = f"OBS:*:*:object:{settings.OBS_BUCKET}/{group_id}/models/{file_key}"
    
    # Standard Huawei Cloud IAM Policy format
    policy_dict = {
        "Version": "1.1",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [obs_action],
                "Resource": [resource_arn]
            }
        ]
    }
    return json.dumps(policy_dict)

def generate_sts_token_for_client(group_id: str, file_key: str, action: str) -> Dict[str, Any]:
    """
    Main entry point for generating STS tokens to be sent to client.
    Requires duration of 3600 seconds.
    """
    if not settings.USE_ECS_AGENCY:
        # Mock STS response for local dev if DEV_AK/SK are used (as they aren't real IAM agency assumed)
        logger.warning("Local development mode: Mocking STS Token generation.")
        return {
            "access_key": settings.DEV_AK,
            "secret_key": settings.DEV_SK,
            "security_token": "mock-sts-security-token-for-local-development",
            "expires_at": "2026-12-31T23:59:59.999Z"
        }

    server_ak, server_sk, server_token = get_server_credentials()
    policy_str = generate_policy(group_id, file_key, action)
    
    # Here we perform the request to IAM to get the temporary token restricted by policy.
    # Note: Creating an STS token via SDK usually requires `CreateTemporaryAccessKeyByAgencyRequest`
    # or by token. For broad compatibility, we interact via IAM API directly or using SDK.
    # Since we are using an SDK, let's configure the IAM Client.
    credentials = GlobalCredentials(server_ak, server_sk)
    if server_token:
        # Currently, huaweicloudsdkcore GlobalCredentials can accept security_token
        # using the property assignment if not supported in constructor.
        credentials.security_token = server_token
        
    # We will use the REST API manually if the SDK version doesn't export the specific STS endpoints we need cleanly,
    # but the canonical STS endpoint for Huawei Cloud is: POST https://iam.{region}.myhuaweicloud.com/v3.0/OS-CREDENTIAL/securitytokens
    
    iam_url = f"https://iam.{settings.IAM_REGION}.myhuaweicloud.com/v3.0/OS-CREDENTIAL/securitytokens"
    
    # We will construct an Auth payload. To request a security token via STS, 
    # we can use the IAM user token or ECS agency token.
    # Actually, a more direct approach since we already have an ECS Token (which is an STS token),
    # is to issue a scoped token? No, Huawei doesn't allow STS-from-STS chaining directly.
    # To issue a token by agency, we call: POST /v3.0/OS-CREDENTIAL/securitytokens
    # payload: { "auth": { "identity": { "methods": ["assume_role"], "assume_role": { "agency_name": "...", "domain_id": "...", "duration_seconds": 3600 }}}}
    
    # But wait, if USE_ECS_AGENCY=False, we have permanent AK/SK. How to generate STS?
    # Usually you cannot just sign an STS request without STS IAM identity payload.
    # We will construct a REST request for token generation, or better: 
    # since we have server AK/SK, we can use standard signature to call STS CreateTemporaryAccessKeyByAgency.
    # We'll use the SDK client to keep it clean.
    
    client = IamClient.new_builder() \
        .with_credentials(credentials) \
        .with_region(IamRegion.value_of(settings.IAM_REGION)) \
        .build()

    # NOTE: Since the requirements demand specific STS capabilities that vary significantly based on whether
    # it's an IAM user AK/SK or ECS Agency token (some IAM APIs forbid AK/SK of agency invoking STS again),
    # and given Huawei Cloud's policy, you usually configure the ECS token directly to the SDK 
    # and it automatically scopes if supported, but here it's an API wrapper.
    # Usually, a direct IAM Token creation API for SDK looks like this:
    from huaweicloudsdkiam.v3.model import CreateTemporaryAccessKeyByAgencyRequest
    from huaweicloudsdkiam.v3.model import AgencyTokenIdentity
    from huaweicloudsdkiam.v3.model import AgencyTokenAuth
    from huaweicloudsdkiam.v3.model import CreateTemporaryAccessKeyByAgencyRequestBody

    # Due to complexity of STS chaining under ECS Agency, the most common practice when 
    # already having an ECS AK/SK/Token is to use it directly, but to scope limit it, you MUST use the STS token creation API.
    # As a simple standalone fallback if SDK types mismatch:
    # We know that ECS STS gives an AK/SK/Token already. If we just pass those, it has full agency permissions.
    # Since we MUST restrict policy, we'll demonstrate the API flow for STS. If the environment
    # does not allow chained STS, this must be run by an AK/SK with assume role permissions.
    

    # If in cloud, the standard way to get a SCOPED token from an existing ECS token is often done via standard AssumeRole API
    # provided by Huawei IAM (AssumeRole / CreateTemporaryAccessKeyByAgency), but usually ECS acts as the delegator.
    # We will raise NotImplemented if we need specific agency_name or domain_id logic, 
    # but let's implement the standard `CreateTemporaryAccessKey...` call.
    try:
        raise RuntimeError("Full STS SDK implementation requires Agency Name/Domain ID which is not in config. Please verify Huawei Cloud STS API specifics.")
    except Exception as e:
        logger.error(f"Error calling Huawei IAM STS API: {e}")
        # Return mocked or re-throw based on strictness. We will re-throw.
        raise
