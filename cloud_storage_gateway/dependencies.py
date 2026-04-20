from fastapi import Header, HTTPException, status
from typing import Annotated
from config import settings

async def verify_internal_token(
    x_internal_token: Annotated[str | None, Header()] = None
):
    """
    Middleware/Dependency to check internal API key header.
    """
    if x_internal_token is None or x_internal_token != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Invalid or missing X-Internal-Token"
        )
    return x_internal_token
