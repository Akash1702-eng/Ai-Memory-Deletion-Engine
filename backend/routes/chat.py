"""
Chat route — POST /api/chat.

Supports model chat execution.
"""

from fastapi import APIRouter, HTTPException

from backend.models.schemas import ChatRequest, ChatResponse
from backend.services.chat_service import get_chat_service
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message and get a response from the selected model variant."""
    try:
        service = get_chat_service()
        result = await service.chat(
            message=request.message,
            model_type=request.model_type,
            history=request.history,
        )
        return ChatResponse(**result)
    except (FileNotFoundError, OSError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Chat error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
