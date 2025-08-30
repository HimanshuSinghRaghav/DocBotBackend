from fastapi import APIRouter, HTTPException, Depends, Body
from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from utils.tts_engine import TTSEngine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
tts_engine = TTSEngine()

class TextToSpeechRequest(BaseModel):
    text: str
    voice_id: str = "en-US-julia"
    
class TextToSpeechResponse(BaseModel):
    audio_base64: Optional[str] = None
    message: str
    success: bool
    debug_info: Optional[Dict[str, Any]] = None
    
class VoiceInfo(BaseModel):
    id: str
    name: Optional[str] = None
    language: Optional[str] = None
    
class VoiceListResponse(BaseModel):
    voices: List[Dict[str, Any]]
    count: int
    debug_info: Optional[Dict[str, Any]] = None

@router.post("/convert", response_model=TextToSpeechResponse)
async def convert_text_to_speech(request: TextToSpeechRequest):
    """
    Convert text to speech using Murf API
    
    Returns base64 encoded audio that can be played directly in browsers
    """
    logger.info(f"TTS request received: '{request.text[:30]}...' with voice: {request.voice_id}")
    
    # Get debug info
    debug_info = tts_engine.get_debug_info()
    
    if not tts_engine.client:
        logger.error("TTS engine not initialized")
        return TextToSpeechResponse(
            audio_base64=None,
            message="TTS engine not initialized. Check MURF_API_KEY environment variable.",
            success=False,
            debug_info=debug_info
        )
    
    if not request.text:
        return TextToSpeechResponse(
            audio_base64=None,
            message="Text cannot be empty",
            success=False,
            debug_info=debug_info
        )
    
    audio_base64 = tts_engine.text_to_speech(request.text, request.voice_id)
    
    if not audio_base64:
        logger.error("Failed to generate audio")
        return TextToSpeechResponse(
            audio_base64=None,
            message="Failed to generate audio",
            success=False,
            debug_info=debug_info
        )
    
    logger.info("Successfully generated audio")
    return TextToSpeechResponse(
        audio_base64=audio_base64,
        message="Audio generated successfully",
        success=True,
        debug_info=debug_info
    )

@router.get("/voices", response_model=VoiceListResponse)
async def get_available_voices():
    """Get list of available Murf voices"""
    logger.info("Voice list request received")
    
    # Get debug info
    debug_info = tts_engine.get_debug_info()
    
    if not tts_engine.client:
        logger.error("TTS engine not initialized")
        return VoiceListResponse(
            voices=[],
            count=0,
            debug_info=debug_info
        )
    
    voices = tts_engine.get_available_voices()
    
    return VoiceListResponse(
        voices=voices if voices else [],
        count=len(voices) if voices else 0,
        debug_info=debug_info
    )

@router.get("/debug")
async def get_tts_debug_info():
    """Get debug information about the TTS engine"""
    logger.info("Debug info request received")
    debug_info = tts_engine.get_debug_info()
    return {"debug_info": debug_info}
