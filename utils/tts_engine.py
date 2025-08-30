import os
from typing import Optional, Dict, Any, List
from murf import Murf
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default voices to use if API doesn't provide them
DEFAULT_VOICES = [
    {"id": "en-US-julia", "name": "Julia", "language": "English (US)"},
    {"id": "en-US-terrell", "name": "Terrell", "language": "English (US)"},
    {"id": "en-GB-charlie", "name": "Charlie", "language": "English (UK)"},
    {"id": "es-ES-maria", "name": "Maria", "language": "Spanish"},
    {"id": "fr-FR-louise", "name": "Louise", "language": "French"},
    {"id": "de-DE-klaus", "name": "Klaus", "language": "German"}
]

class TTSEngine:
    def __init__(self):
        self.murf_api_key = os.getenv("MURF_API_KEY")
        self.client = None
        
        if not self.murf_api_key:
            logger.warning("MURF_API_KEY not found in environment variables")
            return
            
        logger.info(f"Initializing Murf TTS client with API key: {self.murf_api_key[:4]}...")
        
        try:
            self.client = Murf(api_key=self.murf_api_key)
            logger.info("Successfully initialized Murf TTS client")
        except Exception as e:
            logger.error(f"Error initializing Murf TTS client: {str(e)}")
            self.client = None

    def text_to_speech(self, text: str, voice_id: str = "en-US-julia") -> Optional[str]:
        """
        Convert text to speech using Murf API.
        Returns base64 encoded audio string or None if conversion fails.
        
        Args:
            text: Text to convert to speech
            voice_id: Murf voice ID to use (default: en-US-julia)
            
        Returns:
            Base64 encoded audio string or None if conversion fails
        """
        if not self.client:
            logger.error("Murf TTS client not initialized - check MURF_API_KEY environment variable")
            return None
            
        try:
            logger.info(f"Generating speech for text: '{text[:30]}...' with voice: {voice_id}")
            
            # Generate audio with Murf
            response = self.client.text_to_speech.generate(
                text=text,
                voice_id=voice_id,
                # encode_as_base_64=True  # Get base64 encoded audio
            )
            
            # Debug response
            logger.info(f"Murf API response type: {type(response)}")
            logger.info(f"Murf API response attributes: {response.json()}")
            
            # Try different attribute names based on the logs
            if response:
                if hasattr(response, 'audio_file'):
                    logger.info("Found audio_base64 attribute")
                    return response.audio_file
                elif hasattr(response, 'encoded_audio'):
                    logger.info("Found encoded_audio attribute")
                    return response.encoded_audio
                else:
                    # Try to access the attribute as a dictionary
                    try:
                        response_dict = response.model_dump() if hasattr(response, 'model_dump') else vars(response)
                        logger.info(f"Response dict keys: {response_dict.keys()}")
                        
                        if 'encoded_audio' in response_dict:
                            return response_dict['encoded_audio']
                        elif 'audio_base64' in response_dict:
                            return response_dict['audio_base64']
                    except Exception as dict_err:
                        logger.error(f"Error accessing response as dict: {str(dict_err)}")
            
            logger.error("Could not find audio data in response")
            return None
                
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            return None

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available Murf voices."""
        if not self.client:
            logger.error("Cannot get voices - Murf TTS client not initialized")
            return DEFAULT_VOICES
            
        try:
            logger.info("Fetching available Murf voices")
            
            # First try the new API structure
            try:
                # This is for newer versions of the Murf SDK
                voices = self.client.voice.list()
                logger.info(f"Found {len(voices) if voices else 0} voices using voice.list()")
                return voices
            except Exception as e1:
                logger.warning(f"Could not get voices using voice.list(): {str(e1)}")
                
                # Try the old API structure
                try:
                    voices = self.client.voices.list()
                    logger.info(f"Found {len(voices) if voices else 0} voices using voices.list()")
                    return voices
                except Exception as e2:
                    logger.warning(f"Could not get voices using voices.list(): {str(e2)}")
                    
                    # Return default voices as fallback
                    logger.info("Using default voice list as fallback")
                    return DEFAULT_VOICES
        except Exception as e:
            logger.error(f"Error getting voices: {str(e)}")
            return DEFAULT_VOICES
            
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the TTS engine."""
        info = {
            "api_key_set": self.murf_api_key is not None,
            "client_initialized": self.client is not None,
            "api_key_preview": self.murf_api_key[:4] + "..." if self.murf_api_key else None,
            "environment_variables": {k: v[:4] + "..." for k, v in os.environ.items() if "KEY" in k},
            "murf_sdk_version": "1.0.0"  # Add SDK version info
        }
        
        # Try to make a simple API call
        if self.client:
            try:
                # Try to get client version if available
                if hasattr(self.client, 'version'):
                    info["murf_sdk_version"] = self.client.version
                
                # Try to make a simple API call to test connectivity
                try:
                    # Try the direct text-to-speech API first
                    test_response = self.client.text_to_speech.generate(
                        text="Test", 
                        voice_id="en-US-julia",
                        encode_as_base_64=False
                    )
                    info["api_test"] = "success"
                    info["api_test_method"] = "text_to_speech.generate"
                except Exception as e1:
                    logger.warning(f"Direct TTS test failed: {str(e1)}")
                    
                    # Try to get voices as fallback test
                    try:
                        voices = self.get_available_voices()
                        info["api_test"] = "success" 
                        info["api_test_method"] = "get_available_voices"
                        info["voice_count"] = len(voices) if voices else 0
                        info["using_default_voices"] = voices == DEFAULT_VOICES
                    except Exception as e2:
                        info["api_test"] = "failed"
                        info["api_error"] = f"TTS test: {str(e1)}; Voice test: {str(e2)}"
            except Exception as e:
                info["api_test"] = "failed"
                info["api_error"] = str(e)
                
        return info
