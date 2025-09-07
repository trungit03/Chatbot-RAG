import requests
import json
from typing import List, Dict, Any, Optional
import logging
from config import OLLAMA_BASE_URL, DEFAULT_MODEL, TEMPERATURE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaLLM:

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model
        self.temperature = TEMPERATURE
        self._check_connection()

    def _check_connection(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Connected to Ollama server successfully")
                # Check if model is available
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if not any(self.model in name for name in model_names):
                    logger.warning(f"Model {self.model} not found. Available models: {model_names}")
            else:
                logger.error(f"Failed to connect to Ollama server: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama server at {self.base_url}: {str(e)}")
            logger.info("Please ensure Ollama is running: 'ollama serve'")

    def generate_response(self, prompt: str, context: List[str] = None,
                          chat_history: List[Dict[str, str]] = None) -> str:
        """Generate response using Ollama"""
        try:
            # Build the full prompt
            full_prompt = self._build_prompt(prompt, context, chat_history)

            # Prepare request data
            data = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }

            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return "Sorry, I encountered an error while generating a response."

        except requests.exceptions.Timeout:
            logger.error("Request to Ollama timed out")
            return "Sorry, the request timed out. Please try again."
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error while processing your request."

    def _build_prompt(self, user_query: str, context: List[str] = None,
                      chat_history: List[Dict[str, str]] = None) -> str:
        """Build the complete prompt with context and history"""

        prompt_parts = []

        # System message
        prompt_parts.append("""You are a helpful AI assistant that answers questions based on the provided context. 
Follow these guidelines:
1. Use only the information provided in the context to answer questions
2. If the context doesn't contain relevant information, say so clearly
3. Be concise but comprehensive in your answers
4. Maintain a helpful and professional tone
5. If asked about something not in the context, explain that you can only answer based on the provided documents
6. Always answer details
""")

        # Add context if provided
        if context:
            prompt_parts.append("\nContext Information:")
            for i, ctx in enumerate(context, 1):
                prompt_parts.append(f"\n--- Document {i} ---")
                prompt_parts.append(ctx)
            prompt_parts.append("\n" + "=" * 50)

        # Add chat history if provided
        if chat_history:
            prompt_parts.append("\nPrevious Conversation:")
            for exchange in chat_history[-5:]:  # Last 5 exchanges
                prompt_parts.append(f"\nHuman: {exchange.get('human', '')}")
                prompt_parts.append(f"Assistant: {exchange.get('assistant', '')}")
            prompt_parts.append("\n" + "=" * 50)

        # Add current question
        prompt_parts.append(f"\nCurrent Question: {user_query}")
        prompt_parts.append("\nAnswer based on the context provided above:")

        return "\n".join(prompt_parts)

    def stream_response(self, prompt: str, context: List[str] = None,
                        chat_history: List[Dict[str, str]] = None):
        """Generate streaming response (generator)"""
        try:
            full_prompt = self._build_prompt(prompt, context, chat_history)

            data = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": True,
                "options": {
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                stream=True,
                timeout=60
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'response' in chunk:
                                yield chunk['response']
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                yield "Error: Failed to get response from Ollama"

        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield f"Error: {str(e)}"

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception as e:
            logger.error(f"Error getting available models: {str(e)}")
            return []

    def set_model(self, model_name: str):
        """Change the current model"""
        available_models = self.get_available_models()
        if any(model_name in name for name in available_models):
            self.model = model_name
            logger.info(f"Model changed to: {model_name}")
        else:
            logger.error(f"Model {model_name} not available. Available models: {available_models}")