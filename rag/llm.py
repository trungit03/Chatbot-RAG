# llm.py
import requests
import json
import logging
from config import OLLAMA_BASE_URL, DEFAULT_MODEL, TEMPERATURE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaLLM:
    def __init__(self, base_url=OLLAMA_BASE_URL, model=DEFAULT_MODEL):
        self.base_url = base_url
        self.model = model
        self.temperature = TEMPERATURE
        self._check_connection()

    def _check_connection(self):
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Connected to Ollama server successfully")
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if not any(self.model in name for name in model_names):
                    logger.warning(f"Model {self.model} not found. Available models: {model_names}")
            else:
                logger.error(f"Failed to connect to Ollama server: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama server at {self.base_url}: {str(e)}")
            logger.info("Please ensure Ollama is running: 'ollama serve'")

    def generate_response(self, prompt, context=None, chat_history=None):
        try:
            full_prompt = self._build_prompt(prompt, context, chat_history)

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

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=3600
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

    def _build_prompt(self, user_query, context=None, chat_history=None, conversation_context=None):
        prompt_parts = []

        # Enhanced system message with citation instructions
        prompt_parts.append("""You're a helpful research assistant who answers questions based on provided research documents. 
        Follow these guidelines STRICTLY:

1. Provide detailed, coherent answers in natural paragraphs.
2. ALWAYS include specific citations in the format: [Document: filename, Page: X]
3. If the context includes section headings, chapter numbers, or reference information, cite those as well (e.g., [Document: filename, Page: X, Section: Y] or [Document: filename, Chapter: Z, Page: X]).
4. If the information comes from a reference section, cite it as such (e.g., [Document: filename, Page: X, Reference: ...]).
5. If the document doesn't contain relevant information, state that clearly.
6. Maintain a professional, clear style.
7. Only answer based on the provided documents.

EXAMPLE CITATIONS:
- [Document: research.pdf, Page: 5]
- [Document: manual.pdf, Pages: 12, 15]
- [Document: rep.pdf, Page: 3, Section: Introduction]
- [Document: paper.pdf, Page: 6, Section: 1.2]
- [Document: experiment.pdf, Page: 1, Section: A]
- [Document: predict.pdf, Page: 4, Section: II]
""")

        if conversation_context:
            prompt_parts.append("\n=== CONVERSATION CONTEXT ===")
            prompt_parts.append(conversation_context)
            prompt_parts.append("=" * 50)

        if context:
            prompt_parts.append("\n=== CONTEXT DOCUMENTS ===")
            for i, ctx_item in enumerate(context, 1):
                if isinstance(ctx_item, dict):
                    content = ctx_item['content']
                    metadata = ctx_item.get('metadata', {})
                    filename = metadata.get('filename', 'Unknown')
                    page_number = metadata.get('page_number', 'N/A')
                    source_info = f" [Document: {filename}, Page: {page_number}]"
                else:
                    content = ctx_item
                    source_info = ""
                
                prompt_parts.append(f"\n--- Document {i}{source_info} ---")
                # Truncate long content to avoid overwhelming the prompt
                if len(content) > 1000:
                    content = content[:1000] + "..."
                prompt_parts.append(content)
            prompt_parts.append("=" * 50)

        if chat_history:
            prompt_parts.append("\nPrevious Conversation:")
            for exchange in chat_history[-3:]:  
                prompt_parts.append(f"\nHuman: {exchange.get('human', '')}")
                prompt_parts.append(f"Assistant: {exchange.get('assistant', '')}")
            prompt_parts.append("\n" + "=" * 50)

        prompt_parts.append(f"\nCurrent Question: {user_query}")
        prompt_parts.append("\nAnswer based on the context provided above, with accurate citations:")

        return "\n".join(prompt_parts)

    def stream_response(self, prompt, context=None, chat_history=None, conversation_context=None):
        try:
            full_prompt = self._build_prompt(prompt, context, chat_history, conversation_context)

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
                timeout=3600
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

    # def get_available_models(self):
    #     try:
    #         response = requests.get(f"{self.base_url}/api/tags", timeout=5)
    #         if response.status_code == 200:
    #             models = response.json().get('models', [])
    #             return [model['name'] for model in models]
    #         return []
    #     except Exception as e:
    #         logger.error(f"Error getting available models: {str(e)}")
    #         return []

    # def set_model(self, model_name):
    #     available_models = self.get_available_models()
    #     if any(model_name in name for name in available_models):
    #         self.model = model_name
    #         logger.info(f"Model changed to: {model_name}")
    #     else:
    #         logger.error(f"Model {model_name} not available. Available models: {available_models}")