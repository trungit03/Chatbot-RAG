import unittest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from rag.llm import OllamaLLM


class TestOllamaLLM(unittest.TestCase):
    def setUp(self):
        with patch.object(OllamaLLM, '_check_connection'):
            self.llm = OllamaLLM()

    def test_build_prompt_basic(self):
        prompt = self.llm._build_prompt("What is AI?")

        self.assertIn("What is AI?", prompt)
        self.assertIn("helpful AI assistant", prompt)

    def test_build_prompt_with_context(self):
        context = ["AI is artificial intelligence.", "Machine learning is a subset of AI."]
        prompt = self.llm._build_prompt("What is AI?", context=context)

        self.assertIn("What is AI?", prompt)
        self.assertIn("AI is artificial intelligence.", prompt)
        self.assertIn("Machine learning is a subset of AI.", prompt)
        self.assertIn("Context Information:", prompt)

    def test_build_prompt_with_history(self):
        history = [
            {"human": "Hello", "assistant": "Hi there!"},
            {"human": "How are you?", "assistant": "I'm doing well, thank you!"}
        ]
        prompt = self.llm._build_prompt("What is AI?", chat_history=history)

        self.assertIn("What is AI?", prompt)
        self.assertIn("Hello", prompt)
        self.assertIn("Hi there!", prompt)
        self.assertIn("Previous Conversation:", prompt)

    def test_build_prompt_complete(self):
        """Test prompt building with context and history"""
        context = ["AI is artificial intelligence."]
        history = [{"human": "Hello", "assistant": "Hi!"}]
        prompt = self.llm._build_prompt("What is AI?", context=context, chat_history=history)

        self.assertIn("What is AI?", prompt)
        self.assertIn("AI is artificial intelligence.", prompt)
        self.assertIn("Hello", prompt)
        self.assertIn("Context Information:", prompt)
        self.assertIn("Previous Conversation:", prompt)

    @patch('requests.post')
    def test_generate_response_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "AI stands for Artificial Intelligence."}
        mock_post.return_value = mock_response

        response = self.llm.generate_response("What is AI?")

        self.assertEqual(response, "AI stands for Artificial Intelligence.")
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_generate_response_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        response = self.llm.generate_response("What is AI?")

        self.assertIn("error", response.lower())

    @patch('requests.post')
    def test_generate_response_timeout(self, mock_post):
        mock_post.side_effect = Exception("Timeout")

        response = self.llm.generate_response("What is AI?")

        self.assertIn("error", response.lower())

    @patch('requests.get')
    def test_get_available_models_success(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.1:8b"},
                {"name": "codellama:7b"}
            ]
        }
        mock_get.return_value = mock_response

        models = self.llm.get_available_models()

        self.assertEqual(len(models), 2)
        self.assertIn("llama3.1:8b", models)
        self.assertIn("codellama:7b", models)

    @patch('requests.get')
    def test_get_available_models_error(self, mock_get):
        mock_get.side_effect = Exception("Connection error")

        models = self.llm.get_available_models()

        self.assertEqual(len(models), 0)

    def test_set_model(self):
        with patch.object(self.llm, 'get_available_models', return_value=['llama3.1:8b', 'codellama:7b']):
            self.llm.set_model('codellama:7b')
            self.assertEqual(self.llm.model, 'codellama:7b')

    def test_set_invalid_model(self):
        original_model = self.llm.model
        with patch.object(self.llm, 'get_available_models', return_value=['llama3.1:8b']):
            self.llm.set_model('invalid_model')
            self.assertEqual(self.llm.model, original_model)


if __name__ == '__main__':
    unittest.main()