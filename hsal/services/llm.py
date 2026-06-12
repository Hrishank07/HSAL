from abc import ABC, abstractmethod

from hsal.utils.config import settings


class LLMService(ABC):
    """Abstract LLM interface"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response for prompt"""
        pass

class OpenAILLM(LLMService):
    """OpenAI-based LLM"""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        from openai import OpenAI
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.LLM_MODEL

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        """Generate response using OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

class MockLLM(LLMService):
    """Mock LLM for testing"""

    def generate(self, prompt: str) -> str:
        """Generate mock response"""
        return f"Mock response for: {prompt}"

class OllamaLLM(LLMService):
    """Ollama-based LLM for local usage"""

    def __init__(self, host: str | None = None, model: str | None = None):
        import ollama
        self.host = host or settings.OLLAMA_HOST
        self.model = model or settings.OLLAMA_LLM_MODEL
        self.client = ollama.Client(host=self.host)

    def generate(self, prompt: str) -> str:
        """Generate response using Ollama"""
        response = self.client.chat(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
