from abc import ABC, abstractmethod

from langchain_community.llms.ollama import Ollama
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

PROMPT_TEMPLATE = """
You're an assistant knowledgeable about
...         healthcare. 
Based on the following context:
{context}

---

Only answer healthcare-related questions.
Answer the following question : {question}
"""

class LLM(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def invoke(self, prompt: str) -> str:
        pass

    def generate_response(self, context: str, question: str) -> str:
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context, question=question)
        response_text = self.invoke(prompt)
        return response_text

class OllamaModel(LLM):
    def __int__(self, model_name: str):
        super().__init__(model_name)
        self.model = Ollama(model = model_name)

    def invoke(self, prompt: str) -> str:
        return self.model.invoke(prompt)

class GPTModel(LLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=api_key)

    def invoke(self, prompt: str) -> str:
        messages = [
            {"role":"user", "content":prompt}
        ]
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

class GeminiModel(LLM):
    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name)
        self.model = ChatGoogleGenerativeAI(model = model_name, google_api_key = api_key)

    def invoke(self, prompt: str):
        return self.model.invoke(prompt)


