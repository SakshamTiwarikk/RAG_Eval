import os

from deepeval.models import DeepEvalBaseLLM
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by Groq."""

    def __init__(self):
        self.llm = ChatGroq(
            model=os.getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-120b"),
            temperature=0,
            max_retries=12,
        )
        super().__init__(model=self.llm.model_name)

    def load_model(self):
        return self.llm

    def generate(self, prompt: str, schema=None):
        if schema:
            return self.llm.with_structured_output(schema, method="json_schema").invoke(prompt)
        return self.llm.invoke(prompt).content

    async def a_generate(self, prompt: str, schema=None):
        if schema:
            return await self.llm.with_structured_output(schema, method="json_schema").ainvoke(prompt)
        return (await self.llm.ainvoke(prompt)).content

    def get_model_name(self) -> str:
        return self.llm.model_name


judge = GroqJudge()
