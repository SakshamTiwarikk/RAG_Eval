from dotenv import load_dotenv
import RAG1.tools as tl
from .prompts import system_agent_prompt
import os
from langchain_groq import ChatGroq
from .tools import collecting_info
from langchain.agents import create_agent
from langchain.messages import HumanMessage

load_dotenv()
llm = ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"), temperature=0, max_retries=8)

agent = create_agent(
    model=llm,
    tools=[collecting_info],
    system_prompt=system_agent_prompt,
)

def ask(question) -> dict:
    """ Answer the user's question about the pdf"""

    tl.last_retrieved_context = []

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    # context = the chunks the retriever really returned, not what the LLM claims it used
    return {
        "answer": result["messages"][-1].content,
        "context": tl.last_retrieved_context,
    }

#print(ask("what is the diet of Penguins"))