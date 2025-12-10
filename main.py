from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


class Source(BaseModel):
    """
    Schema for a source used by the agent
    """

    url: str = Field(description="The url of the source")


class AgentResponse(BaseModel):
    """
    Schema for the response of the agent
    """

    answer: str = Field(description="The agent'sanswer to the question")
    sources: List[Source] = Field(
        default_factory=list,
        description="The sources used to generate the answer to the question",
    )


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# llm = ChatOllama(model="qwen3:0.6b")
tools = [TavilySearch()]
agent = create_agent(llm, tools, response_format=AgentResponse)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Who is the father of current f1 world champion?"
            )
        }
    )
    print(f"Agent result: {result}")


if __name__ == "__main__":
    main()
