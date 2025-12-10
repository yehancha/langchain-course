from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# llm = ChatOllama(model="qwen3:0.6b")
tools = [TavilySearch()]
agent = create_agent(llm, tools)


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
