from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from langchain.agents import create_agent
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

# Create ReAct prompt manually (hub.pull("hwchase17/react") is no longer available)
react_prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: "{input}"
Thought:{agent_scratchpad}"""

react_prompt = PromptTemplate.from_template(react_prompt_template)
# end of definitions related to 'Old way of creating an agent'


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


# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
llm = ChatOllama(model="qwen3:0.6b")
tools = [TavilySearch()]
agent = create_agent(llm, tools, response_format=AgentResponse)

# Old way of executing an agent
old_agent = create_react_agent(llm, tools, prompt=react_prompt)
executor = AgentExecutor(agent=old_agent, tools=tools, verbose=True)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Who is the father of current f1 world champion?"
            )
        }
    )
    # Old way of executing an agent
    # result = executor.invoke(
    #     {
    #         "input": HumanMessage(
    #             content="Who is the father of current f1 world champion?"
    #         )
    #     }
    # )
    print(f"Agent result: {result}")


if __name__ == "__main__":
    main()
