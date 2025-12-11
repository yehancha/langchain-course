from typing import List, Union

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_classic.agents.format_scratchpad import format_log_to_str
from langchain_classic.agents.output_parsers import \
    ReActSingleInputOutputParser
from langchain_classic.prompts import PromptTemplate
from langchain_classic.tools.render import render_text_description
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from callbacks import AgentCallbackHandler
load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """
    Returns the length (in characters) of the input text
    """

    return len(text.strip("'\n").strip('"'))


def find_tool_by_name(tools: List[Tool], name: str) -> Tool:
    for tool in tools:
        if tool.name == name:
            return tool
    raise ValueError(f"Tool with name {name} not found")


def main():
    print("Hello, react langchain!")

    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

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
    
    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([tool.name for tool in tools]),
    )

    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, model_kwargs={"stop": ["\nObservation:"]})
    llm = ChatOllama(model="qwen3:0.6b", temperature=0, stop=["Observation:"], callbacks=[AgentCallbackHandler()])
    intermediate_steps = []

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agent_step = None
    while not isinstance(agent_step, AgentFinish):
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of 'dog' in characters?",
                "agent_scratchpad": intermediate_steps,
            }
        )

        if isinstance(agent_step, AgentAction):
            tool_to_use = find_tool_by_name(tools, agent_step.tool)
            observation = tool_to_use.func(str(agent_step.tool_input))
            intermediate_steps.append((agent_step, str(observation)))
    
    print(f"Final answer: {agent_step}")


if __name__ == "__main__":
    main()
