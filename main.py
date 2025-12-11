from typing import List

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
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
    print("Hello, function calling langchain!")

    tools = [get_text_length]

    # llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    llm = ChatOllama(model="qwen3:0.6b", temperature=0, callbacks=[AgentCallbackHandler()])
    
    # Bind tools to the LLM for function calling
    llm_with_tools = llm.bind_tools(tools)

    # Initialize conversation with the user's question
    messages = [HumanMessage(content="What is the length of 'dog' in characters?")]
    
    max_iterations = 10
    iteration = 0
    
    while iteration < max_iterations:
        # Invoke the LLM with the current conversation
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Check if the LLM wants to call any tools
        if response.tool_calls:
            # Execute all tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                # Find and execute the tool
                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)
                
                # Add the tool result as a ToolMessage
                messages.append(
                    ToolMessage(
                        content=str(observation),
                        tool_call_id=tool_id
                    )
                )
        else:
            # No tool calls, the LLM has provided a final answer
            print(f"Final answer: {response.content}")
            break
        
        iteration += 1
    
    if iteration >= max_iterations:
        print("Maximum iterations reached")


if __name__ == "__main__":
    main()
