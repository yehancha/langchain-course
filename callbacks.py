from typing import Any
from langchain_classic.callbacks.base import BaseCallbackHandler
from langchain_classic.schema import LLMResult

class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> Any:
        print(f"\n*** Prompt to LLM: *** \n{prompts[0]}")
        print("\n--------------------------------")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print(f"\n*** Response from LLM: *** \n{response.generations[0][0].text}")
        print("\n--------------------------------")