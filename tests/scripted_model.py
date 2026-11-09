"""A model that follows a script, so a real Strands ``Agent`` can be driven without any provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from strands.models.model import Model


class ScriptedModel(Model):
    """Answers every invocation of the agent with one tool call, then a closing sentence.

    ``calls`` holds the input of that tool call, one entry per invocation.
    """

    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name
        self.calls: list[dict[str, Any]] = []

    def update_config(self, **model_config: Any) -> None:
        return None

    def get_config(self) -> dict[str, Any]:
        return {}

    async def structured_output(self, output_model: Any, prompt: Any, system_prompt: Any = None, **kwargs: Any) -> Any:
        raise NotImplementedError("the scripted model has no structured output")
        yield {}

    async def stream(
        self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        answered = any("toolResult" in block for block in messages[-1]["content"])
        yield {"messageStart": {"role": "assistant"}}
        if answered:
            yield {"contentBlockStart": {"start": {}}}
            yield {"contentBlockDelta": {"delta": {"text": "Done."}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
        else:
            tool_use = {"toolUseId": f"scripted-{len(messages)}", "name": self.tool_name}
            yield {"contentBlockStart": {"start": {"toolUse": tool_use}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(self.calls.pop(0))}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        usage = {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
        yield {"metadata": {"usage": usage, "metrics": {"latencyMs": 1}}}
