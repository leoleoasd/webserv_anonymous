"""
Tool Call Parser for Tau2-Bench

Parses LLM responses into tool calls using sglang's function call parser.
Adapted from the original tau-bench example.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.managers.io_struct import Function, Tool

logger = logging.getLogger(__name__)

DEFAULT_TOOL_CALL_PARSER = os.environ.get("TOOL_CALL_PARSER", "qwen25")


def _strip_thinking(text: str) -> tuple[str, str]:
    """Strip <think>...</think> from response, returning (thinking_content, rest)."""
    idx = text.find("</think>")
    if idx == -1:
        return "", text
    return text[: idx + len("</think>")], text[idx + len("</think>") :].lstrip("\n")


def parse_tools(response: str, tools: list[dict[str, Any]], parser: str = DEFAULT_TOOL_CALL_PARSER) -> dict[str, Any]:
    """
    Parse tool calls from LLM response.

    This function mimics the function call parser API from sglang
    but runs locally.

    Args:
        response: Raw response text from the LLM
        tools: List of tool definitions in OpenAI format
        parser: Parser type (default: "qwen25")

    Returns:
        Dictionary with:
        - normal_text: Text content without tool calls
        - calls: List of parsed tool calls
    """
    tools_list = [
        Tool(
            function=Function(
                name=tool["function"]["name"],
                description=tool["function"]["description"],
                parameters=tool["function"]["parameters"],
            ),
            type=tool["type"],
        )
        for tool in tools
    ]

    parser_obj = FunctionCallParser(tools=tools_list, tool_call_parser=parser)

    # Strip <think>...</think> before parsing — some parsers (e.g. llama3) can't
    # handle thinking prefixes, while others (e.g. qwen25) use explicit tags.
    # Stripping is safe for all parsers since thinking content is never a tool call.
    thinking, stripped_response = _strip_thinking(response)

    normal_text, calls = parser_obj.parse_non_stream(stripped_response)

    # Prepend thinking content back to normal_text so it's preserved in output
    if thinking:
        normal_text = thinking + "\n\n" + normal_text if normal_text else thinking

    return {
        "normal_text": normal_text,
        "calls": [call.model_dump() for call in calls],
    }


@dataclass
class OpenAIToolCall:
    """OpenAI format tool call structure."""

    id: str
    type: str = "function"
    function: dict[str, Any] = None


@dataclass
class OpenAIAssistantMessage:
    """OpenAI format assistant message structure."""

    role: str = "assistant"
    content: str | None = None
    tool_calls: list[OpenAIToolCall] | None = None


class OpenAICompatibleToolCallAdapter:
    """
    Adapter that converts sglang tool call parsing results to OpenAI compatible format.
    """

    def __init__(self, tools_info: list[dict[str, Any]], parser_type: str = DEFAULT_TOOL_CALL_PARSER):
        """
        Initialize adapter.

        Args:
            tools_info: List of tool information in OpenAI format
            parser_type: Parser type (default: "qwen25")
        """
        self.tools_info = tools_info
        self.parser_type = parser_type

    def parse_response_to_openai_format(self, response: str) -> dict[str, Any]:
        """
        Parse sglang response to OpenAI compatible format.

        Args:
            response: Raw response text from sglang

        Returns:
            Dictionary containing OpenAI format message and parsing results
        """
        try:
            parsed = parse_tools(response, self.tools_info, self.parser_type)
            normal_text = parsed["normal_text"]
            calls = parsed["calls"]
            openai_message = self._convert_to_openai_message(normal_text, calls)

            return {
                "openai_message": openai_message,
                "parsed_result": parsed,
                "success": True,
            }
        except Exception as e:
            logger.warning(f"Parsing failed with error: {e!s}")
            return {
                "openai_message": None,
                "parsed_result": None,
                "success": False,
                "error": str(e),
            }

    def _convert_to_openai_message(self, normal_text: str, calls: list[dict[str, Any]]) -> OpenAIAssistantMessage:
        """Convert parsing results to OpenAI format assistant message."""
        if not calls:
            return OpenAIAssistantMessage(
                role="assistant",
                content=normal_text,
                tool_calls=None,
            )

        openai_tool_calls = []
        for i, call in enumerate(calls):
            openai_tool_call = OpenAIToolCall(
                id=f"call_{i}_{call.get('name', 'unknown')}",
                type="function",
                function={
                    "name": call.get("name", ""),
                    "arguments": call.get("parameters", "{}"),
                },
            )
            openai_tool_calls.append(openai_tool_call)

        return OpenAIAssistantMessage(
            role="assistant",
            content=normal_text if normal_text.strip() else None,
            tool_calls=openai_tool_calls,
        )

    def get_openai_tools_format(self) -> list[dict[str, Any]]:
        """Get OpenAI format tool definitions."""
        openai_tools = []
        for tool in self.tools_info:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "parameters": tool["function"]["parameters"],
                },
            }
            openai_tools.append(openai_tool)
        return openai_tools


def create_openai_adapter(
    tools_info: list[dict[str, Any]], parser_type: str = DEFAULT_TOOL_CALL_PARSER
) -> OpenAICompatibleToolCallAdapter:
    """
    Factory function to create OpenAI compatible tool call adapter.

    Args:
        tools_info: List of tool information
        parser_type: Parser type

    Returns:
        Configured adapter instance
    """
    return OpenAICompatibleToolCallAdapter(tools_info, parser_type)
