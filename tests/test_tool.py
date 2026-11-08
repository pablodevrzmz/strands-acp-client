"""Tests for acp_client_tool."""

from strands_acp_client import acp_client_tool


def test_acp_client_tool_is_registered():
    """The @tool decorator exposes the function under its tool name."""
    assert acp_client_tool.tool_name == "acp_client_tool"

    # TODO: Add behavior tests once you implement the tool body.
