from __future__ import annotations

from typing import Sequence

from langchain_core.messages import BaseMessage


def get_tool_call_names(messages):
    """
    Method to extract the tool call names from a list of LangChain messages.
    
    Parameters:
    ----------
    messages : list
        A list of LangChain messages.
        
    Returns:
    -------
    tool_calls : list
        A list of tool call names.
    
    """
    tool_calls = []
    for message in messages:
        try: 
            if "tool_call_id" in list(dict(message).keys()):
                tool_calls.append(message.name)
        except:
            pass
    return tool_calls


def get_last_user_message_content(messages: Sequence[BaseMessage]) -> str:
    """
    Returns the content of the most recent human/user message in a list.
    Falls back to an empty string when missing.
    """
    for msg in reversed(messages or []):
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role in ("human", "user"):
            return (getattr(msg, "content", "") or "").strip()
    return ""
