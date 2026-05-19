"""
Jarvis AI Brain — Claude-powered conversation loop.

Supports two modes:
  • Director Mode  — fully autonomous: analyze footage → plan → execute
  • Assistant Mode — user gives commands, Jarvis executes them via tools
"""

import json
from typing import Any

import anthropic

from .tools import TOOLS


ASSISTANT_SYSTEM_PROMPT = """You are Jarvis, an expert AI video editor assistant.
You control CapCut's web app on behalf of the user via a set of tools.

Guidelines:
- Confirm each action concisely before executing it
- If a request is ambiguous, ask one clarifying question before proceeding
- Chain multiple tool calls when the user's request implies several steps
- After executing, briefly describe what was done
- Use your creative judgment when the user gives vague style directions
- If something goes wrong, tell the user clearly and suggest an alternative"""


class JarvisAssistant:
    """Claude-powered assistant that handles natural language video editing commands."""

    def __init__(self, client: anthropic.Anthropic, capcut, console=None):
        self.client = client
        self.capcut = capcut
        self.console = console
        self._history: list[dict] = []

    def _log(self, msg: str, style: str = "cyan"):
        if self.console:
            self.console.print(f"[{style}]Jarvis ▸[/{style}] {msg}")
        else:
            print(f"Jarvis ▸ {msg}")

    async def chat(self, user_message: str) -> str:
        """
        Send a message to Jarvis and let it interpret + execute editing commands.
        Returns Jarvis's final text reply.
        """
        self._history.append({"role": "user", "content": user_message})

        while True:
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=ASSISTANT_SYSTEM_PROMPT,
                tools=TOOLS,
                messages=self._history,
            )

            # Collect any text in this response turn
            reply_text = ""
            tool_calls: list[dict] = []

            for block in response.content:
                if block.type == "text":
                    reply_text += block.text
                elif block.type == "tool_use":
                    tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

            # No more tool calls — we have the final reply
            if not tool_calls or response.stop_reason == "end_turn":
                self._history.append({"role": "assistant", "content": response.content})
                return reply_text

            # Claude wants to call tools — execute them in CapCut
            self._history.append({"role": "assistant", "content": response.content})

            tool_results = []
            for call in tool_calls:
                self._log(f"Executing: {call['name']}({self._format_args(call['input'])})")
                result = await self.capcut.execute_tool(call["name"], call["input"])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": json.dumps(result),
                })

            self._history.append({"role": "user", "content": tool_results})
            # Loop back to get Claude's next response

    def reset(self):
        """Clear conversation history."""
        self._history = []

    @staticmethod
    def _format_args(args: dict[str, Any]) -> str:
        parts = []
        for k, v in args.items():
            if isinstance(v, str) and len(v) > 40:
                v = v[:37] + "..."
            parts.append(f"{k}={v!r}")
        return ", ".join(parts)
