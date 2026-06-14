import json
import textwrap
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).with_name("agent_trace.log")


def clear_trace_log() -> None:
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    LOG_FILE.touch(exist_ok=True)


def _format_text(value: str) -> str:
    if not value:
        return ""
    if "\n" in value:
        parts = []
        for line in value.splitlines():
            if not line.strip():
                parts.append("")
            else:
                parts.append(
                    textwrap.fill(
                        line,
                        width=120,
                        subsequent_indent="  ",
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                )
        return "\n".join(parts)

    return textwrap.fill(
        value,
        width=120,
        subsequent_indent="  ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def _preview_text(value: object, limit: int = 700) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _summarize_state(state: dict) -> str:
    messages = state.get("messages", []) or []
    last_message = messages[-1] if messages else None
    last_type = type(last_message).__name__ if last_message else "None"
    last_content = ""
    if last_message is not None:
        content = getattr(last_message, "content", "")
        if isinstance(content, list):
            preview_items = []
            for item in content[:3]:
                preview_items.append(str(item))
            last_content = " | ".join(preview_items)
        else:
            last_content = str(content)
    last_content = _preview_text(last_content.replace("\n", " "), limit=500)

    lines = [
        f"messages: {len(messages)}",
        f"last_message_type: {last_type}",
        f"last_message_preview: {last_content or '<empty>'}",
        f"origin_iata: {state.get('origin_iata')}",
        f"destination_iata: {state.get('destination_iata')}",
        f"flight_results: {state.get('flight_results')}",
        f"agent_steps: {len(state.get('agent_steps', []) or [])}",
    ]
    return "\n".join(lines)


def _summarize_prompt(prompt: str) -> str:
    preview = _preview_text(prompt.replace("\n", "\\n"), limit=1800)
    return preview


def _fmt_value(value: object) -> str:
    if isinstance(value, str):
        return _format_text(value)
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, indent=2, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)
    return _format_text(str(value))


def log_trace(message: str, step: str | None = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{step.upper()}]" if step else "[TRACE]"
    line = f"{timestamp} | INFO | {prefix} {_fmt_value(message)}"
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def log_state(label: str, state: dict) -> None:
    log_trace(f"{label}:\n{_summarize_state(state)}", step="STATE")


def log_prompt(prompt: str) -> None:
    log_trace(f"System prompt prepared:\n{_summarize_prompt(prompt)}", step="PROMPT")


def log_llm_decision(message: str) -> None:
    log_trace(message, step="LLM")


def log_tool_call(tool_name: str, args: dict) -> None:
    log_trace(f"Calling tool={tool_name} args={_fmt_value(args)}", step="TOOL")


def log_tool_result(tool_name: str, result: object) -> None:
    log_trace(f"Tool={tool_name} result={_fmt_value(result)}", step="RESULT")


def log_final_answer(answer: str) -> None:
    log_trace(f"Final answer:\n{answer}", step="FINAL")


clear_trace_log()
log_trace("Agent trace logging initialized. Log file: " + str(LOG_FILE), step="TRACE")
