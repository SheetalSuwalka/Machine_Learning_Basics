"""
graph.py — LangGraph agent wiring.

Nodes:
  llm   → Gemini decides what to do next (reasons + picks a tool, or writes final answer)
  tools → Executes whichever tool the LLM chose

Flow:
  START → llm → (tool call?) → tools → llm → ... → final answer → END
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from state import FlightAgentState
from tools import ALL_TOOLS

load_dotenv()


def _build_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY is not set in your .env file.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
        temperature=0.1,
        max_tokens=800,
    )
    return llm.bind_tools(ALL_TOOLS)


# ── Nodes ──────────────────────────────────────────────────────────────────────

def llm_node(state: FlightAgentState) -> dict:
    llm = _build_llm()
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state: FlightAgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tools"
    return "end"


# ── Graph ──────────────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(FlightAgentState)
    graph.add_node("llm", llm_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "llm")
    return graph.compile()


flight_agent = build_graph()
