"""
main.py — CLI entry point for the Flight Booking Agent.

Usage:
    python main.py

The conversation persists in memory for the session.
Type 'quit', 'exit', or 'bye' to end.
"""

from langchain_core.messages import HumanMessage, AIMessage
from agent import flight_agent, FlightAgentState


def _fresh_state() -> FlightAgentState:
    """Return a clean state dict for a new conversation."""
    return {
        "messages": [],
        "origin_city": None,
        "destination_city": None,
        "departure_date": None,
        "return_date": None,
        "passengers": None,
        "origin_iata": None,
        "destination_iata": None,
        "raw_flight_data": None,
        "search_attempted": False,
    }


def _print_banner():
    print("\n" + "═" * 55)
    print("  ✈️   FlightBot — AI-powered Flight Search Agent")
    print("═" * 55)
    print("  Tell me where you want to fly and when.")
    print("  Type 'new' to start a fresh search.")
    print("  Type 'quit' to exit.")
    print("═" * 55 + "\n")


def chat():
    _print_banner()

    state = _fresh_state()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFlightBot: Safe travels! 👋")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye"):
            print("\nFlightBot: Safe travels! ✈️\n")
            break

        if user_input.lower() == "new":
            state = _fresh_state()
            print("\nFlightBot: Starting a fresh search. Where would you like to fly?\n")
            continue

        # Append the human message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run the graph — it will loop internally until a final answer is produced
        try:
            result = flight_agent.invoke(state)
        except Exception as e:
            print(f"\nFlightBot: Sorry, something went wrong: {e}\n")
            continue

        # Update state with the full result (preserves all accumulated messages)
        state = result

        # Print the last AIMessage
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\nFlightBot: {msg.content}\n")
                break


if __name__ == "__main__":
    chat()
