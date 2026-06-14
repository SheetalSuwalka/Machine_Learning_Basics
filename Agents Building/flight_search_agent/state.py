from typing import TypedDict, Annotated, Optional
from langchain_core.messages import BaseMessage
import operator


class FlightAgentState(TypedDict):
    # Full conversation — messages accumulate across every node
    messages: Annotated[list[BaseMessage], operator.add]

    # Resolved IATA codes (filled by lookup_airport_code tool)
    origin_iata: Optional[str]
    destination_iata: Optional[str]

    # Raw parsed flights from search_robust_flights
    flight_results: Optional[list[dict]]

    # UI: log of reasoning steps shown to the user
    agent_steps: Annotated[list[str], operator.add]
