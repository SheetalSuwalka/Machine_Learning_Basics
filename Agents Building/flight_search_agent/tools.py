"""
tools.py — Two focused tools:
  1. lookup_airport_code  : fuzzy city → IATA via local pandas DataFrame
  2. search_flights       : IATA codes → Google Flights API results
"""

import os
import re
import json
import requests
import pandas as pd
from functools import lru_cache
from langchain_core.tools import tool

from agent_trace import log_trace, log_tool_call, log_tool_result


SEARCHAPI_KEY = os.getenv("FLIGHT_SEARCH_API_key")


# ─── Airport data loader ──────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_airports() -> pd.DataFrame:
    """
    Load airport data from the CSV specified in AIRPORTS_DATA_PATH env var.
    Falls back to downloading OpenFlights public dataset if file not found.

    Expected columns (flexible — we remap common variants):
        name, city, country, iata
    """
    df = pd.read_csv('airports_list.dat', header=None
            ,names=['id', 'name', 'city', 'country', 'iata', 'icao', 'latitude', 'longitude', 'altitude', 'timezone', 'dst','tz_database_time_zone', 'type', 'source'])

    airport_code_df = df[["name", "city", "country", "iata"]].copy()

    # df = df[["name", "city", "country", "iata"]].copy()

    airport_code_df = airport_code_df.fillna("")
    return airport_code_df.reset_index(drop=True)


def _normalize(text: str) -> str:
    """Lowercase, strip extra spaces, remove punctuation for fuzzy matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text)       # collapse whitespace
    return text


# ─── Tool 1: Airport code lookup ─────────────────────────────────────────────

@tool
def lookup_airport_code(city_name: str) -> str:
    """
    Find the IATA airport code for a given city name.
    Handles variations in case, spacing, and common alternate spellings.

    Args:
        city_name: The city or airport name to look up (e.g. "Delhi", "new york", "mumbai")

    Returns:
        JSON with matched airports including IATA code, city, country.
        Returns top match first. If multiple matches exist, all are returned
        so the LLM can pick the most appropriate one.
    """
    log_tool_call("lookup_airport_code", {"city_name": city_name})
    df = _load_airports()
    query = _normalize(city_name)

    # Build normalized lookup columns (computed once, cached via the df reference)
    if "_city_norm" not in df.columns:
        df["_city_norm"] = df["city"].apply(_normalize)
        df["_name_norm"] = df["name"].apply(_normalize)

    # Priority 1: Exact city match
    exact = df[df["_city_norm"] == query]

    # Priority 2: City starts with query
    starts = df[df["_city_norm"].str.startswith(query)] if exact.empty else pd.DataFrame()

    # Priority 3: City contains query anywhere
    contains_city = (
        df[df["_city_norm"].str.contains(query, regex=False)]
        if exact.empty and starts.empty
        else pd.DataFrame()
    )

    # Priority 4: Airport name contains query
    contains_name = (
        df[df["_name_norm"].str.contains(query, regex=False)]
        if exact.empty and starts.empty and contains_city.empty
        else pd.DataFrame()
    )

    # Merge in priority order, deduplicate
    candidates = pd.concat([exact, starts, contains_city, contains_name])
    candidates = candidates.drop_duplicates(subset=["iata"]).head(5)

    if candidates.empty:
        result = json.dumps({
            "status": "not_found",
            "message": f"No airport found matching '{city_name}'. "
                       "Please try the full city name or an alternate spelling.",
            "matches": []
        })
        log_tool_result("lookup_airport_code", result)
        return result

    matches = candidates[["name", "city", "country", "iata"]].to_dict(orient="records")
    result = json.dumps({
        "status": "found",
        "query": city_name,
        "matches": matches,
        "best_match": matches[0]   # highest-priority hit
    }, indent=2)
    log_tool_result("lookup_airport_code", result)
    return result

    if candidates.empty:
        return json.dumps({
            "status": "not_found",
            "message": f"No airport found matching '{city_name}'. "
                       "Please try the full city name or an alternate spelling.",
            "matches": []
        })

    matches = candidates[["name", "city", "country", "iata"]].to_dict(orient="records")
    return json.dumps({
        "status": "found",
        "query": city_name,
        "matches": matches,
        "best_match": matches[0]   # highest-priority hit
    }, indent=2)


# ─── Tool 2: Google Flights search ───────────────────────────────────────────

@tool
def search_robust_flights(departure_id: str, arrival_id: str, outbound_date: str) -> dict:
    """
    Search for one-way flights between two airports on a given date.

    Args:
        departure_id: Three-letter IATA code for the origin airport.
        arrival_id: Three-letter IATA code for the destination airport.
        outbound_date: Departure date in YYYY-MM-DD format.

    Returns:
        A dictionary containing search status and a list of flight options.
    """
    log_tool_call("search_robust_flights", {"departure_id": departure_id, "arrival_id": arrival_id, "outbound_date": outbound_date})
    url = "https://www.searchapi.io/api/v1/search"
    
    params = {
        "engine": "google_flights",
        "departure_id": departure_id.upper().strip(),
        "arrival_id": arrival_id.upper().strip(),
        "outbound_date": outbound_date.strip(),
        "flight_type": "one_way", 
        "currency": "INR",          
        "hl": "en",                 
        "gl": "IN",                 
        "api_key": SEARCHAPI_KEY
    }
    
    try:
        log_trace(f"Querying Google Flights for {params['departure_id']} ➡️ {params['arrival_id']} on {params['outbound_date']}", step="TOOL")
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            result = {"status": "error", "message": f"API HTTP Error {response.status_code}"}
            log_tool_result("search_robust_flights", result)
            return result
            
        data = response.json()
        raw_flights = data.get("best_flights") or data.get("other_flights") or []
        
        if not raw_flights:
            result = {"status": "success", "flights": []}
            log_tool_result("search_robust_flights", result)
            return result
            
        final_selections = []
        
        # INCREASED LIMIT: We pull up to 15 flights from the API so we can sort them later
        for option in raw_flights[:15]:
            segments = option.get("flights", [])
            if not segments:
                continue
                
            num_stops = len(segments) - 1
            stops_text = "Non-stop" if num_stops == 0 else f"{num_stops} Stop(s)"
            primary_airline = segments[0].get("airline", "Unknown Airline")
            
            dep_info = segments[0].get("departure_airport", {})
            departure_time = dep_info.get("time", "N/A")
            
            arr_info = segments[-1].get("arrival_airport", {})
            arrival_time = arr_info.get("time", "N/A")
            
            total_minutes = option.get("total_duration") or segments[0].get("duration") or 0
            
            flight_payload = {
                "airline": primary_airline,
                "price_raw": int(option.get('price', 0)), # Keep integer for sorting
                "duration_mins": total_minutes,            # Keep integer for sorting
                "duration_str": f"{total_minutes // 60}h {total_minutes % 60}m",
                "stops": stops_text,
                "departure_time": departure_time,
                "arrival_time": arrival_time
            }
            final_selections.append(flight_payload)


            
        result = {"status": "success", "flights": final_selections}
        log_tool_result("search_robust_flights", result)
        return result
        
    except Exception as e:
        result = {"status": "error", "message": str(e)}
        log_tool_result("search_robust_flights", result)
        return result


# ─── Exported list ────────────────────────────────────────────────────────────

ALL_TOOLS = [lookup_airport_code, search_robust_flights]
