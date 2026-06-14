"""
app.py — Streamlit web interface for the Flight Booking Agent.

Run:  streamlit run app.py
"""

import os
import json
import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from agent_trace import clear_trace_log, log_trace, log_state, log_prompt

clear_trace_log()

load_dotenv()
log_trace("Streamlit app loaded")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FlightBot — AI Flight Search",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header */
.hero-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    padding: 2.5rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(56,189,248,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1rem;
    color: #94a3b8;
    margin-top: 0.4rem;
}
.hero-plane {
    font-size: 2.8rem;
    margin-bottom: 0.5rem;
}

/* Search form */
.search-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.75rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* Step trace */
.step-box {
    background: #f8fafc;
    border-left: 3px solid #38bdf8;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.875rem;
    color: #334155;
    font-family: 'Inter', sans-serif;
}
.step-tool {
    border-left-color: #a78bfa;
    background: #faf5ff;
}
.step-done {
    border-left-color: #34d399;
    background: #f0fdf4;
}
.step-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 0.2rem;
}

/* Flight cards */
.flight-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.85rem;
    transition: box-shadow 0.15s;
}
.flight-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.08);
}
.flight-card.best {
    border-color: #38bdf8;
    background: linear-gradient(to right, #f0f9ff, #ffffff);
}
.badge-best {
    background: #0ea5e9;
    color: white;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 999px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-cheap {
    background: #10b981;
    color: white;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 999px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-fast {
    background: #8b5cf6;
    color: white;
    font-size: 0.68rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 999px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.airline-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #0f172a;
}
.price-tag {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #0f172a;
}
.price-label {
    font-size: 0.75rem;
    color: #64748b;
}
.meta-chip {
    display: inline-block;
    background: #f1f5f9;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.8rem;
    color: #475569;
    margin-right: 6px;
}
.time-display {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
}
.time-arrow {
    color: #cbd5e1;
    font-size: 1.2rem;
    margin: 0 0.5rem;
}
.book-btn {
    display: inline-block;
    background: #0f172a;
    color: white !important;
    text-decoration: none !important;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 7px 18px;
    border-radius: 8px;
    letter-spacing: 0.03em;
    transition: background 0.15s;
}
.book-btn:hover {
    background: #1e3a5f;
}
.rec-box {
    background: linear-gradient(135deg, #eff6ff, #f0fdf4);
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
}
.rec-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: #1e40af;
    margin-bottom: 0.4rem;
}

/* Divider with label */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 1.5rem 0 0.75rem;
}

/* Streamlit overrides */
div[data-testid="stButton"] button {
    background: #0f172a;
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: background 0.15s;
}
div[data-testid="stButton"] button:hover {
    background: #1e3a5f;
}
</style>
""", unsafe_allow_html=True)


# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-plane">✈️</div>
    <div class="hero-title">FlightBot</div>
    <div class="hero-sub">AI-powered flight search — tell it what matters to you</div>
</div>
""", unsafe_allow_html=True)


# ── Helper: normalize AI message content ───────────────────────────────────
def normalize_text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
                elif isinstance(text, list):
                    parts.append(normalize_text_content(text))
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


# ── Helper: load city suggestions from airport data ────────────────────────
def load_city_suggestions() -> list[str]:
    try:
        df = pd.read_csv("airports_list.dat", header=None,
                         names=['id', 'name', 'city', 'country', 'iata', 'icao', 'latitude', 'longitude', 'altitude', 'timezone', 'dst', 'tz_database_time_zone', 'type', 'source'])
        cities = df["city"].dropna().astype(str).str.strip()
        cities = sorted({city for city in cities if city})
        return cities
    except Exception:
        return ["Delhi", "Mumbai", "Bengaluru", "Dubai", "Singapore", "London"]


# ── Helper: format price Indian style ─────────────────────────────────────────
def fmt_inr(amount: int) -> str:
    """Format 123456 → ₹1,23,456"""
    s = str(amount)
    if len(s) <= 3:
        return f"₹{s}"
    last3 = s[-3:]
    rest = s[:-3]
    chunks = []
    while len(rest) > 2:
        chunks.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        chunks.append(rest)
    chunks.reverse()
    return "₹" + ",".join(chunks) + "," + last3


# ── Helper: build Google Flights booking URL ───────────────────────────────────
def booking_url(origin: str, dest: str, date: str) -> str:
    return (
        f"https://www.google.com/flights?hl=en#search;"
        f"f={origin};t={dest};d={date};tt=o"
    )


# ── Search form ────────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="search-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1])

    city_suggestions = load_city_suggestions()

    with col1:
        origin = st.selectbox(
            "🛫 From (city name)",
            options=city_suggestions,
            index=city_suggestions.index("Delhi") if "Delhi" in city_suggestions else 0,
            help="Choose a city from the list to avoid typo-based errors"
        )
    with col2:
        destination = st.selectbox(
            "🛬 To (city name)",
            options=city_suggestions,
            index=city_suggestions.index("Dubai") if "Dubai" in city_suggestions else 0,
            help="Choose a destination from the list"
        )
    with col3:
        min_date = datetime.date.today() + datetime.timedelta(days=1)
        travel_date = st.date_input(
            "📅 Date",
            value=min_date,
            min_value=min_date
        )
    with col4:
        adults = st.number_input("👤 Adults", min_value=1, max_value=9, value=1)

    preference = st.text_area(
        "💬 What matters most to you?",
        placeholder=(
            "e.g. I want the cheapest non-stop flight  •  "
            "Shortest travel time  •  "
            "Prefer IndiGo or Air India  •  "
            "Best value with max 1 stop"
        ),
        height=80,
    )

    search_clicked = st.button("Search Flights →", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Run agent on search ────────────────────────────────────────────────────────
if search_clicked:
    log_trace(f"Search button clicked: from={origin.strip()} to={destination.strip()} date={travel_date.strftime('%Y-%m-%d')}")
    # Validate inputs
    if not origin.strip():
        st.error("Please enter an origin city.")
        st.stop()
    if not destination.strip():
        st.error("Please enter a destination city.")
        st.stop()
    if not preference.strip():
        preference = "Show me the best overall flights balancing price and duration."

    date_str = travel_date.strftime("%Y-%m-%d")

    # Import here so API keys are loaded first
    from prompts import build_system_prompt
    from graph import flight_agent
    from state import FlightAgentState
    from agent_trace import log_trace

    log_trace(f"Search started for {origin.strip()} → {destination.strip()} on {date_str}", step="UI")

    system_prompt = build_system_prompt(
        origin=origin.strip(),
        destination=destination.strip(),
        travel_date=date_str,
        adults=adults,
        user_preference=preference.strip(),
    )

    initial_state: FlightAgentState = {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Find flights from {origin.strip()} to {destination.strip()} "
                f"on {date_str} for {adults} adult(s). "
                f"User preference: {preference.strip()}"
            )),],
        "origin_iata": None,
        "destination_iata": None,
        "flight_results": None,
        "agent_steps": [],
    }

    log_state("Initial state", initial_state)
    log_prompt(system_prompt)

    # ── Two-column layout: trace | results ────────────────────────────────────
    trace_col, result_col = st.columns([1, 1.8])

    with trace_col:
        st.markdown('<div class="section-label">🧠 Agent reasoning</div>', unsafe_allow_html=True)
        trace_placeholder = st.empty()

    with result_col:
        st.markdown('<div class="section-label">🔍 Search results</div>', unsafe_allow_html=True)
        result_placeholder = st.empty()

    # ── Stream agent steps using LangGraph stream ──────────────────────────────
    steps_log = []
    final_state = None

    with st.spinner(""):
        try:
            for event in flight_agent.stream(initial_state, stream_mode="values"):
                messages = event.get("messages", [])
                if not messages:
                    continue

                last_msg = messages[-1]

                # ── Capture reasoning trace ───────────────────────────────────
                if isinstance(last_msg, AIMessage):
                    tool_calls = getattr(last_msg, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            name = tc.get("name", "tool")
                            args = tc.get("args", {})

                            # Human-readable step label
                            if name == "lookup_airport_code":
                                city = args.get("city_name", "?")
                                step_html = (
                                    f'<div class="step-box step-tool">'
                                    f'<div class="step-label">Airport lookup</div>'
                                    f'🔍 Resolving IATA code for <strong>{city}</strong>'
                                    f'</div>'
                                )
                            elif name == "search_robust_flights":
                                dep = args.get("departure_id", "?")
                                arr = args.get("arrival_id", "?")
                                dt  = args.get("outbound_date", "?")
                                step_html = (
                                    f'<div class="step-box step-tool">'
                                    f'<div class="step-label">Flight search</div>'
                                    f'📡 Searching <strong>{dep} → {arr}</strong> on {dt}'
                                    f'</div>'
                                )
                            else:
                                step_html = (
                                    f'<div class="step-box step-tool">'
                                    f'<div class="step-label">Tool call</div>'
                                    f'⚙️ {name}({json.dumps(args)[:60]})'
                                    f'</div>'
                                )
                            steps_log.append(step_html)

                    elif last_msg.content:
                        # Final answer from LLM
                        steps_log.append(
                            '<div class="step-box step-done">'
                            '<div class="step-label">Done</div>'
                            '✅ Analysis complete — generating results'
                            '</div>'
                        )

                elif isinstance(last_msg, ToolMessage):
                    # Show what the tool returned (compact)
                    try:
                        result_data = json.loads(last_msg.content)
                        if isinstance(result_data, dict):
                            status = result_data.get("status", "")
                            if status == "found":
                                best = result_data.get("best_match", {})
                                step_html = (
                                    f'<div class="step-box">'
                                    f'<div class="step-label">Result</div>'
                                    f'✅ Found: <strong>{best.get("iata")}</strong> — '
                                    f'{best.get("name")}, {best.get("city")}'
                                    f'</div>'
                                )
                            elif status == "success":
                                n = len(result_data.get("flights", []))
                                step_html = (
                                    f'<div class="step-box">'
                                    f'<div class="step-label">Result</div>'
                                    f'✅ Retrieved <strong>{n} flights</strong>'
                                    f'</div>'
                                )
                            elif status == "not_found":
                                step_html = (
                                    f'<div class="step-box" style="border-left-color:#f87171">'
                                    f'<div class="step-label">Result</div>'
                                    f'❌ {result_data.get("message", "Not found")}'
                                    f'</div>'
                                )
                            elif status == "error":
                                step_html = (
                                    f'<div class="step-box" style="border-left-color:#f87171">'
                                    f'<div class="step-label">Error</div>'
                                    f'⚠️ {result_data.get("message", "Unknown error")}'
                                    f'</div>'
                                )
                            else:
                                step_html = (
                                    f'<div class="step-box">'
                                    f'<div class="step-label">Tool result</div>'
                                    f'↩️ Received response'
                                    f'</div>'
                                )
                        else:
                            step_html = (
                                '<div class="step-box">'
                                '<div class="step-label">Tool result</div>'
                                '↩️ Received response'
                                '</div>'
                            )
                    except (json.JSONDecodeError, TypeError):
                        step_html = (
                            '<div class="step-box">'
                            '<div class="step-label">Tool result</div>'
                            '↩️ Received response'
                            '</div>'
                        )
                    steps_log.append(step_html)

                # Update trace panel live
                trace_placeholder.markdown(
                    "".join(steps_log),
                    unsafe_allow_html=True
                )
                final_state = event

        except Exception as e:
            st.error(f"Agent error: {e}")
            st.stop()

    # ── Extract flights and final LLM answer from final state ──────────────────
    if final_state is None:
        result_placeholder.warning("No response from agent. Check your API keys.")
        st.stop()

    all_messages = final_state.get("messages", [])

    # Find raw flight data from ToolMessages
    raw_flights = []
    for msg in all_messages:
        if isinstance(msg, ToolMessage):
            try:
                d = json.loads(msg.content)
                if isinstance(d, dict) and d.get("status") == "success":
                    raw_flights = d.get("flights", [])
            except (json.JSONDecodeError, TypeError):
                pass

    # Find resolved IATA codes from tool results
    origin_iata, dest_iata = "", ""
    found_count = 0
    for msg in all_messages:
        if isinstance(msg, ToolMessage):
            try:
                d = json.loads(msg.content)
                if isinstance(d, dict) and d.get("status") == "found":
                    iata = d.get("best_match", {}).get("iata", "")
                    if found_count == 0:
                        origin_iata = iata
                    else:
                        dest_iata = iata
                    found_count += 1
            except (json.JSONDecodeError, TypeError):
                pass

    # Find final LLM text
    final_text = ""
    for msg in reversed(all_messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_text = normalize_text_content(msg.content)
            break

    # ── Render flight cards ────────────────────────────────────────────────────
    with result_col:
        if raw_flights:
            # Sort flights based on preference keywords
            pref_lower = preference.lower()
            if any(w in pref_lower for w in ["cheap", "price", "cost", "budget", "affordable", "lowest"]):
                sorted_flights = sorted(raw_flights, key=lambda x: x.get("price_raw", 999999))
                sort_reason = "sorted by price (cheapest first)"
            elif any(w in pref_lower for w in ["fast", "quick", "short", "duration", "time"]):
                sorted_flights = sorted(raw_flights, key=lambda x: x.get("duration_mins", 9999))
                sort_reason = "sorted by duration (fastest first)"
            elif any(w in pref_lower for w in ["non-stop", "nonstop", "direct"]):
                sorted_flights = sorted(
                    raw_flights,
                    key=lambda x: (0 if x.get("stops") == "Non-stop" else 1, x.get("price_raw", 999999))
                )
                sort_reason = "non-stop flights first"
            else:
                # Default: balance price and duration (score = price_rank + duration_rank)
                price_sorted = sorted(raw_flights, key=lambda x: x.get("price_raw", 999999))
                dur_sorted   = sorted(raw_flights, key=lambda x: x.get("duration_mins", 9999))
                price_rank = {f["airline"] + str(i): i for i, f in enumerate(price_sorted)}
                dur_rank   = {f["airline"] + str(i): i for i, f in enumerate(dur_sorted)}
                for i, f in enumerate(raw_flights):
                    key = f["airline"] + str(i)
                    f["_score"] = price_rank.get(key, 5) + dur_rank.get(key, 5)
                sorted_flights = sorted(raw_flights, key=lambda x: x.get("_score", 99))
                sort_reason = "sorted by best value (price + duration balance)"

            result_placeholder.empty()

            # Route header
            st.markdown(
                f'<div style="font-family:Space Grotesk,sans-serif;font-size:1.1rem;'
                f'font-weight:600;color:#0f172a;margin-bottom:0.25rem;">'
                f'✈️ {origin.strip().title()} → {destination.strip().title()}'
                f'</div>'
                f'<div style="font-size:0.8rem;color:#64748b;margin-bottom:1rem;">'
                f'{date_str} &nbsp;·&nbsp; {adults} adult(s) &nbsp;·&nbsp; {sort_reason}'
                f'</div>',
                unsafe_allow_html=True
            )

            cheapest_price = min(f.get("price_raw", 999999) for f in raw_flights)
            fastest_dur    = min(f.get("duration_mins", 9999) for f in raw_flights)

            for i, flight in enumerate(sorted_flights[:6]):
                price    = flight.get("price_raw", 0)
                dur_mins = flight.get("duration_mins", 0)
                airline  = flight.get("airline", "Unknown")
                stops    = flight.get("stops", "—")
                dep_time = flight.get("departure_time", "—")
                arr_time = flight.get("arrival_time", "—")
                dur_str  = flight.get("duration_str", "—")

                is_cheapest = price == cheapest_price
                is_fastest  = dur_mins == fastest_dur
                is_best     = i == 0  # top of sorted list

                # Badge
                badge = ""
                if is_best and is_cheapest:
                    badge = '<span class="badge-cheap">⭐ Best pick</span>'
                elif is_cheapest:
                    badge = '<span class="badge-cheap">💰 Cheapest</span>'
                elif is_fastest:
                    badge = '<span class="badge-fast">⚡ Fastest</span>'
                elif is_best:
                    badge = '<span class="badge-best">⭐ Recommended</span>'

                card_class = "flight-card best" if i == 0 else "flight-card"
                book_link = booking_url(origin_iata, dest_iata, date_str)

                st.markdown(f"""
<div class="{card_class}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.75rem;">
    <div>
      <span class="airline-name">{airline}</span>
      &nbsp; {badge}
    </div>
    <div style="text-align:right;">
      <div class="price-tag">{fmt_inr(price)}</div>
      <div class="price-label">per adult</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;margin-bottom:0.75rem;">
    <span class="time-display">{dep_time}</span>
    <span class="time-arrow">──────✈──────▶</span>
    <span class="time-display">{arr_time}</span>
  </div>
  <div style="margin-bottom:0.85rem;">
    <span class="meta-chip">⏱ {dur_str}</span>
    <span class="meta-chip">🛑 {stops}</span>
  </div>
  <a href="{book_link}" target="_blank" class="book-btn">Book on Google Flights ↗</a>
</div>
""", unsafe_allow_html=True)

            # ── AI Recommendation box ──────────────────────────────────────────
            if final_text:
                # Extract just the recommendation section if LLM wrote one
                rec_text = final_text
                if "recommendation" in final_text.lower():
                    for line in final_text.split("\n"):
                        if "recommendation" in line.lower() or "⭐" in line:
                            # Find this line onwards
                            idx = final_text.find(line)
                            rec_text = final_text[idx:idx+500]
                            break

                st.markdown(f"""
<div class="rec-box">
  <div class="rec-title">⭐ AI Recommendation</div>
  <div style="font-size:0.88rem;color:#1e40af;line-height:1.6;">{rec_text[:400]}</div>
</div>
""", unsafe_allow_html=True)

        elif final_text:
            # No structured flight data but LLM gave a text response
            result_placeholder.empty()
            st.markdown(
                f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:12px;'
                f'padding:1.5rem;line-height:1.7;font-size:0.9rem;">{final_text}</div>',
                unsafe_allow_html=True
            )
        else:
            result_placeholder.warning(
                "No flights were returned. Try different dates or check your API key."
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;font-size:0.75rem;color:#94a3b8;">
  FlightBot uses Google Flights via SearchAPI · Airport data from OpenFlights<br>
  Prices are indicative — verify on the airline or booking site before purchasing.
</div>
""", unsafe_allow_html=True)


# run the code
# /Users/sheetalsuwalka/Documents/Personal Projects/.venv/bin/python -m streamlit run app.py