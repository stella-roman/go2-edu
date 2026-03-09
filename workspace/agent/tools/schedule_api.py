import os
import json
from datetime import datetime, timedelta

SCHEDULE_FILE = "schedule.json"

def normalize_date(date_str: str | None):
    """Convert natural language date expressions into YYYY-MM-DD."""
    if date_str is None:
        return None

    ds = date_str.lower().strip()

    if ds == "today":
        return datetime.now().strftime("%Y-%m-%d")

    if ds == "tomorrow":
        return (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    return ds


def _load_schedule():
    """Load schedule from a local JSON file."""
    if not os.path.exists(SCHEDULE_FILE):
        return {}

    with open(SCHEDULE_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def _save_schedule(data):
    """Save schedule back to the JSON file."""
    with open(SCHEDULE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def manage_schedule(action: str, event: str = None, date: str = None, time: str = None):
    """
    Manage schedule operations.
    action: "add" or "get_today"
    event: event description (required for add)
    date: natural text or YYYY-MM-DD (required for add)
    time: HH:MM (required for add)
    """

    schedule = _load_schedule()

    # Normalize date
    date = normalize_date(date)

    # --- Add a new schedule ---
    if action == "add":
        if not event or not date or not time:
            return {"error": "Missing event, date, or time for adding schedule."}

        # Ensure date exists
        if date not in schedule:
            schedule[date] = {}

        # Ensure time slot exists
        if time not in schedule[date]:
            schedule[date][time] = []  # allow multiple events at same time

        # Add event
        schedule[date][time].append(event)

        _save_schedule(schedule)

        return {
            "status": "success",
            "message": f"Added schedule for {date} at {time}: {event}"
        }

    # --- Get today's schedule ---
    elif action == "get_today":
        today = datetime.now().strftime("%Y-%m-%d")
        events = schedule.get(today, {})

        return {
            "date": today,
            "events": events,
            "message": "Here is your schedule for today." if events else "You have no schedules today."
        }

    return {"error": "Unknown action."}