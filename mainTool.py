import datetime
import os.path
from typing import List, Dict, Any, Tuple
from zoneinfo import ZoneInfo
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Allow local insecure transport
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
class GoogleCalendarAPI:
    """A class to interact with Google Calendar API."""
    
    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    WORK_START_HOUR = 9
    WORK_END_HOUR = 18  # 6 PM in 24-hour format
    SLOT_DURATION_MINUTES = 90  # 1.5 hours

    def __init__(self):
        self.creds = self.get_credentials()
        self.service = build("calendar", "v3", credentials=self.creds)
        self.timezone = self.get_calendar_timezone(self.service)

    def get_credentials(self):
        """Get Google Calendar API credentials."""
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json")
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", self.SCOPES
                )
                creds = flow.run_local_server(port=8000)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        return creds
    
    def get_calendar_timezone(self, service):
        """Get the timezone of the user's primary calendar."""
        calendar = service.calendars().get(calendarId='primary').execute()
        return calendar.get('timeZone')
    
    def find_free_slots(self) -> List[Tuple[datetime.datetime, datetime.datetime]]:
        """Identify available time slots of 1.5 hours for scheduling interviews within standard work hours, starting from tomorrow and extending to the end of the current month.

        This method queries the Google Calendar API to find periods when the user is free, taking into account both busy slots and actual events. It ensures that the identified slots do not overlap with any existing commitments.

        Args:

        Returns:
            A list of tuples, where each tuple contains the start and end datetime of a free slot, formatted in the user's local timezone.
        """
        # Get current date in the user's timezone
        now = datetime.datetime.now(datetime.timezone.utc)
        calendar_tz = ZoneInfo(self.timezone)
        
        print(f"Current time (UTC): {now}")
        
        # Get tomorrow's date as the start date (in the calendar's timezone)
        local_now = now.astimezone(calendar_tz)
        tomorrow = local_now + datetime.timedelta(days=1)
        start_date = datetime.datetime(
            tomorrow.year, 
            tomorrow.month, 
            tomorrow.day, 
            0, 0, 0, 
            tzinfo=calendar_tz
        )
        
        # Get the end of the current month
        if local_now.month == 12:
            end_of_month = datetime.datetime(local_now.year + 1, 1, 1, tzinfo=calendar_tz)
        else:
            end_of_month = datetime.datetime(local_now.year, local_now.month + 1, 1, tzinfo=calendar_tz)
        
        print(f"Searching for free slots from {start_date.strftime('%Y-%m-%d')} to {end_of_month.strftime('%Y-%m-%d')}")
        
        # Convert to UTC for the API
        time_min = start_date.astimezone(datetime.timezone.utc).isoformat()
        time_max = end_of_month.astimezone(datetime.timezone.utc).isoformat()
        
        # Retrieve both busy time slots AND actual events for more accurate checking
        # First get freebusy information
        freebusy_query = {
            "timeMin": time_min,
            "timeMax": time_max,
            "timeZone": self.timezone,
            "items": [{"id": "primary"}]
        }
        
        print("Fetching freebusy information...")
        freebusy_result = self.service.freebusy().query(body=freebusy_query).execute()
        busy_slots = freebusy_result["calendars"]["primary"]["busy"]
        
        print(f"Found {len(busy_slots)} busy slots in freebusy API")
        for i, busy in enumerate(busy_slots):
            print(f"  Busy {i+1}: {busy['start']} to {busy['end']}")
        
        # Also fetch actual events to be double sure
        print("Fetching actual calendar events...")
        events_result = self.service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        print(f"Found {len(events)} events:")
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            summary = event.get('summary', 'No title')
            print(f"  {summary}: {start} to {end}")
        
        # Create a combined list of busy times from both sources
        combined_busy_slots = []
        
        # Add freebusy results
        for busy in busy_slots:
            busy_start = datetime.datetime.fromisoformat(busy["start"].replace('Z', '+00:00'))
            busy_end = datetime.datetime.fromisoformat(busy["end"].replace('Z', '+00:00'))
            combined_busy_slots.append((busy_start, busy_end))
        
        # Add events
        for event in events:
            # Skip all-day events or events without specific times
            if 'dateTime' not in event['start'] or 'dateTime' not in event['end']:
                continue
                
            # Get event times
            event_start = datetime.datetime.fromisoformat(
                event['start']['dateTime'].replace('Z', '+00:00')
            )
            event_end = datetime.datetime.fromisoformat(
                event['end']['dateTime'].replace('Z', '+00:00')
            )
            
            # Add to our busy times if not already covered
            is_new = True
            for start, end in combined_busy_slots:
                # If this event is completely contained within an existing busy slot, skip it
                if event_start >= start and event_end <= end:
                    is_new = False
                    break
                    
            if is_new:
                combined_busy_slots.append((event_start, event_end))
        
        print(f"Combined busy slots: {len(combined_busy_slots)}")
        
        # Generate all possible 1.5-hour slots during work hours
        free_slots = []
        current_date = start_date
        
        # Iterate through each day until the end of the month
        while current_date < end_of_month:
            # Skip weekends (5=Saturday, 6=Sunday)
            if current_date.weekday() < 5:  # Only weekdays
                # Set work hours for the current day in the calendar's timezone
                work_start = datetime.datetime(
                    current_date.year, 
                    current_date.month, 
                    current_date.day, 
                    self.WORK_START_HOUR, 
                    0,
                    tzinfo=calendar_tz
                )
                
                work_end = datetime.datetime(
                    current_date.year, 
                    current_date.month, 
                    current_date.day, 
                    self.WORK_END_HOUR, 
                    0,
                    tzinfo=calendar_tz
                )
                
                # Generate potential interview slots (1.5 hours each)
                # Use 90-minute increments to avoid overlapping slots
                slot_start = work_start
                day_str = current_date.strftime("%Y-%m-%d")
                print(f"Checking day: {day_str}")
                
                while slot_start + datetime.timedelta(minutes=self.SLOT_DURATION_MINUTES) <= work_end:
                    slot_end = slot_start + datetime.timedelta(minutes=self.SLOT_DURATION_MINUTES)
                    is_free = True
                    
                    # Check if slot overlaps with any busy period
                    for busy_start, busy_end in combined_busy_slots:
                        # Check for any overlap between the slot and busy time
                        # A slot is free if either:
                        # 1. The slot ends before or at the busy period starts, OR
                        # 2. The slot starts after or at the busy period ends
                        # Otherwise, there's an overlap
                        if not (slot_end <= busy_start or slot_start >= busy_end):
                            is_free = False
                            break
                            
                    if is_free:
                        # Keep times in calendar timezone for output
                        free_slots.append((slot_start, slot_end))
                        
                        # Log the free slot we found
                        slot_start_str = slot_start.strftime("%I:%M %p")
                        slot_end_str = slot_end.strftime("%I:%M %p")
                        print(f"  Found free slot: {day_str} {slot_start_str} - {slot_end_str}")
                        
                        # Stop once we have 10 free slots
                        if len(free_slots) >= 10:
                            return free_slots
                    
                    # Move to next non-overlapping slot (move by full slot duration)
                    slot_start += datetime.timedelta(minutes=self.SLOT_DURATION_MINUTES)
            
            # Move to next day
            current_date += datetime.timedelta(days=1)
        
        return free_slots
    
    def create_calendar_event_for_interview(self, summary: str, description: str, start_datetime: str, end_datetime: str, email: str) -> Dict[str, Any]:
        """Create and add an event to the user's Google Calendar for scheduling an interview.

        This function constructs an event with the provided details and inserts it into the user's primary calendar. It requires the event's summary, location, description, start and end times, attendees' email addresses, and reminder settings.

        Args:
            summary (str): The title of the event.
            description (str): A brief description of the event.
            start_datetime (str): The start time of the event in ISO format (e.g., '2025-04-16T12:00:00+05:30').
            end_datetime (str): The end time of the event in ISO format (e.g., '2025-04-16T13:30:00+05:30').
            email (str): Email address of the attendee.

        Returns:
            Dict[str, Any]: The created event details, including a link to the event in the calendar.
        """
        import uuid
        request_id = str(uuid.uuid4())
        event = {
            'summary': summary,
            'location': "Online",
            'description': description,
            'start': {
                'dateTime': start_datetime,
                'timeZone': self.timezone,
            },
            'end': {
                'dateTime': end_datetime,
                'timeZone': self.timezone,
            },
            'recurrence': [],
            'attendees': [{'email': email}],
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 24 * 60},
                    {'method': 'popup', 'minutes': 10},
                ],
            },
            'conferenceData': {
                'createRequest': {
                    'requestId': request_id,
                    'conferenceSolutionKey': {
                        'type': 'hangoutsMeet'
                    }
                }
            }
        }
        
        try:
            event = self.service.events().insert(
            calendarId='primary',
            body=event,
            conferenceDataVersion=1 , # <- Important!
            sendUpdates="all"
             ).execute()
            print("Event created:", event.get("htmlLink"))
            return event
        except HttpError as error:
            print(f"An error occurred: {error}")
            return None
        
def main():
    api = GoogleCalendarAPI()
    try:
        free_slots = api.find_free_slots()
        if free_slots:
            print("Free slots found!")
            # Just an example of how to use the first free slot
            if len(free_slots) > 0:
                first_slot = free_slots[0]
                start_time = first_slot[0].strftime("%Y-%m-%dT%H:%M:%S%z")
                end_time = first_slot[1].strftime("%Y-%m-%dT%H:%M:%S%z")
                
                # Format timezone for Google Calendar API
                start_time = start_time[:-2] + ":" + start_time[-2:]
                end_time = end_time[:-2] + ":" + end_time[-2:]
                
                api.create_calendar_event_for_interview(
                    summary="Interview with Candidate",
                    description="Interview for software developer position",
                    start_datetime=start_time,
                    end_datetime=end_time,
                    email="abc@gmail.com",
                )
                print("Event created for the first available slot")
    except HttpError as error:
        print(f"An error occurred: {error}")

if __name__ == "__main__":
    main()