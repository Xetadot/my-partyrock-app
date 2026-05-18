import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-opus-4-6-v1"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-1")

SYSTEM_PROMPT = """You are a GovTech travel itinerary planner. Create a realistic, structured day-by-day travel plan.

CRITICAL FORMAT RULES:
- Structure the output with clear "## Day 1: Title" headers for each day
- Under each day header, list activities as bullet points in this format:
  - **08:00-09:00** - Activity Name @ Location (Notes/Cost)
- Do NOT use markdown tables. Use bullet point lists only.
- Each day MUST start with "## Day X" as a header on its own line.

Example format:
## Day 1: Arrival & Exploration
- **10:00-12:00** - Arrive at Airport @ Airport (Budget flight)
- **12:30-13:30** - Lunch @ Local Restaurant (RM10-15/person)
- **14:00-16:00** - Check-in @ Hotel (RM50-80/night)
- **17:00-19:00** - Walk Heritage Sites @ Old Town (FREE)

## Day 2: Culture & Food
- **08:00-09:00** - Breakfast @ Local Kopitiam (RM5-8/person)
- **09:30-12:00** - Visit Temple @ Temple Area (FREE)

ADAPTATION RULES:
- Students: Budget-friendly, energetic activities
- Adults: Balanced mix
- Seniors: Slower pace, accessible locations, rest breaks
- Family with Kids: Kid-friendly attractions, shorter activities
- Mixed Group: Diverse activities

BUDGET ALIGNMENT:
- Low budget: Free or cheap attractions, local food
- Reasonable: Mix of paid and free, mid-range dining
- High: Premium experiences, fine dining

Include morning, afternoon, and evening plans. Be realistic about timing and distances."""

def handler(event, context):
body = json.loads(event.get("body", "{}"))
    state = body.get("state", "")
    city = body.get("city", "")
    budget = body.get("budget", "")
    currency = body.get("currency", "Ringgit Malaysia")
    num_travelers = body.get("num_travelers", "")
    travel_dates = body.get("travel_dates", "")
    traveler_group = body.get("traveler_group", "Adult")
    purpose = body.get("purpose", "")
    interests = body.get("interests", "")

    user_message = f"""User Inputs:
- State: {state}
- City: {city}
- Travel Dates: {travel_dates}
- Budget: {currency} {budget}
- Travelers: {num_travelers}
- Group Type: {traveler_group}
- Purpose: {purpose}
- Interests: {interests}

Please create a detailed day-by-day itinerary in table format."""

    try:
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 8000,
                "system": [{"type": "text", "text": SYSTEM_PROMPT}],
                "messages": [{"role": "user", "content": user_message}],
            }),
        )
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "text/plain; charset=utf-8"},
            "body": text,
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(e)}),
        }
