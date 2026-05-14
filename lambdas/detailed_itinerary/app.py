import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

SYSTEM_PROMPT = """You are a GovTech travel itinerary planner. Create a realistic, structured day-by-day travel plan.

Create an itinerary in TABLE format:

| Day | Time | Activity | Location | Notes |
|-----|------|----------|----------|-------|

ADAPTATION RULES:
- Students: Budget-friendly, energetic activities
- Adults: Balanced mix
- Seniors: Slower pace, accessible locations, rest breaks
- Family with Kids: Kid-friendly attractions, shorter activities
- Mixed Group: Diverse activities

Purpose alignment:
- Tourism: Attractions, culture, food
- Business: Include work time, professional venues
- Study: Educational sites, libraries, institutions

BUDGET ALIGNMENT:
- Low budget: Free or cheap attractions, local food
- Reasonable: Mix of paid and free, mid-range dining
- High: Premium experiences, fine dining

Include morning, afternoon, and evening plans. Be realistic about timing and distances."""

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

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
                "max_tokens": 4096,
                "system": [{"text": SYSTEM_PROMPT}],
                "messages": [{"role": "user", "content": user_message}],
            }),
        )
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]

        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": "text/plain; charset=utf-8"},
            "body": text,
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(e)}),
        }
