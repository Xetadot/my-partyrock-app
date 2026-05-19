import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

SYSTEM_PROMPT = """You are a travel accommodation expert for Malaysia. Given a destination, find 5 real accommodations available in that area.

Return ONLY a JSON array with exactly 5 objects. Each object must have:
- "name": real hotel/hostel/resort name that actually exists
- "type": "Hotel" or "Resort" or "Hostel" or "Homestay"
- "rating": rating out of 5 (e.g., "4.5")
- "price_range": approximate price per night in MYR (e.g., "RM120-180")
- "address": real street address or area
- "source": "Expedia" or "TripAdvisor" or "Trip.com" (alternate between them)
- "url": real booking URL on that platform (use format like https://www.expedia.com/Hotel-Search?destination=CITY+Malaysia or https://www.tripadvisor.com/Hotels-CITY-Malaysia or https://www.trip.com/hotels/CITY-hotel)
- "image_query": search term to find this hotel image (e.g., "Shangri-La Hotel Kuala Lumpur exterior")

Use REAL hotel names that exist in the specified location. Mix budget, mid-range, and luxury options.
Return ONLY valid JSON array, no other text."""

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def handler(event, context):
    if event.get("requestContext",{}).get("http",{}).get("method","") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    body = json.loads(event.get("body", "{}"))
    state = body.get("state", "Selangor")
    city = body.get("city", "")
    budget = body.get("budget", "")
    traveler_group = body.get("traveler_group", "Adult")

    location = city if city else state
    user_message = f"Find 5 real accommodations in {location}, Malaysia. Budget: RM{budget}. Group: {traveler_group}. Return JSON array only."

    try:
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 3000,
                "system": [{"type": "text", "text": SYSTEM_PROMPT}],
                "messages": [{"role": "user", "content": user_message}],
            }),
        )
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]

        try:
            start = text.index('[')
            end = text.rindex(']') + 1
            accommodations = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            accommodations = []

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"accommodations": accommodations}),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
