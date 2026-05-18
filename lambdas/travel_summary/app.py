import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-opus-4-6-v1"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-1")

SYSTEM_PROMPT = """You are a travel summary assistant. Create a personalized travel summary based on the user's specific inputs. Do NOT include general knowledge - only summarize what's relevant to their trip.

Return a JSON object with these fields:
{
  "summary": "2-3 sentence personalized summary of their trip plan",
  "highlights": ["highlight 1", "highlight 2", "highlight 3"],
  "budget_verdict": "one sentence about their budget adequacy",
  "best_time_to_visit": "specific recommendation for their dates",
  "weather": {
    "condition": "sunny" or "rainy" or "cloudy" or "hazy",
    "description": "Brief weather description for their travel dates and location",
    "temperature": "estimated temperature range e.g. 28-33°C"
  },
  "pro_tips": ["tip 1", "tip 2"]
}

For weather, consider:
- Malaysia's monsoon seasons (Nov-Mar: East Coast rainy, West Coast less affected)
- June-Aug: Generally drier across most states
- Haze season: Usually Aug-Oct (from regional fires)
- Default tropical: Hot and humid year-round (28-33°C)

Return ONLY valid JSON, no other text."""

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
    currency = body.get("currency", "Ringgit Malaysia")
    travel_dates = body.get("travel_dates", "")
    traveler_group = body.get("traveler_group", "Adult")
    purpose = body.get("purpose", "")
    interests = body.get("interests", "")
    num_travelers = body.get("num_travelers", "")

    user_message = f"""Create a travel summary for:
- Destination: {state}, {city}
- Dates: {travel_dates}
- Budget: {currency} {budget} for {num_travelers} travelers
- Group: {traveler_group}
- Purpose: {purpose}
- Interests: {interests}

Return JSON only with summary, highlights, budget_verdict, best_time_to_visit, weather (condition/description/temperature), and pro_tips."""

    try:
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": [{"type": "text", "text": SYSTEM_PROMPT}],
                "messages": [{"role": "user", "content": user_message}],
            }),
        )
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]

        try:
            start = text.index('{')
            end = text.rindex('}') + 1
            summary = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            summary = {"summary": text, "weather": {"condition": "sunny", "description": "Tropical weather", "temperature": "28-33°C"}, "highlights": [], "budget_verdict": "", "best_time_to_visit": "", "pro_tips": []}

        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps(summary),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
