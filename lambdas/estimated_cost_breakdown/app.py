import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

SYSTEM_PROMPT = """You are a GovTech financial planner. Create a detailed cost breakdown that matches the user's stated budget.

Create a cost breakdown in TABLE format:

| Category | Estimated Cost (RM) |
|----------|---------------------|
| Accommodation | XXX |
| Food and Dining | XXX |
| Transportation | XXX |
| Attractions and Activities | XXX |
| Shopping and Souvenirs | XXX |
| Emergency Buffer | XXX |
| TOTAL | MUST equal the budget provided |

CRITICAL REQUIREMENT: The total MUST equal the user's budget amount.

ALLOCATION GUIDELINES:
- Accommodation: 30-40% of budget
- Food: 25-30%
- Transport: 15-20%
- Activities: 15-20%
- Shopping: 5-10%
- Emergency: 5-10%

Adjust percentages based on traveler group, budget level, and state. Provide brief notes explaining the allocation strategy below the table."""

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
- Total Budget: {currency} {budget}
- Travelers: {num_travelers}
- Travel Dates: {travel_dates}
- Group Type: {traveler_group}
- State: {state}
- City: {city}
- Purpose: {purpose}
- Interests: {interests}

Please create a detailed cost breakdown table that totals exactly to the budget provided."""

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
