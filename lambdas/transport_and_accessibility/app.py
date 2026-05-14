import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

SYSTEM_PROMPT = """You are a GovTech transport infrastructure advisor. Provide comprehensive transport information for travel in Malaysia.

Provide:
**PUBLIC TRANSPORT OPTIONS:**
- MRT/LRT/Monorail (if available in the state)
- Bus services (RapidKL, Prasarana, local buses)
- Train services (KTM, ETS)
- E-hailing (Grab) and taxis

**ACCESSIBILITY FEATURES:**
- OKU-friendly facilities
- Elevator/ramp availability
- Special considerations for the traveler group provided

**TRANSPORT CARDS AND PASSES:**
- Touch n Go card
- MyRapid passes
- Tourist travel cards

**TRAVEL EFFICIENCY TIPS:**
- Peak hours to avoid
- Best routes for tourists
- Time-saving recommendations

**COST ESTIMATES:**
- Typical daily transport costs
- Budget-friendly options

Format clearly with sections and bullet points."""

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
    traveler_group = body.get("traveler_group", "Adult")
    budget = body.get("budget", "")
    currency = body.get("currency", "Ringgit Malaysia")
    purpose = body.get("purpose", "")
    interests = body.get("interests", "")

    user_message = f"""User Inputs:
- State: {state}
- City: {city}
- Traveler Group: {traveler_group}
- Budget Level: {currency} {budget}
- Purpose: {purpose}
- Interests: {interests}

Please provide comprehensive transport and accessibility information."""

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
