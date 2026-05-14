import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

SYSTEM_PROMPT = """You are a GovTech travel assistant for Malaysia. Based on the user inputs, provide a welcoming introduction to their destination.

Generate a structured overview with:
1. **DESTINATION INTRODUCTION**: Brief welcome message about the state, incorporating city/place if provided
2. **WHY VISIT**: 3-4 key highlights of the destination
3. **BEST FOR**: Match destination to their purpose and interests

Keep it concise, official, and encouraging. Use proper formatting with headers and bullet points."""

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
    purpose = body.get("purpose", "")
    interests = body.get("interests", "")

    user_message = f"""User Inputs:
- State: {state}
- City: {city}
- Purpose: {purpose}
- Interests: {interests}

Please provide a welcoming travel overview for this destination."""

    try:
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "system": [{"type": "text", "text": SYSTEM_PROMPT}],
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
