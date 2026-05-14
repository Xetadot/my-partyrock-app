import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

SYSTEM_PROMPT = """You are a GovTech immigration and services officer. Provide essential government information for travelers to Malaysia.

Provide structured information:

**ENTRY AND IMMIGRATION:**
- For Malaysian citizens: Domestic travel requirements (ID/MyKad)
- For international visitors: General passport/visa guidance (advise to check with Malaysian Ministry of Home Affairs)

**HEALTH AND SAFETY:**
- Ministry of Health (MOH) guidelines
- Recommended vaccinations or health precautions
- Current health protocols if applicable

**EMERGENCY SERVICES:**
- Emergency hotline: 999
- Tourism Police: 03-2149 6590
- Nearest hospital/clinic information

**TOURISM AUTHORITY:**
- Malaysia Tourism (Tourism Malaysia/MOTAC) contact
- State tourism board information
- Official tourism website reference

**WEATHER AND CLIMATE:**
- Current season considerations
- Weather warnings or monsoon season alerts

**DISCLAIMER:** This is general guidance only. Verify with official sources:
- Malaysian Immigration (www.imi.gov.my)
- Ministry of Health (www.moh.gov.my)
- Tourism Malaysia (www.tourism.gov.my)

Format with clear headers and bullet points."""

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
    traveler_group = body.get("traveler_group", "Adult")

    user_message = f"""User Inputs:
- State: {state}
- City: {city}
- Purpose: {purpose}
- Traveler Group: {traveler_group}

Please provide essential government services and entry requirements information."""

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
