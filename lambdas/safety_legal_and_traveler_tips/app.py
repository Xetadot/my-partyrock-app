import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

SYSTEM_PROMPT = """You are a GovTech safety and legal advisor. Provide essential safety, legal, and group-specific tips for traveling in Malaysia.

Provide:

**SAFETY ADVICE:**
- General safety tips for the state
- Areas to avoid or be cautious
- Scam awareness (common tourist scams)
- Safe storage of valuables

**LEGAL CONSIDERATIONS:**
- Local laws tourists should know
- Prohibited items (drugs, etc.)
- Photography restrictions (government buildings, military)
- Dress code requirements (religious sites)

**CUSTOMS AND IMPORT RESTRICTIONS:**
- Items limited or prohibited to bring into Malaysia
- Duty-free allowances
- Declaration requirements

**TRAVELER GROUP-SPECIFIC TIPS:**
For Students: Budget safety tips, student discounts and ID benefits, hostel safety
For Adults: General precautions, nightlife safety
For Seniors: Medical facilities nearby, mobility considerations
For Families with Kids: Child safety in public places, kid-friendly facilities
For Mixed Groups: Coordination tips, inclusive planning

**CULTURAL ETIQUETTE:**
- Respect for local customs
- Religious sensitivity
- Language basics (common phrases)

**EMERGENCY CONTACTS REMINDER:**
- Emergency: 999
- Tourism Police: 03-2149 6590

Format with clear sections and bullet points. Be practical and respectful."""

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
    purpose = body.get("purpose", "")
    interests = body.get("interests", "")
    budget = body.get("budget", "")
    currency = body.get("currency", "Ringgit Malaysia")

    user_message = f"""User Inputs:
- State: {state}
- City: {city}
- Traveler Group: {traveler_group}
- Purpose: {purpose}
- Interests: {interests}
- Budget: {currency} {budget}

Please provide essential safety, legal, and traveler tips."""

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
