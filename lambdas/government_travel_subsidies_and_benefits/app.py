import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-opus-4-6-v1"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-1")

SYSTEM_PROMPT = """You are a GovTech benefits advisor. Provide comprehensive, realistic information about Malaysian government subsidies, discounts, and travel benefits available to different groups.

Provide:

**PUBLIC TRANSPORT SUBSIDIES:**
- KTM/ETS: Student discounts (50% off), Senior discounts, monthly passes
- RapidKL: Concession fares for students, seniors, OKU (disabled persons)
- MyBAS: Free bus services in certain areas
- Touch n Go eWallet: Occasional cashback campaigns

**TOURISM INCENTIVES:**
- Domestic tourism packages and promotions
- State-specific tourism vouchers
- Hotel tax exemptions during promotional periods

**GROUP-SPECIFIC BENEFITS:**
For Students: Student ID discounts at museums (50-70% off), attractions discounts, youth hostel programs
For Seniors (60+): MyKad senior citizen discounts, government hospital rates (RM1 consultation), priority lanes
For Families: Child discounts (under 12), family packages at theme parks

**FREE OR SUBSIDIZED ATTRACTIONS:**
- National museums: Free or RM2-5 entry for Malaysians
- National parks: Subsidized entry for citizens
- Public beaches and recreational parks: Free

**HOW TO CLAIM:**
1. MyKad Verification: Bring IC for Malaysian citizen rates
2. Student ID: Valid student card for discounts
3. Senior Citizen: 60+ with MyKad automatically qualifies

Format with clear sections, specific RM amounts where applicable, and practical instructions."""

def handler(event, context):
body = json.loads(event.get("body", "{}"))
    state = body.get("state", "")
    city = body.get("city", "")
    budget = body.get("budget", "")
    currency = body.get("currency", "Ringgit Malaysia")
    traveler_group = body.get("traveler_group", "Adult")
    purpose = body.get("purpose", "")
    interests = body.get("interests", "")

    user_message = f"""User Inputs:
- Traveler Group: {traveler_group}
- State: {state}
- City: {city}
- Purpose: {purpose}
- Budget: {currency} {budget}
- Interests: {interests}

Please provide comprehensive information about government subsidies and travel benefits available."""

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
            "headers": {"Content-Type": "text/plain; charset=utf-8"},
            "body": text,
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(e)}),
        }
