import json
import os
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6-20260217-v1:0"
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
- State-specific tourism vouchers (check with state tourism boards)
- Hotel tax exemptions during promotional periods
- Tourism Malaysia campaigns and special offers

**GROUP-SPECIFIC BENEFITS:**
For Students: Student ID discounts at museums (50-70% off), attractions discounts, youth hostel programs, educational site free entry, KTMB student passes
For Seniors (60+): MyKad senior citizen discounts, government hospital rates (RM1 consultation), priority lanes, attraction discounts (50% at national museums)
For Families: Child discounts (under 12), family packages at theme parks, MySalam health protection scheme
For Mixed: Combination of applicable benefits

**FREE OR SUBSIDIZED ATTRACTIONS:**
- National museums: Free or RM2-5 entry for Malaysians
- National parks: Subsidized entry for citizens
- Public beaches and recreational parks: Free
- Government heritage sites: Reduced rates with MyKad

**HEALTHCARE COVERAGE:**
- Government clinics: RM1 consultation for citizens
- MySalam: Free health protection for B40 group
- Vaccination programs: Free or subsidized

**HOW TO CLAIM:**
1. MyKad Verification: Bring IC for Malaysian citizen rates
2. Student ID: Valid student card for discounts
3. Senior Citizen: 60+ with MyKad automatically qualifies
4. Digital Wallets: Register for government e-wallet initiatives

**ESTIMATED SAVINGS:**
Calculate approximate savings based on group type and budget.

**IMPORTANT:** These are general benefits. Availability may vary. Check with specific providers before travel.

Format with clear sections, specific RM amounts where applicable, and practical instructions."""


def build_cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST,OPTIONS",
    }


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=build_cors_headers())


@app.route("/", methods=["POST"])
def generate():
    body = request.get_json(force=True)
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

    def stream():
        response = bedrock.invoke_model_with_response_stream(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": [{"text": SYSTEM_PROMPT}],
                "messages": [{"role": "user", "content": user_message}],
            }),
        )
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                yield chunk["delta"].get("text", "")

    headers = build_cors_headers()
    headers["Content-Type"] = "text/plain; charset=utf-8"
    return Response(stream_with_context(stream()), headers=headers)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
