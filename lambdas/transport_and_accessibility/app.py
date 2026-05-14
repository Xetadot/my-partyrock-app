import json
import os
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6-20260217-v1:0"
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
