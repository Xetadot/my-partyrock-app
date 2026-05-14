import json
import os
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6-20260217-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-1")

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
    purpose = body.get("purpose", "")
    traveler_group = body.get("traveler_group", "Adult")

    user_message = f"""User Inputs:
- State: {state}
- City: {city}
- Purpose: {purpose}
- Traveler Group: {traveler_group}

Please provide essential government services and entry requirements information."""

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
