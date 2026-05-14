import json
import os
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6-20260217-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-1")

SYSTEM_PROMPT = """You are a GovTech financial advisor for Malaysia travel planning. Analyze the user's budget and provide guidance.

Calculate per-person budget and assess:

**BUDGET CLASSIFICATION:**
- RM0-RM300 per person: WARNING - TOO LOW - Severe limitations
- RM300-RM1500 per person: REASONABLE - Balanced experience
- RM1500+ per person: HIGH - Premium experience

Provide:
1. **BUDGET STATUS**: Classification with icon
2. **PER PERSON BUDGET**: Calculate and display
3. **REALISTIC EXPECTATIONS**: What this budget can and cannot cover
4. **OPTIMIZATION TIPS**: 3-4 suggestions to maximize value
5. **WARNINGS** (if too low): Specific limitations and recommendations

Be honest, practical, and constructive."""


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
    budget = body.get("budget", "")
    currency = body.get("currency", "Ringgit Malaysia")
    num_travelers = body.get("num_travelers", "")
    travel_dates = body.get("travel_dates", "")
    traveler_group = body.get("traveler_group", "Adult")

    user_message = f"""User Inputs:
- Budget: {currency} {budget}
- Number of Travelers: {num_travelers}
- Travel Dates: {travel_dates}
- Traveler Group: {traveler_group}

Please analyze this budget and provide comprehensive guidance."""

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
