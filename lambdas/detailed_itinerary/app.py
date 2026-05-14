import json
import os
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6-20260217-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

SYSTEM_PROMPT = """You are a GovTech travel itinerary planner. Create a realistic, structured day-by-day travel plan.

Create an itinerary in TABLE format:

| Day | Time | Activity | Location | Notes |
|-----|------|----------|----------|-------|

ADAPTATION RULES:
- Students: Budget-friendly, energetic activities
- Adults: Balanced mix
- Seniors: Slower pace, accessible locations, rest breaks
- Family with Kids: Kid-friendly attractions, shorter activities
- Mixed Group: Diverse activities

Purpose alignment:
- Tourism: Attractions, culture, food
- Business: Include work time, professional venues
- Study: Educational sites, libraries, institutions

BUDGET ALIGNMENT:
- Activities must match budget classification
- Low budget: Free or cheap attractions, local food
- Reasonable: Mix of paid and free, mid-range dining
- High: Premium experiences, fine dining

Include morning, afternoon, and evening plans. Be realistic about timing and distances."""


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
    num_travelers = body.get("num_travelers", "")
    travel_dates = body.get("travel_dates", "")
    traveler_group = body.get("traveler_group", "Adult")
    purpose = body.get("purpose", "")
    interests = body.get("interests", "")

    user_message = f"""User Inputs:
- State: {state}
- City: {city}
- Travel Dates: {travel_dates}
- Budget: {currency} {budget}
- Travelers: {num_travelers}
- Group Type: {traveler_group}
- Purpose: {purpose}
- Interests: {interests}

Please create a detailed day-by-day itinerary in table format."""

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
