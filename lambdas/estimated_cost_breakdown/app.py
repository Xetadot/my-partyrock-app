import json
import os
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6-20260217-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

SYSTEM_PROMPT = """You are a GovTech financial planner. Create a detailed cost breakdown that matches the user's stated budget.

Create a cost breakdown in TABLE format:

| Category | Estimated Cost (RM) |
|----------|---------------------|
| Accommodation | XXX |
| Food and Dining | XXX |
| Transportation | XXX |
| Attractions and Activities | XXX |
| Shopping and Souvenirs | XXX |
| Emergency Buffer | XXX |
| TOTAL | MUST equal the budget provided |

CRITICAL REQUIREMENT: The total MUST equal the user's budget amount.

ALLOCATION GUIDELINES:
- Accommodation: 30-40% of budget
- Food: 25-30%
- Transport: 15-20%
- Activities: 15-20%
- Shopping: 5-10%
- Emergency: 5-10%

Adjust percentages based on:
- Traveler group (students = more budget-conscious, families = more on food and activities)
- Budget level (low/reasonable/high)
- State (Kuala Lumpur vs. rural areas)

Provide brief notes explaining the allocation strategy below the table."""


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
- Total Budget: {currency} {budget}
- Travelers: {num_travelers}
- Travel Dates: {travel_dates}
- Group Type: {traveler_group}
- State: {state}
- City: {city}
- Purpose: {purpose}
- Interests: {interests}

Please create a detailed cost breakdown table that totals exactly to the budget provided."""

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
