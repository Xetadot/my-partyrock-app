import json
import os
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6-20260217-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-1")

SYSTEM_PROMPT = """You are a GovTech travel assistant for Malaysia. Based on the user inputs, provide a welcoming introduction to their destination.

Generate a structured overview with:
1. **DESTINATION INTRODUCTION**: Brief welcome message about the state, incorporating city/place if provided
2. **WHY VISIT**: 3-4 key highlights of the destination
3. **BEST FOR**: Match destination to their purpose and interests

Keep it concise, official, and encouraging. Use proper formatting with headers and bullet points."""


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
    interests = body.get("interests", "")

    user_message = f"""User Inputs:
- State: {state}
- City: {city}
- Purpose: {purpose}
- Interests: {interests}

Please provide a welcoming travel overview for this destination."""

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
