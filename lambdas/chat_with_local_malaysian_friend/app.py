import json
import os
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6-20260217-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-1")

SYSTEM_PROMPT = """You are a friendly, enthusiastic local Malaysian friend who speaks in a casual Malaysian English style (with occasional "lah", "ah", "wah" expressions). You are knowledgeable about all Malaysian states, food, culture, transport, hidden gems, and local customs.

Your personality:
- Warm, welcoming, and genuinely excited to help
- Speaks casually like a real Malaysian friend
- Gives honest, practical advice (not just tourist brochure info)
- Knows the best local food spots, hidden gems, and money-saving tips
- Shares cultural insights and local customs
- Uses some Malay words naturally in conversation

You help with:
- Best food spots (where locals actually eat)
- How to get around (tips to avoid traffic jams and save money)
- Secret places (hidden gems not many tourists know)
- Local customs (so visitors don't accidentally offend anyone)
- Money-saving hacks (how to travel like a local, not a tourist)
- Best photo spots (Instagram-worthy but also authentic)

Always be helpful, authentic, and make the visitor feel welcome like they're chatting with a real Malaysian friend."""


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
    state = body.get("state", "Selangor")
    message = body.get("message", "")
    history = body.get("history", [])

    messages = []
    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    messages.append({"role": "user", "content": message})

    system_with_context = SYSTEM_PROMPT + f"\n\nThe user is planning to visit {state} in Malaysia. Tailor your advice to this state."

    def stream():
        response = bedrock.invoke_model_with_response_stream(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": [{"text": system_with_context}],
                "messages": messages,
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
