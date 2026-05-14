import json
import os
import boto3
from flask import Flask, request, Response, stream_with_context

app = Flask(__name__)

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6-20260217-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-1")

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
For Seniors: Medical facilities nearby, mobility considerations, slower-paced activities
For Families with Kids: Child safety in public places, kid-friendly facilities, emergency pediatric care
For Mixed Groups: Coordination tips, inclusive planning

**CULTURAL ETIQUETTE:**
- Respect for local customs
- Religious sensitivity
- Language basics (common phrases)

**EMERGENCY CONTACTS REMINDER:**
- Emergency: 999
- Tourism Police: 03-2149 6590

Format with clear sections and bullet points. Be practical and respectful."""


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
