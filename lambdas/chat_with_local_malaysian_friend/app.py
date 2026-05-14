import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

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

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def handler(event, context):
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    body = json.loads(event.get("body", "{}"))
    state = body.get("state", "Selangor")
    message = body.get("message", "")
    history = body.get("history", [])

    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    system_with_context = SYSTEM_PROMPT + f"\n\nThe user is planning to visit {state} in Malaysia. Tailor your advice to this state."

    try:
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "system": [{"type": "text", "text": system_with_context}],
                "messages": messages,
            }),
        )
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]

        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": "text/plain; charset=utf-8"},
            "body": text,
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": CORS_HEADERS,
            "body": json.dumps({"error": str(e)}),
        }
