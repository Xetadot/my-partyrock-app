import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-opus-4-6-v1"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-1")

SYSTEM_PROMPT = """You are a travel photo curator. Given a destination, find 5 real, famous photographs or iconic views of that location.

Return ONLY a JSON array with exactly 5 objects. Each object must have:
- "title": short description of the photo/scene
- "location": specific place name
- "author": photographer name or "Tourism Malaysia" or "Local Photographer"
- "date": approximate year or season (e.g., "2024", "Monsoon Season")
- "search_query": a specific search term to find this image on Unsplash (e.g., "petronas towers night kuala lumpur")

Return ONLY valid JSON array, no other text. Example:
[{"title":"Sunset at KLCC Park","location":"KLCC Park, KL","author":"Ahmad Razali","date":"2024","search_query":"klcc park sunset kuala lumpur"}]"""

def handler(event, context):
body = json.loads(event.get("body", "{}"))
    state = body.get("state", "Selangor")
    city = body.get("city", "")
    interests = body.get("interests", "")

    location = city if city else state
    user_message = f"Find 5 iconic photographs of {location}, Malaysia. Interests: {interests}. Return JSON array only."

    try:
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "system": [{"type": "text", "text": SYSTEM_PROMPT}],
                "messages": [{"role": "user", "content": user_message}],
            }),
        )
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]

        # Try to parse JSON from response
        try:
            # Find JSON array in response
            start = text.index('[')
            end = text.rindex(']') + 1
            photos = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            photos = []

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"photos": photos}),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
