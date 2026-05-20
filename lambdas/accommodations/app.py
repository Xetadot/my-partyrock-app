import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}

# Agoda city IDs for Malaysian cities
AGODA_CITY_IDS = {
    "kuala lumpur": 14014, "kl": 14014, "georgetown": 16087, "george town": 16087,
    "penang": 16087, "langkawi": 15246, "melaka": 15562, "malacca": 15562,
    "johor bahru": 18628, "jb": 18628, "kota kinabalu": 17170, "kk": 17170,
    "kuching": 15564, "cameron highlands": 15247, "ipoh": 15248,
    "kuantan": 15249, "kota bharu": 15250, "kuala terengganu": 15251,
    "seremban": 15252, "shah alam": 14014, "petaling jaya": 14014,
    "sabah": 17170, "sarawak": 15564, "selangor": 14014, "pahang": 15249,
    "perak": 15248, "kedah": 15246, "kelantan": 15250, "terengganu": 15251,
    "negeri sembilan": 15252, "johor": 18628, "perlis": 15257,
}


def get_agoda_city_id(city, state):
    for key in [city.lower(), state.lower()]:
        if key in AGODA_CITY_IDS:
            return AGODA_CITY_IDS[key]
    for k, v in AGODA_CITY_IDS.items():
        if k in city.lower() or city.lower() in k:
            return v
    return 14014


def handler(event, context):
    method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method", "")
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    body = json.loads(event.get("body", "{}"))
    state = body.get("state", "Selangor")
    city = body.get("city", "")
    budget = body.get("budget", "")
    traveler_group = body.get("traveler_group", "Adult")
    travel_dates = body.get("travel_dates", "")
    num_travelers = body.get("num_travelers", "2")

    # Input safety check
    all_input = f"{state} {city} {budget} {traveler_group}".lower()
    blocked = ['hitler','nazi','terrorism','bomb','kill','murder','genocide','porn','nude','sex','drugs','weapon','gun','rape','isis','gore','blood','violence','torture']
    if any(term in all_input for term in blocked):
        return {
            "statusCode": 400,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"error": "Inappropriate input.", "accommodations": []}),
        }

    location = city if city else state
    agoda_city_id = get_agoda_city_id(location, state)

    # Use Converse API with web search tool
    system_prompt = f"""You are a hotel search assistant. Search the web for REAL hotels currently available in {location}, {state}, Malaysia.

CRITICAL: Hotels MUST be located in {location}, {state}. Do NOT return hotels from other cities or states.

RULES:
- Search ONCE for "hotels in {location} {state} Malaysia 2025" — do NOT search multiple times
- Return ONLY hotels that are physically located in {location}, {state}
- If a hotel is in Kuala Lumpur but user asked for Penang, DO NOT include it
- Return exactly 5 hotels as a JSON array
- Mix budget and luxury options
- Double-check: every hotel must be in {location}

Each hotel object must have:
- "name": exact hotel name from search results
- "type": "Hotel" or "Resort" or "Hostel" or "Boutique"
- "rating": rating if found, otherwise "4.0"
- "price_range": price if found, otherwise estimate (e.g., "RM150-250")
- "address": location/area from search results

Return ONLY the JSON array, nothing else. Example:
[{{"name":"Eastern & Oriental Hotel","type":"Hotel","rating":"4.7","price_range":"RM400-800","address":"10 Lebuh Farquhar, Georgetown"}}]"""

    try:
        response = bedrock.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": system_prompt}],
            messages=[{
                "role": "user",
                "content": [{"text": f"Search for 5 real hotels ONLY in {location}, {state}, Malaysia. NOT in KL or other cities. Budget: RM{budget}. Return JSON array only."}]
            }],
            toolConfig={
                "tools": [{
                    "toolSpec": {
                        "name": "web_search",
                        "description": "Search the web for current hotel information",
                        "inputSchema": {
                            "json": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "Search query"}
                                },
                                "required": ["query"]
                            }
                        }
                    }
                }]
            },
            inferenceConfig={"maxTokens": 2000},
        )

        # Extract text from response
        text = ""
        for block in response.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                text += block["text"]

        # Parse JSON
        accommodations = []
        try:
            start = text.index('[')
            end = text.rindex(']') + 1
            accommodations = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            pass

        # If web search didn't return results, fallback to regular invoke
        if not accommodations:
            fallback_response = bedrock.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 2000,
                    "system": [{"type": "text", "text": f"List 5 real, well-known hotels ONLY in {location}, {state}, Malaysia. NOT hotels from other cities. Return ONLY a JSON array with objects having: name, type, rating, price_range, address. Use real hotel names only."}],
                    "messages": [{"role": "user", "content": f"5 real hotels in {location}, {state}, Malaysia only. Budget RM{budget}. JSON array only."}],
                }),
            )
            fb_result = json.loads(fallback_response["body"].read())
            fb_text = fb_result["content"][0]["text"]
            try:
                start = fb_text.index('[')
                end = fb_text.rindex(']') + 1
                accommodations = json.loads(fb_text[start:end])
            except (ValueError, json.JSONDecodeError):
                pass

        # Add Agoda URLs
        depart = travel_dates.split(" to ")[0] if " to " in travel_dates else "2026-06-01"
        ret = travel_dates.split(" to ")[1] if " to " in travel_dates else "2026-06-02"
        for accom in accommodations:
            accom["source"] = "Agoda"
            accom["url"] = f"https://www.agoda.com/search?city={agoda_city_id}&checkIn={depart}&checkOut={ret}&rooms=1&adults={num_travelers}&children=0&currency=MYR&textToSearch={location}"

        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"accommodations": accommodations}),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"error": str(e), "accommodations": []}),
        }
