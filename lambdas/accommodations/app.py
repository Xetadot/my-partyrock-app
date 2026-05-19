import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

# Agoda city IDs for Malaysian cities
AGODA_CITY_IDS = {
    "kuala lumpur": 14014, "kl": 14014, "georgetown": 16087, "george town": 16087,
    "penang": 16087, "langkawi": 15246, "melaka": 15562, "malacca": 15562,
    "johor bahru": 18628, "jb": 18628, "kota kinabalu": 17170, "kk": 17170,
    "kuching": 15564, "cameron highlands": 15247, "ipoh": 15248,
    "kuantan": 15249, "kota bharu": 15250, "kuala terengganu": 15251,
    "seremban": 15252, "shah alam": 14014, "petaling jaya": 14014,
    "genting highlands": 15253, "port dickson": 15254, "tioman": 15255,
    "perhentian": 15256, "sabah": 17170, "sarawak": 15564,
    "selangor": 14014, "pahang": 15249, "perak": 15248, "kedah": 15246,
    "kelantan": 15250, "terengganu": 15251, "negeri sembilan": 15252,
    "johor": 18628, "perlis": 15257,
}

SYSTEM_PROMPT = """You are a travel accommodation expert for Malaysia. Given a destination, find 8 real accommodations that CURRENTLY EXIST and are OPERATIONAL in that area.

CRITICAL RULES:
- ONLY suggest hotels that are CURRENTLY OPERATING (not closed/demolished)
- Use EXACT official hotel names as they appear on Agoda
- Mix: 2 budget (under RM150), 3 mid-range (RM150-400), 3 luxury (RM400+)
- Include the EXACT street/area name

Return ONLY a JSON array with exactly 8 objects:
- "name": exact official hotel name as on Agoda (e.g., "Eastern & Oriental Hotel", "Tune Hotel Georgetown")
- "type": "Hotel" or "Resort" or "Hostel" or "Boutique"
- "rating": rating out of 5 (e.g., "4.5")
- "price_range": price per night in MYR (e.g., "RM120-180")
- "address": specific street/area (e.g., "10 Lebuh Farquhar, Georgetown")

EXAMPLES of REAL hotels by city:
- Georgetown: Eastern & Oriental Hotel, Bayview Hotel Georgetown, Hard Rock Hotel Penang, Tune Hotel Georgetown, Hotel & You, Armenian Street Heritage Hotel, Ren i Tang Heritage Inn, Cititel Penang
- KL: Hilton KL, Shangri-La KL, Traders Hotel, Tune Hotel KLIA2, Hotel Istana, Mandarin Oriental KL, The RuMa Hotel
- Langkawi: The Datai, Meritus Pelangi Beach, Sheraton Langkawi, The Andaman, Four Seasons Langkawi
- Melaka: Hatten Hotel, Casa del Rio, Holiday Inn Melaka, The Majestic Malacca, Hotel & You Melaka
- Sabah: Shangri-La Tanjung Aru, Hyatt Regency Kinabalu, Le Meridien KK, Gaya Island Resort

Return ONLY valid JSON array, no other text."""

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def get_agoda_city_id(city, state):
    """Get Agoda city ID from city/state name."""
    for key in [city.lower(), state.lower()]:
        if key in AGODA_CITY_IDS:
            return AGODA_CITY_IDS[key]
    # Try partial match
    for k, v in AGODA_CITY_IDS.items():
        if k in city.lower() or city.lower() in k:
            return v
    return 14014  # Default to KL


def handler(event, context):
    method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method", "")
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    body = json.loads(event.get("body", "{}"))
    state = body.get("state", "Selangor")
    city = body.get("city", "")
    budget = body.get("budget", "")
    traveler_group = body.get("traveler_group", "Adult")
    depart_date = body.get("travel_dates", "").split(" to ")[0] if " to " in body.get("travel_dates", "") else ""
    return_date = body.get("travel_dates", "").split(" to ")[1] if " to " in body.get("travel_dates", "") else ""
    num_travelers = body.get("num_travelers", "2")

    # Input safety check
    all_input = f"{state} {city} {budget} {traveler_group}".lower()
    blocked = ['hitler','nazi','terrorism','bomb','kill','murder','genocide','porn','nude','sex','drugs','cocaine','weapon','gun','rape','isis','gore','blood','violence','torture','abuse']
    if any(term in all_input for term in blocked):
        return {
            "statusCode": 400,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"error": "Input contains inappropriate content.", "accommodations": []}),
        }

    location = city if city else state
    user_message = f"Find 8 real, currently operating accommodations in {location}, Malaysia. Budget: RM{budget}. Group: {traveler_group}. Return JSON array only."

    try:
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 3000,
                "system": [{"type": "text", "text": SYSTEM_PROMPT}],
                "messages": [{"role": "user", "content": user_message}],
            }),
        )
        result = json.loads(response["body"].read())
        text = result["content"][0]["text"]

        try:
            start = text.index('[')
            end = text.rindex(']') + 1
            accommodations = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            accommodations = []

        # Add Agoda booking URL to each accommodation
        agoda_city_id = get_agoda_city_id(location, state)
        for accom in accommodations:
            check_in = depart_date or "2026-06-01"
            check_out = return_date or "2026-06-02"
            adults = num_travelers or "2"
            accom["source"] = "Agoda"
            accom["url"] = f"https://www.agoda.com/search?city={agoda_city_id}&checkIn={check_in}&checkOut={check_out}&rooms=1&adults={adults}&children=0&currency=MYR&textToSearch={location}"

        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"accommodations": accommodations}),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
