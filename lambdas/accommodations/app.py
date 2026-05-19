import json
import boto3

BEDROCK_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

SYSTEM_PROMPT = """You are a travel accommodation expert for Malaysia. Given a destination, find 8 real accommodations that CURRENTLY EXIST and are OPERATIONAL in that area.

CRITICAL RULES:
- ONLY suggest hotels that are CURRENTLY OPERATING (not closed/demolished)
- Use EXACT official hotel names as they appear on booking sites
- Mix: 2 budget (under RM150), 3 mid-range (RM150-400), 3 luxury (RM400+)
- Include the EXACT street/area name

Return ONLY a JSON array with exactly 8 objects:
- "name": exact official hotel name (e.g., "Hilton Kuala Lumpur", "Tune Hotel Georgetown")
- "type": "Hotel" or "Resort" or "Hostel" or "Boutique"
- "rating": rating out of 5 (e.g., "4.5")
- "price_range": price per night in MYR (e.g., "RM120-180")
- "address": specific street/area (e.g., "Jalan Sultan Ismail, KL")
- "source": alternate between "Expedia", "TripAdvisor", "Trip.com"
- "url": real URL (e.g., "https://www.tripadvisor.com/Hotel_Review-HOTELNAME")

EXAMPLES of REAL hotels by state:
- KL/Selangor: Hilton KL, Shangri-La KL, Traders Hotel, Tune Hotel KLIA, Hotel Istana
- Penang: Eastern & Oriental Hotel, Bayview Hotel Georgetown, Hard Rock Hotel Penang, Tune Hotel Penang
- Langkawi: The Datai, Meritus Pelangi Beach, Sheraton Langkawi
- Melaka: Hatten Hotel, Casa del Rio, Holiday Inn Melaka
- Sabah: Shangri-La Tanjung Aru, Hyatt Regency Kinabalu, Le Meridien KK
- Johor: Thistle JB, DoubleTree JB, Legoland Hotel

Return ONLY valid JSON array, no other text."""

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def geocode_photon(name, city):
    """Geocode using Amazon Location Service."""
    location_client = boto3.client("location", region_name="ap-southeast-5")
    q = f"{name}, {city}, Malaysia"
    try:
        response = location_client.search_place_index_for_text(
            IndexName="SmartTravelMalaysiaPlaces",
            Text=q,
            MaxResults=1,
        )
        results = response.get("Results", [])
        if results:
            coords = results[0]["Place"]["Geometry"]["Point"]  # [lng, lat]
            return coords[1], coords[0]
    except Exception:
        pass
    return None


def handler(event, context):
    method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method", "")
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    body = json.loads(event.get("body", "{}"))
    state = body.get("state", "Selangor")
    city = body.get("city", "")
    budget = body.get("budget", "")
    traveler_group = body.get("traveler_group", "Adult")

    # Input safety check
    all_input = f"{state} {city} {budget} {traveler_group}".lower()
    blocked = ['hitler','nazi','terrorism','bomb','kill','murder','genocide','porn','nude','sex','drugs','cocaine','weapon','gun','rape','isis','gore','blood','violence','torture','abuse']
    if any(term in all_input for term in blocked):
        return {
            "statusCode": 400,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"error": "Input contains inappropriate content. Please modify your request.", "accommodations": []}),
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

        # Geocode each accommodation using Photon, only keep found ones
        validated = []
        for accom in accommodations:
            coords = geocode_photon(accom.get("name", ""), location)
            if coords:
                accom["lat"] = coords[0]
                accom["lng"] = coords[1]
                validated.append(accom)

        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"accommodations": validated}),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
