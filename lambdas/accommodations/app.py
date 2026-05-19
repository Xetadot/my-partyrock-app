import json
import boto3
import urllib.request
import urllib.parse

BEDROCK_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

SYSTEM_PROMPT = """You are a travel accommodation expert for Malaysia. Given a destination, find 5 real, well-known accommodations in that area.

IMPORTANT: Only suggest hotels/hostels that are FAMOUS and WELL-KNOWN in the area. Use major chain hotels or popular landmarks that would appear on any map service. Avoid obscure or recently closed places.

Return ONLY a JSON array with exactly 5 objects. Each object must have:
- "name": real hotel/hostel/resort name (use the official name)
- "type": "Hotel" or "Resort" or "Hostel" or "Homestay"
- "rating": rating out of 5 (e.g., "4.5")
- "price_range": approximate price per night in MYR (e.g., "RM120-180")
- "address": specific area/street name in the city
- "source": "Expedia" or "TripAdvisor" or "Trip.com"
- "url": booking URL (https://www.tripadvisor.com/Hotels-g-CITY or https://www.expedia.com/CITY-Hotels.d-Hotel-Search)

ONLY use well-known hotels like: Shangri-La, Hilton, Holiday Inn, Hard Rock, Sunway, Dorsett, DoubleTree, Marriott, Hyatt, Tune Hotel, OYO, etc. or famous local boutique hotels.
Return ONLY valid JSON array, no other text."""

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def geocode(name, city):
    """Geocode a place using Nominatim. Returns (lat, lng) or None."""
    q = f"{name}, {city}, Malaysia"
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(q)}&format=json&limit=1&countrycodes=my"
        req = urllib.request.Request(url, headers={"User-Agent": "SmartTravelMalaysia/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            if data and len(data) > 0:
                return float(data[0]["lat"]), float(data[0]["lon"])
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

    location = city if city else state
    user_message = f"Find 5 real, well-known accommodations in {location}, Malaysia. Budget: RM{budget}. Group: {traveler_group}. Return JSON array only."

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

        # Geocode each accommodation, only keep ones we can find
        validated = []
        for accom in accommodations:
            coords = geocode(accom["name"], location)
            if coords:
                accom["lat"] = coords[0]
                accom["lng"] = coords[1]
                validated.append(accom)
            if len(validated) >= 5:
                break

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
