import json
import boto3
import urllib.request
import urllib.parse

BEDROCK_MODEL_ID = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

# Coordinates for Malaysian cities/states for weather lookup
CITY_COORDS = {
    "selangor": (3.07, 101.52), "kuala lumpur": (3.14, 101.69), "penang": (5.42, 100.33),
    "george town": (5.42, 100.33), "langkawi": (6.35, 99.73), "johor": (1.49, 103.76),
    "johor bahru": (1.49, 103.76), "melaka": (2.19, 102.25), "malacca": (2.19, 102.25),
    "sabah": (5.98, 116.08), "kota kinabalu": (5.98, 116.08), "sarawak": (1.55, 110.36),
    "kuching": (1.55, 110.36), "pahang": (3.81, 103.33), "cameron highlands": (4.47, 101.38),
    "perak": (4.59, 101.09), "ipoh": (4.60, 101.07), "kedah": (6.12, 100.37),
    "kelantan": (6.13, 102.24), "kota bharu": (6.13, 102.24), "terengganu": (5.31, 103.13),
    "negeri sembilan": (2.73, 101.94), "perlis": (6.44, 100.20),
}

SYSTEM_PROMPT = """You are a travel summary assistant. Create a personalized travel summary based on the user's specific inputs. Do NOT include general knowledge - only summarize what's relevant to their trip.

Return a JSON object with these fields:
{
  "summary": "2-3 sentence personalized summary of their trip plan",
  "highlights": ["highlight 1", "highlight 2", "highlight 3"],
  "budget_verdict": "one sentence about their budget adequacy",
  "best_time_to_visit": "specific recommendation for their dates",
  "pro_tips": ["tip 1", "tip 2"]
}

Return ONLY valid JSON, no other text."""

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def get_weather(city, state, travel_dates):
    """Fetch real-time weather from Open-Meteo API."""
    location = city.lower() if city else state.lower()
    coords = CITY_COORDS.get(location, CITY_COORDS.get(state.lower(), (3.14, 101.69)))
    lat, lng = coords

    try:
        # Get 7-day forecast
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode&timezone=Asia/Kuala_Lumpur&forecast_days=7"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())

        daily = data.get("daily", {})
        temps_max = daily.get("temperature_2m_max", [])
        temps_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        codes = daily.get("weathercode", [])

        if not temps_max:
            return None

        avg_max = sum(temps_max) / len(temps_max)
        avg_min = sum(temps_min) / len(temps_min)
        total_rain = sum(precip)
        # Weather code interpretation
        # 0-1: clear, 2-3: cloudy, 45-48: fog/haze, 51-67: rain, 71-77: snow, 80-99: heavy rain/storm
        avg_code = sum(codes) / len(codes) if codes else 0

        if avg_code <= 3 and total_rain < 5:
            condition = "sunny"
            desc = "Clear skies with minimal rainfall expected"
        elif avg_code <= 48 or (total_rain < 15):
            condition = "cloudy"
            desc = "Partly cloudy with occasional light showers"
        elif total_rain > 30:
            condition = "rainy"
            desc = "Significant rainfall expected, bring an umbrella"
        else:
            condition = "cloudy"
            desc = "Mixed weather with some rain possible"

        return {
            "condition": condition,
            "description": desc,
            "temperature": f"{avg_min:.0f}-{avg_max:.0f}°C",
            "source": "Open-Meteo",
            "source_url": f"https://open-meteo.com/en/docs#latitude={lat}&longitude={lng}",
            "forecast_days": daily.get("time", [])[:3],
            "is_realtime": True,
        }
    except Exception:
        return None


def handler(event, context):
    method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method", "")
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    body = json.loads(event.get("body", "{}"))
    state = body.get("state", "Selangor")
    city = body.get("city", "")
    budget = body.get("budget", "")
    currency = body.get("currency", "Ringgit Malaysia")
    travel_dates = body.get("travel_dates", "")
    traveler_group = body.get("traveler_group", "Adult")
    purpose = body.get("purpose", "")
    interests = body.get("interests", "")
    num_travelers = body.get("num_travelers", "")

    # Fetch real weather
    weather = get_weather(city, state, travel_dates)
    if not weather:
        weather = {"condition": "sunny", "description": "Tropical climate, hot and humid", "temperature": "28-33°C", "source": "AI Estimate (monsoon patterns)", "source_url": "", "is_realtime": False}

    user_message = f"""Create a travel summary for:
- Destination: {state}, {city}
- Dates: {travel_dates}
- Budget: {currency} {budget} for {num_travelers} travelers
- Group: {traveler_group}
- Purpose: {purpose}
- Interests: {interests}

Return JSON only with summary, highlights, budget_verdict, best_time_to_visit, and pro_tips."""

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
            start = text.index('{')
            end = text.rindex('}') + 1
            summary = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            summary = {"summary": text, "highlights": [], "budget_verdict": "", "best_time_to_visit": "", "pro_tips": []}

        # Add real weather data
        summary["weather"] = weather

        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps(summary),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
