import json
import boto3

location_client = boto3.client("location", region_name="ap-southeast-5")
INDEX_NAME = "SmartTravelMalaysiaPlaces"

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST,OPTIONS",
}


def handler(event, context):
    method = event.get("httpMethod") or event.get("requestContext", {}).get("http", {}).get("method", "")
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    body = json.loads(event.get("body", "{}"))
    query = body.get("query", "")
    bias_lat = body.get("lat", 3.14)  # Default: Malaysia center
    bias_lng = body.get("lng", 101.69)

    if not query:
        return {
            "statusCode": 400,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"error": "query is required"}),
        }

    try:
        response = location_client.search_place_index_for_text(
            IndexName=INDEX_NAME,
            Text=query,
            MaxResults=5,
            BiasPosition=[bias_lng, bias_lat],
        )

        results = []
        for place in response.get("Results", []):
            p = place["Place"]
            coords = p["Geometry"]["Point"]  # [lng, lat]
            results.append({
                "name": p.get("Label", query),
                "lat": coords[1],
                "lng": coords[0],
                "address": p.get("Label", ""),
                "category": p.get("Categories", [""])[0] if p.get("Categories") else "",
            })

        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"results": results}),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
