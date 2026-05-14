import json
import base64
import random
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-5")

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
    specific_place = body.get("specific_place", "")
    image_prompt_input = body.get("image_prompt", "")

    place_detail = ""
    if specific_place:
        place_detail = f" Feature the specific landmark: {specific_place}."
    if image_prompt_input:
        place_detail += f" {image_prompt_input}"

    prompt = (
        f"A hyper realistic photograph of {state} in Malaysia.{place_detail} "
        f"The image should capture the essence of Malaysian tourism with bright colors, "
        f"cultural elements, and iconic scenery. Style: clean, modern tourism poster aesthetic "
        f"similar to official Malaysia Tourism promotional materials. Show landmarks, cultural "
        f"heritage, natural beauty, or urban landscapes as appropriate. "
        f"Shot from a 40mm camera, color image. Vibrant and inviting."
    )

    seed = random.randint(0, 2147483647)

    request_body = json.dumps({
        "taskType": "TEXT_IMAGE",
        "textToImageParams": {"text": prompt},
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": 1280,
            "width": 720,
            "cfgScale": 2.0,
            "seed": seed,
        },
    })

    try:
        response = bedrock.invoke_model(
            modelId="amazon.nova-canvas-v1:0",
            body=request_body,
        )
        result = json.loads(response["body"].read())
        image_base64 = result["images"][0]

        return {
            "statusCode": 200,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"image": image_base64}),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {**CORS_HEADERS, "Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
