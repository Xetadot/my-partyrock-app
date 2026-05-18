import json
import random
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

def handler(event, context):
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
        f"The image captures the essence of Malaysian tourism with bright colors, "
        f"cultural elements, and iconic scenery. Clean, modern tourism poster aesthetic "
        f"similar to official Malaysia Tourism promotional materials. Landmarks, cultural "
        f"heritage, natural beauty, urban landscapes. "
        f"Shot from a 40mm camera, color image. Vibrant and inviting."
    )

    seed = random.randint(0, 2147483647)

    request_body = json.dumps({
        "prompt": prompt,
        "mode": "text-to-image",
        "aspect_ratio": "9:16",
        "output_format": "png",
        "seed": seed,
    })

    try:
        response = bedrock.invoke_model(
            modelId="stability.stable-image-ultra-v1:1",
            body=request_body,
            accept="application/json",
            contentType="application/json",
        )
        result = json.loads(response["body"].read())
        image_base64 = result["images"][0]

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"image": image_base64}),
        }
    except Exception as e:
        import traceback
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e), "trace": traceback.format_exc()}),
        }
