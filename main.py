#!/usr/bin/env python3
"""
Estimate calories from an image using:
- Gemini (to detect food + portions)
- CalorieNinjas (to get nutrition info)

Usage:
    python calories_from_image.py
    # Make sure image.png is in the same folder
"""

import os
import requests
from google import genai


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBbvfLeskknidSbSGV0CzzOSLcpkRV4ZTY")
CALORIE_NINJAS_API_KEY = os.environ.get("CALORIE_NINJAS_API_KEY", "2NbmZe21MRuJlobJ6RQN1g==xy8H9xYNR8kbPMXV")

if not GEMINI_API_KEY:
    raise RuntimeError("Please set GEMINI_API_KEY environment variable.")
if not CALORIE_NINJAS_API_KEY:
    raise RuntimeError("Please set CALORIE_NINJAS_API_KEY environment variable.")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def build_food_query_from_image(image_path: str) -> str:
    """
    Uploads the image to Gemini and asks it to describe the visible food
    as a single text query suitable for CalorieNinjas.
    """
    # Upload the image (similar to your example, but using image.png)
    uploaded_file = client.files.upload(file=image_path)

    prompt = """
You are helping to estimate calories via the CalorieNinjas API.

Look at this image and identify all visible food and drinks.
Return a single English sentence that could be used as the `query` parameter
for CalorieNinjas, including approximate but realistic quantities and units.

Examples of valid outputs:
- "2 scrambled eggs and 1 slice of white toast and 1 tablespoon butter"
- "1 medium cheeseburger and 1 small serving of french fries and 1 can of cola"

Rules:
- ONLY output the query sentence.
- Do NOT add explanations, labels, bullets, or extra text.
- Do NOT include newlines.
"""

    response = client.models.generate_content(
        model="gemini-3-pro",
        contents=[prompt, uploaded_file],
    )

    query = (response.text or "").strip().replace("\n", " ")
    if not query:
        raise RuntimeError("Gemini returned an empty query. Try another image or prompt.")
    return query


def get_calories_from_query(query: str):
    """
    Calls CalorieNinjas API with the given food query and returns:
    - total_calories (float)
    - items (list of dicts with per-item nutrition)
    """
    api_url = "https://api.calorieninjas.com/v1/nutrition?query="
    url = api_url + requests.utils.quote(query)

    headers = {"X-Api-Key": CALORIE_NINJAS_API_KEY}
    response = requests.get(url, headers=headers)

    if response.status_code != requests.codes.ok:
        raise RuntimeError(
            f"CalorieNinjas error: {response.status_code} {response.text}"
        )

    data = response.json()
    items = data.get("items", [])
    total_calories = sum(float(item.get("calories", 0)) for item in items)
    return total_calories, items


def main():
    image_path = "images/image3.png"  # change if needed

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"📷 Analyzing image: {image_path}")

    # 1) Use Gemini to build a CalorieNinjas-friendly query
    query = build_food_query_from_image(image_path)
    print(f"\n🧠 Gemini food query:\n  {query}")

    # 2) Send query to CalorieNinjas and get nutrition info
    total_calories, items = get_calories_from_query(query)

    # 3) Print breakdown
    print("\n🥗 Calorie breakdown:")
    if not items:
        print("No food items returned from CalorieNinjas.")
    else:
        for item in items:
            name = item.get("name", "unknown item")
            calories = item.get("calories", 0)
            serving_size = item.get("serving_size_g", "N/A")
            print(f"  - {name}: {calories} kcal (serving: {serving_size} g)")

    print(f"\n🔥 Estimated total calories: {total_calories:.0f} kcal")
    print("\nFull item data from CalorieNinjas API:")
    print(items)

if __name__ == "__main__":
    main()
