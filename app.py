import os
import tempfile

import streamlit as st
import requests
from google import genai


# ---------- CONFIG ----------
# Recommended: set these as environment variables before running:
#   export GEMINI_API_KEY="..."
#   export CALORIE_NINJAS_API_KEY="..."
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CALORIE_NINJAS_API_KEY = os.environ.get("CALORIE_NINJAS_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set in environment variables.")
if not CALORIE_NINJAS_API_KEY:
    st.error("CALORIE_NINJAS_API_KEY is not set in environment variables.")

client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)


# ---------- HELPERS ----------
def build_food_query_from_image(image_path: str) -> str:
    """Use Gemini to describe the food in a CalorieNinjas-friendly way."""
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

    uploaded_file = client.files.upload(file=image_path)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, uploaded_file],
    )

    query = (response.text or "").strip().replace("\n", " ")
    if not query:
        raise RuntimeError("Gemini returned an empty query.")
    return query


def call_calorieninjas(query: str):
    """Call CalorieNinjas API and return JSON + total calories."""
    api_url = "https://api.calorieninjas.com/v1/nutrition?query="
    url = api_url + requests.utils.quote(query)

    headers = {"X-Api-Key": CALORIE_NINJAS_API_KEY}
    resp = requests.get(url, headers=headers)

    if resp.status_code != requests.codes.ok:
        raise RuntimeError(
            f"CalorieNinjas error {resp.status_code}: {resp.text}"
        )

    data = resp.json()
    items = data.get("items", [])
    total_calories = sum(float(item.get("calories", 0)) for item in items)
    return data, total_calories


# ---------- STREAMLIT UI ----------
st.title("🍽️ Image → Calories (Gemini + CalorieNinjas)")

st.markdown(
    "Upload a meal image. The app will:\n"
    "1. Use **Gemini** to describe the food as text.\n"
    "2. Send that text to **CalorieNinjas**.\n"
    "3. Show the total calories and the **full API response**."
)

uploaded_image = st.file_uploader(
    "Upload an image of your meal",
    type=["png", "jpg", "jpeg"]
)

if uploaded_image:
    st.image(uploaded_image, caption="Uploaded image", use_column_width=True)

if st.button("Estimate calories"):
    if not uploaded_image:
        st.warning("Please upload an image first.")
    elif not (GEMINI_API_KEY and CALORIE_NINJAS_API_KEY):
        st.error("Please configure both GEMINI_API_KEY and CALORIE_NINJAS_API_KEY.")
    else:
        try:
            # Save uploaded file to a temporary path for Gemini
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(uploaded_image.getbuffer())
                tmp_path = tmp.name

            with st.spinner("Asking Gemini to describe the food..."):
                query = build_food_query_from_image(tmp_path)

            st.subheader("🧠 Gemini-generated food query")
            st.code(query, language="text")

            with st.spinner("Calling CalorieNinjas..."):
                data, total_calories = call_calorieninjas(query)

            st.subheader("🔥 Estimated total calories")
            st.metric("Total", f"{total_calories:.0f} kcal")

            # Optional: small breakdown
            items = data.get("items", [])
            if items:
                st.subheader("🥗 Item breakdown")
                for item in items:
                    name = item.get("name", "unknown item")
                    calories = item.get("calories", 0)
                    serving = item.get("serving_size_g", "N/A")
                    st.write(f"- **{name}** — {calories} kcal (serving: {serving} g)")

            st.subheader("📦 Full CalorieNinjas JSON response")
            st.json(data)

        except Exception as e:
            st.error(f"Error: {e}")
