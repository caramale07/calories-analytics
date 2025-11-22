#!/usr/bin/env python3
"""
Estimate calories and macros (calories, protein, carbs, fat)
from an image using ONLY Gemini.

- Gemini detects food + portions
- Gemini estimates nutrition values from its own knowledge
- Returns structured JSON validated via Pydantic

Usage:
    streamlit run app_gemini_only.py
"""

import os
import tempfile
from typing import List, Optional

import streamlit as st
from google import genai
from pydantic import BaseModel, Field


# ---------- Pydantic models ----------

class FoodItem(BaseModel):
    name: str = Field(description="Name of the detected food or drink item.")
    quantity_g: float = Field(description="Estimated portion size in grams.")
    calories_kcal: float = Field(description="Estimated calories in kilocalories.")
    protein_g: float = Field(description="Estimated protein in grams.")
    carbs_g: float = Field(description="Estimated carbohydrates in grams.")
    fat_g: float = Field(description="Estimated fat in grams.")
    reasoning_short: Optional[str] = Field(
        default=None,
        description="Short explanation of how this estimate was made (high-level).",
    )


class NutritionEstimate(BaseModel):
    items: List[FoodItem]
    total_calories_kcal: float = Field(description="Sum of calories of all items.")
    total_protein_g: float = Field(description="Sum of protein of all items.")
    total_carbs_g: float = Field(description="Sum of carbohydrates of all items.")
    total_fat_g: float = Field(description="Sum of fat of all items.")
    notes: Optional[str] = Field(
        default=None,
        description="Optional high-level notes or assumptions about the estimates.",
    )


# ---------- Gemini client ----------

# Try to get the API key from Streamlit secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    # Fallback to environment variable or None
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize client only if key is present (handled in main for UI feedback)
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_KEY_HERE":
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    client = None


# ---------- Core function ----------

def estimate_nutrition_with_gemini(image_path: str) -> NutritionEstimate:
    """
    Uploads the image to Gemini and asks it to:
      - detect visible food items
      - estimate mass in grams
      - estimate calories, protein, carbs, fat
      - provide short reasoning
      - return STRICT JSON according to NutritionEstimate schema

    Returns a NutritionEstimate Pydantic model.
    """
    if not client:
        raise ValueError("Gemini API Key not configured.")

    uploaded_file = client.files.upload(file=image_path)

    prompt = """
You are a nutrition expert estimating calories and macros from a meal photo.

TASK:
1. Look at the image and identify each visible food or drink item that a person would reasonably eat or drink.
2. For each item, estimate a realistic portion size in grams.
3. Using typical nutritional values, estimate for EACH item:
   - calories (kcal)
   - protein (g)
   - carbohydrates (g)
   - fat (g)
4. Compute totals by summing over all items.
5. For each item, provide a brief, high-level explanation of the key assumptions.
6. If something is uncertain, make a reasonable assumption and mention it briefly in the explanation or in the overall notes.

CONSTRAINTS:
- Keep explanations concise (no long step-by-step reasoning).
- Use ONLY the provided JSON schema; do not add extra fields.
- If you are unsure about an exact value, pick a reasonable estimate rather than leaving it empty.
"""

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[prompt, uploaded_file],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": NutritionEstimate.model_json_schema(),
        },
    )

    # Gemini should now return pure JSON matching the schema
    return NutritionEstimate.model_validate_json(response.text)


# ---------- Streamlit App ----------

def main():
    st.set_page_config(page_title="Gemini Calories Estimator", page_icon="🥗", layout="wide")
    
    st.title("🥗 Gemini Calories Estimator")
    st.markdown("Upload a food image to estimate calories and macros using Gemini 2.5 Pro.")

    # Check API Key
    if not client:
        st.error("🚨 `GEMINI_API_KEY` is missing.")
        st.info("Please set it in `.streamlit/secrets.toml` or as an environment variable.")
        return

    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        # Layout: Image on left, Results on right
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            analyze_button = st.button("🔍 Analyze Nutrition", type="primary", use_container_width=True)

        if analyze_button:
            with col2:
                st.subheader("� Analysis Logs")
                log_container = st.container()
                
                with log_container:
                    st.write("⚙️ Processing image...")
                    
                    # Save uploaded file to a temporary file
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        st.write(f"📂 Saved temporary file: `{tmp_path}`")
                        st.write("🚀 Sending to Gemini for analysis...")
                        
                        with st.spinner("Gemini is analyzing the food..."):
                            estimate = estimate_nutrition_with_gemini(tmp_path)
                        
                        st.success("✅ Analysis Complete!")
                        
                        # --- Display Results ---
                        st.divider()
                        st.subheader("🔥 Estimated Totals")
                        
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Calories", f"{estimate.total_calories_kcal:.0f} kcal")
                        m2.metric("Protein", f"{estimate.total_protein_g:.1f} g")
                        m3.metric("Carbs", f"{estimate.total_carbs_g:.1f} g")
                        m4.metric("Fat", f"{estimate.total_fat_g:.1f} g")

                        st.subheader("🥗 Item Breakdown")
                        if not estimate.items:
                            st.warning("No items detected.")
                        else:
                            for item in estimate.items:
                                with st.expander(f"**{item.name}** - {item.calories_kcal:.0f} kcal"):
                                    st.markdown(f"""
                                    - **Quantity:** {item.quantity_g:.1f}g
                                    - **Protein:** {item.protein_g:.1f}g
                                    - **Carbs:** {item.carbs_g:.1f}g
                                    - **Fat:** {item.fat_g:.1f}g
                                    """)
                                    if item.reasoning_short:
                                        st.caption(f"_{item.reasoning_short}_")

                        if estimate.notes:
                            st.info(f"**Notes:** {estimate.notes}")

                    except Exception as e:
                        st.error(f"❌ An error occurred: {e}")
                    finally:
                        # Cleanup temp file
                        if 'tmp_path' in locals() and os.path.exists(tmp_path):
                            os.remove(tmp_path)
                            st.write("🧹 Cleanup: Temporary file removed.")


if __name__ == "__main__":
    main()
