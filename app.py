#from openai import OpenAI
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
from IPython.display import display, Markdown
from PIL import Image
import base64
import io

load_dotenv()

openai_api_key = os.getenv("GEMINI_API_KEY")

openai_client = genai.Client(api_key=openai_api_key)
print("OpenAI client initialized successfully.")

def print_markdown(text):
    display(Markdown(text))

image_path = "image/food.jpg"  # Replace with your image path
img = Image.open(image_path)

print(f"Image '{image_path}' loaded successfully.")
print(f"Format {img.format}, Size {img.size}, Mode {img.mode}")

#display(img)

image_to_analyze = img  # Use the loaded image for analysis

def encode_image_to_base64(image):
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"File '{image}' does not exist.")
        with open(image, "rb") as img:
            image_data = img.read()
            return base64.b64encode(image_data).decode("utf-8")
    elif isinstance(image, Image.Image):
        buffered = io.BytesIO()
        image_format = image.format or "JPEG"  # Default to JPEG if format is not set
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        raise ValueError("Input must be a file path or a PIL Image object.")


encoded_image = encode_image_to_base64(image_to_analyze)
print("Image encoded to base64 successfully.")
print(f"Encoded image length: {len(encoded_image)} characters")


def query_openai_vision(client, image, prompt, model="gemini-3.7-flash", max_tokens = 1000):
    encoded_image = encode_image_to_base64(image)
    """ request_payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens
    } """
    image_bytes = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(image_bytes))
    image.load()
    request_payload = {
        "model":model,
        "contents":[
            prompt,
            image
        ],
        "config": types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            )
        )
    }
    print("Querying payload prepared for genai.")
    try:
        response = client.models.generate_content(**request_payload)
        return response.text
    except Exception as e:
        return f"Error querying OpenAI Vision API: {str(e)}"


food_recognition_prompt = """
# Nutritional Analysis Task

## Context
You are a nutrition expert analyzing food images to provide accurate nutritional information.

## Instructions
Analyze the food item in the image and provide estimated nutritional information based on your knowledge.

## Input
- An image of a food item

## Output
Provide the following estimated nutritional information for a typical serving size or per 100g:
- food_name (string)
- serving_description (string, e.g., '1 slice', '100g', '1 cup')
- calories (float)
- fat_grams (float)
- protein_grams (float)
- confidence_level (string: 'High', 'Medium', or 'Low')
- sugar_grams (float)
- fiber_grams (float)

**IMPORTANT:** Respond ONLY with a single JSON object containing these fields. Do not include any other text, explanations, or apologies. The JSON keys must match exactly: "food_name", "serving_description", "calories", "fat_grams", "protein_grams", "confidence_level", "sugar_grams", "fiber_grams". If you cannot estimate a value, use `null`.

Example valid JSON response:
{
  "food_name": "Banana",
  "serving_description": "1 medium banana (approx 118g)",
  "calories": 105.0,
  "fat_grams": 0.4,
  "protein_grams": 1.3,
  "confidence_level": "High",
  "fiber_grams": 3.1,
  "sugar_grams": 14.4
}
"""

openai_description = query_openai_vision(openai_client, image_to_analyze, food_recognition_prompt)
print(openai_description)
print("OpenAI Vision API query completed.")