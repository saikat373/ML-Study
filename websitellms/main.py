#from gradiointerface.mlg import genai_client
import gradio as gr
from google import genai
from google.genai import types
#import google.generativeai as genai
import os
from sarvamai import SarvamAI
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

genai_api_key = os.getenv("GEMINI_API_KEY")
nvidia_api_key = os.getenv("NVIDIA_API_KEY")
client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = nvidia_api_key
)

genai_client = genai.Client(api_key=genai_api_key)
print("GENAI client initialized successfully.")

def query_openai_vision(client, prompt, model="gemini-3.7-flash", max_tokens = 1000):
    request_payload = {
        "model":model,
        "contents":[
            prompt,
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

def openAI(message):
    request = [{
        "role":"user",
        "content":[
            {
                "type":"text",
                "text":message
            }
        ]
    }]


    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/deepseek-v4.1-flash",
            messages=request,
            temperature=1,
            top_p=0.95,
            max_tokens=262144,
            stream=False
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise e

""" sarvam_api_key = os.getenv("SARVAM_API_KEY")
sarvam_client = SarvamAI(api_subscription_key=sarvam_api_key)
print("Sarvam client initialized successfully.") """

test_prompt = "A father is 36 years old, and his son is 6 years old. In how many years will the father be exactly five times as old as his son?"
print("OPEN AI", openAI(test_prompt))

print("GENAI", query_openai_vision(genai_client, test_prompt))
