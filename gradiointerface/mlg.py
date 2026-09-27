import gradio as gr
import os
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

genai_api_key = os.getenv("GEMINI_API_KEY")

genai_client = genai.Client(api_key=genai_api_key)
print("GENAI client initialized successfully.")

explanation_levels = {
    1: "like I'm 5 years old",
    2: "like I'm 10 years old",
    3: "like a high school student",
    4: "like a college student",
    5: "like an expert in the field",
}

def get_ai_tutor_response(user_question,explanation_level):
    level_instruction = explanation_levels.get(explanation_level, "like a high school student")
    system_prompt = (
        "You are an AI tutor that provides clear and concise explanations to help users understand complex topics. "
        "Your responses should be informative, engaging, irratative, lazy and easy to follow. "
        "Provide examples when necessary and avoid unnecessary jargon."
        f"Explain the following concept {level_instruction}"
    )

    request_payload = {
        "model": "gemini-3.5-flash-lite",
        "contents": user_question,
        "config": types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=1000,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            temperature=0.7
        )
    }
    """ This is used by streaming to yield the response in chunks. """
    full_response = ""
    try:
        response = genai_client.models.generate_content_stream(**request_payload)

        for chunk in response:
            if chunk.text:
                full_response += chunk.text
                yield full_response
    except Exception as e:
        yield f"Error querying AI tutor: {str(e)}"




ai_tutor_interface = gr.Interface(
    fn=get_ai_tutor_response,
    inputs=[
        gr.Textbox(lines=5, placeholder="Ask your question here...", label="Your Question"),
        gr.Slider(minimum=1, maximum=5, step=1, label="Explanation Level", value=3)
    ],
    outputs=gr.TextArea(label="AI Tutor Response", lines=10, placeholder="The AI tutor's response will appear here..."),
    title="🙂 AI Tutor with Streaming",
    description="Ask any question and get a clear and concise explanation from the AI tutor.",
    flagging_mode="never"
)
ai_tutor_interface.launch()
