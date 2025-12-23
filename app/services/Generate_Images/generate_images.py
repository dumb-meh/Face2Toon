import os
import json
import openai
from dotenv import load_dotenv
from .generate_images_schema import GenerateImageResponse, GenerateImageRequest

load_dotenv()

class GenerateImages:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_images(self, input_data: dict) -> GenerateImageResponse:
        prompt = self.create_prompt(input_data)
        response = self.get_openai_response(prompt)
        return response

    def create_prompt(self, input_data: dict) -> str:
        return f""""""

    def get_openai_response(self, prompt: str) -> GenerateImageResponse:
        completion = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        response_content = completion.choices[0].message.content.strip()
        response_data = json.loads(response_content)
        
        return GenerateImageResponse(
            image_urls=response_data.get("image_urls", {})
        )
    def get_generate_images(self, input_data: dict) -> GenerateImageResponse:
        prompt = self.create_prompt(input_data)
        response = self.get_openai_response(prompt)
        return response
    
    def generate_first_two_page(self, input_data: dict) -> GenerateImageResponse:
        prompt = self.create_prompt(input_data)
        response = self.get_openai_response(prompt)
        return response