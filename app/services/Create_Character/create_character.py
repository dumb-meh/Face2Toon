from create_character_schema import CreateCharacterRequest, CreateCharacterResponse
import os
from dotenv import load_dotenv
import openai
import json

load_dotenv()
class CreateCharacter:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def create_character (self, request: CreateCharacterRequest) -> CreateCharacterResponse:
        prompt=self.create_prompt(request)
        response=self.get_openai_response(prompt)
        return response

    def create_prompt(self,request: CreateCharacterRequest) -> str:
        # Implement prompt creation logic here
        return "Generated prompt based on past characters"
    
    def get_openai_response(self, prompt: str) -> CreateCharacterResponse:
        completion = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        response_content = completion.choices[0].message.content.strip()
        response_data = json.loads(response_content)
        return CreateCharacterResponse(**response_data)