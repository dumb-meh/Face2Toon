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
        past = ""
        if request.past_characters:
            past = f'''Avoid these past characters: {', '.join(request.past_characters)}. "
        prompt = f"{past}Create a random child character for a children's book. Respond only with a JSON object containing 'name' (a creative child's name), 'age' (an integer between 5 and 12), and 'prompt' (a short description of the character suitable for generating children's book illustrations and stories).
        
        Return the response in the following JSON format:
        {{
            "name": "<child's name>",
            "age": <child's age>,
            "prompt": "<short character description>"
        }}"
        return prompt'''
    
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