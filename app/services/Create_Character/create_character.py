from .create_character_schema import CreateCharacterRequest, CreateCharacterResponse
import os
from dotenv import load_dotenv
import openai
import json
import requests
import time
import base64
from io import BytesIO

load_dotenv()
class CreateCharacter:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.ark_api_key = os.getenv("ARK_API_KEY")
        self.model = "seedream-4-0-250828"
        self.base_url = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
    
    def create_character(self, request: CreateCharacterRequest) -> CreateCharacterResponse:
        # Get character details from OpenAI
        prompt = self.create_prompt(request)
        character_data = self.get_openai_response(prompt)
        
        # Generate realistic image using seedream
        image_prompt = self.create_image_prompt(character_data)
        image_url = self.generate_character_image(image_prompt)
        
        # Add image URL to response
        character_data['image_url'] = image_url
        
        return CreateCharacterResponse(**character_data)

    def create_prompt(self, request: CreateCharacterRequest) -> str:
        past_char = ""
        past_theme = ""
        
        if request.past_characters:
            past_char = f"IMPORTANT: Do NOT generate any character similar to these: {', '.join(request.past_characters)}. Create a completely different character with different name, ethnicity, appearance, and personality.\n"
        
        if request.past_themes:
            past_theme = f"IMPORTANT: Do NOT use any theme similar to these: {', '.join(request.past_themes)}. Create a completely NEW and UNIQUE story theme.\n"
        
        prompt = f"""{past_char}{past_theme}
Create a UNIQUE and DIVERSE child character for a children's book. Generate a completely different character each time.

Respond only with a JSON object containing:

- 'name': a creative child's name (vary across cultures and backgrounds)
- 'age': an integer between 5 and 12  
- 'gender': either 'boy' or 'girl'

- 'story_prompt': This is what a USER would type to generate a book. Write a detailed story prompt/request that describes what story the user wants to create. This should be like user input. Examples:
  * "I want a story about going on a space adventure and meeting friendly aliens"
  * "Create a story about overcoming fear of the dark and discovering bravery"
  * "Write about making new friends at school and learning about kindness"
  * "Tell a story about discovering a magical garden where plants can talk"
  * "Create an adventure about solving a mystery in the neighborhood"
  BE CREATIVE with different scenarios each time!

- 'story_theme': A short 3-5 word theme/category (examples: "space adventure", "overcoming fears", "making friends", "magical discovery", "neighborhood mystery", "helping animals", "learning cultures", "family bonds", "nature exploration")

- 'character': a list of 5-7 strings describing BOTH physical attributes AND personality traits (e.g., ["African American", "Brown eyes", "Curly black hair with two puffs", "Bright smile with dimples", "Wearing yellow sundress", "Curious and brave personality", "Loves asking questions about nature"])

CRITICAL REQUIREMENTS:
1. Ensure MAXIMUM diversity in ethnicity: African, East Asian, South Asian, Hispanic/Latino, Middle Eastern, European, Pacific Islander, Native American, Mixed heritage, etc.
2. Vary story themes across: adventure, learning, friendship, overcoming challenges, discovery, helping others, family, nature, science, arts, sports, cultural traditions, etc.
3. Vary personality types: curious, brave, shy, creative, energetic, thoughtful, kind, determined, imaginative, playful, etc.
4. Include personality traits in the 'character' list along with physical features

Return the response in this exact JSON format:
{{
    "name": "child's name",
    "age": age_number,
    "gender": "boy or girl",
    "story_prompt": "Detailed story prompt like a user would write - what story they want created",
    "story_theme": "short theme category",
    "character": ["race/ethnicity", "eye color", "hair description", "distinguishing features", "clothing style", "personality trait 1", "personality trait 2"]
}}
"""
        return prompt
    
    def get_openai_response(self, prompt: str) -> dict:
        completion = self.client.chat.completions.create(
            model="gpt-4.1",  # Same model as generate_story
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,  # Higher temperature for more variety and creativity
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        response_content = completion.choices[0].message.content.strip()
        response_data = json.loads(response_content)
        return response_data
    
    def create_image_prompt(self, character_data: dict) -> str:
        """Create a detailed prompt for image generation"""
        character_traits = ', '.join(character_data['character'])
        gender = character_data['gender']
        age = character_data['age']
        name = character_data['name']
        
        image_prompt = f"""High-quality realistic portrait photo of a {age} year old {gender} child named {name}. 
Physical features: {character_traits}.
Style: Professional children's portrait photography, soft natural lighting, warm friendly expression, 
waist-up shot, suitable for children's book illustration. Photorealistic, high detail, 
friendly and approachable appearance."""
        
        return image_prompt
    
    def generate_character_image(self, image_prompt: str) -> str:
        """Generate character image using BytePlus ARK API (seedream-4-0-250828 model)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.ark_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "prompt": image_prompt,
                "size": "1024x1024",
                "n": 1,
                "quality": "standard",
                "watermark": False,
                "response_format": "url",
                "negative_prompt": "blurry, distorted, deformed, ugly, cartoon, anime, low quality, bad anatomy, multiple heads, extra limbs"
            }
            
            print(f"Generating character image with ARK API...")
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            print(f"ARK API response: {result}")
            
            # Extract image URL from response
            if 'data' in result and len(result['data']) > 0:
                image_url = result['data'][0].get('url')
                if image_url:
                    return image_url
            
            raise Exception(f"No image URL in response: {result}")
            
        except requests.exceptions.Timeout:
            raise Exception("Image generation timed out after 120 seconds")
        except requests.exceptions.RequestException as e:
            print(f"Error generating image with ARK API: {str(e)}")
            if hasattr(e.response, 'text'):
                print(f"Response text: {e.response.text}")
            raise Exception(f"Failed to generate character image: {str(e)}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise Exception(f"Failed to generate character image: {str(e)}")