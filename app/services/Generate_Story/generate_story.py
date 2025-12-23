import os
import json
import openai
from dotenv import load_dotenv
from .generate_story_schema import GenerateStoryResponse, GenerateStoryRequest

load_dotenv()

class GenerateStory:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_generate_story(self, input_data: dict) -> GenerateStoryResponse:
        prompt = self.create_prompt(input_data)
        response = self.get_openai_response(prompt)
        return response

    def create_prompt(self, input_data: dict) -> str:
        return f"""You are a children's book author and illustrator assistant. Create a complete children's book with 1 cover page and 10 story pages.

Child Information:
- Name: {input_data['name']}
- Age: {input_data['age']}
- Gender: {input_data['gender']}
- Image Style: {input_data['image_style']}
- Language: {input_data['language']}
- Story Theme/Input: {input_data['user_input']}

Requirements:
1. Create a story appropriate for a {input_data['age']}-year-old child
2. The story MUST be completed within 10 pages (pages 1-10)
3. Each page should have 1-3 sentences of story text
4. The story should have a clear beginning, middle, and end
5. Include the child's name ({input_data['name']}) as the main character
6. Make it engaging, educational, and age-appropriate
7. Write the story text in {input_data['language']}

Output Format:
Return a JSON object with three fields:

1. "story": A dictionary with keys "page 0" through "page 10"
   - "page 0": Only the book title (short and catchy) in {input_data['language']}
   - "page 1" to "page 10": Story text for each page in {input_data['language']}

2. "prompt": A dictionary with keys "page 0" through "page 10"
   - Each value is a detailed image generation prompt for that page
   - IMPORTANT: ALL prompts must be written in ENGLISH (regardless of story language)
   - Style: {input_data['image_style']}
   - Prompts should describe the scene, characters, setting, mood, and visual elements
   - Maintain character consistency across pages
   - Each prompt should be 2-3 sentences in English

3. "page_connections": A dictionary mapping pages that should maintain visual consistency
   - Example: {{"page 3": "page 1"}} means page 3's image should reference page 1's generated image
   - Use this for scenes with the same location or character poses
   - Can be null if no specific connections needed

Example output structure:
{{
  "story": {{
    "page 0": "The Adventures of [Name]",
    "page 1": "Once upon a time...",
    "page 2": "And then..."
  }},
  "prompt": {{
    "page 0": "Book cover illustration showing...",
    "page 1": "Illustration of a child named {input_data['name']}...",
    "page 2": "Scene showing..."
  }},
  "page_connections": {{
    "page 3": "page 1",
    "page 5": "page 3"
  }}
}}

Generate the complete children's book now in valid JSON format only."""

    def get_openai_response(self, prompt: str) -> GenerateStoryResponse:
        completion = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000,
            response_format={"type": "json_object"}
        )
        
        response_content = completion.choices[0].message.content.strip()
        response_data = json.loads(response_content)
        
        return GenerateStoryResponse(
            story=response_data.get("story", {}),
            prompt=response_data.get("prompt", {}),
            page_connections=response_data.get("page_connections")
        )
