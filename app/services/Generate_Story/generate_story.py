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
        return f"""You are a children's book author and illustrator assistant. Create a complete children's book with 1 cover page, 11 story pages, and 2 coloring pages.

Child Information:
- Name: {input_data['name']}
- Age: {input_data['age']}
- Gender: {input_data['gender']}
- Image Style: {input_data['image_style']}
- Language: {input_data['language']}
- Story Theme/Input: {input_data['user_input']}
-Character Description: {', '.join(input_data['character_description']) if input_data.get('character_description') else 'N/A'}

Requirements:
1. Create a story appropriate for a {input_data['age']}-year-old child
2. The story MUST be completed within 11 pages (pages 1-11)
3. Each page should have 1-3 sentences of story text
4. The story should have a clear beginning, middle, and end
5. Include the child's name ({input_data['name']}) as the main character
6. Make it engaging, educational, and age-appropriate
7. Write the story text in {input_data['language']}
8. Pages 12-13 will be coloring pages (black and white illustrations) based on memorable scenes from the story

Output Format:
Return a JSON object with three fields:

1. "story": A dictionary with keys "page 0" through "page 13"
   - "page 0": Only the book title (short and catchy) in {input_data['language']}
   - "page 1" to "page 11": Story text for each page in {input_data['language']}
   - "page 12" to "page 13": Text saying "Color this page!" or "Coloring page" in {input_data['language']}

2. "prompt": A dictionary with keys "page 0" through "page 13"
   - Each value is a HIGHLY DETAILED image generation prompt for that page
   - CRITICAL: ALL prompts must be written in ENGLISH (regardless of story language)
   - CRITICAL: A reference image of the child will be provided to the image generator
   - CRITICAL: Image models have limited reasoning - prompts must be extremely specific and descriptive
   
   PROMPT GUIDELINES:
   - For "page 0" (cover): Start with "The child from the reference image..." and describe a compelling cover scene with the story title
   - For story pages (1-11): Start with "The main character {input_data['name']}..." or "The child..." (do NOT say 'from reference image')
   - For coloring pages (12-13): MUST include "black and white coloring page" at the beginning, then describe a memorable scene from the story with clear outlines and simple details suitable for children to color
   - Be very specific about EVERY element in the scene
   - Do NOT describe facial features, skin tone, eye color, hair color/style - these are maintained from previous images
   - DO describe: exact clothing details (colors, patterns, type), specific pose, precise actions, setting details
   - Specify background elements, lighting, mood, other characters (with detailed descriptions)
   - If clothing appears in multiple pages, specify EXACT same colors/patterns to maintain consistency
   - Include composition details: foreground, midground, background elements
   - Each prompt should be 4-6 detailed sentences in English
   - Add negative prompting at the end: "Maintain exact facial features, eyebrows, hair style, and overall character appearance. No changes to face structure, eye shape, or hair color."
   
   COLORING PAGE SPECIFIC GUIDELINES (pages 12-13):
   - Start with: "Black and white coloring page illustration..."
   - Describe a key scene from the story that would be fun to color
   - Specify clear, bold outlines with simple details
   - Mention "line art style, coloring book page, simple shapes, child-friendly"
   - No shading, no color - only black outlines on white background
   - Add: "Simple line drawing, no colors, no shading, no gradients, suitable for children to color"
   
   Style: {input_data['image_style']}
   
   EXAMPLE GOOD PROMPTS:
   Page 0: "The child from the reference image wearing a red t-shirt with blue jeans, standing in a magical forest. [rest of detailed description]"
   Page 1-11: "The child {input_data['name']} wearing a red t-shirt with blue jeans, standing in a sunny park with green grass and tall oak trees in the background. The child is holding a bright yellow kite with a long blue tail, smiling while looking up at the sky. Behind them, there's a wooden bench and a small pond with ducks. Warm afternoon sunlight creates soft shadows on the ground. The scene has a joyful, carefree mood with vibrant colors. Maintain exact facial features, eyebrows, hair style, and overall character appearance. No changes to face structure, eye shape, or hair color."
   Page 12-13: "Black and white coloring page illustration showing the child {input_data['name']} holding a kite in a park. Clear, bold outlines with simple details. The child is in a standing pose with the kite string in hand, surrounded by simple tree outlines and cloud shapes. Line art style, coloring book page, simple shapes, child-friendly. Simple line drawing, no colors, no shading, no gradients, suitable for children to color. Maintain character appearance."

3. "page_connections": A dictionary mapping pages that should maintain visual consistency
   - STRICT CRITERIA: Only create connections when BOTH conditions are met:
     a) Pages feature the SAME character(s) in similar poses/positions, OR
     b) Pages show the EXACT SAME location/setting
   - Example: {{"page 3": "page 1"}} means page 3 visually references page 1
   - Use sparingly - only when visual consistency is critical (same outfit, same room, continuation of action)
   - Do NOT connect pages just because they're sequential
   - Can be null or empty {{}} if no specific visual connections needed
   - Think carefully: does this page show the same people in the same clothes AND/OR the same location?

Example output structure:
{{
  "story": {{
    "page 0": "The Adventures of [Name]",
    "page 1": "Once upon a time...",
    "page 2": "And then..."
  }},
  "prompt": {{
    "page 0": "Book cover illustration showing the child from the reference image wearing...[detailed 4-6 sentence prompt]",
    "page 1": "The child {input_data['name']} wearing a blue striped shirt and khaki shorts...[detailed 4-6 sentence prompt with negative prompting]",
    "page 2": "The main character {input_data['name']} now in different clothing, a green sweater...[detailed prompt]"
  }},
  "page_connections": {{
    "page 3": "page 2"
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
