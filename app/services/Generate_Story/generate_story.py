import os
import json
import openai
import asyncio
from dotenv import load_dotenv
from .generate_story_schema import GenerateStoryResponse, GenerateStoryRequest
from app.utils.image_analysis import analyze_reference_image_from_url

load_dotenv()

class GenerateStory:
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def get_generate_story(self, input_data: dict) -> GenerateStoryResponse:
        # If image(s) provided, attempt to analyze the first image URL to extract visual attributes
        image_url = input_data.get("image")  # This is a string, not a list
        extracted_profile = None
        if image_url and analyze_reference_image_from_url:
            try:
                # analyze_reference_image_from_url is async; await it directly
                extracted_profile = await analyze_reference_image_from_url(image_url)
                # If analyzer returns empty dict => not a single-child image; ignore
                if not extracted_profile:
                    extracted_profile = None
            except Exception as e:
                # Log and continue without profile
                print(f"[GenerateStory] Image analysis failed: {e}")
                extracted_profile = None

        # Attach extracted profile to input_data for prompt generation
        if extracted_profile:
            input_data["image_attributes"] = extracted_profile
            # Also augment character_description list to include canonical_clothing and accessories if available
            if isinstance(input_data.get("character_description"), list):
                if extracted_profile.get("canonical_clothing"):
                    input_data["character_description"].append(extracted_profile.get("canonical_clothing"))
                if extracted_profile.get("unique_attributes"):
                    input_data["character_description"].extend(extracted_profile.get("unique_attributes"))
            else:
                input_data["character_description"] = []
                if extracted_profile.get("canonical_clothing"):
                    input_data["character_description"].append(extracted_profile.get("canonical_clothing"))
                if extracted_profile.get("unique_attributes"):
                    input_data["character_description"].extend(extracted_profile.get("unique_attributes"))

        prompt = self.create_prompt(input_data)
        response = self.get_openai_response(prompt)
        return response

    def create_prompt(self, input_data: dict) -> str:
        # Build extracted attributes string if available
        extracted_attrs = input_data.get("image_attributes")
        attrs_str = "N/A"
        if extracted_attrs:
            attrs_parts = []
            if extracted_attrs.get("facial_features"):
                attrs_parts.append(f"Facial features: {extracted_attrs['facial_features']}")
            if extracted_attrs.get("unique_attributes"):
                attrs_parts.append(f"Unique attributes: {', '.join(extracted_attrs['unique_attributes'])}")
            if extracted_attrs.get("skin_tone"):
                attrs_parts.append(f"Skin tone: {extracted_attrs['skin_tone']}")
            if extracted_attrs.get("ethnicity"):
                attrs_parts.append(f"Ethnicity: {extracted_attrs['ethnicity']}")
            if extracted_attrs.get("dress_color"):
                attrs_parts.append(f"Dress color: {extracted_attrs['dress_color']}")
            if extracted_attrs.get("hair_color"):
                attrs_parts.append(f"Hair color: {extracted_attrs['hair_color']}")
            if extracted_attrs.get("hair_style"):
                attrs_parts.append(f"Hair style: {extracted_attrs['hair_style']}")
            if extracted_attrs.get("eye_color"):
                attrs_parts.append(f"Eye color: {extracted_attrs['eye_color']}")
            if extracted_attrs.get("accessories"):
                attrs_parts.append(f"Accessories: {', '.join(extracted_attrs['accessories'])}")
            if extracted_attrs.get("canonical_clothing"):
                attrs_parts.append(f"Canonical clothing: {extracted_attrs['canonical_clothing']}")
            if extracted_attrs.get("notes"):
                attrs_parts.append(f"Notes: {extracted_attrs['notes']}")
            attrs_str = "; ".join(attrs_parts) if attrs_parts else "N/A"

        return f"""You are a children's book author and illustrator assistant. Create a complete children's book with 1 cover page, 11 story pages, 2 coloring pages, and 1 back cover.

Child Information:
- Name: {input_data['name']}
- Age: {input_data['age']}
- Gender: {input_data['gender']}
- Image Style: {input_data['image_style']}
- Language: {input_data['language']}
- Story Theme/Input: {input_data['user_input']}
-Character Description: {', '.join(input_data['character_description']) if input_data.get('character_description') else 'N/A'}
-Extracted Visual Attributes: {attrs_str}

Requirements:
1. Create a story appropriate for a {input_data['age']}-year-old child
2. The story MUST be completed within 11 pages (pages 1-11)
3. Each page should have 1-3 sentences of story text
4. The story should have a clear beginning, middle, and end
5. Include the child's name ({input_data['name']}) as the main character
6. Make it engaging, educational, and age-appropriate
7. Write the story text in {input_data['language']}
8. Pages 12-13 will be coloring pages (black and white illustrations) based on memorable scenes from the story
9. "page last page" will be the back cover (visual illustration only, no text will be added)

Output Format:
Return a JSON object with three fields:

1. "story": A dictionary with keys "page 0" through "page 13" and "page last page"
   - "page 0": Only the book title (short and catchy) in {input_data['language']}
   - "page 1" to "page 11": Story text for each page in {input_data['language']}
   - "page 12" to "page 13": Text saying "Color this page!" or "Coloring page" in {input_data['language']}
   - "page last page": Empty string "" (back cover has no text overlay)

2. "prompt": A dictionary with keys "page 0" through "page 13" and "page last page"
   - Each value is a HIGHLY DETAILED image generation prompt for that page
   - CRITICAL: ALL prompts must be written in ENGLISH (regardless of story language)
   - CRITICAL: A reference image of the child will be provided to the image generator
   - CRITICAL: Image models have limited reasoning - prompts must be extremely specific and descriptive
   
   PROMPT GUIDELINES:
   - CRITICAL: EVERY image prompt MUST explicitly specify: hair color, hair style, eye color, skin tone, and clothing details. These are REQUIRED in every prompt.
   - If 'Extracted Visual Attributes' are provided (from reference image analysis): Use those exact attributes in all prompts to maintain consistency.
   - If NO extracted attributes: You MUST create and define these visual characteristics yourself based on:
     1. Character Description (if provided)
     2. The child's age, gender, and ethnicity/background that fits the story theme
     3. Your creative judgment to create a consistent, appealing character
   - Once you define the character's appearance (hair color, hair style, eye color, skin tone), maintain it EXACTLY across ALL pages.
   - CRITICAL: ALL image prompts MUST include the image style "{input_data['image_style']}" at the beginning or within the first sentence
   - For "page 0" (cover): Start with "{input_data['image_style']} style book cover illustration showing the child from the reference image..." OR if no reference image "...showing a {input_data['age']}-year-old {input_data['gender']} child with [specify hair color, hair style, eye color, skin tone]...". Include compelling cover scene with the story title.
   - For story pages (1-11): Start with "{input_data['image_style']} style illustration showing the child..." ALWAYS include: hair color, hair style, eye color, skin tone, and clothing (exact colors/patterns). Be very specific about pose, actions, setting, background, lighting, mood, and other characters. Maintain clothing consistency across pages. Each prompt: 4-6 detailed sentences in English, end with negative prompting: "Maintain exact facial features, eyebrows, hair style, and overall character appearance. No changes to face structure, eye shape, or hair color."
   - For coloring pages (12-13): Start with "Black and white coloring page illustration..." (no style needed). Describe a key scene with clear outlines, simple details, line art style, no shading/color. Still mention hair style, facial features for consistency. End with "Simple line drawing, no colors, no shading, no gradients, suitable for children to color. Maintain character appearance."
   - For "page last page" (back cover): Start with "{input_data['image_style']} style back cover illustration showing the child..." Include hair color, hair style, eye color, skin tone. Describe a peaceful closing scene. 4-6 detailed sentences, end with negative prompting.



Example output structure:
{{
  "story": {{
    "page 0": "The Adventures of [Name]",
    "page 1": "Once upon a time...",
    "page 2": "And then...",
    "page last page": ""
  }},
  "prompt": {{
    "page 0": "{input_data['image_style']} style book cover illustration showing a {input_data['age']}-year-old {input_data['gender']} child with [brown curly hair, bright green eyes, warm tan skin tone] wearing...[detailed 4-6 sentence prompt]",
    "page 1": "{input_data['image_style']} style illustration showing the child with brown curly hair, bright green eyes, and warm tan skin tone wearing a blue striped shirt and khaki shorts...[detailed 4-6 sentence prompt with negative prompting]",
    "page 2": "{input_data['image_style']} style illustration showing the child with brown curly hair, bright green eyes, and warm tan skin tone now wearing different clothing, a green sweater...[detailed prompt]",
    "page last page": "{input_data['image_style']} style back cover illustration showing the child with brown curly hair, bright green eyes, and warm tan skin tone...[detailed 4-6 sentence prompt]"
  }}

}}

Generate the complete children's book now in valid JSON format only."""

    def get_openai_response(self, prompt: str) -> GenerateStoryResponse:
        completion = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
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
