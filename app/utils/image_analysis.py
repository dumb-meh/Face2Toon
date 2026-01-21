from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from google import genai
from google.genai import types
import io
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import json
import os
from pathlib import Path
import requests
import base64
import openai
import numpy as np

# Optional OpenCV import for face detection; graceful fallback if not installed
try:
    import cv2
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
    cv2 = None


router = APIRouter()

# Configure Gemini client - use environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
    client = genai.Client()
else:
    client = None

# Hardcoded font metrics for Comic Relief font
# Pre-calculated to avoid runtime computation
FONT_METRICS = {
    20: {'avg_char_width': 9.79, 'line_height': 23, 'max_char_height': 39},
    30: {'avg_char_width': 14.53, 'line_height': 33, 'max_char_height': 57},
    40: {'avg_char_width': 19.40, 'line_height': 44, 'max_char_height': 77},
    50: {'avg_char_width': 24.23, 'line_height': 54, 'max_char_height': 95},
    60: {'avg_char_width': 29.14, 'line_height': 65, 'max_char_height': 115},
    70: {'avg_char_width': 33.93, 'line_height': 77, 'max_char_height': 132},
    80: {'avg_char_width': 38.79, 'line_height': 87, 'max_char_height': 152},
    90: {'avg_char_width': 43.60, 'line_height': 98, 'max_char_height': 170},
    100: {'avg_char_width': 48.53, 'line_height': 108, 'max_char_height': 190},
    120: {'avg_char_width': 58.19, 'line_height': 131, 'max_char_height': 227},
}

# Multipliers for font sizes not in the hardcoded dictionary
AVG_CHAR_WIDTH_MULTIPLIER = 0.485  # multiply by font_size
LINE_HEIGHT_MULTIPLIER = 1.096      # multiply by font_size
MAX_CHAR_HEIGHT_MULTIPLIER = 1.906  # multiply by font_size


class LineCoordinate(BaseModel):
    line_number: int
    text: str
    x: int
    y: int

class TextPlacementRecommendation(BaseModel):
    number_of_lines: int
    line_coordinates: List[LineCoordinate]
    side: str  

async def get_text_placement_recommendation(image_bytes: bytes, text: str, 
                                            font_size: int = 40) -> TextPlacementRecommendation:
    """
    Use Gemini Vision to analyze the image and recommend text placement for a flip book page.
    The image will be split down the middle to create two pages, so text must be on left or right side only.
    
    Args:
        image_bytes: Image data in bytes
        text: The text that will be placed
        font_size: Font size for the text
        
    Returns:
        TextPlacementRecommendation with line count, coordinates for each line, and side placement
    """
    # Get image dimensions
    img = Image.open(io.BytesIO(image_bytes))
    img_width, img_height = img.size
    
    # Get font metrics from hardcoded dictionary or calculate using multipliers
    if font_size in FONT_METRICS:
        # Use pre-calculated metrics
        avg_char_width = FONT_METRICS[font_size]["avg_char_width"]
        line_height = FONT_METRICS[font_size]["line_height"]
        max_char_height = FONT_METRICS[font_size]["max_char_height"]
    else:
        # Calculate using multipliers for custom font sizes
        avg_char_width = font_size * AVG_CHAR_WIDTH_MULTIPLIER
        line_height = int(font_size * LINE_HEIGHT_MULTIPLIER)
        max_char_height = int(font_size * MAX_CHAR_HEIGHT_MULTIPLIER)
    
    # Calculate text dimensions
    text_width = len(text) * avg_char_width
    text_height = line_height
    
    # Create image part for Gemini
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
    
    # Create simplified prompt for text placement
    prompt = f"""Analyze this children's storybook image and recommend where to place text.

Image dimensions: {img_width}x{img_height} pixels
This image will be split vertically in the middle (at x={img_width//2}) to create two separate pages.

Text to place: "{text}"
Font size: {font_size}px
Estimated line height: {int(text_height)}px
Estimated text width per character: {int(avg_char_width)}px

YOUR TASK: Carefully analyze the image content and find the BEST empty area to place text.

CRITICAL RULES FOR TEXT PLACEMENT:
1. ANALYZE the image first - identify:
   - Where are characters/people/animals located?
   - Where are important objects (fish, toys, furniture, etc.)?
   - Which areas are EMPTY (sky, walls, background, plain surfaces)?
   - Which side (left or right) has MORE empty space?

2. AVOID covering ANY important visual elements:
   - NO text on faces, bodies, or limbs of people/characters
   - NO text on animals, pets, or creatures
   - NO text on important objects (toys, furniture, decorative items)
   - NO text on action areas or focal points

3. CHOOSE THE BEST SIDE dynamically:
   - If LEFT side has more empty space → place text on LEFT
   - If RIGHT side has more empty space → place text on RIGHT
   - Prefer areas with plain backgrounds (sky, walls, empty floor)

4. POSITIONING ZONES:
   - LEFT side: x coordinate between 150 and {img_width//2 - 200}
   - RIGHT side: x coordinate between {img_width//2 + 200} and {img_width - 150}
   - DO NOT use the exact same x coordinate every time - adjust based on image content
   - Y coordinates: Avoid top 15% and bottom 15% - use the middle 70% of the image
   - Find the emptiest vertical region (top-middle, center, or bottom-middle)

5. TEXT FORMATTING:
   - Split text into 2-4 lines for readability
   - All lines stack vertically with the SAME x coordinate
   - Space lines {int(text_height + 10)}px apart vertically
   - Ensure longest line fits within the chosen side

6. DYNAMIC PLACEMENT:
   - Adjust x position based on where empty space actually is
   - If character is on far left, push text more to the center-left
   - If character is on far right, push text more to the center-right
   - If there's empty space at the top, place text higher (y starting around {img_height // 4})
   - If there's empty space at the bottom, place text lower (y starting around {img_height * 2 // 3})

EXAMPLE 1 - Character on left side, empty sky on right:
{{
    "side": "right",
    "number_of_lines": 3,
    "lines": [
        {{"line_number": 1, "text": "First part of", "x": {img_width - 800}, "y": {img_height // 3}}},
        {{"line_number": 2, "text": "the story text", "x": {img_width - 800}, "y": {img_height // 3 + int(text_height + 10)}}},
        {{"line_number": 3, "text": "goes here", "x": {img_width - 800}, "y": {img_height // 3 + int((text_height + 10) * 2)}}}
    ]
}}

EXAMPLE 2 - Character on right side, empty area on left:
{{
    "side": "left",
    "number_of_lines": 2,
    "lines": [
        {{"line_number": 1, "text": "Text placed in", "x": 400, "y": {img_height // 2 - 40}}},
        {{"line_number": 2, "text": "empty left area", "x": 400, "y": {img_height // 2 + 40}}}
    ]
}}

Return ONLY valid JSON with your analysis-based placement (no markdown, no explanations):"""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[prompt, image_part]
        )
        
        # Parse response
        result_text = response.text.strip()
        
        print(f"[Gemini Vision] Raw response length: {len(result_text)}")
        print(f"[Gemini Vision] Raw response preview: {result_text[:200]}...")
        
        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
        print(f"[Gemini Vision] After cleanup: {result_text[:200]}...")
        
        if not result_text:
            raise Exception("Gemini Vision returned empty response")
        
        result = json.loads(result_text)
        
        # Log the full parsed result to debug coordinate issues
        print(f"[Gemini Vision] Parsed JSON result: {json.dumps(result, indent=2)}")
        
        # Extract line coordinates
        lines = result.get("lines", [])
        
        if not lines:
            raise Exception("Gemini Vision returned no line coordinates")
        
        # Validate that coordinates are present
        line_coordinates = []
        for idx, line in enumerate(lines):
            if "x" not in line or "y" not in line:
                print(f"[Gemini Vision] WARNING: Line {idx+1} missing coordinates! Line data: {line}")
                raise Exception(f"Line {idx+1} is missing x or y coordinates in Gemini response")
            
            line_coordinates.append(LineCoordinate(
                line_number=line.get("line_number", idx + 1),
                text=line.get("text", ""),
                x=line["x"],  # No default - fail if missing
                y=line["y"]   # No default - fail if missing
            ))
        
        # Validate coordinates are reasonable
        for coord in line_coordinates:
            if coord.x < 0 or coord.x > img_width or coord.y < 0 or coord.y > img_height:
                print(f"[Gemini Vision] WARNING: Coordinate out of bounds: x={coord.x}, y={coord.y} (image: {img_width}x{img_height})")
        
        print(f"[Gemini Vision] Extracted {len(line_coordinates)} line coordinates:")
        for coord in line_coordinates:
            print(f"  Line {coord.line_number}: '{coord.text}' at ({coord.x}, {coord.y})")
        
        return TextPlacementRecommendation(
            number_of_lines=result.get("number_of_lines", len(lines)),
            line_coordinates=line_coordinates,
            side=result.get("side", "left")
        )
        
    except Exception as e:
        print(f"[Gemini Vision] Error: {str(e)}")
        print(f"[Gemini Vision] Full error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Text placement recommendation failed: {str(e)}")




@router.post("/add-text-with-face-avoidance")
async def add_text_with_face_avoidance(
    file: UploadFile = File(...),
    text: str = "Sample Text",
    font_size: int = 100,
    color: str = "white"
):
    """
    Add text to a flip book page image using AI-recommended placement.
    Text will be placed on either left or right side (avoiding middle split line).
    Text is automatically split into multiple lines for optimal readability.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    image_bytes = await file.read()
    
    # Get AI recommendation for text placement
    recommendation = await get_text_placement_recommendation(image_bytes, text, font_size)
    
    # Open image and prepare for drawing
    img = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(img)
    
    # Load font with robust path handling
    font_path = Path(__file__).resolve().parents[2] / "fonts" / "Comic_Relief" / "ComicRelief-Regular.ttf"
    try:
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), font_size)
        else:
            print(f"Warning: Font not found at {font_path}, using default font")
            font = ImageFont.load_default()
    except Exception as e:
        print(f"Warning: Error loading font: {e}, using default font")
        font = ImageFont.load_default()
    
    # Get image dimensions
    width, height = img.size
    
    # Step 1: Create glow mask - draw text in white on black background
    glow_mask = Image.new('L', (width, height), 0)
    mask_draw = ImageDraw.Draw(glow_mask)
    
    # Draw all text lines on the mask
    for line_coord in recommendation.line_coordinates:
        mask_draw.text((line_coord.x, line_coord.y), line_coord.text, font=font, fill=255)
    
    # Step 2: Apply multiple Gaussian blurs for strong purple glow effect
    glow1 = glow_mask.filter(ImageFilter.GaussianBlur(radius=40))  # Wide glow
    glow2 = glow_mask.filter(ImageFilter.GaussianBlur(radius=25))  # Medium glow
    glow3 = glow_mask.filter(ImageFilter.GaussianBlur(radius=15))  # Tight glow
    
    # Step 3: Create bright purple glow color
    purple_glow = Image.new('RGB', (width, height), (180, 60, 200))
    
    # Step 4: Apply each glow layer to the image (builds up intensity)
    img.paste(purple_glow, (0, 0), glow1)  # Widest glow
    img.paste(purple_glow, (0, 0), glow2)  # Medium glow (makes it stronger)
    img.paste(purple_glow, (0, 0), glow3)  # Tight glow (adds more intensity)
    
    # Step 5: Draw white text on top of the purple glow
    draw = ImageDraw.Draw(img)
    text_color = (255, 255, 255)
    
    for line_coord in recommendation.line_coordinates:
        draw.text((line_coord.x, line_coord.y), line_coord.text, font=font, fill=text_color)
    
    # Save result
    output = io.BytesIO()
    img.save(output, format='PNG')
    output.seek(0)
    
    # Build response headers with placement info
    headers = {
        "X-Number-Of-Lines": str(recommendation.number_of_lines),
        "X-Text-Side": recommendation.side,
        "Content-Disposition": "attachment; filename=output.png"
    }
    
    # Add line coordinate info to headers
    for idx, line_coord in enumerate(recommendation.line_coordinates):
        headers[f"X-Line-{idx + 1}-Coordinates"] = f"({line_coord.x}, {line_coord.y})"
    
    return StreamingResponse(
        output,
        media_type="image/png",
        headers=headers
    )


# -------------------------
# Reference image analysis
# -------------------------

class CharacterAnalysis(BaseModel):
    is_single_child: bool
    facial_features: Optional[Dict[str, Any]] = None
    unique_attributes: Optional[List[str]] = None
    skin_tone: Optional[str] = None
    ethnicity: Optional[str] = None
    dress_color: Optional[str] = None
    hair_color: Optional[str] = None
    hair_style: Optional[str] = None
    eye_color: Optional[str] = None
    accessories: Optional[List[str]] = None
    canonical_clothing: Optional[str] = None
    confidence: Optional[Dict[str, float]] = None
    notes: Optional[str] = None


def _downscale_image_for_embedding(img: Image.Image, max_size: int = 1024) -> Image.Image:
    """Downscale large images to a sensible size for embedding/vision model."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / float(max(w, h))
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS)


async def _send_image_to_gemini_vision(image_bytes: bytes, prompt_text: str) -> Dict[str, Any]:
    """Send the image + prompt to the Gemini Vision model and return parsed JSON.

    This function centralizes the Gemini call for vision analysis.
    """
    if client is None:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    # Downscale image to reduce size for API limits
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_small = _downscale_image_for_embedding(img)
    buf = io.BytesIO()
    img_small.save(buf, format="JPEG", quality=85)
    image_bytes = buf.getvalue()

    # Create image part for Gemini
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[prompt_text, image_part]
        )

        # Parse response
        result_text = response.text.strip()

        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            parts = result_text.split("```")
            if len(parts) >= 2:
                result_text = parts[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

        if not result_text:
            raise Exception("Gemini Vision returned empty response")

        # Parse JSON
        parsed = json.loads(result_text)
        return parsed

    except Exception as e:
        print(f"[Gemini Vision] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Vision model failed: {e}")


async def analyze_reference_image_from_url(image_url: str) -> Dict[str, Any]:
    """Download image, check single child, and extract visual attributes.

    Returns an empty dict if the image does not contain exactly one face.
    """
    try:
        resp = requests.get(image_url, timeout=60)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download image: status {resp.status_code}")

    image_bytes = resp.content

    # Face detection: prefer OpenCV if available, otherwise ask GPT to verify number of people
    faces_count = None
    notes = ""

    if CV2_AVAILABLE:
        try:
            arr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            faces_count = len(faces)
        except Exception as e:
            print(f"[FaceDetection] OpenCV failed: {e}")
            faces_count = None
            notes += "OpenCV face detection failed; falling back to model. "

    if faces_count is None:
        # Ask the model to count people in the image first
        prompt_for_count = (
            "Analyze the provided image and return JSON: {\n  \"people_count\": <int>\n}. "
            "If you cannot confidently determine count, return people_count as -1. "
            "Do NOT include any extra text or markdown."
        )
        parsed_count = await _send_image_to_gemini_vision(image_bytes, prompt_for_count)
        try:
            people_count = int(parsed_count.get("people_count", -1))
        except Exception:
            people_count = -1
        if people_count == -1:
            raise HTTPException(status_code=500, detail="Vision model could not determine number of people in the image")
        faces_count = people_count

    if faces_count != 1:
        # Per requirement, return empty response (no data) if not a single child image
        print(f"[ImageAnalysis] Detected {faces_count} faces; returning empty response.")
        return {}

    # If single face, ask model for detailed attributes
    prompt_for_attributes = (
        "Analyze the provided image and return ONLY valid JSON (no markdown) with the following fields:\n"
        "{\n"
        "  \"is_single_child\": true,\n"
        "  \"facial_features\": {\"shape\": <str>, \"notable_marks\": <list of str>},\n"
        "  \"unique_attributes\": [<list of key attributes>],\n"
        "  \"skin_tone\": <str or 'unknown'>,\n"
        "  \"ethnicity\": <str or 'unknown'>,\n"
        "  \"dress_color\": <str or 'unknown'>,\n"
        "  \"hair_color\": <str or 'unknown'>,\n"
        "  \"eye_color\": <str or 'unknown'>,\n"
        "  \"accessories\": [<list>],\n"
        "  \"canonical_clothing\": <short canonical string to be used verbatim in prompts>,\n"
        "  \"confidence\": {<field>: <0.0-1.0>},\n"
        "  \"notes\": <any clarifying notes>\n"
        "}\n"
        "Do NOT invent attributes; if you cannot tell from the image, set field to 'unknown' or an empty list."
    )

    attributes = await _send_image_to_gemini_vision(image_bytes, prompt_for_attributes)

    # Ensure boolean field is present
    attributes.setdefault("is_single_child", True)

    return attributes


@router.post("/analyze-reference-image")
async def analyze_reference_image_endpoint(image_url: str):
    """Endpoint: analyze an image by URL and return structured character attributes.

    Returns an empty JSON object {} if the image does not contain exactly one face.
    """
    result = await analyze_reference_image_from_url(image_url)
    return result




