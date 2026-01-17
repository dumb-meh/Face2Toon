from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
import io
from PIL import Image, ImageDraw, ImageFont
import json
import os
from pathlib import Path

router = APIRouter()

# Configure Gemini client - use environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    os.environ["GOOGLE_API_KEY"] = gemini_api_key
client = genai.Client()

# Hardcoded font metrics for Comic Relief font
# Pre-calculated to avoid runtime computation
FONT_METRICS = {
    20: {'avg_char_width': 9.79, 'line_height': 23},
    30: {'avg_char_width': 14.53, 'line_height': 33},
    40: {'avg_char_width': 19.40, 'line_height': 44},
    50: {'avg_char_width': 24.23, 'line_height': 54},
    60: {'avg_char_width': 29.14, 'line_height': 65},
    70: {'avg_char_width': 33.93, 'line_height': 77},
    80: {'avg_char_width': 38.79, 'line_height': 87},
    90: {'avg_char_width': 43.60, 'line_height': 98},
    100: {'avg_char_width': 48.53, 'line_height': 108},
    120: {'avg_char_width': 58.19, 'line_height': 131},
}

# Multipliers for font sizes not in the hardcoded dictionary
AVG_CHAR_WIDTH_MULTIPLIER = 0.485  # multiply by font_size
LINE_HEIGHT_MULTIPLIER = 1.09      # multiply by font_size


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
    else:
        # Calculate using multipliers for custom font sizes
        avg_char_width = font_size * AVG_CHAR_WIDTH_MULTIPLIER
        line_height = int(font_size * LINE_HEIGHT_MULTIPLIER)
    
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

Rules:
1. DO NOT place text over people's faces, bodies, or any animals/creatures
2. Find an empty area (sky, background, plain surface) with good visibility
3. Text must be on LEFT half (x: 100 to {img_width//2 - 150}) OR RIGHT half (x: {img_width//2 + 150} to {img_width - 100})
4. Split text into 2-4 lines for readability
5. All lines must have the SAME x coordinate (they stack vertically)
6. Space lines {int(text_height + 10)}px apart vertically
7. Include ALL words from the original text

Return ONLY valid JSON (no markdown):
{{
    "side": "left" or "right",
    "number_of_lines": <number>,
    "lines": [
        {{"line_number": 1, "text": "...", "x": <coord>, "y": <coord>}},
        {{"line_number": 2, "text": "...", "x": <same x>, "y": <y + {int(text_height + 10)}>}}
    ]
}}"""
    
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
        
        # Extract line coordinates
        lines = result.get("lines", [])
        line_coordinates = [
            LineCoordinate(
                line_number=line.get("line_number", idx + 1),
                text=line.get("text", ""),
                x=line.get("x", 50),
                y=line.get("y", 50 + idx * (font_size + 10))
            )
            for idx, line in enumerate(lines)
        ]
        
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
    
    # Draw each line of text at the recommended coordinates
    outline_color = "black" if color.lower() == "white" else "white"
    
    for line_coord in recommendation.line_coordinates:
        x, y = line_coord.x, line_coord.y
        line_text = line_coord.text
        
        # Draw text with outline for better visibility
        for adj_x in [-2, 0, 2]:
            for adj_y in [-2, 0, 2]:
                draw.text((x + adj_x, y + adj_y), line_text, font=font, fill=outline_color)
        draw.text((x, y), line_text, font=font, fill=color)
    
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




