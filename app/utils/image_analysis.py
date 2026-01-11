from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import openai
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import json
import os
from pathlib import Path

router = APIRouter()

# Configure OpenAI client - use environment variable or fallback
openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key)

# Font metrics cache - stores metrics per font size to avoid recalculation
# Format: {font_size: {"avg_char_width": float, "line_height": int}}
FONT_METRICS_CACHE = {}


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
    Use GPT-4 Vision to analyze the image and recommend text placement for a flip book page.
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
    
    # Get or calculate font metrics for this font size (cached to avoid repeated calculations)
    if font_size in FONT_METRICS_CACHE:
        # Use cached metrics
        avg_char_width = FONT_METRICS_CACHE[font_size]["avg_char_width"]
        line_height = FONT_METRICS_CACHE[font_size]["line_height"]
    else:
        # Calculate and cache metrics for this font size
        font_path = Path(__file__).resolve().parents[2] / "fonts" / "Comic_Relief" / "ComicRelief-Regular.ttf"
        try:
            if font_path.exists():
                temp_font = ImageFont.truetype(str(font_path), font_size)
                # Create temporary image to measure text with sample characters
                temp_img = Image.new('RGB', (100, 100))
                temp_draw = ImageDraw.Draw(temp_img)
                # Use common English characters for average width calculation
                sample_text = "The quick brown fox jumps over the lazy dog"
                sample_bbox = temp_draw.textbbox((0, 0), sample_text, font=temp_font)
                sample_width = sample_bbox[2] - sample_bbox[0]
                sample_height = sample_bbox[3] - sample_bbox[1]
                # Calculate average character width
                avg_char_width = sample_width / len(sample_text)
                line_height = sample_height
            else:
                # Fallback estimates
                avg_char_width = font_size * 0.6
                line_height = font_size + 10
        except Exception as e:
            print(f"Warning: Error calculating font metrics: {e}, using fallback")
            # Fallback estimates
            avg_char_width = font_size * 0.6
            line_height = font_size + 10
        
        # Cache the calculated metrics
        FONT_METRICS_CACHE[font_size] = {
            "avg_char_width": avg_char_width,
            "line_height": line_height
        }
    
    # Calculate text dimensions using cached metrics
    text_width = len(text) * avg_char_width
    text_height = line_height
    
    # Encode image to base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # Create prompt for text placement recommendation
    prompt = f"""CHILDREN'S STORYBOOK - FLIP BOOK PAGE TEXT PLACEMENT

IMAGE CONTEXT:
- This is a children's storybook illustration in flip book format
- Image dimensions: 17 inch x 8.5 inch (width x height)
- Image pixel dimensions: {img_width} x {img_height} pixels
- **CRITICAL**: This image will be SPLIT DOWN THE MIDDLE (at x={img_width//2}) to create TWO separate pages
- Text must be placed ONLY on the LEFT HALF (x < {img_width//2}) OR RIGHT HALF (x > {img_width//2})
- NEVER place text in the middle area where the split occurs

TEXT TO PLACE:
"{text}"

FONT SPECIFICATIONS:
- Font family: Comic Relief (Regular weight)
- Font size: {font_size}px
- Font characteristics: Rounded, friendly, comic-style font with medium spacing
- Approximate line height: {int(text_height)}px
- Average character width: ~{int(avg_char_width)}px per character
- Full text width (if single line): ~{int(text_width)}px
- Total text characters: {len(text)}

TEXT DIMENSION CALCULATIONS:
Based on the font metrics above, when splitting text into lines:
- 2 lines: each line ~{len(text)//2} chars, ~{int((len(text)//2) * avg_char_width)}px width
- 3 lines: each line ~{len(text)//3} chars, ~{int((len(text)//3) * avg_char_width)}px width
- 4 lines: each line ~{len(text)//4} chars, ~{int((len(text)//4) * avg_char_width)}px width

Use these dimensions to ensure text fits within the chosen area and doesn't overflow.

YOUR TASK - TEXT PLACEMENT STRATEGY:
Analyze the image carefully and find the OPTIMAL text placement that balances:
1. **Prominence & Readability** - Text should be clearly visible and prominently placed (this is a storybook!)
2. **Avoid obscuring important elements** - DO NOT place text over:
   - Human characters (faces, bodies, or any part of people)
   - Animals of any kind (birds, pets, creatures, insects, etc.)
   - Key story elements or focal points
   - Important objects that tell the story
3. **Natural reading position** - Text should be in a natural, comfortable reading position
   - Prefer upper-third or middle-area positions over bottom corners
   - Avoid obscure corner placements that make text hard to find
   - Think like a children's book designer - where would kids naturally look?

PLACEMENT PRINCIPLES:
✓ GOOD: Text over clear sky, empty background areas, grass/ground, plain walls, solid color areas, simple textures
✓ GOOD: Text positioned prominently in upper or middle areas where readers naturally look
✓ GOOD: Text that's easy to spot and read at first glance
✗ BAD: Text covering people, animals (including birds), faces, or story-critical objects
✗ BAD: Text hidden in corners or edges just because there's empty space
✗ BAD: Text in awkward positions that require effort to find or read
✗ BAD: Text placement that looks like an afterthought

TECHNICAL CONSTRAINTS:
- Text must be on LEFT side (x: 100 to {img_width//2 - 150}) OR RIGHT side (x: {img_width//2 + 150} to {img_width - 100})
- Stay at least 150px away from the middle split line at x={img_width//2}
- Leave 80-100px margins from top/bottom/side edges
- Each line should be horizontally aligned on the chosen side (same x coordinate for all lines)
- Vertical spacing between lines: {int(text_height + 10)}px
- Ensure each line has enough horizontal space for its calculated width

LINE SPLITTING STRATEGY:
- Split text into 2-4 lines for optimal readability
- Consider the character widths above when breaking lines
- Break at natural phrase boundaries when possible
- Keep related words together
- Ensure each line fits within the available horizontal space on the chosen side

COORDINATE PRECISION:
- Provide exact x, y coordinates where text should START (top-left corner of text)
- Account for the text width of each line to ensure it doesn't overflow into restricted areas
- Verify that (x + estimated_line_width) stays within the allowed boundaries
- For LEFT side: x should allow full line width without crossing x={img_width//2 - 150}
- For RIGHT side: x should allow full line width without exceeding x={img_width - 100}

Return ONLY a JSON object in this exact format (no markdown, no extra text):
{{
    "side": "left" or "right",
    "number_of_lines": <number of lines>,
    "lines": [
        {{
            "line_number": 1,
            "text": "first portion of text",
            "x": <x coordinate>,
            "y": <y coordinate>
        }},
        {{
            "line_number": 2,
            "text": "next portion of text",
            "x": <x coordinate>,
            "y": <y coordinate>
        }}
    ],
    "explanation": "brief explanation: what area was chosen and what important elements (people/animals/objects) were avoided"
}}"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        
        # Parse response
        result_text = response.choices[0].message.content.strip()
        
        # Remove markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
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
        raise HTTPException(status_code=500, detail=f"Text placement recommendation failed: {str(e)}")




@router.post("/add-text-with-face-avoidance")
async def add_text_with_face_avoidance(
    file: UploadFile = File(...),
    text: str = "Sample Text",
    font_size: int = 40,
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




