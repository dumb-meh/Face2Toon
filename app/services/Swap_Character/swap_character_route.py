from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List
import json
from .swap_character import SwapCharacter
from .swap_character_schema import SwapCharacterRequest, SwapCharacterResponse

router = APIRouter()
swap_character_service = SwapCharacter()

@router.post("/swap_character", response_model=SwapCharacterResponse)
async def swap_character_endpoint(
    full_page_urls: str = Form(..., description="JSON string of list of 11 full-page URLs"),
    prompts: str = Form(..., description="JSON string of prompts dictionary"),
    story: str = Form(..., description="JSON string of story dictionary"),
    character_name: str = Form(...),
    gender: str = Form(...),
    age: int = Form(...),
    image_style: str = Form(...),
    reference_images: List[UploadFile] = File(..., description="New character reference images")
):
    try:
        # Debug: Print received data
        print(f"\n=== Swap Character Endpoint Called ===")
        print(f"full_page_urls type: {type(full_page_urls)}")
        print(f"full_page_urls value: {full_page_urls[:200] if full_page_urls else 'None'}...")
        print(f"prompts type: {type(prompts)}")
        print(f"story type: {type(story)}")
        print(f"reference_images count: {len(reference_images)}")
        
        # Parse full_page_urls - handle both JSON array and comma-separated string
        try:
            full_page_urls_list = json.loads(full_page_urls)
        except json.JSONDecodeError:
            # If JSON parsing fails, try splitting by comma (CSV format)
            print("⚠️  full_page_urls is not valid JSON, trying comma-separated format...")
            if ',' in full_page_urls:
                full_page_urls_list = [url.strip() for url in full_page_urls.split(',') if url.strip()]
                print(f"✓ Parsed as CSV: {len(full_page_urls_list)} URLs")
            else:
                # Single URL
                full_page_urls_list = [full_page_urls.strip()] if full_page_urls.strip() else []
                print(f"✓ Parsed as single URL")
        
        # Parse prompts - handle both JSON object and URL-encoded format
        try:
            prompts_dict = json.loads(prompts)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid JSON in prompts: {str(e)}. Received: {prompts[:200]}"
            )
        
        # Parse story - handle both JSON object and URL-encoded format
        try:
            story_dict = json.loads(story)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid JSON in story: {str(e)}. Received: {story[:200]}"
            )
        
        # Validate data
        if not isinstance(full_page_urls_list, list):
            raise HTTPException(status_code=400, detail="full_page_urls must be a JSON array")
        
        if not isinstance(prompts_dict, dict):
            raise HTTPException(status_code=400, detail="prompts must be a JSON object")
        
        if not isinstance(story_dict, dict):
            raise HTTPException(status_code=400, detail="story must be a JSON object")
        
        print(f"✓ Parsed {len(full_page_urls_list)} URLs")
        print(f"✓ Parsed {len(prompts_dict)} prompts")
        print(f"✓ Parsed {len(story_dict)} story entries")
        
        # Call swap service
        response = await swap_character_service.swap_character(
            full_page_urls=full_page_urls_list,
            prompts=prompts_dict,
            story=story_dict,
            character_name=character_name,
            gender=gender,
            age=age,
            image_style=image_style,
            reference_images=reference_images
        )
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
