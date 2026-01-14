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
        # Parse JSON strings
        full_page_urls_list = json.loads(full_page_urls)
        prompts_dict = json.loads(prompts)
        story_dict = json.loads(story)
        
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
