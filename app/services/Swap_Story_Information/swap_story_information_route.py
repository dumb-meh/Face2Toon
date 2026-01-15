from fastapi import APIRouter, HTTPException
from .swap_story_information import SwapStoryInformation
from .swap_story_information_schema import SwapStoryInformationRequest, SwapStoryInformationResponse

router = APIRouter()
swap_story_information_service = SwapStoryInformation()

@router.post("/swap_story_information", response_model=SwapStoryInformationResponse)
async def swap_story_information_endpoint(request: SwapStoryInformationRequest):
    try:
        print(f"\n=== Swap Story Information Endpoint Called ===")
        print(f"Change Type: {request.change_type}")
        print(f"Number of pages: {len(request.full_page_urls)}")
        print(f"Character Name: {request.character_name}")
        print(f"Language: {request.story_language} -> {request.language}")
        
        # Call swap service
        response = await swap_story_information_service.swap_story_information(request)
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
