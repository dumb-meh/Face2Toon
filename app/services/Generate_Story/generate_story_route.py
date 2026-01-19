from fastapi import APIRouter, HTTPException
from .generate_story import GenerateStory
from .generate_story_schema import GenerateStoryRequest, GenerateStoryResponse

router = APIRouter()
generate_story= GenerateStory()     
@router.post("/generate_story", response_model=GenerateStoryResponse)
async def  get_generate_story(request: GenerateStoryRequest):
    try:
        response = await generate_story.get_generate_story(request.dict())
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
