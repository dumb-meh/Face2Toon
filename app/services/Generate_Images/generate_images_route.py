from fastapi import APIRouter, HTTPException, Query
from .generate_images import GenerateImages
from .generate_images_schema import GenerateImageRequest, GenerateImageResponse

router = APIRouter()
generate_images = GenerateImages()     

@router.post("/generate_images_first", response_model=GenerateImageResponse)
async def  generate_first_two_page(request: GenerateImageRequest):
    try:
        response = generate_images.generate_first_two_page(request.dict())
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/generate_images_full", response_model=GenerateImageResponse)
async def  generate_images(request: GenerateImageRequest, coverpage: Query = None):
    try:
        response = generate_images.generate_images(request.dict())
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

