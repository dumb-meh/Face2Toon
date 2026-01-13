from fastapi import APIRouter, HTTPException
from .regenerate_image import ReGenerateImage
from .regenerate_image_schema import ReGenerateImageRequest, ReGenerateImageResponse

router = APIRouter()
regenerate_service = ReGenerateImage()

@router.post("/regenerate_image", response_model=ReGenerateImageResponse)
async def regenerate_image_endpoint(request_data: ReGenerateImageRequest):
    try:
        response = regenerate_service.regenerate_image(request_data)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
