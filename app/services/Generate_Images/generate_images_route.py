from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from .generate_images import GenerateImages
from .generate_images_schema import GenerateImageRequest, GenerateImageResponse
from typing import Dict, Optional
import json

router = APIRouter()
generate_images_service = GenerateImages()     

@router.post("/generate_images_first", response_model=GenerateImageResponse)
async def generate_first_two_page(
    reference_image: UploadFile = File(..., description="Reference image of the child"),
    story:str = Form(..., description="JSON string of story content for each page"),
    prompt: str = Form(..., description="JSON string of prompts for each page"),
    page_connections: Optional[str] = Form(default=None, description="JSON string of page connections"),
    gender: str = Form(..., description="Child's gender"),
    age: int = Form(..., description="Child's age"),
    image_style: str = Form(..., description="Desired illustration style"),
    sequential: str = Query(default="no", description="'yes' to force sequential generation with reference images, 'no' for default behavior")
):
    """Generate images for page 0 (cover) and page 1 only"""
    try:
        story_dict = json.loads(story)
        prompt_dict = json.loads(prompt)
        page_connections_dict = json.loads(page_connections) if page_connections else None
        
        response = generate_images_service.generate_first_two_page(
            prompt_dict,
            page_connections_dict,
            reference_image,
            gender,
            age,
            image_style,
            sequential,
            story_dict
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/generate_images_full", response_model=GenerateImageResponse)
async def generate_images(
    reference_image: UploadFile = File(..., description="Reference image of the child"),
    story:str = Form(..., description="JSON string of story content for each page"),
    prompt: str = Form(..., description="JSON string of prompts for each page"),
    page_connections: Optional[str] = Form(default=None, description="JSON string of page connections"),
    gender: str = Form(..., description="Child's gender"),
    age: int = Form(..., description="Child's age"),
    image_style: str = Form(..., description="Desired illustration style"),
    coverpage: str = Query(default="no", description="'yes' if cover and page 1 already exist, 'no' to generate all pages"),
    sequential: str = Query(default="no", description="'yes' to force sequential generation with reference images, 'no' for default behavior")
):
    """Generate images for all pages. Set coverpage='yes' to skip page 0 and page 1."""
    try:
        story_dict = json.loads(story)
        prompt_dict = json.loads(prompt)
        page_connections_dict = json.loads(page_connections) if page_connections else None
        
        response = generate_images_service.generate_images(
            prompt_dict,
            page_connections_dict,
            reference_image,
            gender,
            age,
            image_style,
            coverpage=coverpage,
            sequential=sequential,
            story=story_dict
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

