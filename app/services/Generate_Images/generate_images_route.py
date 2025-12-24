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
    prompt: str = Form(..., description="JSON string of prompts for each page"),
    page_connections: Optional[str] = Form(default=None, description="JSON string of page connections"),
    gender: str = Form(..., description="Child's gender"),
    age: int = Form(..., description="Child's age"),
    image_style: str = Form(..., description="Desired illustration style"),
    sequential: str = Query(default="no", description="'yes' to force sequential generation with reference images, 'no' for default behavior")
):
    """Generate images for page 0 (cover) and page 1 only"""
    try:
        print(f"DEBUG RAW: prompt type: {type(prompt)}")
        
        # Handle potentially double-encoded JSON
        prompt_dict = prompt
        while isinstance(prompt_dict, str):
            prompt_dict = json.loads(prompt_dict)
            
        page_connections_dict = page_connections
        if page_connections_dict:
            while isinstance(page_connections_dict, str):
                page_connections_dict = json.loads(page_connections_dict)
        
        print(f"DEBUG PARSED: prompt_dict type: {type(prompt_dict)}")
        print(f"DEBUG PARSED: page_connections_dict type: {type(page_connections_dict)}")
        
        response = generate_images_service.generate_first_two_page(
            prompt_dict,
            page_connections_dict,
            reference_image,
            gender,
            age,
            image_style,
            sequential
        )
        return response
    except Exception as e:
        print(f"ERROR in route: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/generate_images_full", response_model=GenerateImageResponse)
async def generate_images(
    reference_image: UploadFile = File(..., description="Reference image of the child"),
    prompt: str = Form(..., description="JSON string of prompts for each page"),
    page_connections: Optional[str] = Form(default=None, description="JSON string of page connections"),
    gender: str = Form(..., description="Child's gender"),
    age: int = Form(..., description="Child's age"),
    image_style: str = Form(..., description="Desired illustration style"),
    coverpage: str = Query(default="no", description="'yes' if cover and page 1 already exist, 'no' to generate all pages"),
    sequential: str = Query(default="no", description="'yes' to force sequential generation with reference images, 'no' for default behavior"),
    existing_pages: Optional[str] = Form(default=None, description="JSON string of already generated page URLs (e.g., {'page 0': 'url', 'page 1': 'url'})")
):
    """Generate images for all pages. Set coverpage='yes' to skip page 0 and page 1."""
    try:
        # Handle potentially double-encoded JSON
        prompt_dict = prompt
        while isinstance(prompt_dict, str):
            prompt_dict = json.loads(prompt_dict)
            
        page_connections_dict = page_connections
        if page_connections_dict:
            while isinstance(page_connections_dict, str):
                page_connections_dict = json.loads(page_connections_dict)
                
        existing_pages_dict = existing_pages
        if existing_pages_dict:
            while isinstance(existing_pages_dict, str):
                existing_pages_dict = json.loads(existing_pages_dict)
        
        response = generate_images_service.generate_images(
            prompt_dict,
            page_connections_dict,
            reference_image,
            gender,
            age,
            image_style,
            coverpage=coverpage,
            sequential=sequential,
            existing_pages=existing_pages_dict
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

