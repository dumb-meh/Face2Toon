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
        # Parse JSON strings
        try:
            story_dict = json.loads(story) if isinstance(story, str) else story
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid story JSON format: {str(e)}")
        
        try:
            prompt_dict = json.loads(prompt) if isinstance(prompt, str) else prompt
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid prompt JSON format: {str(e)}")
        
        try:
            page_connections_dict = json.loads(page_connections) if (page_connections and isinstance(page_connections, str)) else page_connections
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid page_connections JSON format: {str(e)}")
        
        # Handle double-encoded JSON (if client sends JSON string of JSON string)
        if isinstance(prompt_dict, str):
            try:
                prompt_dict = json.loads(prompt_dict)
            except:
                raise HTTPException(status_code=400, detail="prompt appears to be double-encoded. Please send as single JSON string.")
        
        if isinstance(page_connections_dict, str):
            try:
                page_connections_dict = json.loads(page_connections_dict)
            except:
                pass  # It's okay if this fails, might be intentional
        
        if isinstance(story_dict, str):
            try:
                story_dict = json.loads(story_dict)
            except:
                raise HTTPException(status_code=400, detail="story appears to be double-encoded. Please send as single JSON string.")
        
        # Validate types
        if not isinstance(prompt_dict, dict):
            raise HTTPException(status_code=400, detail=f"prompt must be a JSON object (dict), got {type(prompt_dict).__name__}")
        if not isinstance(story_dict, dict):
            raise HTTPException(status_code=400, detail=f"story must be a JSON object (dict), got {type(story_dict).__name__}")
        if page_connections_dict is not None and not isinstance(page_connections_dict, dict):
            raise HTTPException(status_code=400, detail=f"page_connections must be a JSON object (dict), got {type(page_connections_dict).__name__}")
        
        response = generate_images_service.generate_first_two_page(
            prompts=prompt_dict,
            page_connections=page_connections_dict,
            reference_image=reference_image,
            gender=gender,
            age=age,
            image_style=image_style,
            sequential=sequential,
            story=story_dict
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
    story:str = Form(..., description="JSON string of story content for each page"),
    prompt: str = Form(..., description="JSON string of prompts for each page"),
    page_connections: Optional[str] = Form(default=None, description="JSON string of page connections"),
    gender: str = Form(..., description="Child's gender"),
    age: int = Form(..., description="Child's age"),
    image_style: str = Form(..., description="Desired illustration style"),
    coverpage: str = Query(default="no", description="'yes' if cover and page 1 already exist, 'no' to generate all pages"),
    sequential: str = Query(default="no", description="'yes' to force sequential generation with reference images, 'no' for default behavior"),
    page_1_url: Optional[str] = Form(default=None, description="Optional: URL of previously generated page 1 image (required when coverpage='yes' and sequential='yes')")
):
    """Generate images for all pages. Set coverpage='yes' to skip page 0 and page 1. If coverpage='yes', optionally provide page_1_url for sequential generation."""
    try:
        # Parse JSON strings
        try:
            story_dict = json.loads(story) if isinstance(story, str) else story
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid story JSON format: {str(e)}")
        
        try:
            prompt_dict = json.loads(prompt) if isinstance(prompt, str) else prompt
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid prompt JSON format: {str(e)}")
        
        try:
            page_connections_dict = json.loads(page_connections) if (page_connections and isinstance(page_connections, str)) else page_connections
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid page_connections JSON format: {str(e)}")
        
        # Handle double-encoded JSON
        if isinstance(prompt_dict, str):
            try:
                prompt_dict = json.loads(prompt_dict)
            except:
                raise HTTPException(status_code=400, detail="prompt appears to be double-encoded. Please send as single JSON string.")
        
        if isinstance(page_connections_dict, str):
            try:
                page_connections_dict = json.loads(page_connections_dict)
            except:
                pass
        
        if isinstance(story_dict, str):
            try:
                story_dict = json.loads(story_dict)
            except:
                raise HTTPException(status_code=400, detail="story appears to be double-encoded. Please send as single JSON string.")
        
        # Validate types
        if not isinstance(prompt_dict, dict):
            raise HTTPException(status_code=400, detail=f"prompt must be a JSON object (dict), got {type(prompt_dict).__name__}")
        if not isinstance(story_dict, dict):
            raise HTTPException(status_code=400, detail=f"story must be a JSON object (dict), got {type(story_dict).__name__}")
        if page_connections_dict is not None and not isinstance(page_connections_dict, dict):
            raise HTTPException(status_code=400, detail=f"page_connections must be a JSON object (dict), got {type(page_connections_dict).__name__}")
        
        response = generate_images_service.generate_images(
            prompts=prompt_dict,
            page_connections=page_connections_dict,
            reference_image=reference_image,
            gender=gender,
            age=age,
            image_style=image_style,
            coverpage=coverpage,
            sequential=sequential,
            story=story_dict,
            page_1_url=page_1_url
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

