from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from .generate_images_batch import GenerateImages
from .generate_images_batch_schema import GenerateImageRequest, GenerateImageResponse
from typing import Dict, Optional
import json

router = APIRouter()
generate_images_service = GenerateImages()     

@router.post("/generate-images-batch", response_model=GenerateImageResponse)
async def generate_images_batch(
    reference_image_1: Optional[UploadFile] = File(None, description="First reference image of the child"),
    reference_image_2: Optional[UploadFile] = File(None, description="Second reference image of the child"),
    reference_image_3: Optional[UploadFile] = File(None, description="Third reference image of the child"),
    story:str = Form(..., description="JSON string of story content for each page"),
    prompt: str = Form(..., description="JSON string of prompts for each page"),
    page_connections: Optional[str] = Form(default=None, description="JSON string of page connections"),
    gender: str = Form(..., description="Child's gender"),
    age: int = Form(..., description="Child's age"),
    image_style: str = Form(..., description="Desired illustration style"),
    coverpage: str = Query(default="no", description="'yes' if cover and page 1 already exist, 'no' to generate all pages (parameter kept for compatibility but not used in batch mode)"),
    sequential: str = Query(default="no", description="'yes' to force sequential generation with reference images, 'no' for default behavior"),
    page_0_url: Optional[str] = Form(default=None, description="Optional: URL of previously generated cover page image"),
    page_1_url: Optional[str] = Form(default=None, description="Optional: URL of previously generated page 1 image")
):
    """Generate ALL book images at once using batch mode. This endpoint generates cover, story pages, coloring pages, and back cover in a single batch operation."""
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
            # Handle case where page_connections is "string" or other placeholder text
            if page_connections and isinstance(page_connections, str):
                # Skip parsing if it's a placeholder like "string"
                if page_connections.strip().lower() in ['string', 'null', 'none', '']:
                    page_connections_dict = None
                else:
                    page_connections_dict = json.loads(page_connections)
            else:
                page_connections_dict = page_connections
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
        
        # Collect reference images - always use them in batch mode
        reference_images = []
        if reference_image_1:
            reference_images.append(reference_image_1)
        if reference_image_2:
            reference_images.append(reference_image_2)
        if reference_image_3:
            reference_images.append(reference_image_3)
        
        response = await generate_images_service.generate_images(
            prompts=prompt_dict,
            page_connections=page_connections_dict,
            reference_images=reference_images if reference_images else None,
            gender=gender,
            age=age,
            image_style=image_style,
            coverpage=coverpage,
            sequential=sequential,
            story=story_dict,
            page_0_url=page_0_url,
            page_1_url=page_1_url
        )
        
        # Wrap the response if it's already a list
        if isinstance(response, GenerateImageResponse):
            return response
        else:
            return GenerateImageResponse(image_urls=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

