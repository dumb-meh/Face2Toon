from pydantic import BaseModel, Field
from typing import Optional, Dict

class GenerateImageRequest(BaseModel):
   gender: str
   name: str
   age: int
   image_style: str
   language: str
   user_input: str

class GenerateImageResponse(BaseModel):
    story: Dict[str, str] = Field(
        description="Story content for each page. Key format: 'page 0' (cover with title only), 'page 1' through 'page 10' (story pages). Must contain exactly 11 pages."
    )
    prompt: Dict[str, str] = Field(
        description="Image generation prompts for each page. Key format: 'page 0' through 'page 10'. Each prompt should be appropriate for children's book illustrations."
    )
    page_connections: Optional[Dict[str, str]] = Field(
        default=None,
        description="Visual connections between pages. Key is the page being generated, value is the reference page whose generated image should be used. Example: {'page 3': 'page 1'} means page 3 should use page 1's image as reference."
    )