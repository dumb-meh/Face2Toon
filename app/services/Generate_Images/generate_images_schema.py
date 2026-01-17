from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class GenerateImageRequest(BaseModel):
   prompt: Dict[str, str] = Field(description="Image generation prompts for each page")
   page_connections: Optional[Dict[str, str]] = Field(default=None, description="Visual connections between pages")
   gender: str = Field(description="Child's gender")
   age: int = Field(description="Child's age")
   language: str = Field(description="Story language")
   image_style: str = Field(description="Desired illustration style")
   image: Optional[str] = Field(default=None, description="s3 bucket uploaded url of the reference image")

class PageImageUrls(BaseModel):
    name: str = Field(description="Page identifier (e.g., 'page 1')")
    fullPageUrl: str = Field(description="Full original generated image URL")
    leftUrl: Optional[str] = Field(default=None, description="Left half of split image URL")
    rightUrl: Optional[str] = Field(default=None, description="Right half of split image URL")
   
class GenerateImageResponse(BaseModel):
    image_urls: List[PageImageUrls] = Field(description="Generated image URLs for each page with full and split versions")
    pdf_url: Optional[str] = Field(default=None, description="URL of the generated PDF book")