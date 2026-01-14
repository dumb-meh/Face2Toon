from pydantic import BaseModel, Field
from typing import Optional, List
from fastapi import UploadFile

class PageImageUrls(BaseModel):
    name: str = Field(description="Page identifier (e.g., 'page 1')")
    fullPageUrl: str = Field(description="Full original generated image URL")
    leftUrl: Optional[str] = Field(default=None, description="Left half of split image URL")
    rightUrl: Optional[str] = Field(default=None, description="Right half of split image URL")
      
class SwapCharacterRequest(BaseModel):
    full_page_urls: List[str] = Field(description="List of 11 full-page image URLs (pages 1-11)")
    prompts: dict = Field(description="Dictionary of prompts for all pages (page 0 to page 13)")
    story: dict = Field(description="Dictionary of story text for each page")
    character_name: str = Field(description="New character's name")
    gender: str = Field(description="New character's gender (male/female)")
    age: int = Field(description="New character's age")
    image_style: str = Field(description="Image style (e.g., 'watercolor', 'cartoon')")

class SwapCharacterResponse(BaseModel):
    image_urls: List[PageImageUrls] = Field(description="Generated image URLs for each page with full and split versions")
    pdf_url: Optional[str] = Field(default=None, description="URL of the generated PDF book")