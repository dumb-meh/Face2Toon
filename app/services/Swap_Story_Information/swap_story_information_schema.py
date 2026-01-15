from pydantic import BaseModel, Field
from typing import Optional, List

class PageImageUrls(BaseModel):
    name: str = Field(description="Page identifier (e.g., 'page 1')")
    fullPageUrl: str = Field(description="Full original generated image URL")
    leftUrl: Optional[str] = Field(default=None, description="Left half of split image URL")
    rightUrl: Optional[str] = Field(default=None, description="Right half of split image URL")
      
class SwapStoryInformationRequest(BaseModel):
    full_page_urls: List[str] = Field(description="List of full-page image URLs")
    story: dict = Field(description="Dictionary of story text for each page")
    character_name: str = Field(description="New character's name")
    language: str = Field(description="Target language for the story")
    story_language: str = Field(description="Current language of the story")
    change_type: str = Field(description="Type of information to swap: 'language', 'name', or 'both'")

class SwapStoryInformationResponse(BaseModel):
    image_urls: List[PageImageUrls] = Field(description="Generated image URLs for each page with full and split versions")
    pdf_url: Optional[str] = Field(default=None, description="URL of the generated PDF book")