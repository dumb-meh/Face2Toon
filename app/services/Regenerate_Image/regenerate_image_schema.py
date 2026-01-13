from pydantic import BaseModel, Field
from typing import Optional, Dict, List

class ReGenerateImageRequest(BaseModel):
   prompt:str
   story:str
   iamge_url:str = Field(description="URL of the image to be regenerated")
   gender: str = Field(description="Child's gender")
   age: int = Field(description="Child's age")
   image_style: str = Field(description="Desired illustration style")
   page_type:str = Field(description="Type of the page")
   page_number:int = Field(description="Page number to be regenerated")

class PageImageUrls(BaseModel):
    fullPageUrl: str = Field(description="Full original generated image URL")
    leftUrl: Optional[str] = Field(default=None, description="Left half of split image URL")
    rightUrl: Optional[str] = Field(default=None, description="Right half of split image URL")
    
class ReGenerateImageResponse(BaseModel):
    image_url:List[PageImageUrls] = Field(description="Regenerated image URL of the page with full and split versions")