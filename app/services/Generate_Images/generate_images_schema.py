from pydantic import BaseModel, Field
from typing import Optional, Dict

class GenerateImageRequest(BaseModel):
   prompt: Dict[str, str] = Field(description="Image generation prompts for each page")
   page_connections: Optional[Dict[str, str]] = Field(default=None, description="Visual connections between pages")
   gender: str = Field(description="Child's gender")
   age: int = Field(description="Child's age")
   image_style: str = Field(description="Desired illustration style")
   
class GenerateImageResponse(BaseModel):
    image_urls: Dict[str, str] = Field(description="Generated image URLs for each page")