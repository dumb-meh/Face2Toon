from pydantic import BaseModel, Field
from typing import Optional, Dict

class GenerateImageRequest(BaseModel):
   prompt: Dict[str, str]
   
class GenerateImageResponse(BaseModel):
    image_urls: Dict[str, str]