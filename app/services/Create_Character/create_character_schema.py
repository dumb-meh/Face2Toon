from pydantic import BaseModel
from typing import Optional, List

class CreateCharacterRequest(BaseModel):
    past_characters: Optional[List[str]] = None

class CreateCharacterResponse(BaseModel):
    name: str
    age: int
    gender:str
    prompt:str
    character:List[str]
    image_url: str