from pydantic import BaseModel
from typing import Optional, List

class CreateCharacterRequest(BaseModel):
    past_characters: Optional[List[str]] = None

class CreateCharacterResponse(BaseModel):
    name: str
    age: int
    prompt:str