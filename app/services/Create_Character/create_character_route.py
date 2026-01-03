from fastapi import APIRouter, HTTPException
from app.services.Create_Character.create_character_schema import CreateCharacterRequest, CreateCharacterResponse
from app.services.Create_Character.create_character import CreateCharacter


router = APIRouter()
create_character= CreateCharacter()     
@router.post("/create_character", response_model=CreateCharacterResponse)
async def  get_create_character(request: CreateCharacterRequest):
    try:
        response = create_character.create_character(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
