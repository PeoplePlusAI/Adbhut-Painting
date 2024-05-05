from enum import Enum
from pydantic import BaseModel
from typing import Optional

class ExpressionEnum(str, Enum):
    idle = "idle"
    talking = "talking"
    deadface = "deadface"
    surprised = "surprised"

class VoiceResponseModel(BaseModel):
    response: Optional[str] = ""
    audio: Optional[str] = ""
    expression: Optional[ExpressionEnum] = ""