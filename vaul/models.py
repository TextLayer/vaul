from pydantic import BaseModel, Extra


class BaseTool(BaseModel):
    class Config:
        extra = Extra.allow