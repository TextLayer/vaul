from pydantic import BaseModel, ConfigDict


class BaseTool(BaseModel):
    model_config = ConfigDict(extra="allow")
