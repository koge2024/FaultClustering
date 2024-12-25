# config_model.py
from pydantic import BaseModel, Field, validator
from typing import Optional

class Config(BaseModel):
    shapefile_path: str = Field(..., description="Path to the input shapefile")
    k: int = Field(..., gt=0, description="Number of clusters")
    R2: float = Field(..., ge=0.0, le=1.0, description="Threshold for linearity filtering")
    output_directory: str = Field(default="./output", description="Directory to save logs and results")
    log_level: Optional[str] = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    @validator('log_level')
    def validate_log_level(cls, v):
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v not in levels:
            raise ValueError(f"log_level must be one of {levels}")
        return v
