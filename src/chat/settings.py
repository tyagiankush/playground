from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file="../../.env", env_file_encoding="utf-8")
	
	# API Configuration
	api_key: str = Field(..., min_length=32)
	model_name: str = Field(default="gpt-4o-mini")
	
	# Model Configuration
	temperature: float = Field(default=0.7, ge=0.0, le=1.0)
	max_tokens: int = Field(default=2048, gt=0)
	top_p: float = Field(default=1.0, ge=0.0, le=1.0)
	
	# System Configuration
	debug: bool = Field(default=False)
	log_level: str = Field(default="INFO")
	
	@field_validator('api_key')
	def validate_api_key(cls, v: str) -> str:
		if not v.startswith(('sk-', 'hf-')):
			raise ValueError("API key must start with 'sk-' or 'hf-'")
		return v
	
	@field_validator('log_level')
	def validate_log_level(cls, v: str) -> str:
		valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
		if v.upper() not in valid_levels:
			raise ValueError(f"Log level must be one of {valid_levels}")
		return v.upper()


settings = Settings()
