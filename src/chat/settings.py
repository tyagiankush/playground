from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
	"""Configuration for a specific LLM model."""

	name: str
	temperature: float = Field(default=0.7, ge=0.0, le=1.0)
	max_tokens: int = Field(default=2048, gt=0)
	top_p: float = Field(default=1.0, ge=0.0, le=1.0)
	context_window: int = Field(default=4096, gt=0)


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file="../../.env", env_file_encoding="utf-8")

	# API Configuration
	api_key: str = Field(..., min_length=32)

	# Model Selection
	default_model: Literal["deepseek-r1", "mistral", "llama2"] = Field(default="deepseek-r1")

	# Model Configurations
	models: dict[str, ModelConfig] = Field(
		default_factory=lambda: {
			"deepseek-r1": ModelConfig(name="deepseek-r1", temperature=0.7, max_tokens=2048, context_window=4096),
			"mistral": ModelConfig(name="mistral", temperature=0.7, max_tokens=2048, context_window=4096),
			"llama2": ModelConfig(name="llama2", temperature=0.7, max_tokens=2048, context_window=4096),
		}
	)

	# Search Configuration
	use_hybrid_search: bool = Field(default=True)
	semantic_weight: float = Field(default=0.7, ge=0.0, le=1.0)
	keyword_weight: float = Field(default=0.3, ge=0.0, le=1.0)

	# System Configuration
	debug: bool = Field(default=False)
	log_level: str = Field(default="INFO")

	@field_validator("api_key")
	def validate_api_key(cls, v: str) -> str:
		if not v.startswith(("sk-", "hf-")):
			raise ValueError("API key must start with 'sk-' or 'hf-'")
		return v

	@field_validator("log_level")
	def validate_log_level(cls, v: str) -> str:
		valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
		if v.upper() not in valid_levels:
			raise ValueError(f"Log level must be one of {valid_levels}")
		return v.upper()

	@field_validator("semantic_weight", "keyword_weight")
	def validate_weights(cls, v: float) -> float:
		if v < 0 or v > 1:
			raise ValueError("Weights must be between 0 and 1")
		return v


settings = Settings()
