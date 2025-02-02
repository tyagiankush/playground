from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
	model_config = SettingsConfigDict(env_file='../.env', env_file_encoding='utf-8')
	api_key: str
	model_name: str = Field(default='gpt-4o-mini')


settings = Settings()
