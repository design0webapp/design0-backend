from pydantic_settings import BaseSettings


class Config(BaseSettings):
    ideogram_api_key: str
    gemini_api_key: str


conf = Config()
