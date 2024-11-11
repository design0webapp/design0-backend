from pydantic_settings import BaseSettings


class Config(BaseSettings):
    # ideogram
    ideogram_api_key: str
    # google gemini
    gemini_api_key: str
    gemini_model: str = "gemini-1.5-flash"


conf = Config()
