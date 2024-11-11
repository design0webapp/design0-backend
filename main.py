import tempfile
from typing import Optional

import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from core.ai import gen_prompt_from_image_mask_and_user, edit_image_by_mask_and_prompt
from core.img import save_image_and_mask

app = FastAPI()
# URL_SUFFIX used to limit size of image
URL_SUFFIX = "?fm=jpg&q=80&w=1080&h=1080&fit=max"


class Image(BaseModel):
    id: str
    url: str
    category: str
    description: str


class ImagesResponse(BaseModel):
    images: list[Image]


@app.get("/api/image/random")
def image_random(limit: int = 10) -> ImagesResponse:
    pass


@app.get("/api/image/search")
def image_search(
    query: str, limit: int = 10, category: Optional[str] = None
) -> ImagesResponse:
    pass


class EditRequest(BaseModel):
    image_url: str
    polygons: list[list[list[float]]]
    prompt: str


class EditResponse(BaseModel):
    url: str


@app.post("/api/image/edit")
def image_edit(req: EditRequest):
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path, mask_path = save_image_and_mask(
            temp_dir, req.image_url, req.polygons
        )
        prompts = gen_prompt_from_image_mask_and_user(image_path, mask_path, req.prompt)
        data = edit_image_by_mask_and_prompt(image_path, mask_path, prompts)
        return EditResponse(url=data["url"])


if __name__ == "__main__":
    logger.info("Starting server...")
    # run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
