import tempfile

import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from core.ai import edit_image_by_mask_and_prompt, edit_image_by_prompt
from core.img import save_image_and_mask

app = FastAPI()


@app.get("/ping")
def ping():
    return "pong"


class EditRequest(BaseModel):
    image_url: str
    polygons: list[list[list[float]]]
    prompt: str


class EditResponse(BaseModel):
    url: str


@app.post("/api/image/edit")
def image_edit(req: EditRequest):
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path, mask_path = save_image_and_mask(
            temp_dir, req.image_url, req.polygons
        )
        data = edit_image_by_mask_and_prompt(image_path, mask_path, req.prompt)
        return EditResponse(url=data["url"])


class EditWithoutMaskRequest(BaseModel):
    image_url: str
    prompt: str


@app.post("/api/image/edit_without_mask")
def image_edit_without_mask(req: EditWithoutMaskRequest):
    return edit_image_by_prompt(req.image_url, req.prompt)


if __name__ == "__main__":
    logger.info("Starting server...")
    # run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
