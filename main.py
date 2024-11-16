import tempfile

import uvicorn
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel

from core.ai import gen_prompt_from_image_mask_and_user, edit_image_by_mask_and_prompt
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
