import json
from pathlib import Path

import PIL.Image
import google.generativeai as genai
import requests
import typing_extensions as typing
from loguru import logger

from .config import conf

genai.configure(api_key=conf.gemini_api_key)


def gen_prompt_from_image_mask_and_user(
    image_path: Path, mask_path: Path, user_prompt: str
) -> str:
    class PromptSchema(typing.TypedDict):
        prompt: str

    model = genai.GenerativeModel(model_name=conf.gemini_model)

    image = PIL.Image.open(image_path)
    mask = PIL.Image.open(mask_path)

    response = model.generate_content(
        [
            """
You are text-to-image prompt writer:

- Takes three inputs:
    1. Original image
    2. Mask image (black areas indicate edit regions, white areas indicate areas to keep)
    3. User's edit description (in any language)
- Generates FULL English prompt:
    1. Maintains original style and content of original image
    2. Incorporates user's changes, specifies locations and sizes of masked areas clearly
    3. Ensures prompt is clear, concise, and accurate to generate new image
    4. Ensures prompt is English regardless of user's input language
""",
            "### ORIGINAL IMAGE ###\n\n",
            image,
            "### MASK IMAGE ###\n\n",
            mask,
            "### USER'S EDIT DESCRIPTION ###\n\n" + user_prompt.strip(),
        ],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", response_schema=PromptSchema
        ),
    )
    prompt = json.loads(response.text)["prompt"]
    logger.info(f"gen prompt: '{prompt}' from '{user_prompt.strip()}'")
    return prompt


def edit_image_by_mask_and_prompt(
    image_path: Path, mask_path: Path, prompt: str
) -> dict:
    with open(image_path, "rb") as image_file:
        with open(mask_path, "rb") as mask_file:
            resp = requests.post(
                "https://api.ideogram.ai/edit",
                data={
                    "prompt": prompt,
                    "model": "V_2_TURBO",
                    "magic_prompt_option": "AUTO",
                    "style_type": "AUTO",
                },
                files={
                    "image_file": image_file,
                    "mask": mask_file,
                },
                headers={"Api-Key": conf.ideogram_api_key},
            )
            resp.raise_for_status()
    data = resp.json()["data"][0]
    logger.info(f"edited image: {data}")
    return data
