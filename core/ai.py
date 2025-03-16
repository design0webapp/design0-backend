import base64
from io import BytesIO
from pathlib import Path

import requests
from google import genai
from google.genai import types
from loguru import logger
from PIL import Image

from .config import conf

gemini_client = genai.Client(api_key=conf.gemini_api_key)


def edit_image_by_prompt(image_url: str, prompt: str):
    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash-exp",
        contents=[
            prompt,
            Image.open(requests.get(image_url, stream=True).raw),
        ],
        config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),
    )
    image = Image.open(
        BytesIO(response.candidates[0].content.parts[0].inline_data.data)
    )
    # Convert PIL Image to base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


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
                    "magic_prompt_option": "ON",
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


def upscale_image(image_url: str) -> str:
    # download image and send to ideogram
    resp = requests.get(image_url, stream=True)
    resp.raise_for_status()
    resp = requests.post(
        "https://api.ideogram.ai/upscale",
        files={
            "image_file": resp.raw,
        },
        headers={"Api-Key": conf.ideogram_api_key},
    )
    resp.raise_for_status()
    data = resp.json()["data"][0]
    logger.info(f"upscaled image: {data}")
    url = data["url"]
    # turn url to base64
    resp = requests.get(url)
    resp.raise_for_status()
    image = Image.open(resp.content)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
