from pathlib import Path
from typing import List

import cv2
import numpy as np
import requests


def save_image_and_mask(
    folder: str, image_url: str, polygons: List[List[List[float]]]
) -> (Path, Path):
    # Download the image
    response = requests.get(image_url)
    image_path = Path(folder) / "image"
    with open(image_path, "wb") as f:
        f.write(response.content)

    # Load the image
    image = cv2.imread(str(image_path))

    # Create a white mask with the same size as the image
    mask = np.full(image.shape[:2], 255, dtype=np.uint8)

    # Draw polygons in black on the mask
    for polygon in polygons:
        points = np.int32(polygon)
        # Draw filled polygon in black (0)
        cv2.fillPoly(mask, [points], 0)

    # Save mask if needed
    mask_path = Path(folder) / "mask.png"
    cv2.imwrite(str(mask_path), mask)

    return image_path, mask_path
