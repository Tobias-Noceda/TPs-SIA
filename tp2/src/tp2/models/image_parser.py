from functools import cache

from tp2.types.coords import Coords
from tp2.types.individual import Individual

import numpy as np
from PIL import Image, ImageDraw

WHITE = (255, 255, 255, 255)

class ImageParser:
    @staticmethod
    def _open_image(path: str, lt: Coords, rb: Coords | None = None):
        img = ImageParser._get_img(path)

        left, top = lt
        right, bottom = img.size if rb is None else rb

        region = (left, top, right, bottom)
        reference = np.asarray(img.crop(region), dtype=np.float32) / 255.0

        width = right - left
        height = bottom - top

        color = ImageParser._get_dominant_color(img)
        if not isinstance(color, tuple):
            color = WHITE

        return reference, width, height, color

    @staticmethod
    @cache
    def _get_img(path: str):
        return Image.open(path).convert("RGB")

    @staticmethod
    def _get_dominant_color(img: Image.Image):
        colors = img \
            .resize((150, 150)) \
            .convert("P", palette=Image.Palette.ADAPTIVE, colors=256) \
            .convert("RGB") \
            .getcolors(150 * 150)

        return None if colors is None else max(colors, key=lambda item: item[0])[1]

    @staticmethod
    @cache
    def _get_individual_render(individual: Individual, w: int, h: int, bg: tuple[int, ...]) -> np.ndarray:
        img = Image.new("RGBA", (w, h), bg)
        draw = ImageDraw.Draw(img)

        for polygon in individual:
            draw.polygon(polygon.vertices, fill=polygon.color.pillow())

        img = img.convert("RGB")

        return np.asarray(img)
