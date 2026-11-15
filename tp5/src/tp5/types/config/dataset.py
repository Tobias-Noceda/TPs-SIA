from typing import List, Literal, NotRequired, TypedDict

class Dataset(TypedDict):
    """
    Attributes:
        type: `font` (a font.h file, default) or `images` (a folder of images, read recursively).
        path: The font.h file or the image folder.
        size: [height, width] every image is resized to (images only).
        channels: 1 = grayscale, 3 = RGB (images only, default 3).
        fit: How images of another aspect ratio are resized: crop, pad or stretch (images only, default stretch).
    """
    type: NotRequired[Literal["font", "images"]]
    path: str
    size: NotRequired[List[int]]
    channels: NotRequired[int]
    fit: NotRequired[Literal["crop", "pad", "stretch"]]
