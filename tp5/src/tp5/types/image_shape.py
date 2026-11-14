from typing import NamedTuple

class ImageShape(NamedTuple):
    """
    Shape of the images of a dataset. A sample is the image flattened row-major and channels-last
    (pixel (r, c) channel k at index (r · width + c) · channels + k), values in [0, 1].

    Attributes:
        height: Rows.
        width: Columns.
        channels: 1 = grayscale, 3 = RGB.
    """
    height: int
    width: int
    channels: int = 1

    @property
    def size(self) -> int:
        return self.height * self.width * self.channels
