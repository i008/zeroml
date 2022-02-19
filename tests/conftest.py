import numpy as np
import PIL.Image
import pytest


@pytest.fixture
def random_pil_image() -> PIL.Image:
    image = PIL.Image.fromarray(np.random.randn(224, 224, 3).astype("uint8"))
    return image
