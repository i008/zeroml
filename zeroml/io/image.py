import io
import uuid
from pathlib import Path
from typing import Union

import fsspec
import PIL.Image


def save_array_image(np_array, target_path, file_name=None) -> Path:
    """


    Parameters
    ----------
    np_array
    target_path
    file_name

    Returns
    -------

    """
    pil_image = PIL.Image.fromarray(np_array)
    return save_pil(pil_image, target_path, file_name)


def save_pil(
    pil_image: PIL.Image.Image,
    target_path: Union[str, Path],
    file_name=None,
    image_type="PNG",
) -> Path:
    """

    Saves a pil image to disk, cloud or anywhere supported by fsspec.

    Parameters
    ----------
    pil_image
    target_paths
    file_name
    image_type

    Returns
    -------

    """
    if file_name is None:
        file_name = str(uuid.uuid4())

    if isinstance(target_path, str):
        target_path = Path(target_path)

    target = target_path / file_name
    pil_image = pil_image.convert("RGB")
    bio = io.BytesIO()
    pil_image.save(bio, image_type)
    pil_image.close()
    with fsspec.open(target, "wb") as storage:
        storage.write(bio.getvalue())

    return target
