import uuid

import numpy as np
import pandas as pd
import PIL.Image
import pytest

from zeroml.dataloading.image.datasets import ImageClfAlbuDataset
from zeroml.dataloading.image.transforms import build_training_transform


@pytest.fixture
def create_test_dataframe(tmp_path):

    images = [PIL.Image.fromarray(np.ones((224, 224, 3)).astype("uint8")) for _ in range(10)]
    labels = [i for i in range(10)]
    fns = [str(uuid.uuid4()) + '.png' for _ in range(10)]
    print(fns)

    for image, f in zip(images, fns):
        image.save(tmp_path / f)

    df = pd.DataFrame({'labels': labels, 'file_names': fns})
    base_path = tmp_path
    return df, base_path


def test_clf_albu_dataset(create_test_dataframe):

    df, base_path = create_test_dataframe

    transform = build_training_transform(224, model=None, augment_level='noaug')

    ds = ImageClfAlbuDataset(
        df, base_path=base_path, transform=transform, file_column='file_names', label_column='labels'
    )

    image, label = ds[5]

    assert label == 5
