"""VAEX image preprocessing pipeline and HDF5 cache loader."""

import pathlib

from typing import cast

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import h5py

from src.config.experiment import Experiment


class Preprocessor:
    """VAEX image preprocessor for CelebA facial images with HDF5 dataset caching."""

    def __init__(self) -> None:
        cfg = Experiment.get_config()
        self.crop_size = cfg.data.crop_size
        self.image_size = cfg.data.x_dim
        self.chunk_size = 256
        return

    def transform_image(self, image: Image.Image) -> torch.Tensor:
        w, h = image.size
        crop_dim = min(w, h, self.crop_size)
        transform = transforms.Compose(
            [
                transforms.CenterCrop(crop_dim),
                transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )
        # the last transform turns the image into a tensor
        return cast(torch.Tensor, transform(image))

    def _tensor_to_uint8_numpy(self, image: torch.Tensor) -> np.ndarray:
        """Convert a float tensor in [0, 1] to a uint8 numpy array."""
        return image.mul(255).clamp(0, 255).to(torch.uint8).numpy()

    def _create_hdf5_cache(self, img_dir: pathlib.Path, h5_file: pathlib.Path) -> None:
        """Create an HDF5 cache from a directory of JPEG images.

        The cache is written under a temporary name and renamed when complete, so that an
        interrupted run leaves no partial cache behind.
        """
        files_map = {int(path.stem): path for path in img_dir.glob('*.jpg')}
        if not files_map:
            raise FileNotFoundError(f'No CelebA images found in {img_dir}')

        total_images = max(files_map)
        h5_file.parent.mkdir(parents=True, exist_ok=True)
        partial_file = h5_file.with_suffix('.partial')
        with h5py.File(partial_file, 'w') as h5f:
            dataset = h5f.create_dataset(
                'images',
                shape=(total_images, 3, self.image_size, self.image_size),
                dtype=np.uint8,
                chunks=(self.chunk_size, 3, self.image_size, self.image_size),
            )
            for img_id, img_path in files_map.items():
                with Image.open(img_path) as image:
                    image = image.convert('RGB')
                    image = self.transform_image(image)

                dataset[img_id - 1] = self._tensor_to_uint8_numpy(image)

        partial_file.rename(h5_file)
        return

    def preload_uint8_images(
        self,
        img_dir: pathlib.Path,
        df_subset: pd.DataFrame,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Pre-load CelebA images on a device as a uint8 tensor (N, 3, H, W)."""
        h5_file = img_dir.parent / f'celeba_{self.image_size}x{self.image_size}_crop{self.crop_size}_bicubic.h5'
        if not h5_file.exists():
            self._create_hdf5_cache(img_dir, h5_file)

        image_ids = df_subset['image_id'].to_numpy(dtype=np.int64) - 1
        n_images = len(image_ids)
        images = torch.empty(
            (n_images, 3, self.image_size, self.image_size),
            dtype=torch.uint8,
            device=device,
        )
        sort_idx = np.argsort(image_ids)
        sorted_ids = image_ids[sort_idx]
        load_chunk_size = self.chunk_size * 16
        with h5py.File(h5_file, 'r', locking=False, rdcc_nbytes=1024 * 1024) as h5f:
            h5_dataset = h5f['images']
            if not isinstance(h5_dataset, h5py.Dataset):
                msg = f'{h5_file} holds no image dataset.'
                raise TypeError(msg)

            for start in range(0, n_images, load_chunk_size):
                end = min(start + load_chunk_size, n_images)
                ids_chunk = sorted_ids[start:end]
                if ids_chunk[-1] - ids_chunk[0] + 1 == len(ids_chunk):
                    images_np = h5_dataset[ids_chunk[0] : ids_chunk[-1] + 1]
                else:
                    images_np = h5_dataset[ids_chunk]

                orig_idx = sort_idx[start:end]
                images[orig_idx] = torch.from_numpy(images_np).to(device)

        return images


def get_preprocessor() -> Preprocessor:
    """Instantiate VAEX preprocessor from global configuration."""
    return Preprocessor()
