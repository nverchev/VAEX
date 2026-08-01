"""CelebA dataset data builder and directory setup."""

from typing import ClassVar

import pandas as pd

from src.config.experiment import Experiment
from src.data.databuilder import Singleton
from src.data.preprocess import get_preprocessor
from src.data.split import CelebaDataset, Partitions


class CelebaDataBuilder(metaclass=Singleton):
    """Data builder managing CelebA attribute loading, pre-loading, and partition splits."""

    folder_name: ClassVar[str] = 'celeba_torchvision'
    train_val_split: ClassVar[int] = 162770
    val_test_split: ClassVar[int] = 182637

    def __init__(self) -> None:
        cfg = Experiment.get_config()
        ds_cfg = cfg.data.dataset
        self.classes = ds_cfg.classes
        self.data_dir = cfg.user.path.data_dir / self.folder_name
        self.target_attr = ds_cfg.target_attr
        self._ensure_dataset_downloaded()
        self.img_dir = self.data_dir / 'img_align_celeba'
        self.preprocessor = get_preprocessor()
        self.attributes = self._read_attributes()
        return

    def _ensure_dataset_downloaded(self) -> None:
        if self.data_dir.exists():
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            import torchvision.datasets as datasets

            datasets.CelebA(root=self.data_dir.parent, split='train', download=True)
        except Exception as err:
            print(
                f'Automatic torchvision CelebA download failed:\n\n\t{err}\n\n'
                'Please download the dataset manually. See the README for instructions.'
            )

        return

    def _read_attributes(self) -> pd.DataFrame:
        attr_file = self.data_dir / 'list_attr_celeba.txt'
        if not attr_file.exists():
            raise FileNotFoundError(
                f'CelebA attributes file not found at {attr_file}. Please ensure CelebA is downloaded correctly.'
            )

        df = pd.read_csv(attr_file, sep=r'\s+', skiprows=1, usecols=[self.target_attr])
        df.reset_index(inplace=True)
        df.columns = ['image_id', self.target_attr]
        df['image_id'] = df['image_id'].astype(str).str.replace('.jpg', '', regex=False).astype(int)
        unique_classes = df[self.target_attr].nunique()
        if unique_classes != len(self.classes):
            msg = 'Configured num_classes ({}) does not match existing number of classes ({})'
            raise ValueError(msg.format(len(self.classes), unique_classes))

        return df

    def get_split(self, partition: Partitions | str) -> CelebaDataset:
        total = len(self.attributes)
        tr_split = min(self.train_val_split, int(0.8 * total))
        val_split = min(self.val_test_split, int(0.9 * total))

        if isinstance(partition, Partitions):
            key = partition.name.lower()
        else:
            key = str(partition).lower()

        if key == 'train':
            df_subset = self.attributes.iloc[:tr_split]
        elif key in ('val', 'validation'):
            df_subset = self.attributes.iloc[tr_split:val_split]
        elif key == 'test':
            df_subset = self.attributes.iloc[val_split:]
        elif key in ('trainval', 'train_val'):
            df_subset = self.attributes.iloc[:val_split]
        else:
            df_subset = self.attributes.iloc[:tr_split]

        df_subset = df_subset.reset_index(drop=True)
        images = self.preprocessor.preload_uint8_images(self.img_dir, df_subset)
        return CelebaDataset(
            images=images,
            metadata=df_subset,
            target_attr=self.target_attr,
        )
