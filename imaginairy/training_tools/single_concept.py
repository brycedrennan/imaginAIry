import os
import random
import re
from abc import abstractmethod
from collections import defaultdict

from einops import rearrange
from omegaconf import ListConfig
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision.transforms import transforms

from imaginairy.utils import instantiate_from_config


class Txt2ImgIterableBaseDataset(IterableDataset):
    """
    Define an interface to make the IterableDatasets for text2img data chainable.
    """

    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f"{self.__class__.__name__} dataset contains {self.__len__()} examples.")

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass


def _rearrange(x):
    return rearrange(x * 2.0 - 1.0, "c h w -> h w c")


class SingleConceptDataset(Dataset):
    """
    Dataset for finetuning a model on a single concept.

    Similar to "dreambooth"
    """

    def __init__(
        self,
        concept_label,
        class_label,
        concept_images_dir,
        class_images_dir,
        image_transforms=None,
    ):
        self.concept_label = concept_label
        self.class_label = class_label
        self.concept_images_dir = concept_images_dir
        self.class_images_dir = class_images_dir

        if isinstance(image_transforms, (ListConfig, list)):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend(
            [
                transforms.ToTensor(),
                transforms.Lambda(_rearrange),
            ]
        )
        image_transforms = transforms.Compose(image_transforms)

        self.image_transforms = image_transforms

        self._concept_image_filename_groups = None
        self._class_image_filenames = None

    def __len__(self):
        return len(self.concept_image_filename_groups) * 2

    def __getitem__(self, idx):
        if idx % 2:
            img_group = self._concept_image_filename_groups[int(idx / 2)]
            img_filename = random.choice(img_group)
            img_path = os.path.join(self.concept_images_dir, img_filename)

            txt = self.concept_label
        else:
            img_path = os.path.join(
                self.class_images_dir, self.class_image_filenames[int(idx / 2)]
            )
            txt = self.class_label
        try:
            image = Image.open(img_path).convert("RGB")
        except RuntimeError as e:
            msg = f"Could not read image {img_path}"
            raise RuntimeError(msg) from e
        image = self.image_transforms(image)
        data = {"image": image, "txt": txt}
        return data

    @property
    def concept_image_filename_groups(self):
        if self._concept_image_filename_groups is None:
            self._concept_image_filename_groups = _load_image_filenames_and_alts(
                self.concept_images_dir
            )
        return self._concept_image_filename_groups

    @property
    def class_image_filenames(self):
        if self._class_image_filenames is None:
            self._class_image_filenames = _load_image_filenames(self.class_images_dir)
        return self._class_image_filenames

    @property
    def num_records(self):
        return len(self)


def _load_image_filenames_and_alts(img_dir, image_extensions=(".jpg", ".jpeg", ".png")):
    """Loads images into groups (filenames tagged with `[alt-{n:02d}]` are grouped together)."""
    image_filenames = _load_image_filenames(img_dir, image_extensions)
    grouped_img_filenames = defaultdict(list)
    for filename in image_filenames:
        base_filename = re.sub(r"\[alt-\d*\]", "", filename)
        grouped_img_filenames[base_filename].append(filename)
    return list(grouped_img_filenames.values())


def _load_image_filenames(img_dir, image_extensions=(".jpg", ".jpeg", ".png")):
    image_filenames = []
    for filename in os.listdir(img_dir):
        if filename.lower().endswith(image_extensions) and not filename.startswith("."):
            image_filenames.append(filename)
    random.shuffle(image_filenames)
    return image_filenames
