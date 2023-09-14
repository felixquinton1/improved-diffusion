from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from monai import data, transforms
import os

from crop.crop import RandCropByLabelClassesd

def load_data(
    *, data_dir, json_list, batch_size, image_size, roi, class_cond=False, debug=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    # if class_cond:
    #     # Assume classes are the first part of the filename,
    #     # before an underscore.
    #     class_names = [bf.basename(path).split("_")[0] for path in all_files]
    #     sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    #     classes = [sorted_classes[x] for x in class_names]
    # dataset = ImageDataset(
    #     image_size,
    #     all_files,
    #     classes=classes,
    #     shard=MPI.COMM_WORLD.Get_rank(),
    #     num_shards=MPI.COMM_WORLD.Get_size(),
    # )
    datalist_json = os.path.join(data_dir, json_list)
    files = data.load_decathlon_datalist(datalist_json,
                                    True,
                                    "training",
                                    base_dir=data_dir)
    transform = transforms_list(roi[0], roi[1], roi[2])

    if debug:
        dataset = data.Dataset(
            data=files,
            transform=transform
            #   copy_cache=True,
        )

    else:
        dataset = data.CacheDataset(
            data=files,
            transform=transform,
            cache_rate=1.0,
            cache_num=20,
            num_workers=8,
            #   copy_cache=True,
        )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    # return  loader
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict

def transforms_list(roi_x, roi_y, roi_z):
    all_keys = ["image", "label"]
    transforms_list = [transforms.LoadImaged(keys=all_keys, image_only=True),
                       transforms.EnsureChannelFirstd(keys=all_keys, channel_dim='no_channel'),
                       transforms.Orientationd(keys=all_keys, axcodes="RAS"),
                       transforms.SpatialPadd(keys=all_keys, spatial_size=(roi_x, roi_y, roi_z)),
                       transforms.RandRotated(keys=all_keys,
                                              mode=["bilinear", "nearest"],
                                              range_x=0.26,
                                              range_y=0.26, range_z=0.26,
                                              prob=0.2, keep_size=True,
                                              align_corners=False, allow_missing_keys=False),
                       transforms.RandFlipd(keys=all_keys, prob=0.1, spatial_axis=(0)),
                       transforms.RandFlipd(keys=all_keys, prob=0.1, spatial_axis=(1)),
                       transforms.RandFlipd(keys=all_keys, prob=0.1, spatial_axis=(2)),
                       # RandSpatialCropd(keys=all_keys, roi_size=(96, 64, 64), random_size=False),
                       RandCropByLabelClassesd(keys=all_keys, label_key="label", ratios=[0,0,1], num_classes=3, spatial_size=(64, 64, 64)),
                       transforms.ToTensord(keys=all_keys),
                       ]
    return transforms.Compose(transforms_list)

