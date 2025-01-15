import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

import OpenEXR, Imath
import os
import re
# os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def read_openexr_to_numpy(file_path):
    # 打开 EXR 文件
    exr_file = OpenEXR.InputFile(file_path)
    
    # 获取图像尺寸
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 读取 RGB 通道
    channels = ['R', 'G', 'B']
    data = {channel: exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT)) for channel in channels}

    # 转换为 NumPy 数组
    img = np.zeros((height, width, 3), dtype=np.float32)
    for i, channel in enumerate(channels):
        img[..., i] = np.fromstring(data[channel], dtype=np.float32).reshape(height, width)

    exr_file.close()
    return img


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
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
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)
    # all_dirs = _list_image_dir_recursively(data_dir)
    all_rgb_exr_path = _list_rgb_path_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_rgb_exr_path,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
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

def _list_image_dir_recursively(data_dir):
    all_items = os.listdir(data_dir)
    # 过滤出所有子目录并获取其完整路径
    subdirectories = [os.path.join(data_dir, item) for item in all_items if os.path.isdir(os.path.join(data_dir, item))]
    return subdirectories

def _list_rgb_path_recursively(data_dir):
    all_rgb_paths = []
    for scene in os.listdir(data_dir):
        scene_path = os.path.join(data_dir, scene)
        for filename in os.listdir(scene_path):
            if filename.startswith('rgb') and filename.endswith('.exr'):
                all_rgb_paths.append(os.path.join(scene_path, filename))
    return all_rgb_paths


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    # def __getitem__(self, idx):
    #     path = self.local_images[idx]
    #     with bf.BlobFile(path, "rb") as f:
    #         pil_image = Image.open(f)
    #         pil_image.load()
    #     pil_image = pil_image.convert("RGB")

    #     if self.random_crop:
    #         arr = random_crop_arr(pil_image, self.resolution)
    #     else:
    #         arr = center_crop_arr(pil_image, self.resolution)

    #     if self.random_flip and random.random() < 0.5:
    #         arr = arr[:, ::-1]

    #     arr = arr.astype(np.float32) / 127.5 - 1

    #     out_dict = {}
    #     if self.local_classes is not None:
    #         out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
    #     return np.transpose(arr, [2, 0, 1]), out_dict

    def __getitem__(self, idx):
        # TODO: not support class
        path = self.local_images[idx]
        # numbers = int(re.findall(r'\d+', os.path.basename(path))[0])
        # with bf.BlobFile(os.path.dirname(path) + "/rgb_{}.exr".format(numbers), "rb") as f:
        #     arr = read_openexr_to_numpy(f)

        with bf.BlobFile(path, "rb") as f:
           arr = read_openexr_to_numpy(f)


        # if self.random_crop:
        #     arr = random_crop_arr(rgb_pil_image, self.resolution)
        #     depth_arr =  random_crop_arr(depth_pil_image, self.resolution)
        # else:
        #     arr = center_crop_arr(rgb_pil_image, self.resolution)
        #     depth_arr =  center_crop_arr(depth_pil_image, self.resolution)

        if self.random_flip and random.random() < 0.5:
            arr = arr[:, ::-1]
            # depth_arr = depth_arr[:, ::-1]
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        # combined_arr = np.concatenate([arr, depth_arr[:, :, 0:1]], axis=-1)
        combined_arr = arr * 2.0 - 1.0
        # print(  combined_arr.shape)
        return np.transpose(combined_arr, [2, 0, 1]), out_dict



def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    arr = np.array(pil_image)
    
    # 如果图像的宽高已经符合目标大小，不进行裁剪
    if arr.shape[0] == image_size and arr.shape[1] == image_size:
        return arr

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
