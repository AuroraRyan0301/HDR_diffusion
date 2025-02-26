"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import OpenEXR
import Imath

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
def save_exr(filename, image):
    # 获取图像的形状
    height, width, channels = image.shape

    # 创建一个 OpenEXR 输出文件
    exr_file = OpenEXR.OutputFile(filename, OpenEXR.Header(width, height))

    # 为每个通道创建一个数组
    r = image[..., 0].astype(np.float32).tobytes()
    g = image[..., 1].astype(np.float32).tobytes()
    b = image[..., 2].astype(np.float32).tobytes()

    # 写入通道数据
    exr_file.writePixels({'R': r, 'G': g, 'B': b})

    # 关闭文件
    exr_file.close()

def save_depth_exr(filename, image):
    # 获取图像的形状
    height, width = image.shape

    # 创建一个 OpenEXR 输出文件
    exr_file = OpenEXR.OutputFile(filename, OpenEXR.Header(width, height))

    # 为每个通道创建一个数组
    r = image.astype(np.float32).tobytes()
    g = image.astype(np.float32).tobytes()
    b = image.astype(np.float32).tobytes()

    # 写入通道数据
    exr_file.writePixels({'R': r, 'G': g, 'B': b})

    # 关闭文件
    exr_file.close()


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 4, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join('/root/guided-diffusion/output', f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        for i, arr_slice in enumerate(arr):
            save_exr(os.path.join('/root/guided-diffusion/output', f"samples_{i}.exr"), arr_slice[:,:,:3]/255.0)
            save_depth_exr(os.path.join('/root/guided-diffusion/output', f"samples_depth_{i}.exr"), arr_slice[:,:,3]/255.0)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()