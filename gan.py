"""DCGAN for 64x64 painting images and the image loader."""
import glob
import os
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

IMG_SIZE = 64


def _load_one(path):
    try:
        im = Image.open(path)
        im.draft("RGB", (IMG_SIZE * 2, IMG_SIZE * 2))# fast JPEG decode at reduced size
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        return np.asarray(im.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC))
    except Exception:
        return None # skip unreadable files


def load_images(image_dir, cache="data/images_64.npy"):
    if not os.path.exists(cache):
        files = sorted(f for ext in ("jpg", "jpeg", "png")
                       for f in glob.glob(os.path.join(image_dir, "**", f"*.{ext}"), recursive=True))
        print(f"loading {len(files)} images from {image_dir} ...")
        with Pool(os.cpu_count()) as pool:
            imgs = [x for x in pool.map(_load_one, files, chunksize=64) if x is not None]
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, np.stack(imgs))
    return torch.from_numpy(np.load(cache)).permute(0, 3, 1, 2).contiguous()


class Generator(nn.Module):
    """z (100) -> 512x4x4 -> 256x8x8 -> 128x16x16 -> 64x32x32 -> 3x64x64 in [-1, 1]"""
    def __init__(self, z_dim=100, ch=64):
        super().__init__()
        self.z_dim = z_dim
        def up(c_in, c_out, stride=2, pad=1):
            return [nn.ConvTranspose2d(c_in, c_out, 4, stride, pad, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(True)]
        self.net = nn.Sequential(
            *up(z_dim, ch * 8, 1, 0), *up(ch * 8, ch * 4), *up(ch * 4, ch * 2), *up(ch * 2, ch),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(z.view(-1, self.z_dim, 1, 1))


class Discriminator(nn.Module):
    """3x64x64 -> 64x32x32 -> 128x16x16 -> 256x8x8 -> 512x4x4 -> 1 logit (real vs fake)"""
    def __init__(self, ch=64):
        super().__init__()
        def down(c_in, c_out, bn=True):
            return [nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False)] + ([nn.BatchNorm2d(c_out)] if bn else []) + [nn.LeakyReLU(0.2, True)]
        self.net = nn.Sequential(
            *down(3, ch, bn=False), *down(ch, ch * 2), *down(ch * 2, ch * 4), *down(ch * 4, ch * 8),
            nn.Conv2d(ch * 8, 1, 4, 1, 0))

    def forward(self, x):
        return self.net(x).view(-1)


def init_weights(m): # DCGAN initialization: N(0, 0.02)
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
