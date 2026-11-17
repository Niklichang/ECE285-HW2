import json

import torch
import torch.nn.functional as F
from ignite.metrics import FID, InceptionScore
from torchvision.utils import save_image

from gan import Generator, load_images

IMAGE_DIR = "data/wikiart/subset"
OUT_DIR = "results"
BATCH_SIZE = 64

torch.manual_seed(123)
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(f"{OUT_DIR}/generator.pt", map_location=device)
G = Generator(ckpt["z_dim"]).to(device)
G.load_state_dict(ckpt["G"])
G.eval()

# Inception Score and FID of generated paintings, using pytorch-ignite

def to_inception(x):
    return F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)


def is_and_fid(images, reference):
    fid, inception_score = FID(device=device), InceptionScore(device=device)
    for i in range(0, len(images), BATCH_SIZE):
        x = to_inception(images[i:i + BATCH_SIZE].to(device))
        r = to_inception(reference[i:i + BATCH_SIZE].to(device))
        fid.update((x, r))
        inception_score.update(x)
    return float(inception_score.compute()), float(fid.compute())


# Split the real images into halves A and B.
# Generated images are compared with A; real images B are compared with A as a reference.
real = load_images(IMAGE_DIR).float() / 255
real = real[torch.randperm(len(real))]
n = len(real) // 2
real_a, real_b = real[:n], real[n:2 * n]

with torch.no_grad():
    generated = torch.cat([G(torch.randn(BATCH_SIZE, ckpt["z_dim"], device=device)).cpu()
                           for _ in range(0, n, BATCH_SIZE)])[:n] * 0.5 + 0.5   # to [0, 1]

is_gen, fid_gen = is_and_fid(generated, real_a)
is_real, fid_real = is_and_fid(real_b, real_a)

results = {"n_images": n,
           "generated": {"IS": is_gen, "FID": fid_gen},
           "real_reference": {"IS": is_real, "FID": fid_real}}
print(json.dumps(results, indent=2))
json.dump(results, open(f"{OUT_DIR}/results.json", "w"), indent=2)

save_image(generated[:32], f"{OUT_DIR}/samples.png", nrow=8, padding=2, pad_value=1.0)
