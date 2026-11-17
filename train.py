import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from gan import Discriminator, Generator, init_weights, load_images

# hyperparams
IMAGE_DIR = "data/wikiart/subset"   # folder with the training images
OUT_DIR = "results"
Z_DIM = 100
BATCH_SIZE = 128
EPOCHS = 100
LR = 2e-4
BETAS = (0.5, 0.999)
SEED = 0


# train DCGAN on WikiArt paintings

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu" # for datahub
os.makedirs(OUT_DIR, exist_ok=True)

data = load_images(IMAGE_DIR).to(device) # uint8 [N,3,64,64], kept on the GPU
n = len(data)
print(f"images={n} device={device}")

G, D = Generator(Z_DIM).to(device), Discriminator().to(device)
G.apply(init_weights)
D.apply(init_weights)
opt_g = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
opt_d = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)
print(f"G params: {sum(p.numel() for p in G.parameters()) / 1e6:.2f}M, "
      f"D params: {sum(p.numel() for p in D.parameters()) / 1e6:.2f}M")

fixed_z = torch.randn(64, Z_DIM, device=device)
hist = {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": []}   # per iteration
iters_per_epoch = n // BATCH_SIZE
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(n, device=device)
    for i in range(iters_per_epoch):
        real = data[perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]].float() / 127.5 - 1 # to [-1, 1]
        flip = torch.rand(len(real), 1, 1, 1, device=device) < 0.5
        real = torch.where(flip, real.flip(3), real) # random horizontal flip
        ones = torch.ones(len(real), device=device)
        zeros = torch.zeros(len(real), device=device)

        #discriminator: real -> 1, fake -> 0
        fake = G(torch.randn(len(real), Z_DIM, device=device))
        d_real, d_fake = D(real), D(fake.detach())
        d_loss = F.binary_cross_entropy_with_logits(d_real, ones) + F.binary_cross_entropy_with_logits(d_fake, zeros)
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # generator: make D output 1 on fakes (non-saturating loss)
        g_loss = F.binary_cross_entropy_with_logits(D(fake), ones)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        hist["d_loss"].append(d_loss.item())
        hist["g_loss"].append(g_loss.item())
        hist["d_real"].append(torch.sigmoid(d_real).mean().item())
        hist["d_fake"].append(torch.sigmoid(d_fake).mean().item())

    last = slice(-iters_per_epoch, None)
    print(f"ep {epoch:3d} | D loss {np.mean(hist['d_loss'][last]):.3f} | G loss {np.mean(hist['g_loss'][last]):.3f} | "
          f"D(real) {np.mean(hist['d_real'][last]):.2f} | D(fake) {np.mean(hist['d_fake'][last]):.2f} | "
          f"{time.time() - t0:.0f}s", flush=True)

    if epoch in (1, 10, 25, 50) or epoch == EPOCHS:
        G.eval()
        with torch.no_grad():
            save_image(G(fixed_z) * 0.5 + 0.5, f"{OUT_DIR}/samples_epoch{epoch:03d}.png", nrow=8, padding=2, pad_value=1.0)
        G.train()

torch.save({"G": G.state_dict(), "D": D.state_dict(), "z_dim": Z_DIM}, f"{OUT_DIR}/generator.pt")
json.dump(hist, open(f"{OUT_DIR}/history.json", "w"))
print(f"done in {(time.time() - t0) / 60:.1f} min")

# loss curves: light = per iteration, dark = average per epoch
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
it = np.arange(1, len(hist["d_loss"]) + 1) / iters_per_epoch
ep = np.arange(1, EPOCHS + 1)
for ax, keys, title in [(axes[0], ["d_loss", "g_loss"], "Losses"), (axes[1], ["d_real", "d_fake"], "Discriminator outputs")]:
    for k, c, name in zip(keys, ["C0", "C1"], ["D" if "loss" in keys[0] else "D(real)", "G" if "loss" in keys[0] else "D(fake)"]):
        v = np.array(hist[k])
        ax.plot(it, v, color=c, alpha=0.2, lw=0.5)
        ax.plot(ep, v[:EPOCHS * iters_per_epoch].reshape(EPOCHS, -1).mean(1), color=c, lw=1.8, label=name)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/loss_curves.png", dpi=200)
