# Saves the first 15,000 WikiArt paintings
import os
from datasets import load_dataset

os.makedirs("data/wikiart/subset", exist_ok=True)
ds = load_dataset("huggan/wikiart", split="train", streaming=True)
for i, ex in enumerate(ds):
    if i == 15000:
        break
    im = ex["image"].convert("RGB")
    im.thumbnail((256, 256))
    im.save(f"data/wikiart/subset/{i:05d}.jpg", quality=95)
    if i % 1000 == 0:
        print(i, flush=True)
