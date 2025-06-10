import numpy as np
from PIL import Image
import io
import os

data = np.load("tapvid3d_9142545919543484617_86_000_106_000_2_5AKc-TYQochsSWXpv376cA.npz")

frames_bytes = data["images_jpeg_bytes"]

output_dir = "images"
os.makedirs(output_dir, exist_ok=True)

# Decode and save frames
for idx, jpeg_bytes in enumerate(frames_bytes):
  img = Image.open(io.BytesIO(jpeg_bytes))
  filename = os.path.join(output_dir, f"{idx:05d}.jpeg")
  img.save(filename, "JPEG")

queries = data["queries_xyt"]  # shape (N, 3)
np.save(os.path.join(output_dir, "queries_xyt.npy"), queries)

print(f"Saved queries to '{output_dir}/queries_xyt.npy'")


