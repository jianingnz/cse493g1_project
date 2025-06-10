import numpy as np
import torch
import os

# ==== Configuration ====
PRED_FILE = "./outputs/drivetrack/image/tapvid3d_9142545919543484617_86_000_106_000_2_5AKc-TYQochsSWXpv376cA.npz"
SAVE_FILE = "./outputs/drivetrack/image/optimized_tracks.npz"
LR = 1e-2
EPOCHS = 500

# ==== Load Data ====
data = np.load(PRED_FILE, allow_pickle=True)
tracks_xyz = data['tracks_XYZ']         # shape: (T, N, 3)
visibility_np = data['visibility']      # shape: (T, N)

# Convert to torch tensors
tracks = torch.tensor(tracks_xyz, dtype=torch.float32, requires_grad=True)
visibility = torch.tensor(visibility_np, dtype=torch.float32)  # keep float for masking

# ==== Optimizer ====
optimizer = torch.optim.AdamW([tracks], lr=LR)

# ==== Loss Function: Acceleration ====
def acceleration_loss(tracks, visibility):
    acc = tracks[2:] - 2 * tracks[1:-1] + tracks[:-2]  # shape: (T-2, N, 3)
    visible_mask = visibility[2:] * visibility[1:-1] * visibility[:-2]  # shape: (T-2, N)
    acc_loss = (acc ** 2).sum(dim=2)  # shape: (T-2, N)
    weighted_loss = acc_loss * visible_mask  # apply mask
    return weighted_loss.mean()

# ==== Optimization Loop ====
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    loss = acceleration_loss(tracks, visibility)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# ==== Save Optimized Results ====
optimized_tracks = tracks.detach().numpy()
# Convert visibility to boolean for indexing compatibility
visibility_bool = (visibility_np > 0.5)

os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
np.savez(SAVE_FILE, tracks_XYZ=optimized_tracks, visibility=visibility_bool)
print(f"Optimization complete. Saved to {SAVE_FILE}")
