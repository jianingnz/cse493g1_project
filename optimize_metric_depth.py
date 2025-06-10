import numpy as np
import torch
import os
import argparse

# ==== Args ====
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--depth_mode', choices=['video', 'image'], required=True)
args = parser.parse_args()

# ==== Device ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==== Load Data ====
data1 = np.load(f"./preprop/tapnet/{args.dataset}/tapir_all_query_tracks_fast.npz")
tracks = data1['tracks']       # [N, F, 2]
visibles = data1['visibles']   # [N, F]

if args.depth_mode == 'video':
    data2 = np.load(f"./preprop/DepthCrafter/demo_output/{args.dataset}/images.npz")
    depths = data2['depth']    # [F, H, W]
elif args.depth_mode == 'image':
    depths = np.load(f"./preprop/Distill-Any-Depth/output/{args.dataset}/depth_maps.npy")
else:
    raise TypeError("Depth map mode not available.")

data3 = np.load(f"./preprop/mega-sam/outputs/{args.dataset}_droid.npz")
intrinsic = torch.tensor(data3['intrinsic'], dtype=torch.float32, device=device)  # [3, 3]
cam_c2w = torch.tensor(data3['cam_c2w'], dtype=torch.float32, device=device)      # [F, 4, 4]

# ==== Setup ====
depths = torch.tensor(depths, dtype=torch.float32, requires_grad=True, device=device)  # [F, H, W]
tracks = torch.tensor(tracks, dtype=torch.float32, device=device)                      # [N, F, 2]
visibles = torch.tensor(visibles, dtype=torch.float32, device=device)                  # [N, F]

optimizer = torch.optim.AdamW([depths], lr=1e-2)
epochs = 10

H, W = depths.shape[1:]

# ==== Loss Function ====
def depth_consistency_loss(depths, tracks, visibles, intrinsic, cam_c2w):
    K_inv = torch.inverse(intrinsic)
    F = depths.shape[0]
    N = tracks.shape[0]

    loss_total = 0.0
    count = 0

    for i in range(N):
        for j in range(F):
            for k in range(j+1, F):
                if visibles[i, j] > 0.5 and visibles[i, k] > 0.5:
                    pj = tracks[i, j]
                    pk = tracks[i, k]

                    yj, xj = int(pj[1]), int(pj[0])
                    yk, xk = int(pk[1]), int(pk[0])
                    if not (0 <= yj < H and 0 <= xj < W and 0 <= yk < H and 0 <= xk < W):
                        continue

                    dj = depths[j, yj, xj]
                    dk = depths[k, yk, xk]

                    pj_h = pj.new_tensor([pj[0], pj[1], 1.0])
                    pk_h = pk.new_tensor([pk[0], pk[1], 1.0])

                    Ej_inv = torch.inverse(cam_c2w[j])
                    Ek_inv = torch.inverse(cam_c2w[k])

                    Pj = Ej_inv[:3, :3] @ (K_inv @ pj_h * dj) + Ej_inv[:3, 3]
                    Pk = Ek_inv[:3, :3] @ (K_inv @ pk_h * dk) + Ek_inv[:3, 3]

                    loss_total += torch.sum((Pj - Pk) ** 2)
                    count += 1

    return loss_total / max(count, 1)

# ==== Optimization Loop ====
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = depth_consistency_loss(depths, tracks, visibles, intrinsic, cam_c2w)
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# ==== Save Result ====
save_path = f"./preprop/Distill-Any-Depth/output/{args.dataset}/optimzed_depth_maps.npy"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.save(save_path, depths.detach().cpu().numpy())
print(f"Optimization complete. Saved to {save_path}")
