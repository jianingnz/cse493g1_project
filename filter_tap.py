import numpy as np
import os

# Load the original tracking data
data = np.load("./preprop/tapnet/bear/tapir_grid_tracks_data.npz")
tracks = data["tracks"]   # shape (1024, 82, 2)
visibles = data["visibles"]  # shape (1024, 82)

# Path to the folder containing the masks
mask_folder = "./preprop/Track-Anything/result/mask/bear_input"  # Replace with your actual path

num_points, num_frames, _ = tracks.shape

# Apply the masks frame by frame
for frame_idx in range(num_frames):
    mask = np.load(os.path.join(mask_folder, f"{frame_idx:05d}.npy"))  # shape (256, 512)

    coords = tracks[:, frame_idx]  # shape (1024, 2)
    x = coords[:, 0].astype(np.int32)
    y = coords[:, 1].astype(np.int32)

    in_bounds = (x >= 0) & (x < mask.shape[1]) & (y >= 0) & (y < mask.shape[0])
    inside_mask = np.zeros(num_points, dtype=bool)
    inside_mask[in_bounds] = mask[y[in_bounds], x[in_bounds]] == 1

    # Update visibility
    visibles[:, frame_idx] &= inside_mask

# Optional: keep only points visible in at least one frame
valid_points = np.any(visibles, axis=1)
filtered_tracks = tracks[valid_points]
filtered_visibles = visibles[valid_points]

# Print shape of filtered tracks
print("Filtered tracks shape:", filtered_tracks.shape)  # (n, 82, 2)
print("Filtered visibles shape:", filtered_visibles.shape)  # (n, 82)


# Save to file
np.savez("tapir_masked_tracks_data.npz", tracks=filtered_tracks, visibles=filtered_visibles)
print("Saved filtered tracks and visibles to tapir_masked_tracks_data.npz")
