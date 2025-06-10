import numpy as np
import argparse
import os
from scipy.interpolate import interp1d


def backproject_tracks_to_world(tracks, visibles, depths, intrinsic, cam_c2w, mode, dataset):
  """
  returns:
    points_3d (np.ndarray): [N, F, 3] array of 3D points (NaN where not visible)
  """
  N, F, _ = tracks.shape
  if mode == "image":
    H, W = depths.shape[1:]
    H_d, W_d = H, W
  elif mode == "video":
    H_d, W_d = depths.shape[1:]
    if dataset == "pstudio":
      H, W = 360., 640.
    elif dataset == "drivetrack":
      H, W = 1280., 1920.
    elif dataset == "bear":
      H, W = 854., 480.
    else:
      raise ValueError("Unsupported dataset selection")
  else:
    raise ValueError("Unsupported depth map mode")
  fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

  # Prepare output
  points_3d = np.full((N, F, 3), np.nan, dtype=np.float32)

  for f in range(F):
    # Get depth and cam2world
    depth = depths[f]
    pose = cam_c2w[f]

    for i in range(N):
      if not visibles[i, f]:
        continue

      x, y = tracks[i, f]

      if mode == "video":
        x_d = x * (W_d / W)
        y_d = y * (H_d / H)
      else:
        x_d, y_d = x, y
  
      # Round and clamp to valid range
      x_pix = int(np.clip(round(x_d), 0, W_d - 1))
      y_pix = int(np.clip(round(y_d), 0, H_d - 1))

      # Retrieve depth value
      z = depth[y_pix, x_pix]

      # Pixel to camera coordinates
      X = (x - cx) * z / fx
      Y = (y - cy) * z / fy
      Z = z
      cam_point = np.array([X, Y, Z, 1.0])  # homogeneous

      # Transform to world coordinates
      world_point = pose @ cam_point  # [4]
      # print(world_point)
      points_3d[i, f] = world_point[:3]

  return points_3d

def interpolate_missing_points(points, visibility):
    """
    points: [F, N, 3]
    visibility: [F, N]
    Fills NaNs in time for each trajectory using linear interpolation.
    """
    F, N, _ = points.shape
    points_interp = np.copy(points)

    for i in range(N):  # for each point track
        visible_indices = np.where(visibility[:, i])[0]
        if len(visible_indices) < 2:
            continue  # cannot interpolate with fewer than 2 points

        for dim in range(3):
            values = points[:, i, dim]
            interp_func = interp1d(
                visible_indices,
                values[visible_indices],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )
            missing_indices = np.where(~visibility[:, i])[0]
            points_interp[missing_indices, i, dim] = interp_func(missing_indices)

    return points_interp

def parse_args():
  parser = argparse.ArgumentParser(description="Backproject 2D tracks to 3D points.")
  parser.add_argument('--depth_mode', type=str, required=True, help='Depth map usage mode')
  parser.add_argument('--dataset', type=str, required=True, help='dataset to generate')
  return parser.parse_args()

if __name__ == "__main__":
  args = parse_args()
  # Load your data
  # from ./tapir_masked_tracks_data.npz
  data1 = np.load(f"./preprop/tapnet/{args.dataset}/tapir_all_query_tracks_fast.npz")
  tracks = data1['tracks']       # shape [N, F, 2]
  visibles = data1['visibles']   # shape [N, F]
  
  if args.depth_mode == 'video':
    data2 = np.load(f"./preprop/DepthCrafter/demo_output/{args.dataset}/images.npz")
    depths = data2['depth']      # shape [F, H, W]
  elif args.depth_mode == 'image':
    depths = np.load(f"./preprop/Distill-Any-Depth/output/{args.dataset}/depth_maps.npy")       # shape [F, H, W]
  else:
    raise TypeError("Depth map mode not available.")
  
  # from ./preprop/mega-sam/outputs/bear_droid.npz
  data3 = np.load(f"./preprop/mega-sam/outputs/{args.dataset}_droid.npz")
  intrinsic = data3['intrinsic'] # shape [3, 3]
  cam_c2w = data3['cam_c2w']     # shape [F, 4, 4]

  points_3d = backproject_tracks_to_world(tracks, visibles, depths, intrinsic, cam_c2w, args.depth_mode, args.dataset)

  tracks_XYZ_transposed = points_3d.transpose(1, 0, 2)     # (F, N, 3)
  visibility_transposed = visibles.transpose(1, 0)         # (F, N)


  tracks_XYZ_interpolated = interpolate_missing_points(tracks_XYZ_transposed, visibility_transposed)

  # Create output directory
  output_dir = f"./outputs/{args.dataset}/{args.depth_mode}"
  os.makedirs(output_dir, exist_ok=True)

  # Save results
  np.savez_compressed(f"{output_dir}/backprojected_points.npz", tracks_XYZ=tracks_XYZ_interpolated, visibility=visibility_transposed)
  print(f"Saved 3D points and visibility mask to {output_dir}/backprojected_points.npz")
