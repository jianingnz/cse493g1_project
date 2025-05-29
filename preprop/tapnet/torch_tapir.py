import torch
import torch.nn.functional as F
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
import cv2

from tapnet.torch import tapir_model
from tapnet.utils import transforms


# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load video
video = media.read_video("../DepthCrafter/demo_output/bear_input.mp4")

# Resize frames
resize_height, resize_width, stride = 256, 256, 8
frames = media.resize_video(video, (resize_height, resize_width))
height, width = frames.shape[1:3]

# Preprocessing
def preprocess_frames(frames):
    frames = frames.float()
    return frames / 255 * 2 - 1

def draw_tracks_on_frames(frames, tracks, visibles, radius=2, color=(0, 255, 0)):
    """Draw visible tracked points as anti-aliased round dots."""
    output = []
    for i, frame in enumerate(frames):
        img = frame.copy()
        for j, (pt, visible) in enumerate(zip(tracks[:, i], visibles[:, i])):
            if visible:
                x, y = int(round(pt[0])), int(round(pt[1]))
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    cv2.circle(img, (x, y), radius, color, thickness=-1, lineType=cv2.LINE_AA)
        output.append(img)
    return np.stack(output)

# Grid sampling
def sample_grid_points(frame_idx, height, width, stride=8):
    yx = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
    yx = yx.transpose(1, 2, 0).reshape(-1, 2)
    t = np.full((yx.shape[0], 1), frame_idx, dtype=np.int32)
    return np.concatenate([t, yx], axis=-1)

# Load model
model = tapir_model.TAPIR(pyramid_level=1, use_casual_conv=True)
model.load_state_dict(torch.load("tapnet/checkpoints/causal_bootstapir_checkpoint.pt"))
model = model.to(device).eval()
torch.set_grad_enabled(False)

# Inference functions
def postprocess_occlusions(occlusions, expected_dist):
    return (1 - torch.sigmoid(occlusions)) * (1 - torch.sigmoid(expected_dist)) > 0.5

def online_model_init(frames, query_points):
    frames = preprocess_frames(frames)
    feature_grids = model.get_feature_grids(frames, is_training=False)
    return model.get_query_features(frames, False, query_points, feature_grids)

def online_model_predict(frames, query_features, causal_context):
    frames = preprocess_frames(frames)
    feature_grids = model.get_feature_grids(frames, is_training=False)
    trajectories = model.estimate_trajectories(
        frames.shape[-3:-1],
        False,
        feature_grids,
        query_features,
        None,
        query_chunk_size=64,
        causal_context=causal_context,
        get_causal_context=True
    )
    return (
        trajectories["tracks"][-1],
        postprocess_occlusions(trajectories["occlusion"][-1], trajectories["expected_dist"][-1]),
        trajectories["causal_context"]
    )

# Main demo
frames = torch.tensor(frames).to(device)
query_points = torch.tensor(sample_grid_points(0, resize_height, resize_width, stride)).to(device)

query_features = online_model_init(frames[None, 0:1], query_points[None])
causal_state = model.construct_initial_causal_state(query_points.shape[0], len(query_features.resolutions) - 1)
for i in range(len(causal_state)):
    for k in causal_state[i]:
        causal_state[i][k] = causal_state[i][k].to(device)

predictions = []
for i in range(frames.shape[0]):
    tracks, visibles, causal_state = online_model_predict(
        frames=frames[None, i : i + 1],
        query_features=query_features,
        causal_context=causal_state,
    )
    predictions.append({"tracks": tracks, "visibles": visibles})

tracks = torch.cat([x["tracks"][0] for x in predictions], dim=1).cpu().numpy()
visibles = torch.cat([x["visibles"][0] for x in predictions], dim=1).cpu().numpy()

# Convert coordinates to original video size
tracks = transforms.convert_grid_coordinates(tracks, (resize_width, resize_height), (video.shape[2], video.shape[1]))

# Visualize
# video_viz = viz_utils.paint_point_track(video, tracks, visibles)
video_viz = draw_tracks_on_frames(video, tracks, visibles, radius=1)
np.savez("tapir_grid_tracks_data.npz", tracks=tracks, visibles=visibles)
print("Saved tracks and visibles to tapir_grid_tracks_data.npz")
media.write_video("tapir_grid_tracks.mp4", video_viz, fps=10)
print("Saved output to tapir_grid_tracks.mp4")
