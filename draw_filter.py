import numpy as np
import mediapy as media
import cv2

# Load original video
video = media.read_video("./preprop/DepthCrafter/demo_output/bear/_input.mp4")

# Load filtered tracking data
masked_data = np.load("tapir_masked_tracks_data.npz")
tracks = masked_data["tracks"]       # shape (N, 82, 2)
visibles = masked_data["visibles"]   # shape (N, 82)

# Visualization function (same as in your main script)
def draw_tracks_on_frames(frames, tracks, visibles, radius=2, color=(255, 0, 0)):
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

# Draw and save video
video_viz = draw_tracks_on_frames(video, tracks, visibles, radius=2)
media.write_video("tapir_masked_tracks.mp4", video_viz, fps=10)
print("Saved masked visualization to tapir_masked_tracks.mp4")
