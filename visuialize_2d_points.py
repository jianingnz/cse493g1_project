import numpy as np
import cv2
import argparse
from matplotlib import cm
from matplotlib.colors import Normalize
import os

def load_data(npz_path):
    data = np.load(npz_path)
    return data['tracks'], data['visibles']

def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)

def save_video(frames, output_path, fps=15):
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()

def visualize_2d_tracks_on_video(video, tracks, visibles, output_path, leave_trace=16, show_occ=True):
    N, F, _ = tracks.shape
    H, W = video.shape[1:3]

    # Generate rainbow colors
    cmap = cm.get_cmap('hsv', N)
    colors = (cmap(np.arange(N))[:, :3] * 255).astype(np.uint8)

    frames_out = []
    for t in range(F):
        frame = video[t].copy()

        # Draw trajectory lines
        for i in range(N):
            color = tuple(map(int, colors[i]))
            for k in range(max(0, t - leave_trace), t):
                if visibles[i, k] and visibles[i, k + 1]:
                    pt1 = tuple(map(int, np.round(tracks[i, k])))
                    pt2 = tuple(map(int, np.round(tracks[i, k + 1])))
                    cv2.line(frame, pt1, pt2, color=color, thickness=1, lineType=cv2.LINE_AA)

            # Draw current point
            if visibles[i, t]:
                pt = tuple(map(int, np.round(tracks[i, t])))
                cv2.circle(frame, pt, radius=2, color=color, thickness=-1)
            elif show_occ:
                pt = tuple(map(int, np.round(tracks[i, t])))
                cv2.circle(frame, pt, radius=2, color=color, thickness=1)

        frames_out.append(frame)

    save_video(frames_out, output_path)
    print(f"Saved visualized video to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument('--npz', type=str, required=True, help="Path to .npz file (with 'points' and 'visibles')")
    parser.add_argument("--output", default="video_2d_tracks.mp4", help="Output video path")
    args = parser.parse_args()

    video = load_video(args.video)
    tracks, visibles = load_data(args.npz)

    if tracks.shape[1] > video.shape[0]:
        tracks = tracks[:, :video.shape[0]]
        visibles = visibles[:, :video.shape[0]]
    elif video.shape[0] > tracks.shape[1]:
        video = video[:tracks.shape[1]]

    visualize_2d_tracks_on_video(video, tracks, visibles, args.output)
