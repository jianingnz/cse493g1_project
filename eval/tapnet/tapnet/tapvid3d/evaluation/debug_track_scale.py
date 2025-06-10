import os
import numpy as np
import sys


def load_npz(file_path):
    with open(file_path, 'rb') as f:
        data = np.load(f, allow_pickle=True)
        return data['tracks_XYZ'], data['visibility']

def compute_depth_stats(tracks_xyz, visibility):
    visible_mask = visibility.astype(bool)
    depths = tracks_xyz[..., 2]
    return {
        'mean_z': np.mean(depths[visible_mask]),
        'median_z': np.median(depths[visible_mask]),
        'num_visible': np.sum(visible_mask)
    }

def compare_scale(gt_xyz, pred_xyz, visibility):
    mask = visibility.astype(bool)
    gt_z = gt_xyz[..., 2][mask]
    pred_z = pred_xyz[..., 2][mask]

    if len(gt_z) == 0 or len(pred_z) == 0:
        return None

    scale_ratio = np.median(gt_z) / np.median(pred_z)
    pred_xyz_scaled = pred_xyz * scale_ratio
    error = np.linalg.norm(gt_xyz[mask] - pred_xyz_scaled[mask], axis=-1)

    return {
        'scale_ratio': scale_ratio,
        'mean_error_scaled': np.mean(error),
        'median_error_scaled': np.median(error),
        'num_points': len(error)
    }

def analyze_prediction_vs_gt(gt_dir, pred_dir):
    results = []
    all_gt_data = {}

    for filename in sorted(os.listdir(gt_dir)):
        if not filename.endswith('.npz'):
            continue

        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)
        if not os.path.exists(pred_path):
            print(f"[Missing prediction] {filename}")
            continue

        gt_xyz, gt_vis = load_npz(gt_path)
        pred_xyz, pred_vis = load_npz(pred_path)

        if gt_xyz.shape[0] < gt_xyz.shape[1]:  # (N, T, 3)
            gt_xyz = gt_xyz.transpose(1, 0, 2)
            gt_vis = gt_vis.transpose(1, 0)
        if pred_xyz.shape[0] < pred_xyz.shape[1]:
            pred_xyz = pred_xyz.transpose(1, 0, 2)
            pred_vis = pred_vis.transpose(1, 0)

        joint_vis = np.logical_and(gt_vis, pred_vis)
        stats = compare_scale(gt_xyz, pred_xyz, joint_vis)
        if stats is not None:
            stats['video'] = filename
            results.append(stats)

        all_gt_data[filename] = (gt_xyz, pred_xyz, gt_vis, pred_vis)

    return results, all_gt_data

def print_summary(results):
    scale_ratios = [r['scale_ratio'] for r in results]
    mean_errors = [r['mean_error_scaled'] for r in results]
    median_errors = [r['median_error_scaled'] for r in results]

    print("\n=== Summary ===")
    print(f"Avg. scale ratio: {np.mean(scale_ratios):.3f}")
    print(f"Avg. mean error (scaled): {np.mean(mean_errors):.3f}")
    print(f"Avg. median error (scaled): {np.mean(median_errors):.3f}")
    print(f"Videos evaluated: {len(results)}")

    print("\nSample Results (first 5):")
    for r in results[:5]:
        print(f"{r['video']}: scale_ratio={r['scale_ratio']:.3f}, mean_err={r['mean_error_scaled']:.3f}")

def print_detailed_pointwise_diff(gt_xyz, pred_xyz, gt_vis, pred_vis, filename):
    print(f"\n--- Detailed Point Comparison: {filename} ---")
    if gt_xyz.shape[0] < gt_xyz.shape[1]:
        gt_xyz = gt_xyz.transpose(1, 0, 2)
        gt_vis = gt_vis.transpose(1, 0)
    if pred_xyz.shape[0] < pred_xyz.shape[1]:
        pred_xyz = pred_xyz.transpose(1, 0, 2)
        pred_vis = pred_vis.transpose(1, 0)

    joint_vis = np.logical_and(gt_vis, pred_vis)
    T, N = joint_vis.shape
    count = 0

    for t in range(T):
        for n in range(N):
            gt_point = gt_xyz[t, n]
            pred_point = pred_xyz[t, n]
            vis = joint_vis[t, n]

            if vis:
                dist = np.linalg.norm(gt_point - pred_point)
                print(f"Frame {t:03d}, Track {n:03d} | GT: {gt_point.round(3)} | Pred: {pred_point.round(3)} | Dist: {dist:.4f} | Visible")
                count += 1
            else:
                print(f"Frame {t:03d}, Track {n:03d} | GT: {gt_point.round(3)} | Pred: {pred_point.round(3)} | Not Visible")

    print(f"\nTotal visible comparisons: {count}/{T * N}")

# === Run ===
if __name__ == "__main__":
    # Set your paths here
    gt_folder = "/home/jianing/research/cse493g1/eval/evaluation/ground_truth/drivetrack"
    pred_folder = "/home/jianing/research/cse493g1/eval/evaluation/predictions/drivetrack"

    results, all_gt_data = analyze_prediction_vs_gt(gt_folder, pred_folder)
    print_summary(results)

    # Inspect a specific example in detail
    sample_filename = results[0]['video'] if results else None
    if sample_filename:
        gt_xyz, pred_xyz, gt_vis, pred_vis = all_gt_data[sample_filename]
        print_detailed_pointwise_diff(gt_xyz, pred_xyz, gt_vis, pred_vis, sample_filename)
