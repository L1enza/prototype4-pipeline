#!/usr/bin/env python3
"""Guarded tiny VGGT world-reconstruction artifact exporter.

This script is intentionally separate from the normal Prototype 4 dry-run. It may
instantiate VGGT and may download model weights only when --allow-download-weights
is explicitly provided.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Run a tiny guarded VGGT inference artifact export.")
    parser.add_argument("--run-id", default="nll-test1", help="Existing Prototype 4 dry-run id.")
    parser.add_argument("--repo", default="../vggt", help="Path to local VGGT repository.")
    parser.add_argument("--frame-count", type=int, default=4, help="Number of sampled frames to use, from 4 to 8.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Inference device. Default requires CUDA.")
    parser.add_argument("--checkpoint", default="facebook/VGGT-1B", help="VGGT checkpoint id for from_pretrained.")
    parser.add_argument("--confidence-threshold", type=float, default=0.0, help="Confidence threshold for point-cloud summary counts.")
    parser.add_argument("--max-preview-points", type=int, default=50000, help="Maximum points saved in downsampled preview NPZ.")
    parser.add_argument(
        "--allow-download-weights",
        action="store_true",
        help="Required. Acknowledges that VGGT.from_pretrained may download model weights.",
    )
    return parser.parse_args()


def tensor_summary(value):
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, dict):
        return {key: tensor_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_summary(item) for item in value]
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def tensor_to_numpy(tensor):
    return tensor.detach().to("cpu", dtype=None).float().numpy()


def save_npy(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)
    return str(path)


def write_failure_outputs(output_dir, args, repo, frames, device, dtype, error):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "stage": "vggt_inference_smoke",
        "checkpoint": args.checkpoint,
        "repo": str(repo),
        "device": str(device),
        "dtype": str(dtype),
        "frame_count": len(frames),
        "frames": [str(path) for path in frames],
        "output_dir": str(output_dir),
        "error": {
            "type": error.__class__.__name__,
            "message": str(error),
        },
        "artifacts": {},
        "did_download_weights_flag_acknowledged": bool(args.allow_download_weights),
        "full_video_inference": False,
    }
    metadata_path = output_dir / "metadata.json"
    manifest_path = output_dir / "world_reconstruction_manifest.json"
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "status": "failed",
        "stage": "vggt_inference_smoke_artifact_export",
        "run_id": args.run_id,
        "checkpoint": args.checkpoint,
        "frame_count": len(frames),
        "device": str(device),
        "output_dir": str(output_dir),
        "metadata": str(metadata_path),
        "artifacts": {},
        "error": payload["error"],
        "safety": {
            "allow_download_weights_required": True,
            "allow_download_weights_supplied": bool(args.allow_download_weights),
            "full_video_inference": False,
            "normal_dry_run_instantiates_vggt": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Wrote failure metadata {}".format(metadata_path))
    print("Wrote failure manifest {}".format(manifest_path))


def finite_range(points):
    finite = np.isfinite(points).all(axis=-1)
    if not finite.any():
        return {"min_xyz": None, "max_xyz": None}
    valid = points[finite]
    return {
        "min_xyz": valid.min(axis=0).astype(float).tolist(),
        "max_xyz": valid.max(axis=0).astype(float).tolist(),
    }


def build_point_cloud_preview(world_points, world_points_conf, threshold, max_preview_points, output_dir):
    # Inputs are expected as [1, S, H, W, 3] and [1, S, H, W].
    points = world_points[0]
    conf = world_points_conf[0]
    frame_count = points.shape[0]

    valid_mask = np.isfinite(points).all(axis=-1) & np.isfinite(conf) & (conf > threshold)
    per_frame_valid_counts = valid_mask.reshape(frame_count, -1).sum(axis=1).astype(int).tolist()
    total_valid = int(valid_mask.sum())

    ranges = finite_range(points[valid_mask]) if total_valid else {"min_xyz": None, "max_xyz": None}

    flat_points = points.reshape(-1, 3)
    flat_conf = conf.reshape(-1)
    flat_valid = valid_mask.reshape(-1)
    flat_frame_index = np.repeat(np.arange(frame_count), points.shape[1] * points.shape[2])
    valid_indices = np.flatnonzero(flat_valid)

    if len(valid_indices) > max_preview_points:
        selected = np.linspace(0, len(valid_indices) - 1, max_preview_points).round().astype(np.int64)
        valid_indices = valid_indices[selected]

    preview_path = output_dir / "point_cloud_preview.npz"
    np.savez_compressed(
        preview_path,
        points=flat_points[valid_indices].astype(np.float32),
        confidence=flat_conf[valid_indices].astype(np.float32),
        frame_index=flat_frame_index[valid_indices].astype(np.int16),
    )

    return {
        "confidence_threshold": threshold,
        "max_preview_points": max_preview_points,
        "total_valid_points": total_valid,
        "saved_preview_points": int(len(valid_indices)),
        "per_frame_valid_point_counts": per_frame_valid_counts,
        "coordinate_ranges": ranges,
        "preview_npz": str(preview_path),
    }


def main():
    args = parse_args()
    if not args.allow_download_weights:
        raise SystemExit(
            "Refusing to run VGGT inference smoke test without --allow-download-weights. "
            "VGGT.from_pretrained may download checkpoint weights."
        )
    if args.frame_count < 4 or args.frame_count > 8:
        raise SystemExit("--frame-count must be between 4 and 8 for this tiny smoke test.")

    repo = Path(args.repo).resolve()
    if not repo.exists():
        raise SystemExit("VGGT repo not found: {}".format(repo))
    sys.path.insert(0, str(repo))

    try:
        import torch
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except Exception as exc:
        raise SystemExit("VGGT imports failed before inference: {}: {}".format(exc.__class__.__name__, exc))

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Re-run with --device cpu only if you intentionally want a CPU smoke test.")
    device = torch.device(args.device)

    frame_dir = PROJECT_ROOT / "outputs" / args.run_id / "sampled_frames"
    frames = sorted(frame_dir.glob("frame_*.jpg"))[: args.frame_count]
    if len(frames) < args.frame_count:
        raise SystemExit("Not enough sampled frames in {}. Run dry-run first.".format(frame_dir))

    output_dir = PROJECT_ROOT / "outputs" / args.run_id / "world_reconstruction" / "vggt_inference_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float32
    if device.type == "cuda":
        major, _minor = torch.cuda.get_device_capability(device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16

    print("WARNING: this may download VGGT weights for checkpoint {}.".format(args.checkpoint))
    print("Using {} frames on device {}.".format(len(frames), device))

    try:
        model = VGGT.from_pretrained(args.checkpoint).to(device)
        model.eval()
        images = load_and_preprocess_images([str(path) for path in frames]).to(device)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = model(images)
            else:
                predictions = model(images)
    except Exception as exc:
        write_failure_outputs(output_dir, args, repo, frames, device, dtype, exc)
        raise SystemExit("VGGT inference smoke export failed: {}: {}".format(exc.__class__.__name__, exc))

    artifact_paths = {}
    depth = tensor_to_numpy(predictions["depth"])
    depth_conf = tensor_to_numpy(predictions["depth_conf"])
    world_points = tensor_to_numpy(predictions["world_points"])
    world_points_conf = tensor_to_numpy(predictions["world_points_conf"])
    pose_enc = tensor_to_numpy(predictions["pose_enc"])

    artifact_paths["depth"] = save_npy(output_dir / "depth.npy", depth)
    artifact_paths["depth_conf"] = save_npy(output_dir / "depth_conf.npy", depth_conf)
    artifact_paths["world_points"] = save_npy(output_dir / "world_points.npy", world_points)
    artifact_paths["world_points_conf"] = save_npy(output_dir / "world_points_conf.npy", world_points_conf)
    artifact_paths["pose_enc"] = save_npy(output_dir / "pose_enc.npy", pose_enc)

    image_hw = tuple(int(value) for value in predictions["depth"].shape[2:4])
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], image_hw)
    extrinsic_np = tensor_to_numpy(extrinsic)
    intrinsic_np = tensor_to_numpy(intrinsic)
    artifact_paths["camera_extrinsics"] = save_npy(output_dir / "camera_extrinsics.npy", extrinsic_np)
    artifact_paths["camera_intrinsics"] = save_npy(output_dir / "camera_intrinsics.npy", intrinsic_np)

    camera_metadata = {
        "status": "complete",
        "pose_encoding_type": "absT_quaR_FoV",
        "image_size_hw": list(image_hw),
        "pose_enc_shape": list(pose_enc.shape),
        "extrinsics_shape": list(extrinsic_np.shape),
        "intrinsics_shape": list(intrinsic_np.shape),
        "extrinsics_opencv_camera_from_world": extrinsic_np[0].astype(float).tolist(),
        "intrinsics_pixels": intrinsic_np[0].astype(float).tolist(),
        "artifacts": {
            "pose_enc": artifact_paths["pose_enc"],
            "camera_extrinsics": artifact_paths["camera_extrinsics"],
            "camera_intrinsics": artifact_paths["camera_intrinsics"],
        },
    }
    camera_metadata_path = output_dir / "camera_metadata.json"
    camera_metadata_path.write_text(json.dumps(camera_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_paths["camera_metadata"] = str(camera_metadata_path)

    point_cloud_preview = build_point_cloud_preview(
        world_points,
        world_points_conf,
        args.confidence_threshold,
        args.max_preview_points,
        output_dir,
    )
    artifact_paths["point_cloud_preview"] = point_cloud_preview["preview_npz"]

    metadata = {
        "status": "complete",
        "stage": "vggt_inference_smoke",
        "warning": "This script may download weights only when --allow-download-weights is supplied.",
        "checkpoint": args.checkpoint,
        "repo": str(repo),
        "device": str(device),
        "dtype": str(dtype),
        "frame_count": len(frames),
        "frames": [str(path) for path in frames],
        "output_dir": str(output_dir),
        "prediction_keys": sorted(predictions.keys()),
        "prediction_summaries": {key: tensor_summary(value) for key, value in predictions.items()},
        "artifacts": artifact_paths,
        "camera_metadata": camera_metadata,
        "point_cloud_preview": point_cloud_preview,
        "did_download_weights_flag_acknowledged": bool(args.allow_download_weights),
        "full_video_inference": False,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "status": "complete",
        "stage": "vggt_inference_smoke_artifact_export",
        "run_id": args.run_id,
        "checkpoint": args.checkpoint,
        "frame_count": len(frames),
        "device": str(device),
        "output_dir": str(output_dir),
        "metadata": str(metadata_path),
        "artifacts": artifact_paths,
        "prediction_keys": sorted(predictions.keys()),
        "tensor_shapes": {
            key: value["shape"]
            for key, value in metadata["prediction_summaries"].items()
            if isinstance(value, dict) and "shape" in value
        },
        "point_cloud_preview": point_cloud_preview,
        "camera_metadata": str(camera_metadata_path),
        "safety": {
            "allow_download_weights_required": True,
            "allow_download_weights_supplied": bool(args.allow_download_weights),
            "full_video_inference": False,
            "normal_dry_run_instantiates_vggt": False,
        },
    }
    manifest_path = output_dir / "world_reconstruction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Wrote {}".format(metadata_path))
    print("Wrote {}".format(manifest_path))


if __name__ == "__main__":
    main()
