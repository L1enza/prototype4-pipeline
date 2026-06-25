from prototype4_pipeline.artifacts import write_json
from prototype4_pipeline.integrations.sam3_runner import SAM3Runner


def run_player_segmentation_tracking(context, frames):
    result = SAM3Runner(context).run(frames)
    write_json(context.dirs["tracks"] / "tracks_manifest.json", result)
    write_json(
        context.dirs["masks"] / "masks_manifest.json",
        {
            "status": result.get("status"),
            "stage": "sam3_mask_generation",
            "output_dir": str(context.dirs["masks"]),
            "input_video": str(context.video_path),
            "frame_count": len(frames),
            "candidate_track_count": len(result.get("tracks", [])),
            "mask_count": result.get("masks", {}).get("mask_count", 0),
            "masks": result.get("masks", {}),
        },
    )
    return result
