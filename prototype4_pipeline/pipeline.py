from pathlib import Path

from prototype4_pipeline.artifacts import write_json
from prototype4_pipeline.stages.active_filtering import run_active_player_filtering
from prototype4_pipeline.stages.alignment import run_coordinate_alignment
from prototype4_pipeline.stages.crops import run_player_crop_extraction
from prototype4_pipeline.stages.frame_sampling import run_frame_sampling
from prototype4_pipeline.stages.ingestion import run_video_ingestion
from prototype4_pipeline.stages.meshes import run_player_mesh_reconstruction
from prototype4_pipeline.stages.segmentation import run_player_segmentation_tracking
from prototype4_pipeline.stages.visualization import run_visualization_export
from prototype4_pipeline.stages.world import run_world_reconstruction


OUTPUT_DIRS = {
    "world": "world_reconstruction",
    "masks": "player_masks",
    "tracks": "player_tracks",
    "crops": "player_crops",
    "meshes": "player_meshes",
    "visualizations": "visualizations",
    "metadata": "metadata",
    "frames": "sampled_frames",
}


class PipelineContext:
    def __init__(self, config, video_path, run_id, run_root, dirs):
        self.config = config
        self.video_path = video_path
        self.run_id = run_id
        self.run_root = run_root
        self.dirs = dirs

    @classmethod
    def from_config(cls, config, video_path, run_id):
        output_root = Path(config.get("paths", {}).get("output_root", "outputs")).expanduser()
        run_root = output_root / run_id
        dirs = {key: run_root / value for key, value in OUTPUT_DIRS.items()}
        return cls(config, video_path, run_id, run_root, dirs)


def run_pipeline(context):
    for directory in context.dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    output_dirs = {key: str(value) for key, value in context.dirs.items()}
    write_json(
        context.dirs["metadata"] / "run_config.json",
        {
            "status": "started",
            "stage": "run_configuration",
            "run_id": context.run_id,
            "video_path": str(context.video_path),
            "config": context.config,
            "output_dirs": output_dirs,
        },
    )

    ingestion = run_video_ingestion(context)
    frames = run_frame_sampling(context, ingestion)
    world = run_world_reconstruction(context, frames)
    tracks = run_player_segmentation_tracking(context, frames)
    active_tracks = run_active_player_filtering(context, tracks, world)
    crops = run_player_crop_extraction(context, frames, active_tracks)
    meshes = run_player_mesh_reconstruction(context, crops, active_tracks)
    alignment = run_coordinate_alignment(context, world, active_tracks, meshes)
    run_visualization_export(context, world, active_tracks, meshes, alignment)

    write_json(
        context.dirs["metadata"] / "run_summary.json",
        {
            "status": "complete",
            "stage": "run_summary",
            "run_id": context.run_id,
            "video_path": str(context.video_path),
            "output_root": str(context.run_root),
            "output_dirs": output_dirs,
            "counts": {
                "sampled_frames": len(frames),
                "candidate_tracks": len(tracks.get("tracks", [])),
                "active_tracks": len(active_tracks.get("tracks", [])),
                "rejected_tracks": len(active_tracks.get("rejected_tracks", [])),
                "player_crops": len(crops.get("crops", [])),
                "mesh_sequences": len(meshes.get("mesh_sequences", {})),
                "alignment_records": len(alignment.get("aligned_tracks", [])),
            },
            "stage_manifests": {
                "video_ingestion": str(context.dirs["metadata"] / "video_ingestion.json"),
                "frame_sampling": str(context.dirs["metadata"] / "sampled_frames.json"),
                "world_reconstruction": str(context.dirs["world"] / "world_reconstruction_manifest.json"),
                "segmentation_tracking": str(context.dirs["tracks"] / "tracks_manifest.json"),
                "active_player_filtering": str(context.dirs["tracks"] / "active_player_tracks.json"),
                "crop_extraction": str(context.dirs["crops"] / "crops_manifest.json"),
                "mesh_reconstruction": str(context.dirs["meshes"] / "meshes_manifest.json"),
                "coordinate_alignment": str(context.dirs["metadata"] / "coordinate_alignment.json"),
                "visualization_export": str(context.dirs["visualizations"] / "visualization_manifest.json"),
            },
        },
    )
