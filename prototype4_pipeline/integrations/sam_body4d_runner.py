from pathlib import Path


class SAMBody4DRunner:
    def __init__(self, context):
        self.context = context
        self.config = context.config.get("models", {}).get("sam_body4d", {})
        self.repo = Path(context.config.get("paths", {}).get("sam_body4d_repo", "../sam-body4d"))

    def run(self, crops, active_tracks):
        mesh_sequences = {}
        for track in active_tracks.get("tracks", []):
            track_id = track["track_id"]
            mesh_sequences[track_id] = {
                "status": "placeholder",
                "crop_inputs": [
                    crop["path"] for crop in crops.get("crops", []) if crop["track_id"] == track_id
                ],
                "output_dir": str(self.context.dirs["meshes"] / track_id),
                "mesh_format": "OBJ/PLY/GLB sequence plus temporal metadata",
            }

        return {
            "status": "placeholder" if not self.config.get("enabled") else "not_executed_by_prototype",
            "stage": "sam_body4d_mesh_reconstruction",
            "dry_run": not self.config.get("enabled"),
            "repo": str(self.repo),
            "repo_present": self.repo.exists(),
            "clone_command_if_missing": "git clone https://github.com/gaomingqi/sam-body4d.git ../sam-body4d",
            "entrypoint": str(self.repo / "scripts" / "offline_app.py"),
            "input_summary": {
                "active_track_count": len(active_tracks.get("tracks", [])),
                "crop_count": len(crops.get("crops", [])),
                "crop_manifest": str(self.context.dirs["crops"] / "crops_manifest.json"),
            },
            "output_dir": str(self.context.dirs["meshes"]),
            "would_run": [
                "python3",
                str(self.repo / "scripts" / "offline_app.py"),
                "--input_video",
                "<per-track-player-crop-video-or-frame-directory>",
            ],
            "counts": {
                "mesh_sequences": len(mesh_sequences),
            },
            "mesh_sequences": mesh_sequences,
        }
