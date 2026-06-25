from pathlib import Path


class SAM3Runner:
    def __init__(self, context):
        self.context = context
        self.config = context.config.get("models", {}).get("sam3", {})
        self.repo = Path(context.config.get("paths", {}).get("sam3_repo", "../sam3"))

    def run(self, frames):
        prompts = self.context.config.get("segmentation", {}).get("text_prompts", {})
        tracks = self._placeholder_tracks(frames)
        observation_count = sum(len(track.get("observations", [])) for track in tracks)
        return {
            "status": "placeholder" if not self.config.get("enabled") else "not_executed_by_prototype",
            "stage": "sam3_player_segmentation_tracking",
            "dry_run": not self.config.get("enabled"),
            "repo": str(self.repo),
            "repo_present": self.repo.exists(),
            "clone_command_if_missing": "git clone https://github.com/facebookresearch/sam3.git ../sam3",
            "entrypoint": "sam3.model_builder.build_sam3_video_predictor",
            "input_summary": {
                "source_video": str(self.context.video_path),
                "sampled_frame_count": len(frames),
                "sampled_frame_paths": [frame["path"] for frame in frames],
            },
            "would_call": {
                "start_session": {"resource_path": str(self.context.video_path)},
                "add_prompt": {
                    "frame_index": 0,
                    "text": prompts.get("players", "active lacrosse player on the field"),
                },
            },
            "counts": {
                "candidate_tracks": len(tracks),
                "observations": observation_count,
                "masks": 0,
            },
            "tracks": tracks,
            "masks": {
                "directory": str(self.context.dirs["masks"]),
                "format": "per-track/per-frame binary mask PNG or RLE JSON",
                "placeholder": True,
                "mask_count": 0,
                "expected_future_layout": "outputs/<run_id>/player_masks/<track_id>/frame_<frame_index>.png",
            },
            "negative_prompts": {
                "referees": prompts.get("referees"),
                "bench": prompts.get("bench"),
                "crowd": prompts.get("crowd"),
            },
        }

    def _placeholder_tracks(self, frames):
        if not frames:
            return []

        observations = []
        for frame in frames[: min(6, len(frames))]:
            width = frame["width"]
            height = frame["height"]
            x1 = int(width * 0.42)
            y1 = int(height * 0.35)
            x2 = int(width * 0.52)
            y2 = int(height * 0.82)
            observations.append(
                {
                    "frame_index": frame["frame_index"],
                    "timestamp_sec": frame["timestamp_sec"],
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "mask_path": None,
                    "mask_area_px": (x2 - x1) * (y2 - y1),
                    "field_overlap": 1.0,
                    "classification": "candidate_player",
                    "field_coordinate_xy": None,
                }
            )

        return [
            {
                "track_id": "dryrun_player_0001",
                "source": "sam3_placeholder",
                "role_candidates": ["player"],
                "rejection_scores": {"referee": 0.0, "bench_or_crowd": 0.0},
                "observations": observations,
                "metadata_slots": {
                    "jersey_number": None,
                    "roster_id": None,
                    "player_name": None,
                    "team": None,
                    "future_heatmap_group": None,
                },
            }
        ]
