def filter_active_players(config, tracks, world):
    segmentation_cfg = config.get("segmentation", {})
    min_track_frames = int(segmentation_cfg.get("min_track_frames", 3))
    min_field_overlap = float(segmentation_cfg.get("min_field_overlap", 0.25))

    accepted = []
    rejected = []
    for track in tracks.get("tracks", []):
        observations = track.get("observations", [])
        overlaps = [float(obs.get("field_overlap") or 0.0) for obs in observations]
        mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        rejection_scores = track.get("rejection_scores", {})

        reasons = []
        if len(observations) < min_track_frames:
            reasons.append("track_too_short")
        if mean_overlap < min_field_overlap:
            reasons.append("insufficient_playing_surface_overlap")
        if rejection_scores.get("referee", 0.0) >= 0.5:
            reasons.append("referee_rejection")
        if rejection_scores.get("bench_or_crowd", 0.0) >= 0.5:
            reasons.append("bench_crowd_suppression")

        enriched = dict(track)
        enriched["active_player_filter"] = {
            "mean_field_overlap": mean_overlap,
            "min_track_frames": min_track_frames,
            "min_field_overlap": min_field_overlap,
            "world_reference_status": world.get("status"),
        }

        if reasons:
            enriched["rejection_reasons"] = reasons
            rejected.append(enriched)
        else:
            enriched["field_coordinates"] = [
                {
                    "frame_index": obs["frame_index"],
                    "timestamp_sec": obs.get("timestamp_sec"),
                    "xy": obs.get("field_coordinate_xy"),
                    "status": "pending_vggt_field_alignment",
                }
                for obs in observations
            ]
            accepted.append(enriched)

    return {
        "status": "complete",
        "stage": "active_player_filtering",
        "input_counts": {
            "candidate_tracks": len(tracks.get("tracks", [])),
        },
        "counts": {
            "active_tracks": len(accepted),
            "rejected_tracks": len(rejected),
        },
        "tracks": accepted,
        "rejected_tracks": rejected,
        "hooks": {
            "field_boundary_filter": config.get("filters", {}).get("enable_field_boundary_filter", True),
            "referee_rejection": config.get("filters", {}).get("enable_referee_rejection", True),
            "bench_crowd_suppression": config.get("filters", {}).get("enable_bench_crowd_suppression", True),
            "future_jersey_number_recognition": config.get("metadata_hooks", {}).get("jersey_number_recognition", False),
            "future_roster_player_metadata": config.get("metadata_hooks", {}).get("roster_player_metadata", False),
            "future_heatmaps": config.get("metadata_hooks", {}).get("heatmap_generation", False),
        },
    }
