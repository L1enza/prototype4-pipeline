from prototype4_pipeline.artifacts import write_json


def run_coordinate_alignment(context, world, active_tracks, meshes):
    aligned_tracks = []
    for track in active_tracks.get("tracks", []):
        aligned_tracks.append(
            {
                "track_id": track["track_id"],
                "field_coordinates": track.get("field_coordinates", []),
                "mesh_sequence": meshes.get("mesh_sequences", {}).get(track["track_id"]),
                "alignment_status": "placeholder_pending_vggt_field_calibration",
            }
        )

    result = {
        "status": "placeholder",
        "stage": "coordinate_alignment",
        "input_counts": {
            "active_tracks": len(active_tracks.get("tracks", [])),
            "mesh_sequences": len(meshes.get("mesh_sequences", {})),
        },
        "output_dir": str(context.dirs["metadata"]),
        "coordinate_system": context.config.get("field", {}).get("coordinate_system"),
        "world_reference": world.get("outputs", {}),
        "aligned_tracks": aligned_tracks,
        "notes": [
            "Map 2D player footpoints through VGGT camera geometry and lacrosse field calibration.",
            "Attach SAM-Body4D mesh roots to the same field coordinate system.",
        ],
    }
    write_json(context.dirs["metadata"] / "coordinate_alignment.json", result)
    return result
