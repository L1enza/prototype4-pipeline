from prototype4_pipeline.artifacts import write_json
from prototype4_pipeline.integrations.sam_body4d_runner import SAMBody4DRunner


def run_player_mesh_reconstruction(context, crops, active_tracks):
    result = SAMBody4DRunner(context).run(crops, active_tracks)
    write_json(context.dirs["meshes"] / "meshes_manifest.json", result)
    return result
