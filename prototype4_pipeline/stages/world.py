from prototype4_pipeline.artifacts import write_json
from prototype4_pipeline.integrations.vggt_runner import VGGTRunner


def run_world_reconstruction(context, frames):
    result = VGGTRunner(context).run(frames)
    write_json(context.dirs["world"] / "world_reconstruction_manifest.json", result)
    return result
