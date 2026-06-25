from prototype4_pipeline.artifacts import write_json
from prototype4_pipeline.filters.active_players import filter_active_players


def run_active_player_filtering(context, tracks, world):
    result = filter_active_players(context.config, tracks, world)
    write_json(context.dirs["tracks"] / "active_player_tracks.json", result)
    return result
