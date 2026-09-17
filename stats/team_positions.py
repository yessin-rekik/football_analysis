"""
Shared primitive for Phase 4b (team shape, Voronoi, pressing heatmap) --
the "positions grouped by team" utility the project plan calls out as the
common input all three share.

Deliberately its own tiny module rather than a method on FrameResult or a
copy-pasted loop in each of the three consumers: each of team_shape.py,
voronoi.py, and pressing_heatmap.py calls this once per frame in whatever
window they were handed, rather than re-implementing "how do I pull
team-labeled world positions out of a FrameResult" three separate times.

Same dependency rule as stats/radar.py: imports ONLY from schemas/ and
config/ (config isn't even needed here) -- nothing from tracking/, even
though tracking/team_classifier.py already defines an equivalent
TEAM_ELIGIBLE_CLASSES constant. That constant is NOT imported here; a
separate, independently-defined one is declared below instead. Reusing
tracking's constant would create exactly the sibling-stage import the
project's dependency-direction rule forbids -- stats/ is downstream of
schemas/, not of tracking/, and should stay buildable/testable without
tracking/ existing at all (same reasoning `coordinates/transform.py`
already established for Phase 3).
"""

from typing import Dict, List, Optional, Set, Tuple

from ..schemas.enums import ObjectClass, Team
from ..schemas.frame_result import FrameResult

# Object classes eligible to be grouped by team at all. Independently
# defined from tracking/team_classifier.py's TEAM_ELIGIBLE_CLASSES (see
# module docstring for why) -- happens to currently hold the same two
# classes, but there's no code-level link between them, so a future change
# to one has no silent effect on the other. If they ever need to diverge
# (e.g. a stat that deliberately wants goalkeepers excluded), that's a
# parameter override here, not a reason to import the other constant.
TEAM_POSITION_ELIGIBLE_CLASSES: Set[ObjectClass] = {ObjectClass.PLAYER, ObjectClass.GOALKEEPER}


def group_positions_by_team(
    frame: FrameResult,
    object_classes: Optional[Set[ObjectClass]] = None,
) -> Dict[Team, List[Tuple[int, Tuple[float, float]]]]:
    """
    Groups one frame's team-labeled world positions by Team.

    An object is included only if ALL of the following hold:
      - obj.object_class is in `object_classes` (default:
        TEAM_POSITION_ELIGIBLE_CLASSES -- players and goalkeepers; the
        ball and referees are never team-eligible regardless of override,
        since Team is genuinely not applicable to them)
      - obj.team is not None (team classification hasn't reached this
        object yet, or it's a class Team doesn't apply to)
      - obj.world_position is not None (NOT_CALIBRATED this frame, or
        Phase 3 didn't project this object for some other reason)

    No confidence filtering happens here -- a low-confidence (e.g.
    PROPAGATED or out-of-bounds-zeroed) world_position is still included.
    That's a deliberate scope boundary: confidence-based filtering is a
    per-consumer decision (a heatmap might want to weight by confidence,
    team shape might want to exclude below some floor entirely), not
    something this shared primitive should decide unilaterally for every
    caller.

    Returns a dict with BOTH Team.HOME and Team.AWAY always present as
    keys (each mapped to an empty list if no eligible object was found for
    that team this frame) -- so callers can safely do
    `grouped[Team.HOME]` without a defensive `.get(...)` check on every
    use, even on a frame where one team has nobody eligible (e.g. a
    NOT_CALIBRATED frame, or the very start of a clip before team
    classification has fit).

    Each entry is (track_id, (world_x, world_y)) -- track_id is included
    (not just the bare coordinate) because team_shape.py needs it to
    accumulate a running per-track average; a caller that doesn't need it
    can simply ignore the first element of the tuple.
    """
    eligible_classes = object_classes if object_classes is not None else TEAM_POSITION_ELIGIBLE_CLASSES

    grouped: Dict[Team, List[Tuple[int, Tuple[float, float]]]] = {Team.HOME: [], Team.AWAY: []}

    for obj in frame.tracked_objects:
        if obj.object_class not in eligible_classes:
            continue
        if obj.team is None:
            continue
        if obj.world_position is None:
            continue
        grouped[obj.team].append((obj.track_id, obj.world_position.as_tuple()))

    return grouped