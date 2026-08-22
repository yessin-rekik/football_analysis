"""
Match-level metadata: the "header" that accompanies a stream/list of
FrameResults. This is what the Phase 5 API returns alongside the frame
data, and what stats code (Phase 4) needs for context that doesn't belong
on every single frame (pitch dimensions, video properties, which model
versions produced this data).
"""

from typing import Optional

from pydantic import BaseModel

from ..config.pitch_config import PitchConfig


class ModelVersions(BaseModel):
    """Which model/version produced this data -- important for a
    microservice: lets consumers know what to expect (e.g. accuracy
    characteristics) and makes results reproducible/debuggable."""
    keypoint_model: Optional[str] = None
    scene_classifier: Optional[str] = None
    detection_model: Optional[str] = None
    tracker: Optional[str] = None
    reid_model: Optional[str] = None


class MatchMetadata(BaseModel):
    schema_version: int = 1

    match_id: str
    video_fps: float
    video_width: int
    video_height: int

    pitch_config: PitchConfig
    model_versions: ModelVersions = ModelVersions()
