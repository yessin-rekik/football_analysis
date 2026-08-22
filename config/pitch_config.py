"""
Pitch configuration schema.

This is the single source of truth for pitch geometry. It is created once
per match/video (from API input, or FIFA-standard defaults) and passed down
through every stage of the pipeline -- calibration, coordinate transform,
and eventually stats -- so nothing downstream hardcodes 105x68 or any other
dimension.
"""

from enum import IntEnum
from typing import Dict, Tuple

import numpy as np
from pydantic import BaseModel, Field, model_validator


class PitchKeypointName(IntEnum):
    """Fixed semantic ordering of the 29 pitch keypoints. Values match the
    index convention already used by the keypoint detection model, so this
    enum can be used directly as an array index."""
    SIDELINE_TOP_LEFT = 0
    BIG_RECT_LEFT_TOP_PT1 = 1
    BIG_RECT_LEFT_TOP_PT2 = 2
    BIG_RECT_LEFT_BOTTOM_PT1 = 3
    BIG_RECT_LEFT_BOTTOM_PT2 = 4
    SMALL_RECT_LEFT_TOP_PT1 = 5
    SMALL_RECT_LEFT_TOP_PT2 = 6
    SMALL_RECT_LEFT_BOTTOM_PT1 = 7
    SMALL_RECT_LEFT_BOTTOM_PT2 = 8
    SIDELINE_BOTTOM_LEFT = 9
    LEFT_SEMICIRCLE_RIGHT = 10
    CENTER_LINE_TOP = 11
    CENTER_LINE_BOTTOM = 12
    CENTER_CIRCLE_TOP = 13
    CENTER_CIRCLE_BOTTOM = 14
    FIELD_CENTER = 15
    SIDELINE_TOP_RIGHT = 16
    BIG_RECT_RIGHT_TOP_PT1 = 17
    BIG_RECT_RIGHT_TOP_PT2 = 18
    BIG_RECT_RIGHT_BOTTOM_PT1 = 19
    BIG_RECT_RIGHT_BOTTOM_PT2 = 20
    SMALL_RECT_RIGHT_TOP_PT1 = 21
    SMALL_RECT_RIGHT_TOP_PT2 = 22
    SMALL_RECT_RIGHT_BOTTOM_PT1 = 23
    SMALL_RECT_RIGHT_BOTTOM_PT2 = 24
    SIDELINE_BOTTOM_RIGHT = 25
    RIGHT_SEMICIRCLE_LEFT = 26
    CENTER_CIRCLE_LEFT = 27
    CENTER_CIRCLE_RIGHT = 28


PITCH_KEYPOINT_NAMES: Dict[int, str] = {kp.value: kp.name.lower() for kp in PitchKeypointName}

NUM_PITCH_KEYPOINTS = len(PitchKeypointName)


class PitchConfig(BaseModel):
    """
    Pitch geometry, in meters. Defaults are FIFA/UEFA standard senior-pitch
    dimensions. Every field is overridable so this same schema covers youth
    pitches, non-standard stadiums, or other sports' pitch conventions later
    if needed.

    Coordinate convention (must stay consistent across every stage that
    consumes this config):
        x in [0, pitch_length]   0 = left goal line, pitch_length = right goal line
        y in [0, pitch_width]    0 = "bottom" sideline, pitch_width = "top" sideline
    """

    config_version: int = Field(
        default=1,
        description="Schema version for this config shape. Bump on breaking field changes.",
    )

    pitch_length: float = Field(default=105.0, gt=0, description="Meters, goal line to goal line.")
    pitch_width: float = Field(default=68.0, gt=0, description="Meters, sideline to sideline.")

    penalty_area_depth: float = Field(default=16.5, gt=0, description="Meters, goal line to 18-yard line.")
    penalty_area_width: float = Field(default=40.32, gt=0, description="Meters, full width of the penalty area.")

    goal_area_depth: float = Field(default=5.5, gt=0, description="Meters, goal line to 6-yard line.")
    goal_area_width: float = Field(default=18.32, gt=0, description="Meters, full width of the goal area.")

    center_circle_radius: float = Field(default=9.15, gt=0, description="Meters.")
    penalty_spot_distance: float = Field(default=11.0, gt=0, description="Meters, goal line to penalty spot.")

    goal_width: float = Field(default=7.32, gt=0, description="Meters, inside of posts.")

    @model_validator(mode="after")
    def _check_dimensions_are_consistent(self) -> "PitchConfig":
        if self.penalty_area_width >= self.pitch_width:
            raise ValueError("penalty_area_width must be smaller than pitch_width")
        if self.goal_area_width >= self.penalty_area_width:
            raise ValueError("goal_area_width must be smaller than penalty_area_width")
        if self.goal_area_depth >= self.penalty_area_depth:
            raise ValueError("goal_area_depth must be smaller than penalty_area_depth")
        if self.penalty_area_depth >= self.pitch_length / 2:
            raise ValueError("penalty_area_depth must be smaller than half the pitch length")
        if self.goal_width >= self.goal_area_width:
            raise ValueError("goal_width must be smaller than goal_area_width")
        return self

    def world_keypoints(self) -> np.ndarray:
        """
        Build the (29, 2) array of real-world meter coordinates for the
        standard pitch keypoint schema, using this config's dimensions.

        This is the ONLY place pitch geometry should be translated into
        keypoint coordinates -- the calibration layer (Phase 1) consumes
        this rather than recomputing geometry itself.
        """
        L, W = self.pitch_length, self.pitch_width
        half_pa_w = self.penalty_area_width / 2.0
        half_ga_w = self.goal_area_width / 2.0
        cy = W / 2.0
        r = self.center_circle_radius
        pa_depth = self.penalty_area_depth
        ga_depth = self.goal_area_depth
        pen_spot = self.penalty_spot_distance

        pts = np.zeros((NUM_PITCH_KEYPOINTS, 2), dtype=np.float32)
        K = PitchKeypointName

        pts[K.SIDELINE_TOP_LEFT] = (0, W)
        pts[K.BIG_RECT_LEFT_TOP_PT1] = (0, cy + half_pa_w)
        pts[K.BIG_RECT_LEFT_TOP_PT2] = (pa_depth, cy + half_pa_w)
        pts[K.BIG_RECT_LEFT_BOTTOM_PT1] = (0, cy - half_pa_w)
        pts[K.BIG_RECT_LEFT_BOTTOM_PT2] = (pa_depth, cy - half_pa_w)
        pts[K.SMALL_RECT_LEFT_TOP_PT1] = (0, cy + half_ga_w)
        pts[K.SMALL_RECT_LEFT_TOP_PT2] = (ga_depth, cy + half_ga_w)
        pts[K.SMALL_RECT_LEFT_BOTTOM_PT1] = (0, cy - half_ga_w)
        pts[K.SMALL_RECT_LEFT_BOTTOM_PT2] = (ga_depth, cy - half_ga_w)
        pts[K.SIDELINE_BOTTOM_LEFT] = (0, 0)
        pts[K.LEFT_SEMICIRCLE_RIGHT] = (pen_spot + r, cy)
        pts[K.CENTER_LINE_TOP] = (L / 2.0, W)
        pts[K.CENTER_LINE_BOTTOM] = (L / 2.0, 0)
        pts[K.CENTER_CIRCLE_TOP] = (L / 2.0, cy + r)
        pts[K.CENTER_CIRCLE_BOTTOM] = (L / 2.0, cy - r)
        pts[K.FIELD_CENTER] = (L / 2.0, cy)
        pts[K.SIDELINE_TOP_RIGHT] = (L, W)
        pts[K.BIG_RECT_RIGHT_TOP_PT1] = (L, cy + half_pa_w)
        pts[K.BIG_RECT_RIGHT_TOP_PT2] = (L - pa_depth, cy + half_pa_w)
        pts[K.BIG_RECT_RIGHT_BOTTOM_PT1] = (L, cy - half_pa_w)
        pts[K.BIG_RECT_RIGHT_BOTTOM_PT2] = (L - pa_depth, cy - half_pa_w)
        pts[K.SMALL_RECT_RIGHT_TOP_PT1] = (L, cy + half_ga_w)
        pts[K.SMALL_RECT_RIGHT_TOP_PT2] = (L - ga_depth, cy + half_ga_w)
        pts[K.SMALL_RECT_RIGHT_BOTTOM_PT1] = (L, cy - half_ga_w)
        pts[K.SMALL_RECT_RIGHT_BOTTOM_PT2] = (L - ga_depth, cy - half_ga_w)
        pts[K.SIDELINE_BOTTOM_RIGHT] = (L, 0)
        pts[K.RIGHT_SEMICIRCLE_LEFT] = (L - pen_spot - r, cy)
        pts[K.CENTER_CIRCLE_LEFT] = (L / 2.0 - r, cy)
        pts[K.CENTER_CIRCLE_RIGHT] = (L / 2.0 + r, cy)

        return pts

    def in_bounds(self, world_xy: Tuple[float, float], margin: float = 5.0) -> bool:
        """Sanity check used by the calibration layer to flag implausible
        pixel->world results (see the earlier out-of-bounds diagnostic)."""
        x, y = world_xy
        return (-margin <= x <= self.pitch_length + margin) and (-margin <= y <= self.pitch_width + margin)
