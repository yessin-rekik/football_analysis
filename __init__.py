"""
football_analysis
==================

Modular pipeline for football match calibration, tracking, and (eventually)
tactical analysis, built as independently-testable stages:

    config      -- pitch/match configuration schema (Phase 0)
    schemas     -- canonical per-frame output schema (Phase 0)
    calibration -- scene routing, keypoint model, homography, camera motion
                   propagation (Phase 1)
    tracking    -- player/ball detection, multi-object tracking, team
                   classification, re-identification (Phase 2)
    coordinates -- pixel -> world transform, export (Phase 3)
    api         -- FastAPI service layer (Phase 5)

Every stage after `config` and `schemas` communicates using ONLY the models
defined in those two packages. No stage should import internals from another
stage's implementation -- if stage B needs something from stage A, it should
be expressible in the shared schema.
"""

__version__ = "0.1.0"
