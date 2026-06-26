# External Public 0.5917 notebook review

The notebook is useful as a strategy note, not as a reproducible pipeline. It claims Public `0.5917` with LGBM+XGB anchor blending, Subject-hole CV, stability feature selection, and 23,177 -> 1,682 feature reduction, but it does not include the V152 anchor artifacts or full executable feature construction.

Transferable pieces: conservative anchor blending, stability-selected features, subject-aware validation, and window/screen-status feature directions. It should guide the next clean representation search rather than replace the current artifact ladder.

## Follow-up transfer checks

Two lightweight transfer diagnostics were run after this review. Sleep-guideline overlays on ID221/226/231 selected `base` for ID226/ID231 under both public-interleaved and Subject-Hole lenses. A shallow stability-selected LGBM/XGB fast scan also selected `base` for every target, even under optimistic full-OOF selection. These results make the notebook useful as validation philosophy, but not as a direct source of the next promoted artifact.
