# Thermal Field Optimization Design

## Scope

Optimize `environment_variables/environment_variables/信息转换.py` without changing the thermal-field shape, range, or downstream API.

## Design

Use a 4x spatial downsample before Gaussian filtering, scale `sigma` from 60 to 15, retain `truncate=4.0`, and resize the blurred field back to the source grid. Cache the low-resolution blurred field rather than the full-resolution output.

Replace the fire-cell-count cache key with a BLAKE2b digest of the packed binary fire mask. This distinguishes masks that contain the same number of fire cells at different positions.

Keep the existing `Scene loaded` diagnostic disabled. No training-loop or reward changes are included.

## Acceptance Criteria

- Repeated identical masks execute Gaussian filtering once.
- Equal-count masks at different positions do not share a cache entry.
- Output shape matches the source map and values remain in `[0, 100]`.
- Against the original full-resolution implementation on representative Self dataset states:
  - mean absolute error is at most `0.5`;
  - `0.5` and `0.8` threshold disagreement is at most `0.2%`.
- Cold thermal-field computation is at least 20x faster in a manual benchmark.
- Existing tests and a short training smoke test pass.

## Dependencies

Declare the directly imported SciPy and OpenCV packages in `environment_variables/requirements.txt`.
