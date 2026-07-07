# Thermal Field Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce thermal-field computation time while preserving the current full-resolution output contract.

**Architecture:** Build and blur a quarter-resolution fire-intensity map, cache that small blurred map by an exact mask digest, and resize it to the original grid on access. Keep Gaussian `truncate=4.0` so the approximation remains numerically close to the original field.

**Tech Stack:** Python, NumPy, SciPy, OpenCV, unittest/pytest

---

### Task 1: Add thermal-field regression tests

**Files:**
- Create: `environment_variables/environment_variables/test_thermal_field_optimization.py`
- Test: `environment_variables/environment_variables/test_thermal_field_optimization.py`

- [x] **Step 1: Write failing cache and output tests**

Create tests that instantiate `FireSceneData` without loading raster files, supply synthetic `intensity` and fire masks, and assert:

```python
self.assertEqual(mocked_filter.call_count, 1)
self.assertEqual(len(scene._thermal_field_cache), 2)
self.assertEqual(scene.thermal_field.shape, scene.shape)
self.assertGreaterEqual(float(scene.thermal_field.min()), 0.0)
self.assertLessEqual(float(scene.thermal_field.max()), 100.0)
```

- [x] **Step 2: Run tests and verify RED**

Run:

```powershell
C:\Users\cameliar\.conda\envs\pytorch\python.exe -B test_thermal_field_optimization.py
```

Expected: failures because equal-count masks currently collide and the cache stores full-resolution fields.

### Task 2: Implement low-resolution filtering and reliable caching

**Files:**
- Modify: `environment_variables/environment_variables/信息转换.py:1-15`
- Modify: `environment_variables/environment_variables/信息转换.py:753-776`
- Test: `environment_variables/environment_variables/test_thermal_field_optimization.py`

- [x] **Step 1: Add required imports**

```python
import hashlib
import cv2
```

- [x] **Step 2: Replace the count key and full-resolution filter**

Use:

```python
fire_mask = np.ascontiguousarray(self.fire_binary_map > 0)
cache_key = hashlib.blake2b(
    np.packbits(fire_mask).tobytes(),
    digest_size=16,
).digest()
```

On a cache miss, resize `current_fire` by `0.25`, call:

```python
gaussian_filter(small_fire, sigma=15, truncate=4.0)
```

and cache the small result. On every call, resize the small result back to `(width, height)` and clip it after multiplying by `800.0`.

- [x] **Step 3: Run focused tests and verify GREEN**

Run:

```powershell
C:\Users\cameliar\.conda\envs\pytorch\python.exe -B test_thermal_field_optimization.py
```

Expected: all tests pass.

### Task 3: Declare direct dependencies

**Files:**
- Modify: `environment_variables/requirements.txt`

- [x] **Step 1: Add SciPy and OpenCV**

Add:

```text
scipy>=1.10.0
opencv-python>=4.8.0
```

### Task 4: Verify regression, accuracy, and performance

**Files:**
- Verify: `environment_variables/environment_variables/信息转换.py`
- Verify: `environment_variables/environment_variables/test_thermal_field_optimization.py`

- [x] **Step 1: Run the full test suite**

Run:

```powershell
C:\Users\cameliar\.conda\envs\pytorch\python.exe -B -m unittest discover -s . -p "test_*.py" -v
```

Expected: all tests pass.

- [x] **Step 2: Run dataset accuracy and benchmark checks**

Compare the optimized output against `gaussian_filter(current_fire, sigma=60, truncate=4.0)` on train scenes at steps 0, 300, and 600.

Expected:

```text
MAE <= 0.5
threshold disagreement <= 0.2%
speedup >= 20x
```

- [x] **Step 3: Run a short training smoke test**

Run:

```powershell
C:\Users\cameliar\.conda\envs\pytorch\python.exe ctde_ppo_baseline_train.py --single-train --episodes 2 --no-eval --no-plot
```

Expected: exit code 0 and a completed two-episode training log.

## Plan Self-Review

- The plan covers cache correctness, numerical behavior, dependencies, full regression, performance, and training integration.
- No unrelated refactoring or configuration options are included.
- Production behavior remains behind the existing `_compute_thermal_field()` interface.
