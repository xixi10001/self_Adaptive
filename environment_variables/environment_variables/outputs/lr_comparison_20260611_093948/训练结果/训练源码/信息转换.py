from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import rasterio
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, gaussian_filter


class InvalidSceneError(RuntimeError):
    """Raised when a scene cannot provide a valid t=0 fire boundary."""


class DatasetIndex:
    """Dataset index backed by dataset/dataset_index.json and scene_key ids."""

    MODE_ALIASES = {
        "train": "train",
        "validation": "validation",
        "generalization": "generalization",
        "stress": "stress",
        "test": "generalization",
        "eval": "generalization",
    }

    def __init__(
        self, data_dir: str = "./dataset", index_name: str = "dataset_index.json"
    ):
        self.data_dir = self._resolve_data_dir(data_dir)
        self.index_path = self.data_dir / index_name
        if not self.index_path.is_file():
            raise FileNotFoundError(
                f"dataset_index.json not found: {self.index_path}\n"
                f"Current working directory: {os.getcwd()}"
            )

        with self.index_path.open("r", encoding="utf-8") as f:
            self.index = json.load(f)

        source_root = Path(self.index.get("source_root", ""))
        if not source_root.is_absolute():
            source_root = (self.index_path.parent / source_root).resolve()
        self.source_root = source_root

        self.splits: Dict[str, List[str]] = {
            str(name): [str(key) for key in keys]
            for name, keys in self.index.get("splits", {}).items()
        }
        self.scenes: Dict[str, Dict] = {
            str(key): dict(record)
            for key, record in self.index.get("scenes", {}).items()
        }

        self.all_scene_keys: List[str] = []
        for split_name in ["train", "validation", "generalization", "stress"]:
            self.all_scene_keys.extend(self.splits.get(split_name, []))
        for key in self.scenes:
            if key not in self.all_scene_keys:
                self.all_scene_keys.append(key)

    @staticmethod
    def _resolve_data_dir(data_dir: str) -> Path:
        path = Path(data_dir)
        if path.is_absolute():
            return path

        cwd_path = (Path.cwd() / path).resolve()
        if cwd_path.exists():
            return cwd_path

        script_path = (Path(__file__).resolve().parent / path).resolve()
        return script_path

    def normalize_mode(self, mode: str) -> str:
        mode_key = str(mode).lower()
        if mode_key not in self.MODE_ALIASES:
            raise ValueError(
                f"Unknown scene mode {mode!r}. "
                f"Expected one of: {sorted(self.MODE_ALIASES)}"
            )
        return self.MODE_ALIASES[mode_key]

    def scene_keys(self, mode: str = "train") -> List[str]:
        split = self.normalize_mode(mode)
        keys = list(self.splits.get(split, []))
        if not keys:
            raise ValueError(f"No scenes configured for split {split!r}")
        return keys

    def get_record(self, scene_key: str) -> Dict:
        scene_key = str(scene_key)
        if scene_key not in self.scenes:
            raise KeyError(f"Unknown scene_key: {scene_key}")

        record = dict(self.scenes[scene_key])
        record["scene_key"] = scene_key
        record["scene_dir_abs"] = str(self.scene_dir(scene_key))
        record["scene_index"] = self.scene_index(scene_key)
        return record

    def scene_dir(self, scene_key: str) -> Path:
        record = self.scenes[str(scene_key)]
        scene_dir = Path(record["scene_dir"])
        if not scene_dir.is_absolute():
            scene_dir = self.source_root / scene_dir
        return scene_dir.resolve()

    def scene_index(self, scene_key: str) -> int:
        scene_key = str(scene_key)
        if scene_key in self.all_scene_keys:
            return self.all_scene_keys.index(scene_key) + 1
        return 0

    def required_file_paths(self, scene_key: str) -> List[Tuple[str, Path]]:
        record = self.scenes[str(scene_key)]
        scene_dir = self.scene_dir(scene_key)
        required = [
            ("metadata", record.get("metadata", "metadata.json")),
        ]
        for key in ["intensity", "length", "time", "speedRate"]:
            required.append((key, record.get("rasters", {}).get(key)))
        for key in ["spread_direction", "heat_per_unit_area", "crown_fire"]:
            required.append((key, record.get("rasters", {}).get(key)))
        required.extend(
            [
                (
                    "ignition",
                    record.get("vectors", {}).get("ignition", "vectors/ignition.shp"),
                ),
                (
                    "fire_perimeter",
                    record.get("vectors", {}).get(
                        "fire_perimeter", "vectors/fire_perimeter.shp"
                    ),
                ),
                (
                    "weather_stream",
                    record.get("inputs", {}).get(
                        "weather_stream", "inputs/weather_stream.wxs"
                    ),
                ),
                (
                    "fuel_moisture",
                    record.get("inputs", {}).get(
                        "fuel_moisture", "inputs/fuel_moisture_dry.fms"
                    ),
                ),
                (
                    "fire_growth_report",
                    record.get("reports", {}).get(
                        "fire_growth_report", "reports/fire_growth_report.csv"
                    ),
                ),
                (
                    "run_log",
                    record.get("reports", {}).get("run_log", "reports/Run_log.txt"),
                ),
            ]
        )

        paths = []
        for label, rel_path in required:
            if not rel_path:
                paths.append((label, scene_dir / "__missing_path__"))
                continue
            path = Path(str(rel_path).replace("\\", "/"))
            if not path.is_absolute():
                path = scene_dir / path
            paths.append((label, path.resolve()))
        return paths


class FireEnvironmentData:
    """Load one FARSITE scene from dataset_index.json."""

    CORE_KEYS = ["intensity", "length", "time", "speedRate"]
    EXTRA_RASTER_KEYS = ["spread_direction", "heat_per_unit_area", "crown_fire"]

    def __init__(
        self,
        data_dir: str = "./dataset",
        scene_key: str = None,
        scene_record: Dict = None,
        dataset_index: DatasetIndex = None,
    ):
        self.dataset_index = dataset_index or DatasetIndex(data_dir)
        if scene_record is None:
            if scene_key is None:
                scene_key = self.dataset_index.scene_keys("train")[0]
            scene_record = self.dataset_index.get_record(scene_key)

        self.base_dir = str(self.dataset_index.data_dir)
        self.scene_key = str(scene_record["scene_key"])
        self.scene_id = int(
            scene_record.get(
                "scene_index", self.dataset_index.scene_index(self.scene_key)
            )
        )
        self.scene_record = dict(scene_record)
        self.data_dir = Path(scene_record["scene_dir_abs"]).resolve()

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {self.data_dir}")

        self.metadata = self._load_metadata()
        self.file_paths = self._build_file_paths()

        self.data: Dict[str, np.ndarray] = {}
        self.transform = None
        self.crs = None
        self.shape = None
        self.nodata_value = None

        self.norm_params = {
            "intensity_max": 626.94,
            "dem_min": 0.0,
            "dem_max": 1.0,
            "slope_max": 1.0,
            "wind_speed_max": 50.49,
            "fire_threshold": 1.0,
        }

        self.boundary_points = None
        self.fire_binary_map = None
        self.sdf = None
        self.thermal_field = None
        self.last_boundary_sim_time = None
        self.training_start_sim_time = None
        self.last_init_area_stats = None
        self.is_valid_scene = True
        self.invalid_reason = None

        self.load_all_data()
        self._initialize_boundary()
        self._compute_sdf()
        self._compute_thermal_field()

    def _load_metadata(self) -> Dict:
        metadata_path = self.data_dir / self.scene_record.get(
            "metadata", "metadata.json"
        )
        if not metadata_path.is_file():
            raise FileNotFoundError(f"metadata.json missing: {metadata_path}")
        with metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _path_from_scene(self, rel_path: str) -> Path:
        path = Path(str(rel_path).replace("\\", "/"))
        if path.is_absolute():
            return path
        return (self.data_dir / path).resolve()

    def _build_file_paths(self) -> Dict[str, Path]:
        rasters = self.scene_record.get("rasters", {})
        file_paths: Dict[str, Path] = {}
        for key in self.CORE_KEYS + self.EXTRA_RASTER_KEYS:
            if key in rasters:
                file_paths[key] = self._path_from_scene(rasters[key])

        inputs = self.scene_record.get("inputs", {})
        file_paths["weather_stream"] = self._path_from_scene(
            inputs.get("weather_stream", "inputs/weather_stream.wxs")
        )
        file_paths["fuel_moisture"] = self._path_from_scene(
            inputs.get("fuel_moisture", "inputs/fuel_moisture_dry.fms")
        )
        file_paths["wind_direction_asc"] = self._path_from_scene("wind/wdir.asc")
        file_paths["wind_speed_asc"] = self._path_from_scene("wind/wspd.asc")
        return file_paths

    def load_raster(self, filepath: Path) -> Tuple[np.ndarray, dict]:
        try:
            with rasterio.open(filepath) as src:
                data = src.read()
                if data.shape[0] == 1:
                    data = data[0]
                metadata = {
                    "transform": src.transform,
                    "crs": src.crs,
                    "shape": (src.height, src.width),
                    "nodata": src.nodata,
                    "bounds": src.bounds,
                    "count": src.count,
                }
                return data, metadata
        except Exception as exc:
            raise RuntimeError(f"Failed to read raster: {filepath}\n{exc}") from exc

    def load_asc(self, filepath: Path) -> np.ndarray:
        try:
            with filepath.open("r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            data_lines = [line.strip().split() for line in lines[6:] if line.strip()]
            return np.array(
                [[float(x) for x in line] for line in data_lines], dtype=np.float32
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to read ASC file: {filepath}\n{exc}") from exc

    def _parse_weather_stream(self) -> Tuple[float, float]:
        weather_path = self.file_paths["weather_stream"]
        if not weather_path.is_file():
            wind = self.metadata.get("wind", {})
            return (
                float(
                    wind.get("wind_speed_mph", wind.get("wind_speed_mps_approx", 0.0))
                ),
                float(wind.get("wind_direction_deg", 0.0)),
            )

        rows = []
        header_seen = False
        with weather_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.lower().startswith("year"):
                    header_seen = True
                    continue
                if not header_seen:
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                try:
                    rows.append((float(parts[7]), float(parts[8])))
                except ValueError:
                    continue

        if not rows:
            wind = self.metadata.get("wind", {})
            return (
                float(
                    wind.get("wind_speed_mph", wind.get("wind_speed_mps_approx", 0.0))
                ),
                float(wind.get("wind_direction_deg", 0.0)),
            )

        speeds = np.asarray([row[0] for row in rows], dtype=np.float64)
        dirs = np.radians([row[1] for row in rows])
        mean_dir = float(
            np.degrees(np.arctan2(np.mean(np.sin(dirs)), np.mean(np.cos(dirs)))) % 360.0
        )
        return float(np.mean(speeds)), mean_dir

    def _load_wind_fields(self):
        speed_asc = self.file_paths["wind_speed_asc"]
        direction_asc = self.file_paths["wind_direction_asc"]
        if speed_asc.is_file() and direction_asc.is_file():
            self.data["wind_speed"] = self.load_asc(speed_asc)
            self.data["wind_direction"] = self.load_asc(direction_asc)
            return

        if self.shape is None:
            raise RuntimeError(
                "Cannot generate wind fields before raster shape is known"
            )

        wind_speed, wind_direction = self._parse_weather_stream()
        self.data["wind_speed"] = np.full(self.shape, wind_speed, dtype=np.float32)
        self.data["wind_direction"] = np.full(
            self.shape, wind_direction, dtype=np.float32
        )

    def _load_placeholder_terrain(self):
        if self.shape is None:
            raise RuntimeError(
                "Cannot generate placeholder terrain before raster shape is known"
            )
        self.data["dem"] = np.zeros(self.shape, dtype=np.float32)
        self.data["slope"] = np.zeros(self.shape, dtype=np.float32)
        self.data["aspect"] = np.zeros(self.shape, dtype=np.float32)

    def load_all_data(self):
        print(f"Loading scene {self.scene_key}...")

        files_loaded = 0
        for key in self.CORE_KEYS:
            filepath = self.file_paths.get(key)
            if filepath is None or not filepath.is_file():
                raise FileNotFoundError(
                    f"Required raster missing for {self.scene_key}: {filepath}"
                )

            data, metadata = self.load_raster(filepath)
            data = np.asarray(data)
            data[data < 0] = 0
            self.data[key] = data
            files_loaded += 1

            if self.transform is None:
                self.transform = metadata["transform"]
                self.crs = metadata["crs"]
                self.shape = metadata["shape"]
                self.nodata_value = metadata["nodata"]
            elif metadata["shape"] != self.shape:
                raise RuntimeError(
                    f"Core raster shape mismatch in {self.scene_key}: "
                    f"{filepath} has {metadata['shape']}, expected {self.shape}"
                )

        for key in self.EXTRA_RASTER_KEYS:
            filepath = self.file_paths.get(key)
            if filepath is not None and filepath.is_file():
                data, metadata = self.load_raster(filepath)
                if metadata["shape"] != self.shape:
                    raise RuntimeError(
                        f"Raster shape mismatch in {self.scene_key}: "
                        f"{filepath} has {metadata['shape']}, expected {self.shape}"
                    )
                data = np.asarray(data)
                data[data < 0] = 0
                self.data[key] = data
                files_loaded += 1

        self._load_placeholder_terrain()
        self._load_wind_fields()

        if (
            self.data["wind_speed"].shape != self.shape
            or self.data["wind_direction"].shape != self.shape
        ):
            raise RuntimeError(
                f"Wind field shape mismatch in {self.scene_key}: "
                f"speed={self.data['wind_speed'].shape}, direction={self.data['wind_direction'].shape}, "
                f"expected={self.shape}"
            )

        print(
            f"Scene {self.scene_key} loaded | shape={self.shape} | rasters={files_loaded}"
        )

    def _initialize_boundary(self):
        self.boundary_points = self.detect_fire_boundary(time_step=0)
        self.training_start_sim_time = None
        if len(self.boundary_points) == 0:
            self.is_valid_scene = False
            self.invalid_reason = (
                f"Scene {self.scene_key} has empty t=0 fire boundary. "
                "Training must stop instead of falling back to the final-state boundary."
            )
            raise InvalidSceneError(self.invalid_reason)
        print(
            f"Scene {self.scene_key} t=0 boundary points: {len(self.boundary_points)}"
        )

    def initialize_training_boundary(
        self,
        init_percentile: Optional[float] = 5.0,
        init_area_percent: Optional[float] = None,
    ) -> List[Tuple[int, int]]:
        area_percent = init_area_percent if init_area_percent is not None else init_percentile
        if area_percent is None:
            self.boundary_points = self.detect_fire_boundary(time_step=0)
            self.training_start_sim_time = None
        else:
            self.boundary_points = self.detect_fire_boundary(
                time_step=0,
                init_area_percent=float(area_percent),
            )
            self.training_start_sim_time = self.last_boundary_sim_time

        if len(self.boundary_points) == 0:
            self.is_valid_scene = False
            self.invalid_reason = (
                f"Scene {self.scene_key} has empty training fire boundary "
                f"(init_area_percent={area_percent})."
            )
            raise InvalidSceneError(self.invalid_reason)
        return self.boundary_points

    def _select_fire_by_area_percent(
        self,
        base_binary: np.ndarray,
        time_map: np.ndarray,
        init_area_percent: float,
    ) -> np.ndarray:
        valid_fire_mask = (base_binary > 0) & (time_map >= -1)
        total_fire_cells = int(np.count_nonzero(valid_fire_mask))
        if total_fire_cells == 0:
            self.last_boundary_sim_time = None
            self.last_init_area_stats = {
                "total_fire_cells": 0,
                "init_fire_cells": 0,
                "actual_init_area_percent": 0.0,
                "cutoff_time": None,
            }
            return np.zeros_like(base_binary)

        pct = float(np.clip(init_area_percent, 0.0, 100.0))
        target_cells = max(1, int(np.ceil(total_fire_cells * pct / 100.0)))
        fire_times = time_map[valid_fire_mask].astype(np.float64)
        cutoff_time = float(np.partition(fire_times, target_cells - 1)[target_cells - 1])
        fire_binary = (valid_fire_mask & (time_map <= cutoff_time)).astype(np.uint8)
        init_fire_cells = int(np.count_nonzero(fire_binary))

        self.last_boundary_sim_time = cutoff_time
        self.last_init_area_stats = {
            "total_fire_cells": total_fire_cells,
            "init_fire_cells": init_fire_cells,
            "actual_init_area_percent": float(100.0 * init_fire_cells / max(total_fire_cells, 1)),
            "cutoff_time": cutoff_time,
        }
        return fire_binary

    def _compute_sdf(self):
        if self.fire_binary_map is None:
            raise RuntimeError("Cannot compute SDF: fire binary map is not initialized")
        dist_outside = distance_transform_edt(1 - self.fire_binary_map)
        dist_inside = distance_transform_edt(self.fire_binary_map)
        self.sdf = dist_outside - dist_inside

    def _compute_thermal_field(self):
        if self.fire_binary_map is None:
            raise RuntimeError(
                "Cannot compute thermal field: fire binary map is not initialized"
            )
        intensity_map = self.data.get("intensity")
        if intensity_map is None:
            raise RuntimeError("Cannot compute thermal field: missing intensity data")

        current_fire = np.zeros_like(intensity_map)
        current_fire[self.fire_binary_map > 0] = intensity_map[self.fire_binary_map > 0]
        self.thermal_field = gaussian_filter(current_fire, sigma=60)
        self.thermal_field = np.clip(self.thermal_field * 800.0, 0, 100)

    def detect_fire_boundary(
        self,
        time_step: int = 0,
        fire_threshold: Optional[float] = None,
        init_percentile: Optional[float] = None,
        init_area_percent: Optional[float] = None,
        start_sim_time: Optional[float] = None,
    ) -> List[Tuple[int, int]]:
        if fire_threshold is None:
            fire_threshold = self.norm_params["fire_threshold"]

        intensity_map = self.get_full_map("intensity")
        if intensity_map is None:
            return []

        base_binary = (intensity_map > fire_threshold).astype(np.uint8)
        time_map = self.get_full_map("time")
        area_percent = init_area_percent if init_area_percent is not None else init_percentile
        self.last_init_area_stats = None

        if time_map is not None and time_step < 999999:
            valid_times = time_map[base_binary > 0]
            if len(valid_times) > 0:
                nonnegative_times = valid_times[valid_times >= 0]
                min_time = (
                    np.min(nonnegative_times)
                    if len(nonnegative_times) > 0
                    else 0.0
                )
                max_time = np.max(valid_times)
                time_range = max_time - min_time
                sim_time_delta = (time_range * 0.8) / 800.0 if time_range > 0 else 1.0
                if start_sim_time is not None:
                    current_sim_time = float(start_sim_time) + time_step * sim_time_delta
                elif area_percent is not None and time_step == 0:
                    fire_binary = self._select_fire_by_area_percent(
                        base_binary,
                        time_map,
                        float(area_percent),
                    )
                    self.fire_binary_map = fire_binary
                    eroded = ndimage.binary_erosion(fire_binary)
                    boundary = fire_binary - eroded
                    boundary_points = np.argwhere(boundary > 0)
                    return [tuple(point) for point in boundary_points]
                else:
                    current_sim_time = min_time + time_step * sim_time_delta
                current_sim_time = min(float(current_sim_time), float(max_time))
                self.last_boundary_sim_time = current_sim_time
                valid_time_mask = (time_map <= current_sim_time) & (time_map >= -1)
                fire_binary = base_binary & valid_time_mask.astype(np.uint8)
            else:
                self.last_boundary_sim_time = None
                fire_binary = base_binary
        else:
            self.last_boundary_sim_time = None
            fire_binary = base_binary

        self.fire_binary_map = fire_binary
        eroded = ndimage.binary_erosion(fire_binary)
        boundary = fire_binary - eroded
        boundary_points = np.argwhere(boundary > 0)
        return [tuple(point) for point in boundary_points]

    def get_sdf_value(self, row: int, col: int) -> float:
        if getattr(self, "sdf", None) is None or not self._check_bounds(row, col):
            return 0.0
        return float(self.sdf[row, col])

    def get_thermal_value(self, row: int, col: int) -> float:
        if getattr(self, "thermal_field", None) is None or not self._check_bounds(
            row, col
        ):
            return 0.0
        return float(self.thermal_field[row, col])

    def get_local_thermal_gradient(self, row: int, col: int) -> Tuple[float, float]:
        if getattr(self, "thermal_field", None) is None or not self._check_bounds(
            row, col
        ):
            return 0.0, 0.0

        curr_heat = self.get_thermal_value(row, col)
        if curr_heat < 0.5:
            return 0.0, 0.0

        h_up = (
            self.get_thermal_value(row - 1, col)
            if self._check_bounds(row - 1, col)
            else curr_heat
        )
        h_down = (
            self.get_thermal_value(row + 1, col)
            if self._check_bounds(row + 1, col)
            else curr_heat
        )
        h_left = (
            self.get_thermal_value(row, col - 1)
            if self._check_bounds(row, col - 1)
            else curr_heat
        )
        h_right = (
            self.get_thermal_value(row, col + 1)
            if self._check_bounds(row, col + 1)
            else curr_heat
        )

        dy = h_down - h_up
        dx = h_right - h_left
        norm = np.sqrt(dy**2 + dx**2)
        if norm > 1e-6:
            return float(dy / norm), float(dx / norm)
        return 0.0, 0.0

    def get_circular_neighborhood(
        self,
        row: int,
        col: int,
        radius: int,
        time_step: int = 0,
    ) -> Optional[Dict[str, np.ndarray]]:
        if not self._check_bounds(row, col):
            return None

        row_start = max(0, row - radius)
        row_end = min(self.shape[0], row + radius + 1)
        col_start = max(0, col - radius)
        col_end = min(self.shape[1], col + radius + 1)

        view_height = row_end - row_start
        view_width = col_end - col_start
        center_row_local = row - row_start
        center_col_local = col - col_start

        y_grid, x_grid = np.ogrid[:view_height, :view_width]
        distances = np.sqrt(
            (y_grid - center_row_local) ** 2 + (x_grid - center_col_local) ** 2
        )
        circular_mask = distances <= radius

        neighborhood = {}
        for key, data in self.data.items():
            if data is None:
                neighborhood[key] = None
                continue
            if len(data.shape) == 2:
                view = data[row_start:row_end, col_start:col_end].copy()
                view[~circular_mask] = 0
                neighborhood[key] = view
            elif len(data.shape) == 3:
                if key == "time":
                    if time_step < data.shape[0]:
                        view = data[
                            time_step, row_start:row_end, col_start:col_end
                        ].copy()
                        view[~circular_mask] = 0
                        neighborhood[key] = view
                else:
                    view = data[0, row_start:row_end, col_start:col_end].copy()
                    view[~circular_mask] = 0
                    neighborhood[key] = view

        neighborhood["circular_mask"] = circular_mask
        neighborhood["center_local"] = (center_row_local, center_col_local)
        if getattr(self, "fire_binary_map", None) is not None:
            view = self.fire_binary_map[row_start:row_end, col_start:col_end].copy()
            view[~circular_mask] = 0
            neighborhood["fire_binary_mask"] = view
        return neighborhood

    def get_local_fire_info(
        self, row: int, col: int, radius: int, time_step: int = 0
    ) -> Dict:
        neighborhood = self.get_circular_neighborhood(row, col, radius, time_step)
        if neighborhood is None or "intensity" not in neighborhood:
            return {
                "fire_count": 0,
                "boundary_count": 0,
                "avg_intensity": 0.0,
                "max_intensity": 0.0,
                "fire_direction": (0.0, 0.0),
                "nearest_fire_distance": float("inf"),
            }

        intensity = neighborhood["intensity"]
        mask = neighborhood["circular_mask"]
        fire_threshold = self.norm_params["fire_threshold"]
        if "fire_binary_mask" in neighborhood:
            fire_mask = (neighborhood["fire_binary_mask"] > 0) & mask
        else:
            fire_mask = (intensity > fire_threshold) & mask

        fire_count = np.sum(fire_mask)
        if fire_count > 0:
            fire_binary_local = fire_mask.astype(np.uint8)
            eroded_local = ndimage.binary_erosion(fire_binary_local)
            boundary_local = fire_binary_local - eroded_local
            boundary_count = np.sum(boundary_local)
            fire_positions = np.argwhere(fire_mask)
            center = neighborhood["center_local"]
            centroid = np.mean(fire_positions, axis=0)
            fire_direction = (centroid[0] - center[0], centroid[1] - center[1])
            distances = np.sqrt(
                (fire_positions[:, 0] - center[0]) ** 2
                + (fire_positions[:, 1] - center[1]) ** 2
            )
            nearest_distance = np.min(distances)
        else:
            boundary_count = 0
            fire_direction = (0.0, 0.0)
            nearest_distance = float("inf")

        return {
            "fire_count": int(fire_count),
            "boundary_count": int(boundary_count),
            "avg_intensity": float(np.mean(intensity[fire_mask]))
            if fire_count > 0
            else 0.0,
            "max_intensity": float(np.max(intensity[fire_mask]))
            if fire_count > 0
            else 0.0,
            "fire_direction": fire_direction,
            "nearest_fire_distance": float(nearest_distance),
        }

    def get_wind_effect(
        self, row: int, col: int, movement_direction: Tuple[int, int]
    ) -> Dict[str, float]:
        if not self._check_bounds(row, col):
            return {"wind_resistance": 0.0, "battery_penalty": 0.0}

        wind_speed = (
            float(self.data["wind_speed"][row, col])
            if "wind_speed" in self.data
            else 0.0
        )
        wind_direction = (
            float(self.data["wind_direction"][row, col])
            if "wind_direction" in self.data
            else 0.0
        )
        if movement_direction == (0, 0):
            return {"wind_resistance": 0.0, "battery_penalty": 0.0}

        move_angle = np.arctan2(movement_direction[1], -movement_direction[0])
        move_angle_deg = np.degrees(move_angle) % 360
        angle_diff = abs(wind_direction - move_angle_deg)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        angle_diff_rad = np.radians(angle_diff)
        wind_resistance = np.cos(angle_diff_rad)
        normalized_wind_speed = wind_speed / self.norm_params["wind_speed_max"]

        if angle_diff < 90:
            battery_penalty = normalized_wind_speed * wind_resistance * 0.5
        else:
            battery_penalty = 0.0

        return {
            "wind_resistance": float(wind_resistance),
            "battery_penalty": float(max(0.0, battery_penalty)),
            "wind_speed_norm": float(normalized_wind_speed),
            "angle_diff": float(angle_diff),
        }

    def check_boundary_closure(
        self, discovered_boundary: Set[Tuple[int, int]], closure_threshold: float = 0.8
    ) -> Dict:
        if self.boundary_points is None or len(self.boundary_points) == 0:
            return {
                "is_closed": False,
                "coverage": 0.0,
                "total_boundary": 0,
                "discovered": 0,
            }
        total_boundary = len(self.boundary_points)
        discovered_count = len(discovered_boundary)
        coverage = discovered_count / total_boundary if total_boundary > 0 else 0.0
        return {
            "is_closed": coverage >= closure_threshold,
            "coverage": coverage,
            "total_boundary": total_boundary,
            "discovered": discovered_count,
        }

    def get_normalized_features(
        self, row: int, col: int, time_step: int = 0
    ) -> Optional[Dict]:
        if not self._check_bounds(row, col):
            return None

        features = {}
        if "intensity" in self.data:
            intensity = float(self.data["intensity"][row, col])
            if (
                getattr(self, "fire_binary_map", None) is not None
                and self.fire_binary_map[row, col] == 0
            ):
                intensity = 0.0
            features["intensity_norm"] = intensity / self.norm_params["intensity_max"]
        if "dem" in self.data:
            dem = float(self.data["dem"][row, col])
            denom = max(
                float(self.norm_params["dem_max"] - self.norm_params["dem_min"]), 1.0
            )
            features["dem_norm"] = (dem - self.norm_params["dem_min"]) / denom
        if "slope" in self.data:
            features["slope_norm"] = (
                float(self.data["slope"][row, col]) / self.norm_params["slope_max"]
            )
        if "wind_speed" in self.data:
            features["wind_speed_norm"] = (
                float(self.data["wind_speed"][row, col])
                / self.norm_params["wind_speed_max"]
            )
        if "wind_direction" in self.data:
            wind_dir_rad = np.radians(float(self.data["wind_direction"][row, col]))
            features["wind_dir_sin"] = np.sin(wind_dir_rad)
            features["wind_dir_cos"] = np.cos(wind_dir_rad)
        if getattr(self, "sdf", None) is not None:
            features["sdf_norm"] = np.tanh(self.get_sdf_value(row, col) / 50.0)
        if getattr(self, "thermal_field", None) is not None:
            features["thermal_norm"] = np.tanh(self.get_thermal_value(row, col) / 50.0)
        return features

    def get_cell_info(self, row: int, col: int, time_step: int = 0) -> Optional[Dict]:
        if not self._check_bounds(row, col):
            return None
        info = {"row": row, "col": col, "coordinates": self.get_coordinates(row, col)}
        for key, data in self.data.items():
            if data is None:
                info[key] = None
            elif len(data.shape) > 2:
                if key == "time":
                    info[key] = (
                        float(data[time_step, row, col])
                        if time_step < data.shape[0]
                        else None
                    )
                else:
                    info[key] = float(data[0, row, col])
            else:
                info[key] = float(data[row, col])
        return info

    def get_coordinates(self, row: int, col: int) -> Optional[Tuple[float, float]]:
        if self.transform is None:
            return None
        x, y = rasterio.transform.xy(self.transform, row, col)
        return (x, y)

    def _check_bounds(self, row: int, col: int) -> bool:
        if self.shape is None:
            return False
        return 0 <= row < self.shape[0] and 0 <= col < self.shape[1]

    def get_full_map(self, variable: str, time_step: int = 0) -> Optional[np.ndarray]:
        if variable not in self.data:
            return None
        data = self.data[variable]
        if data is None:
            return None
        if len(data.shape) > 2:
            return data[time_step].copy() if time_step < data.shape[0] else None
        return data.copy()


class SceneManager:
    """Scene manager for train/validation/generalization/stress splits."""

    def __init__(
        self,
        base_dir: str = "./dataset",
        scene_keys_by_split: Optional[Dict[str, List[str]]] = None,
    ):
        self.base_dir = base_dir
        self.dataset_index = DatasetIndex(base_dir)
        self.cache: Dict[str, FireEnvironmentData] = {}
        scene_keys_by_split = scene_keys_by_split or {}
        self.scene_keys_by_split: Dict[str, List[str]] = {}
        for split in ["train", "validation", "generalization", "stress"]:
            override = scene_keys_by_split.get(split)
            self.scene_keys_by_split[split] = (
                [str(key) for key in override]
                if override
                else self.dataset_index.scene_keys(split)
            )
        self.train_scenes = self.scene_keys_by_split["train"]
        self.validation_scenes = self.scene_keys_by_split["validation"]
        self.generalization_scenes = self.scene_keys_by_split["generalization"]
        self.stress_scenes = self.scene_keys_by_split["stress"]

    def get_scene(self, mode: str = "train") -> FireEnvironmentData:
        split = self.dataset_index.normalize_mode(mode)
        scene_key = str(np.random.choice(self.scene_keys_by_split[split]))
        return self.get_specific_scene(scene_key)

    def get_specific_scene(self, scene_key: str) -> FireEnvironmentData:
        scene_key = str(scene_key)
        if scene_key not in self.cache:
            record = self.dataset_index.get_record(scene_key)
            self.cache[scene_key] = FireEnvironmentData(
                self.base_dir,
                scene_key=scene_key,
                scene_record=record,
                dataset_index=self.dataset_index,
            )
        return self.cache[scene_key]


def validate_scene_boundaries(
    base_dir: str = "./dataset",
    scene_keys: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
    init_percentile: Optional[float] = 5.0,
    init_area_percent: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Optional[float]]]:
    dataset_index = DatasetIndex(base_dir)
    if scene_keys is None:
        scene_keys = []
        for split in splits or ["train", "validation", "generalization", "stress"]:
            scene_keys.extend(dataset_index.scene_keys(split))
    scene_keys = [str(key) for key in scene_keys]

    counts: Dict[str, Dict[str, Optional[float]]] = {}
    invalid_messages: List[str] = []

    if verbose:
        print("\n" + "=" * 60)
        print(f"Dataset Preflight: validate scenes ({len(scene_keys)})")
        print("=" * 60)

    for scene_key in scene_keys:
        try:
            missing = [
                f"{label}: {path}"
                for label, path in dataset_index.required_file_paths(scene_key)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError("; ".join(missing))

            record = dataset_index.get_record(scene_key)
            scene = FireEnvironmentData(
                base_dir,
                scene_key=scene_key,
                scene_record=record,
                dataset_index=dataset_index,
            )
            t0_count = len(scene.boundary_points or [])
            area_percent = init_area_percent if init_area_percent is not None else init_percentile
            init_count = None
            init_stats = {}
            if area_percent is not None:
                init_count = len(
                    scene.detect_fire_boundary(
                        time_step=0,
                        init_area_percent=float(area_percent),
                    )
                )
                init_stats = scene.last_init_area_stats or {}
            counts[scene_key] = {
                "t0_boundary_points": t0_count,
                "init_percentile": init_percentile,
                "init_area_percent": area_percent,
                "total_fire_cells": init_stats.get("total_fire_cells"),
                "init_fire_cells": init_stats.get("init_fire_cells"),
                "actual_init_area_percent": init_stats.get("actual_init_area_percent"),
                "cutoff_time": init_stats.get("cutoff_time"),
                "init_boundary_points": init_count,
            }
            if verbose:
                if init_count is None:
                    print(f"  {scene_key}: t=0 boundary points = {t0_count}")
                else:
                    print(
                        f"  {scene_key}: t=0 boundary points = {t0_count} | "
                        f"init_area{float(area_percent):g}% boundary points = {init_count} | "
                        f"actual_area={init_stats.get('actual_init_area_percent', 0.0):.2f}%"
                    )
            if t0_count == 0:
                invalid_messages.append(f"{scene_key}: empty t=0 boundary")
            if init_count is not None and init_count == 0:
                invalid_messages.append(
                    f"{scene_key}: empty init_area{float(area_percent):g}% boundary"
                )
        except Exception as exc:
            invalid_messages.append(f"{scene_key}: {exc}")
            if verbose:
                print(f"  {scene_key}: INVALID | {exc}")

    if invalid_messages:
        raise InvalidSceneError(
            "Dataset preflight failed. Invalid scenes detected: "
            + "; ".join(invalid_messages)
        )
    return counts


if __name__ == "__main__":
    manager = SceneManager(base_dir="./dataset")
    print("Train scenes:", len(manager.train_scenes))
    print("Validation scenes:", len(manager.validation_scenes))
    print("Generalization scenes:", len(manager.generalization_scenes))
    print("Stress scenes:", len(manager.stress_scenes))
    validate_scene_boundaries(base_dir="./dataset", verbose=True)
