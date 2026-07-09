from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import rasterio
from scipy import ndimage
from scipy.ndimage import gaussian_filter


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
        scene_dir = self.scene_dir(scene_key)
        record["scene_key"] = scene_key
        record["scene_dir_abs"] = str(scene_dir)
        metadata_path = Path(str(record.get("metadata", "metadata.json")).replace("\\", "/"))
        if not metadata_path.is_absolute():
            metadata_path = scene_dir / metadata_path
        record["metadata_abs"] = str(metadata_path.resolve())
        if record.get("static_map"):
            static_map_path = Path(str(record["static_map"]).replace("\\", "/"))
            if not static_map_path.is_absolute():
                static_map_path = self.source_root / static_map_path
            record["static_map_abs"] = str(static_map_path.resolve())
        record["rasters_abs"] = {}
        for key, rel_path in record.get("rasters", {}).items():
            raster_path = Path(str(rel_path).replace("\\", "/"))
            if not raster_path.is_absolute():
                raster_path = scene_dir / raster_path
            record["rasters_abs"][key] = str(raster_path.resolve())
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
            ("static_map", record.get("static_map")),
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
                if label == "static_map":
                    path = self.source_root / path
                else:
                    path = scene_dir / path
            paths.append((label, path.resolve()))
        return paths


class _BoundaryPointsAccessor:
    def __init__(self, owner: "FireSceneData"):
        self.owner = owner

    def __call__(self, time_step: int = 0) -> List[Tuple[int, int]]:
        return self.owner._boundary_points_at(time_step)

    def __iter__(self):
        return iter(self.owner.boundary_points_cache)

    def __len__(self) -> int:
        return len(self.owner.boundary_points_cache)

    def __bool__(self) -> bool:
        return bool(self.owner.boundary_points_cache)

    def __getitem__(self, item):
        return self.owner.boundary_points_cache[item]


class FireSceneData:
    """Load one FARSITE scene from dataset_index.json."""

    CORE_KEYS = ["intensity", "length", "time", "speedRate"]
    EXTRA_RASTER_KEYS = ["spread_direction", "heat_per_unit_area", "crown_fire"]
    NORM_RASTER_PARAMS = {
        "intensity": "intensity_max",
        "length": "length_max",
        "speedRate": "speedRate_max",
        "spread_direction": "spread_direction_max",
        "heat_per_unit_area": "heat_per_unit_area_max",
        "crown_fire": "crown_fire_max",
    }
    NORM_ALIASES = {
        "flame_length": "length",
        "ros": "speedRate",
        "heat": "heat_per_unit_area",
    }
    STATIC_BAND_KEYS = [
        "elevation",
        "slope",
        "aspect",
        "fuel_model",
        "canopy_cover",
        "canopy_height",
        "canopy_base_height",
        "canopy_bulk_density",
    ]

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
        self.resolution_m = float(self.metadata.get("resolution_m", 1.0))
        uav = self.metadata.get("uav", {})
        self.sensor_radius_m = float(uav.get("sensor_radius_m", 0.0))
        self.sensor_radius_cells = (
            int(np.ceil(self.sensor_radius_m / self.resolution_m))
            if self.resolution_m > 0
            else 0
        )
        self.max_steps = int(uav.get("max_steps", 0))

        self.data: Dict[str, np.ndarray] = {}
        self.static_map = None
        self.static_bands: Dict[str, np.ndarray] = {}
        self.transform = None
        self.crs = None
        self.shape = None
        self.nodata_value = None

        self.norm_params = {
            "intensity_max": 1.0,
            "length_max": 1.0,
            "speedRate_max": 1.0,
            "spread_direction_max": 360.0,
            "heat_per_unit_area_max": 1.0,
            "crown_fire_max": 1.0,
            "dem_min": 0.0,
            "dem_max": 1.0,
            "slope_max": 1.0,
            "wind_speed_max": 50.49,
            "fire_threshold": 1.0,
        }

        self._boundary_points: List[Tuple[int, int]] = []
        self._boundary_points_accessor = _BoundaryPointsAccessor(self)
        self.fire_binary_map = None
        self.thermal_field = None
        self._thermal_field_unclipped = None
        self._nav_field = None
        self.last_boundary_sim_time = None
        self.training_start_sim_time = None
        self.last_init_area_stats = None
        self.is_valid_scene = True
        self.invalid_reason = None

        self.load_all_data()
        self._initialize_boundary()
        self._compute_thermal_field()

    @property
    def boundary_points(self) -> _BoundaryPointsAccessor:
        accessor = getattr(self, "_boundary_points_accessor", None)
        if accessor is None:
            accessor = _BoundaryPointsAccessor(self)
            self._boundary_points_accessor = accessor
        return accessor

    @boundary_points.setter
    def boundary_points(self, points):
        self._set_boundary_points(points)

    @property
    def boundary_points_cache(self) -> List[Tuple[int, int]]:
        return list(getattr(self, "_boundary_points", []))

    def _set_boundary_points(self, points):
        if points is None:
            self._boundary_points = []
            return
        self._boundary_points = [
            (int(point[0]), int(point[1]))
            for point in points
        ]

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

    def _path_from_source_root(self, rel_path: str) -> Path:
        path = Path(str(rel_path).replace("\\", "/"))
        if path.is_absolute():
            return path
        return (self.dataset_index.source_root / path).resolve()

    def _build_file_paths(self) -> Dict[str, Path]:
        rasters = self.scene_record.get("rasters", {})
        file_paths: Dict[str, Path] = {}
        static_map = self.scene_record.get("static_map")
        if not static_map:
            raise KeyError(f"Scene {self.scene_key} missing static_map in dataset_index.json")
        file_paths["static_map"] = self._path_from_source_root(static_map)
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
                data = src.read().astype(np.float32)
                if src.nodata is not None:
                    data[data == src.nodata] = 0
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                data[data < 0] = 0
                if data.shape[0] == 1:
                    data = data[0]
                metadata = {
                    "transform": src.transform,
                    "crs": src.crs,
                    "shape": (src.height, src.width),
                    "nodata": src.nodata,
                    "bounds": src.bounds,
                    "count": src.count,
                    "descriptions": src.descriptions,
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

    def _load_static_map(self):
        filepath = self.file_paths["static_map"]
        if not filepath.is_file():
            raise FileNotFoundError(f"Static map missing for {self.scene_key}: {filepath}")

        data, metadata = self.load_raster(filepath)
        if len(data.shape) != 3 or data.shape[0] != len(self.STATIC_BAND_KEYS):
            raise RuntimeError(
                f"Static map must have {len(self.STATIC_BAND_KEYS)} bands for "
                f"{self.scene_key}: {filepath} has shape {data.shape}"
            )

        self.static_map = data
        self.transform = metadata["transform"]
        self.crs = metadata["crs"]
        self.shape = metadata["shape"]
        self.nodata_value = metadata["nodata"]
        self.static_band_descriptions = metadata.get("descriptions", ())
        for band_index, key in enumerate(self.STATIC_BAND_KEYS):
            band = data[band_index]
            self.static_bands[key] = band
            self.data[key] = band
        self.data["dem"] = self.static_bands["elevation"]

    def _assert_raster_shape(self, key: str, filepath: Path, shape: Tuple[int, int]):
        if shape != self.shape:
            static_path = self.file_paths.get("static_map")
            raise RuntimeError(
                f"Raster shape mismatch in {self.scene_key}: "
                f"static_map {static_path} has {self.shape}; "
                f"{key} {filepath} has {shape}"
            )

    @staticmethod
    def _positive_values(data: np.ndarray) -> np.ndarray:
        values = np.asarray(data, dtype=np.float32)
        return values[np.isfinite(values) & (values > 0)]

    @staticmethod
    def _clip01(value):
        return np.clip(value, 0.0, 1.0)

    def _percentile_scale(
        self,
        raster_key: str,
        percentile: float = 99.5,
        min_value: float = 1.0,
        clamp_range: Optional[Tuple[float, float]] = None,
    ) -> float:
        values = self._positive_values(self.data.get(raster_key, np.array([])))
        if values.size:
            scale = float(np.percentile(values, percentile))
        else:
            scale = float(min_value)
        if clamp_range is not None:
            scale = float(np.clip(scale, clamp_range[0], clamp_range[1]))
        return max(scale, float(min_value))

    def _derive_norm_params(self):
        dem_values = self._positive_values(self.data.get("dem", np.array([])))
        if dem_values.size:
            dem_min = float(np.min(dem_values))
            dem_max = float(np.max(dem_values))
        else:
            dem_min, dem_max = 0.0, 1.0
        if dem_max <= dem_min:
            dem_max = dem_min + 1.0

        slope_values = self._positive_values(self.data.get("slope", np.array([])))
        slope_max = float(np.max(slope_values)) if slope_values.size else 1.0

        wind_values = self._positive_values(self.data.get("wind_speed", np.array([])))
        wind_speed_max = float(np.max(wind_values)) if wind_values.size else 1.0
        wind = self.metadata.get("wind", {})
        for key in ["wind_speed_mph", "peak_wind_speed_mph"]:
            if key in wind:
                wind_speed_max = max(wind_speed_max, float(wind[key]))
        for key in ["wind_speed_range_mph", "expected_wind_speed_range_mph"]:
            if key in wind and wind[key]:
                wind_speed_max = max(wind_speed_max, float(max(wind[key])))

        norm_params = {
            "intensity_max": self._percentile_scale(
                "intensity", clamp_range=(500.0, 8000.0)
            ),
            "length_max": self._percentile_scale("length"),
            "speedRate_max": self._percentile_scale("speedRate"),
            "spread_direction_max": self._percentile_scale(
                "spread_direction", min_value=360.0
            ),
            "heat_per_unit_area_max": self._percentile_scale("heat_per_unit_area"),
            "crown_fire_max": self._percentile_scale("crown_fire"),
            "dem_min": dem_min,
            "dem_max": dem_max,
            "slope_max": max(slope_max, 1.0),
            "wind_speed_max": max(wind_speed_max, 1.0),
            "fire_threshold": 1.0,
        }
        norm_params["flame_length_max"] = norm_params["length_max"]
        norm_params["ros_max"] = norm_params["speedRate_max"]
        norm_params["heat_max"] = norm_params["heat_per_unit_area_max"]
        self.norm_params = norm_params

    def _log_norm_params(self):
        keys = [
            "intensity_max",
            "length_max",
            "speedRate_max",
            "heat_per_unit_area_max",
            "crown_fire_max",
            "wind_speed_max",
        ]
        summary = ", ".join(f"{key}={self.norm_params[key]:.4g}" for key in keys)
        print(f"Scene {self.scene_key} norm_params | {summary}")

    def normalized_map(self, variable: str) -> np.ndarray:
        key = self.NORM_ALIASES.get(str(variable), str(variable))
        if key not in self.data:
            raise KeyError(f"Unknown scene raster for normalization: {variable}")
        data = np.asarray(self.data[key], dtype=np.float32)
        if key == "dem":
            denom = max(
                float(self.norm_params["dem_max"] - self.norm_params["dem_min"]),
                1.0,
            )
            return self._clip01((data - float(self.norm_params["dem_min"])) / denom).astype(
                np.float32
            )
        param_key = self.NORM_RASTER_PARAMS.get(key)
        if key == "slope":
            param_key = "slope_max"
        elif key == "wind_speed":
            param_key = "wind_speed_max"
        if param_key is None:
            raise KeyError(f"No normalization parameter configured for: {variable}")
        denom = max(float(self.norm_params[param_key]), 1.0)
        return self._clip01(data / denom).astype(np.float32)

    def load_all_data(self):
        print(f"Loading scene {self.scene_key}...")

        files_loaded = 0
        self._load_static_map()
        files_loaded += 1

        for key in self.CORE_KEYS:
            filepath = self.file_paths.get(key)
            if filepath is None or not filepath.is_file():
                raise FileNotFoundError(
                    f"Required raster missing for {self.scene_key}: {filepath}"
                )

            data, metadata = self.load_raster(filepath)
            self._assert_raster_shape(key, filepath, metadata["shape"])
            self.data[key] = data
            files_loaded += 1

        for key in self.EXTRA_RASTER_KEYS:
            filepath = self.file_paths.get(key)
            if filepath is not None and filepath.is_file():
                data, metadata = self.load_raster(filepath)
                self._assert_raster_shape(key, filepath, metadata["shape"])
                self.data[key] = data
                files_loaded += 1

        self._load_wind_fields()
        self._derive_norm_params()
        self._log_norm_params()

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
        boundary_points = self.detect_fire_boundary(time_step=0)
        self.training_start_sim_time = None
        if len(boundary_points) == 0:
            self.is_valid_scene = False
            self.invalid_reason = (
                f"Scene {self.scene_key} has empty t=0 fire boundary. "
                "Training must stop instead of falling back to the final-state boundary."
            )
            raise InvalidSceneError(self.invalid_reason)
        print(
            f"Scene {self.scene_key} t=0 boundary points: {len(boundary_points)}"
        )

    def initialize_training_boundary(
        self,
        init_percentile: Optional[float] = 5.0,
        init_area_percent: Optional[float] = None,
    ) -> List[Tuple[int, int]]:
        area_percent = init_area_percent if init_area_percent is not None else init_percentile
        if area_percent is None:
            boundary_points = self.detect_fire_boundary(time_step=0)
            self.training_start_sim_time = None
        else:
            boundary_points = self.detect_fire_boundary(
                time_step=0,
                init_area_percent=float(area_percent),
            )
            self.training_start_sim_time = self.last_boundary_sim_time

        if len(boundary_points) == 0:
            self.is_valid_scene = False
            self.invalid_reason = (
                f"Scene {self.scene_key} has empty training fire boundary "
                f"(init_area_percent={area_percent})."
            )
            raise InvalidSceneError(self.invalid_reason)
        return boundary_points

    def _select_fire_by_area_percent(
        self,
        base_binary: np.ndarray,
        time_map: np.ndarray,
        init_area_percent: float,
    ) -> np.ndarray:
        positive_times = time_map[(base_binary > 0) & (time_map >= 0)]
        min_time = float(np.min(positive_times)) if positive_times.size > 0 else 0.0
        valid_fire_mask = (base_binary > 0) & (time_map >= min_time)
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

    def _compute_thermal_field(self):
        """方案 C 热场语义重建：per-scene robust normalization → thermal_potential [0,1]。

        新链路：
          source = fire_mask * clip(intensity / intensity_ref, 0, 1)
          blur   = gaussian_filter(downsample(source), sigma=15)
          ref    = p99(blur[blur > eps])
          potential = clip(blur / ref, 0, 1)
          nav    = log1p(alpha * potential) / log1p(alpha)   # 给梯度用
        """
        if self.fire_binary_map is None:
            raise RuntimeError(
                "Cannot compute thermal field: fire binary map is not initialized"
            )

        fire_mask = np.ascontiguousarray(self.fire_binary_map > 0)
        height, width = fire_mask.shape

        if not np.any(fire_mask):
            self.thermal_field = np.zeros((height, width), dtype=np.float32)
            self._thermal_field_unclipped = np.zeros((height, width), dtype=np.float32)
            self._nav_field = np.zeros((height, width), dtype=np.float32)
            return

        # --- per-scene robust normalization ---
        intensity_map = self.data.get("intensity")
        if intensity_map is None:
            raise RuntimeError("Cannot compute thermal field: missing intensity data")

        intensity_ref = max(float(self.norm_params.get("intensity_max", 1.0)), 1.0)
        source = np.zeros_like(intensity_map, dtype=np.float32)
        source[fire_mask] = np.clip(
            intensity_map[fire_mask].astype(np.float32) / intensity_ref, 0.0, 1.0
        )

        # --- downsample + gaussian blur ---
        small_source = cv2.resize(source, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        small_blur = gaussian_filter(small_source, sigma=15, truncate=4.0)

        # --- upsample back to full resolution ---
        blur_full = cv2.resize(small_blur, (width, height), interpolation=cv2.INTER_LINEAR)
        blur_full = np.maximum(blur_full, 0.0)

        # --- robust potential normalization ---
        eps = 1e-8
        positive_mask = blur_full > eps
        if np.any(positive_mask):
            ref = float(np.percentile(blur_full[positive_mask], 99.0))
        else:
            ref = eps
        ref = max(ref, eps)

        potential_unclipped = blur_full / ref
        self._thermal_field_unclipped = potential_unclipped
        self.thermal_field = np.clip(potential_unclipped, 0.0, 1.0).astype(np.float32)

        # --- log-compressed navigation field for gradient computation ---
        alpha = 20.0
        self._nav_field = (
            np.log1p(alpha * potential_unclipped) / np.log1p(alpha)
        ).astype(np.float32)

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
                    points = [tuple(point) for point in boundary_points]
                    self._set_boundary_points(points)
                    return points
                else:
                    current_sim_time = min_time + time_step * sim_time_delta
                current_sim_time = min(float(current_sim_time), float(max_time))
                self.last_boundary_sim_time = current_sim_time
                valid_time_mask = (time_map <= current_sim_time) & (time_map >= min_time)
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
        points = [tuple(point) for point in boundary_points]
        self._set_boundary_points(points)
        return points

    def _boundary_points_at(self, time_step: int = 0) -> List[Tuple[int, int]]:
        return self.detect_fire_boundary(time_step=time_step)

    def current_fire(self, time_step: int = 0) -> np.ndarray:
        self.detect_fire_boundary(time_step=time_step)
        if self.fire_binary_map is None:
            return np.zeros(self.shape, dtype=np.uint8)
        return self.fire_binary_map.astype(np.uint8).copy()

    def active_front(self, time_step: int = 0) -> np.ndarray:
        fire_binary = self.current_fire(time_step=time_step)
        eroded = ndimage.binary_erosion(fire_binary)
        return (fire_binary - eroded).astype(np.uint8)

    def severity_map(self) -> np.ndarray:
        if "intensity" not in self.data:
            return np.zeros(self.shape, dtype=np.float32)
        intensity_score = self.normalized_map("intensity")
        length_score = self.normalized_map("length")
        ros_score = self.normalized_map("speedRate")
        heat_score = self.normalized_map("heat_per_unit_area")
        crown_score = self.normalized_map("crown_fire")
        severity = (
            0.35 * intensity_score
            + 0.20 * length_score
            + 0.20 * ros_score
            + 0.15 * heat_score
            + 0.10 * crown_score
        )
        return np.clip(severity, 0.0, 1.0).astype(np.float32)

    def get_thermal_value(self, row: int, col: int) -> float:
        if getattr(self, "thermal_field", None) is None or not self._check_bounds(
            row, col
        ):
            return 0.0
        return float(self.thermal_field[row, col])

    def _get_nav_value(self, row: int, col: int) -> float:
        """从 log 压缩后的导航场取值，供梯度计算使用。"""
        if getattr(self, "_nav_field", None) is None or not self._check_bounds(row, col):
            return 0.0
        return float(self._nav_field[row, col])

    def get_local_thermal_gradient(self, row: int, col: int) -> Tuple[float, float]:
        """从 nav_field（log 压缩势场）计算局部梯度，避免高值区梯度消失。"""
        if getattr(self, "_nav_field", None) is None or not self._check_bounds(
            row, col
        ):
            return 0.0, 0.0

        curr_heat = self.get_thermal_value(row, col)
        if curr_heat < 0.005:
            return 0.0, 0.0

        h_up = (
            self._get_nav_value(row - 1, col)
            if self._check_bounds(row - 1, col)
            else self._get_nav_value(row, col)
        )
        h_down = (
            self._get_nav_value(row + 1, col)
            if self._check_bounds(row + 1, col)
            else self._get_nav_value(row, col)
        )
        h_left = (
            self._get_nav_value(row, col - 1)
            if self._check_bounds(row, col - 1)
            else self._get_nav_value(row, col)
        )
        h_right = (
            self._get_nav_value(row, col + 1)
            if self._check_bounds(row, col + 1)
            else self._get_nav_value(row, col)
        )

        dy = h_down - h_up
        dx = h_right - h_left
        norm = np.sqrt(dy**2 + dx**2)
        if norm > 1e-6:
            return float(dy / norm), float(dx / norm)
        return 0.0, 0.0

    def diagnose_thermal_health(self) -> Dict:
        """热场健康诊断。训练前应通过此检查确认热场语义层正常。"""
        field = getattr(self, "thermal_field", None)
        if field is None:
            return {"status": "no_field", "sat_ratio": 1.0}

        total = field.size
        if total == 0:
            return {"status": "empty_field", "sat_ratio": 1.0}

        sat_count = int(np.sum(field >= 0.999))
        high_count = int(np.sum(field >= 0.8))
        nonzero = field > 0.001
        nonzero_count = int(np.sum(nonzero))

        # 高热区零梯度比例
        high_mask = field >= 0.5
        high_cells = int(np.sum(high_mask))
        zero_grad_count = 0
        if high_cells > 0 and getattr(self, "_nav_field", None) is not None:
            h, w = field.shape
            padded = np.pad(self._nav_field, 1, mode="edge")
            gy = padded[2:h+2, 1:w+1] - padded[0:h, 1:w+1]
            gx = padded[1:h+1, 2:w+2] - padded[1:h+1, 0:w]
            grad_norm = np.sqrt(gy**2 + gx**2)
            zero_grad_count = int(np.sum((high_mask) & (grad_norm < 1e-6)))

        potential_vals = field[nonzero] if nonzero_count > 0 else np.array([0.0])

        return {
            "status": "ok",
            "sat_ratio": float(sat_count / total),
            "high_ratio": float(high_count / total),
            "nonzero_ratio": float(nonzero_count / total),
            "zero_grad_in_high_ratio": float(zero_grad_count / max(high_cells, 1)),
            "potential_q50": float(np.percentile(potential_vals, 50)),
            "potential_q90": float(np.percentile(potential_vals, 90)),
            "potential_q99": float(np.percentile(potential_vals, 99)),
            "field_min": float(field.min()),
            "field_max": float(field.max()),
        }

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
        normalized_wind_speed = float(
            self._clip01(wind_speed / max(float(self.norm_params["wind_speed_max"]), 1.0))
        )

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
            features["intensity_norm"] = float(
                self._clip01(intensity / max(float(self.norm_params["intensity_max"]), 1.0))
            )
        if "dem" in self.data:
            dem = float(self.data["dem"][row, col])
            denom = max(
                float(self.norm_params["dem_max"] - self.norm_params["dem_min"]), 1.0
            )
            features["dem_norm"] = float(
                self._clip01((dem - float(self.norm_params["dem_min"])) / denom)
            )
        if "slope" in self.data:
            features["slope_norm"] = float(
                self._clip01(
                    float(self.data["slope"][row, col])
                    / max(float(self.norm_params["slope_max"]), 1.0)
                )
            )
        if "wind_speed" in self.data:
            features["wind_speed_norm"] = float(
                self._clip01(
                    float(self.data["wind_speed"][row, col])
                    / max(float(self.norm_params["wind_speed_max"]), 1.0)
                )
            )
        if "wind_direction" in self.data:
            wind_dir_rad = np.radians(float(self.data["wind_direction"][row, col]))
            features["wind_dir_sin"] = np.sin(wind_dir_rad)
            features["wind_dir_cos"] = np.cos(wind_dir_rad)
        if getattr(self, "thermal_field", None) is not None:
            features["thermal_norm"] = float(
                self._clip01(self.get_thermal_value(row, col))
            )
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


class FireEnvironmentData(FireSceneData):
    """Backward-compatible name for FireSceneData."""


class SceneManager:
    """Scene manager for train/validation/generalization/stress splits."""

    # 跨所有 SceneManager 实例共享的场景缓存，避免 evaluate() 每次
    # 创建新环境时重新读盘、重新计算归一化参数和初始边界。
    _shared_scene_cache: Dict[str, "FireEnvironmentData"] = {}

    def __init__(
        self,
        base_dir: str = "./dataset",
        scene_keys_by_split: Optional[Dict[str, List[str]]] = None,
    ):
        self.base_dir = base_dir
        self.dataset_index = DatasetIndex(base_dir)
        self.cache: Dict[str, FireEnvironmentData] = SceneManager._shared_scene_cache
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
