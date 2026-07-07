# Firemap Visual Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a 23-slide image-enhanced copy of the existing firemap deck, delete the canopy-data slide, and add visual evidence for FlamMap, FARSITE, raster outputs, JSON integration and reinforcement-learning use.

**Architecture:** Import the existing detailed PPTX as the only presentation source, remove its ninth slide, and revise the remaining slides with Artifact Tool. Generate truthful raster previews from the project GeoTIFFs using Rasterio CLI conversion, reuse the four user-provided software screenshots, and use one generated conceptual UAV/fire visual only where no real screenshot can explain the project goal.

**Tech Stack:** `@oai/artifact-tool`, JavaScript ES modules, Rasterio CLI, local GeoTIFF/FMS/WXS/JSON files, PowerPoint `.pptx`.

---

### Task 1: Prepare the working copy and visual assets

**Files:**
- Read: `outputs/森林火场地图创建与项目集成_详细版.pptx`
- Create: `outputs/森林火场地图创建与项目集成_图片增强版.pptx`
- Create: `C:/Users/cameliar/AppData/Local/Temp/codex-presentations/manual-firemap-template/firemap-deck-v4/tmp/assets/*`

- [ ] **Step 1: Create the v4 scratch workspace**

Initialize Artifact Tool package resolution under the v4 temporary workspace.

- [ ] **Step 2: Preserve the original deck**

Use the detailed deck as read-only input. The builder must export to the new `_图片增强版.pptx` path.

- [ ] **Step 3: Generate project raster previews**

Use Rasterio CLI to export scaled 8-bit PNG previews from:

```text
map6.tif bands 1–4: elevation, slope, aspect, fuel model
arrival_time.tif
ros_farsite.tif
spread_direction_farsite.tif
fireline_intensity_farsite.tif
flame_length_farsite.tif
heat_per_unit_area_farsite.tif
crown_fire_activity_farsite.tif
```

The previews must be generated from actual project files and labeled as project data.

- [ ] **Step 4: Prepare text-file evidence**

Use actual FMS, WXS, metadata and dataset-index excerpts as editable PowerPoint text blocks rather than fake screenshots.

- [ ] **Step 5: Generate one conceptual project-background image**

Create one non-photorealistic scientific illustration showing two UAVs tracking a forest-fire boundary. Do not present it as simulation evidence.

### Task 2: Build the 23-slide image-enhanced deck

**Files:**
- Create: `C:/Users/cameliar/AppData/Local/Temp/codex-presentations/manual-firemap-template/firemap-deck-v4/tmp/build_visual_deck.mjs`
- Create: `outputs/森林火场地图创建与项目集成_图片增强版.pptx`

- [ ] **Step 1: Import and copy the detailed deck**

Import the 24-slide detailed deck and delete slide 9, producing a 23-slide working presentation.

- [ ] **Step 2: Enhance slides 1–7**

Add the fire-raster cover treatment, conceptual UAV background visual, FlamMap/FARSITE screenshots, three-stage software relationship and input-file montage.

- [ ] **Step 3: Enhance slides 8–13**

Add the four static-map previews, actual FMS/WXS excerpts, ignition/burn-period evidence, annotated model-settings screenshot and output thumbnails.

- [ ] **Step 4: Enhance slides 14–16**

Add actual project raster previews for arrival time, spread rate, spread direction, fireline intensity, flame length, heat per unit area and crown-fire activity.

- [ ] **Step 5: Enhance slides 17–22**

Add directory evidence, JSON excerpts, JSON-to-environment loading logic, preprocessing comparison and reinforcement-learning fire-map composition.

- [ ] **Step 6: Enhance slide 23**

Use six small visual thumbnails to summarize the complete chain from landscape input to RL environment.

- [ ] **Step 7: Add selection rationale**

Ensure the deck explicitly states:

```text
Why FlamMap: standard landscape inputs, GIS outputs, mature fire-behavior framework.
Why FARSITE: dynamic fire growth, arrival time, weather/ignition/burn-period support.
Why JSON: human-readable nested metadata and file indexing; rasters remain in GeoTIFF.
```

### Task 3: Verify and deliver

**Files:**
- Inspect: `outputs/森林火场地图创建与项目集成_图片增强版.pptx`
- Create: `C:/Users/cameliar/AppData/Local/Temp/codex-presentations/manual-firemap-template/firemap-deck-v4/tmp/preview/*.png`
- Create: `C:/Users/cameliar/AppData/Local/Temp/codex-presentations/manual-firemap-template/firemap-deck-v4/tmp/qa/*`

- [ ] **Step 1: Verify file separation**

Confirm both the original detailed deck and the new image-enhanced deck exist.

- [ ] **Step 2: Verify slide count**

Confirm the new deck has exactly 23 slides and that the canopy-data page is absent.

- [ ] **Step 3: Run slide structure tests**

Run the presentation test utility. Expected result: no overflow or corrupted slide relationships.

- [ ] **Step 4: Inspect all rendered slides**

Review a full contact sheet and full-size dense slides for image stretching, poor crop, tiny labels, text clipping, overlap and inconsistent margins.

- [ ] **Step 5: Verify content**

Confirm the FlamMap, FARSITE and JSON selection rationales are present and that all project-data images come from the project files.

- [ ] **Step 6: Deliver**

Return only the verified image-enhanced PPTX path, state that the original detailed deck was preserved, and identify the main visual additions.
