# Firemap Presentation Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the existing 19-slide firemap presentation to a verified 24-slide deck with detailed explanations of each project data field and its significance.

**Architecture:** Reuse the existing template-derived starter deck and Artifact Tool builder. Extend the starter to 24 inherited slides, replace the slide authoring map with the approved 24-slide structure, export a new PPTX, then render and inspect every slide.

**Tech Stack:** JavaScript ES modules, `@oai/artifact-tool`, PowerPoint `.pptx`, bundled rendering and slide QA tools.

---

### Task 1: Prepare a 24-slide inherited template

**Files:**
- Read: `C:/Users/cameliar/AppData/Local/Temp/codex-presentations/manual-firemap-template/firemap-deck-v2/tmp/template-starter.pptx`
- Create: `C:/Users/cameliar/AppData/Local/Temp/codex-presentations/manual-firemap-template/firemap-deck-v3/tmp/template-starter.pptx`
- Create: `C:/Users/cameliar/AppData/Local/Temp/codex-presentations/manual-firemap-template/firemap-deck-v3/tmp/template-frame-map.json`

- [ ] **Step 1: Create a new scratch workspace**

Create the v3 scratch folders outside the repository and initialize Artifact Tool package resolution.

- [ ] **Step 2: Build a 24-slide frame map**

Map the output cover to the inherited cover slide and map slides 2–24 to the template content-slide frame.

- [ ] **Step 3: Validate the frame map**

Run the template-plan validator. Expected result: 24 valid slide mappings and no unsupported insertion.

- [ ] **Step 4: Generate the inherited starter**

Run the starter-deck preparation script. Expected result: a 24-slide starter with the original header, divider, footer and page style.

### Task 2: Author the expanded deck

**Files:**
- Create: `C:/Users/cameliar/AppData/Local/Temp/codex-presentations/manual-firemap-template/firemap-deck-v3/tmp/build_firemap_deck.mjs`
- Create: `D:/DestopMoren/Desktop/Pyproject/Self-adaptive parameters/outputs/森林火场地图创建与项目集成_详细版.pptx`

- [ ] **Step 1: Reuse the existing builder helpers**

Reuse `setTitle`, `addText`, `addCard`, `addImage`, inherited-header cleanup and export helpers without introducing another visual system.

- [ ] **Step 2: Implement slides 1–7**

Keep the cover, project background, FlamMap, FARSITE, relationship and full workflow, then add an input-data overview separating static map, dynamic inputs and control parameters.

- [ ] **Step 3: Implement slides 8–13**

Explain:

```text
Static terrain/fuel: elevation, slope, aspect, fuel model
Static canopy: cover, height, base height, bulk density
FMS: fuel model, 1h, 10h, 100h, live herbaceous, live woody, 1000h
WXS: date/time, temperature, RH, precipitation, wind speed, wind direction, cloud cover
Ignition/barrier/burn period
FARSITE: perimeter resolution, distance resolution, timestep, acceleration, spotting, foliar moisture, crown-fire method
```

- [ ] **Step 4: Implement slides 14–17**

Explain the selected outputs in three groups:

```text
Temporal/motion: arrival time, rate of spread, spread direction
Intensity/energy: fireline intensity, flame length, heat per unit area
Fire type/geometry: crown fire activity, perimeters
```

Include the verified project note that the sample spread-direction raster is in radians and the FMS final column is 1000-hour fuel moisture.

- [ ] **Step 5: Implement slides 18–23**

Show actual directory organization, file-to-field mapping, metadata/index responsibilities, loader chain, preprocessing rules and derived fire boundary/thermal/severity features.

- [ ] **Step 6: Implement slide 24**

Summarize the end-to-end chain from FlamMap landscape data through FARSITE output to reinforcement-learning state and reward use.

- [ ] **Step 7: Export the PPTX**

Run the builder with `FIREMAP_TMP` pointing to the v3 scratch directory and `FIREMAP_OUTPUT` pointing to the detailed output file. Expected result: a 24-slide PPTX.

### Task 3: Verify content and layout

**Files:**
- Inspect: `D:/DestopMoren/Desktop/Pyproject/Self-adaptive parameters/outputs/森林火场地图创建与项目集成_详细版.pptx`
- Create: `C:/Users/cameliar/AppData/Local/Temp/codex-presentations/manual-firemap-template/firemap-deck-v3/tmp/preview/*.png`
- Create: `C:/Users/cameliar/AppData/Local/Temp/codex-presentations/manual-firemap-template/firemap-deck-v3/tmp/layout/*.json`
- Create: `C:/Users/cameliar/AppData/Local/Temp/codex-presentations/manual-firemap-template/firemap-deck-v3/tmp/qa/*`

- [ ] **Step 1: Verify slide count**

Inspect the exported deck. Expected result: exactly 24 slides.

- [ ] **Step 2: Run structural slide checks**

Run the presentation test utility. Expected result: no overflow, invalid coordinates or corrupted slide relationships.

- [ ] **Step 3: Review the contact sheet**

Inspect all 24 rendered slides for clipped text, unintended overlap, inconsistent margins, broken screenshots and unreadable footnotes.

- [ ] **Step 4: Correct defects**

Adjust only the affected slide frames or text density, regenerate the deck and repeat structural and visual checks.

- [ ] **Step 5: Deliver**

Return only the verified detailed PPTX and identify the main data-definition corrections included.
