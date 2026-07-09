---
kind: external_dependency
name: rasterio 读取 FARSITE 栅格数据
slug: rasterio
category: external_dependency
category_hints:
    - sdk_real_api
scope:
    - '**'
---

信息转换模块通过 rasterio.open 读取 scene 的 static_map 与各时间步栅格（intensity/length/speedRate/time/spread_direction/heat_per_unit_area/crown_fire），从 src.transform/src.crs/src.nodata 提取地理配准元数据。若栅格缺失则抛出 FileNotFoundError；shape 必须与 static_map 一致，否则报 shape mismatch 错误。